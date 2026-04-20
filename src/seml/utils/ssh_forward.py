from __future__ import annotations

import atexit
import contextlib
import logging
import os
import secrets
import subprocess
import sys
import time
import uuid
from typing import TYPE_CHECKING, Any

from seml.document import ExperimentDoc
from seml.settings import SETTINGS
from seml.utils import Hashabledict, assert_package_installed

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

States = SETTINGS.STATES
_STOP_COMMAND = 'stop'


def retried_and_locked_ssh_port_forward(
    retries_max: int = SETTINGS.SSH_FORWARD.RETRIES_MAX,
    retries_delay: int = SETTINGS.SSH_FORWARD.RETRIES_DELAY,
    lock_file: str = SETTINGS.SSH_FORWARD.LOCK_FILE,
    lock_timeout: int = SETTINGS.SSH_FORWARD.LOCK_TIMEOUT,
    **ssh_config,
):
    """
    Attempt to establish an SSH tunnel with retries and a lock file to avoid parallel tunnel establishment.

    Parameters
    ----------
    retries_max: int
        Maximum number of retries to establish the tunnel.
    retries_delay: float
        Initial delay for exponential backoff.
    lock_file: str
        Path to the lock file.
    lock_timeout: int
        Timeout for acquiring the lock.
    ssh_config: dict
        Configuration for the SSH tunnel.

    Returns
    -------
    server: SSHTunnelForwarder
        The SSH tunnel server.
    """
    import random

    from filelock import FileLock, Timeout
    from sshtunnel import BaseSSHTunnelForwarderError, SSHTunnelForwarder

    delay = retries_delay
    error = None
    # disable SSH forward messages
    logging.getLogger('paramiko.transport').disabled = True
    for _ in range(retries_max):
        try:
            lock = FileLock(lock_file, mode=0o666, timeout=lock_timeout)
            with lock:
                server = SSHTunnelForwarder(**ssh_config)
                server.start()
                if not server.tunnel_is_up[server.local_bind_address]:
                    raise BaseSSHTunnelForwarderError()
                return server
        except Timeout as e:
            error = e
            logging.warning(f'Failed to aquire lock for ssh tunnel {lock_file}')
        except BaseSSHTunnelForwarderError as e:
            error = e
            logging.warning(f'Retry establishing ssh tunnel in {delay} s')
            # Jittered exponential retry
            time.sleep(delay)
            delay *= 2
            delay += random.uniform(0, 1)

    logging.error(f'Failed to establish ssh tunnel: {error}')
    exit(1)


def _ssh_forward_process(connection: Connection, ssh_config: dict[str, Any]):
    """
    Establish an SSH tunnel in a separate process. The process periodically checks if the tunnel is still up and
    restarts it if it is not.

    Parameters
    ----------
    connection: Connection
        Connection carrying commands from the parent process.
    ssh_config: dict
        Configuration for the SSH tunnel.
    """
    server = retried_and_locked_ssh_port_forward(**ssh_config)
    # We need to bind to the same local addresses
    server._local_binds = server.local_bind_addresses
    connection.send((str(server.local_bind_host), int(server.local_bind_port)))

    while True:
        # check if we should end the process
        try:
            command = None
            if connection.poll(SETTINGS.SSH_FORWARD.HEALTH_CHECK_INTERVAL):
                command = connection.recv()

            if command == _STOP_COMMAND:
                server.stop()
                break

            # Check for tunnel health
            server.check_tunnels()
            if not server.tunnel_is_up[server.local_bind_address]:
                logging.warning('SSH tunnel was closed unexpectedly. Restarting.')
                server.restart()
        except KeyboardInterrupt:
            server.stop()
            break
        except EOFError:
            server.stop()
            break
        except Exception as e:
            logging.error(f'Error in SSH tunnel health check:\n{e}')
            server.restart()


def _remove_unix_socket(socket_address: str):
    with contextlib.suppress(FileNotFoundError, OSError):
        os.unlink(socket_address)


def _connect_to_ssh_worker(socket_address: str, authkey: bytes, timeout: float):
    from multiprocessing.connection import Client

    deadline = time.monotonic() + max(timeout, 0)
    while True:
        try:
            return Client(socket_address, family='AF_UNIX', authkey=authkey)
        except (FileNotFoundError, ConnectionRefusedError, OSError):
            if time.monotonic() >= deadline:
                raise TimeoutError('Failed to connect to SSH forwarding worker.')
            time.sleep(0.05)


def _start_ssh_forward_subprocess(connect_timeout: float):
    socket_address = f'/tmp/seml_ssh_forward_{uuid.uuid4().hex}.sock'
    authkey = secrets.token_bytes(16)  # avoid other processes connecting to the socket
    proc = subprocess.Popen(
        [
            sys.executable,
            '-m',
            'seml.utils.ssh_forward',
            '--worker',
            socket_address,
            authkey.hex(),
        ]
    )

    try:
        connection = _connect_to_ssh_worker(socket_address, authkey, connect_timeout)
    except Exception:
        if proc.poll() is None:
            proc.terminate()
        _remove_unix_socket(socket_address)
        raise
    return proc, connection, socket_address


def _close_ssh_forward_subprocess(
    proc: subprocess.Popen[Any],
    connection: Connection | None,
    socket_address: str,
):
    try:
        if proc.poll() is None and connection is not None:
            try:
                connection.send(_STOP_COMMAND)
            except (BrokenPipeError, EOFError, OSError):
                pass
    finally:
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        if proc.poll() is None:
            proc.terminate()
        _remove_unix_socket(socket_address)


# To establish only a single connection to a remote
_forwards: dict[Hashabledict, tuple[str, int]] = {}
_forward_processes: dict[
    Hashabledict, tuple[subprocess.Popen[Any], Connection, str]
] = {}


def _get_ssh_forward(ssh_config: dict[str, Any]):
    """
    Establishes an SSH tunnel in a separate process and returns the local address of the tunnel.
    If a connection to the remote host already exists, it is reused.

    Parameters
    ----------
    ssh_config: dict
        Configuration for the SSH tunnel.

    Returns
    -------
    local_address: tuple
        Local address of the SSH tunnel.
    try_close: Callable
        Function to close the SSH tunnel.
    """
    assert_package_installed(
        'sshtunnel',
        'Opening ssh tunnel requires `sshtunnel` (e.g. `pip install sshtunnel`)',
    )
    assert_package_installed(
        'filelock',
        'Opening ssh tunnel requires `filelock` (e.g. `pip install filelock`)',
    )

    global _forwards, _forward_processes

    ssh_config = Hashabledict(ssh_config)
    if ssh_config not in _forwards:
        # Compute the maximum time we should wait
        retries_max = ssh_config.get('retries_max', SETTINGS.SSH_FORWARD.RETRIES_MAX)
        retries_delay = ssh_config.get(
            'retries_delay', SETTINGS.SSH_FORWARD.RETRIES_DELAY
        )
        max_delay = 2 ** (retries_max + 1) * retries_delay
        try:
            proc, connection, socket_address = _start_ssh_forward_subprocess(max_delay)
        except TimeoutError:
            logging.error('Failed to connect to SSH tunnel worker.')
            exit(1)

        # Send stop if we exit the program
        atexit.register(
            lambda: _close_ssh_forward_subprocess(proc, connection, socket_address)
        )

        try:
            connection.send(dict(ssh_config))
        except (BrokenPipeError, EOFError, OSError):
            logging.error('Failed to send SSH tunnel configuration to worker.')
            _close_ssh_forward_subprocess(proc, connection, socket_address)
            exit(1)

        if not connection.poll(max_delay):
            logging.error('Failed to establish SSH tunnel.')
            _close_ssh_forward_subprocess(proc, connection, socket_address)
            exit(1)

        try:
            host, port = connection.recv()
        except (EOFError, OSError, ValueError, TypeError) as e:
            logging.error(f'Failed to receive SSH tunnel worker startup output: {e}')
            _close_ssh_forward_subprocess(proc, connection, socket_address)
            exit(1)

        _forwards[ssh_config] = (str(host), int(port))
        _forward_processes[ssh_config] = (proc, connection, socket_address)
    return _forwards[ssh_config]


def _worker_main(socket_address: str, authkey_hex: str):
    from multiprocessing.connection import Listener

    try:
        authkey = bytes.fromhex(authkey_hex)
    except ValueError:
        logging.error('Invalid SSH worker authkey payload.')
        return 1

    listener = Listener(socket_address, family='AF_UNIX', authkey=authkey)
    connection: Connection | None = None
    try:
        connection = listener.accept()
        ssh_config = connection.recv()
        if not isinstance(ssh_config, dict):
            logging.error('SSH worker expects the configuration payload to be a dict.')
            return 1
        _ssh_forward_process(connection, ssh_config)
    finally:
        if connection is not None:
            connection.close()
        listener.close()
        _remove_unix_socket(socket_address)
    return 0


def get_forwarded_mongo_client(
    db_name: str, username: str, password: str, ssh_config: dict[str, Any], **kwargs
):
    """
    Establish an SSH tunnel and return a forwarded MongoDB client.
    The SSH tunnel is established in a separate process to enable continuously checking for its health.

    Parameters
    ----------
    db_name: str
        Name of the database.
    username: str
        Username for the database.
    password: str
        Password for the database.
    ssh_config: dict
        Configuration for the SSH tunnel.
    kwargs: dict
        Additional arguments for the MongoDB client.

    Returns
    -------
    client: pymongo.MongoClient
        Forwarded MongoDB client.
    """
    import pymongo

    host, port = _get_ssh_forward(ssh_config)

    client = pymongo.MongoClient[ExperimentDoc](
        host,
        int(port),
        username=username,
        password=password,
        authSource=db_name,
        **kwargs,
    )
    return client


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--worker':
        raise SystemExit(
            'Usage: python -m seml.utils.ssh_forward --worker <socket_address> <authkey_hex>'
        )
    raise SystemExit(_worker_main(sys.argv[2], sys.argv[3]))
