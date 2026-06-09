import unittest
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

from seml.commands.start import start_sbatch_job
from seml.document import ExperimentDoc, SBatchOptions
from seml.experiment.command import value_to_string


class TestValueToString(unittest.TestCase):
    def test_literal(self):
        vals = [True, False, None]
        for val in vals:
            str_json = value_to_string(val, use_json=True)
            str_repr = value_to_string(val, use_json=False)
            self.assertEqual(str_json, str_repr)

    def test_list(self):
        vals = [True, False, None]
        lists = [
            [4, "test"],
            ["test", {"a": 5}],
            [[5, 3], {6.5: 2.3}],
        ]
        res_json = [
            ['[{val}, 4, "test"]', '[4, {val}, "test"]', '[4, "test", {val}]'],
            [
                '[{val}, "test", {{"a": 5}}]',
                '["test", {val}, {{"a": 5}}]',
                '["test", {{"a": 5}}, {val}]',
            ],
            [
                '[{val}, [5, 3], {{"6.5": 2.3}}]',
                '[[5, 3], {val}, {{"6.5": 2.3}}]',
                '[[5, 3], {{"6.5": 2.3}}, {val}]',
            ],
        ]
        res_repr = [
            ["[{val}, 4, 'test']", "[4, {val}, 'test']", "[4, 'test', {val}]"],
            [
                "[{val}, 'test', {{'a': 5}}]",
                "['test', {val}, {{'a': 5}}]",
                "['test', {{'a': 5}}, {val}]",
            ],
            [
                "[{val}, [5, 3], {{6.5: 2.3}}]",
                "[[5, 3], {val}, {{6.5: 2.3}}]",
                "[[5, 3], {{6.5: 2.3}}, {val}]",
            ],
        ]
        for ilist, raw_list in enumerate(lists):
            for pos in range(3):
                for val in vals:
                    test_list = raw_list.copy()
                    test_list.insert(pos, val)
                    str_json = value_to_string(test_list, use_json=True)
                    str_repr = value_to_string(test_list, use_json=False)
                    self.assertEqual(str_json, res_json[ilist][pos].format(val=val))
                    self.assertEqual(str_repr, res_repr[ilist][pos].format(val=val))

    def test_dict(self):
        vals = [True, False, None]
        dicts = [
            {1: "test"},
            {"test": {"a": 5}},
            {"a": [6.5, 2.3]},
        ]
        keys = [3, "b", "nest", 4.3]
        res_json = [
            [
                '{{"1": "test", "3": {val}}}',
                '{{"1": "test", "b": {val}}}',
                '{{"1": "test", "nest": {val}}}',
                '{{"1": "test", "4.3": {val}}}',
            ],
            [
                '{{"test": {{"a": 5}}, "3": {val}}}',
                '{{"test": {{"a": 5}}, "b": {val}}}',
                '{{"test": {{"a": 5}}, "nest": {val}}}',
                '{{"test": {{"a": 5}}, "4.3": {val}}}',
            ],
            [
                '{{"a": [6.5, 2.3], "3": {val}}}',
                '{{"a": [6.5, 2.3], "b": {val}}}',
                '{{"a": [6.5, 2.3], "nest": {val}}}',
                '{{"a": [6.5, 2.3], "4.3": {val}}}',
            ],
        ]
        res_repr = [
            [
                "{{1: 'test', 3: {val}}}",
                "{{1: 'test', 'b': {val}}}",
                "{{1: 'test', 'nest': {val}}}",
                "{{1: 'test', 4.3: {val}}}",
            ],
            [
                "{{'test': {{'a': 5}}, 3: {val}}}",
                "{{'test': {{'a': 5}}, 'b': {val}}}",
                "{{'test': {{'a': 5}}, 'nest': {val}}}",
                "{{'test': {{'a': 5}}, 4.3: {val}}}",
            ],
            [
                "{{'a': [6.5, 2.3], 3: {val}}}",
                "{{'a': [6.5, 2.3], 'b': {val}}}",
                "{{'a': [6.5, 2.3], 'nest': {val}}}",
                "{{'a': [6.5, 2.3], 4.3: {val}}}",
            ],
        ]
        for idict, raw_dict in enumerate(dicts):
            for ikey, key in enumerate(keys):
                for val in vals:
                    test_dict = raw_dict.copy()
                    test_dict[key] = val
                    str_json = value_to_string(test_dict, use_json=True)
                    str_repr = value_to_string(test_dict, use_json=False)
                    self.assertEqual(str_json, res_json[idict][ikey].format(val=val))
                    self.assertEqual(str_repr, res_repr[idict][ikey].format(val=val))


class TestStartSbatchJobMaybeSrun(unittest.TestCase):
    def _rendered_srun_prefix(
        self, sbatch_options: SBatchOptions, experiments_per_job: int = 1
    ) -> str:
        captured: dict[str, str] = {}

        def fake_subprocess_run(command, shell, check, capture_output, env):
            self.assertTrue(command.startswith("sbatch "))
            script_path = command.split(" ", maxsplit=1)[1]
            captured["script"] = Path(script_path).read_text()
            result = Mock()
            result.stdout = b"Submitted batch job 123"
            return result

        collection = Mock()
        collection.name = "test_collection"
        exp = cast(
            ExperimentDoc, {"_id": 1, "batch_id": 0, "seml": {"working_dir": "."}}
        )

        with (
            patch(
                "seml.commands.start.load_text_resource", return_value="{maybe_srun}"
            ),
            patch(
                "seml.commands.start.subprocess.run", side_effect=fake_subprocess_run
            ),
        ):
            start_sbatch_job(
                collection=collection,
                exp_array=cast(list[ExperimentDoc], [exp]),
                slurm_options_id=0,
                sbatch_options=sbatch_options,
                unobserved=True,
                output_path=".",
                experiments_per_job=experiments_per_job,
            )

        return captured["script"]

    def test_single_experiment_without_task_count_uses_ntasks_1(self):
        rendered = self._rendered_srun_prefix({"nodes": 1})
        self.assertEqual(rendered, "srun --ntasks=1 ")

    def test_single_experiment_with_task_count_keeps_plain_srun(self):
        rendered = self._rendered_srun_prefix({"ntasks": 1, "nodes": 1})
        self.assertEqual(rendered, "srun ")

    def test_multiple_experiments_per_job_omits_srun_prefix(self):
        rendered = self._rendered_srun_prefix({"nodes": 1}, experiments_per_job=2)
        self.assertEqual(rendered, "")
