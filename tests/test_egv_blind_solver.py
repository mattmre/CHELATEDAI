from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile
import unittest

from egv.canonical import canonical_json, digest_bytes, digest_for
from egv.evaluation import blind_solver
from egv.evaluation.blind_solver import (
    BLIND_SOLVER_VERSION,
    compile_blind_source,
    validate_model_visible_package_bytes,
)
from egv.evaluation.learnable import LearnableCorpusV2
from egv.evaluation.sandbox import LocalTestSandbox, validate_candidate_source_contract


def _detached(value):
    return json.loads(canonical_json(value))


def _rebind(package):
    rebound = _detached(package)
    task = rebound["task"]
    task["public_spec_digest"] = digest_for(task["public_spec"])
    task["starter_source_digest"] = digest_bytes(task["starter_source"].encode("utf-8"))
    task["record_digest"] = digest_for(
        {key: value for key, value in task.items() if key != "record_digest"}
    )
    rebound["package_digest"] = digest_for(
        {key: value for key, value in rebound.items() if key != "package_digest"}
    )
    return canonical_json(rebound).encode("utf-8")


def _field_value(spec, name):
    expression = spec.get("program", spec)
    return next(field["value"] for field in expression["fields"] if field["name"] == name)


class BlindSolverTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.seed_path = cls.root / "evaluator-seed.bin"
        cls.seed_path.write_bytes(bytes(range(32)))
        cls.corpus = LearnableCorpusV2.generate(secret_seed_file=cls.seed_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def package(self, task):
        return self.corpus.model_visible_package(task.task_id)

    def package_bytes(self, task):
        return canonical_json(self.package(task)).encode("utf-8")


class TestIndependentBlindCompiler(BlindSolverTestCase):
    def test_compiler_has_no_trusted_oracle_or_private_corpus_import(self) -> None:
        module_path = Path(blind_solver.__file__)
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported_modules = []
        imported_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_modules.append(node.module or "")
                imported_names.extend(alias.name for alias in node.names)
        self.assertFalse(any(name.endswith("learnable") for name in imported_modules))
        self.assertFalse(
            {
                "LearnableCorpusV2",
                "build_reference_source",
                "solve_public_spec",
                "validate_model_visible_package",
            }
            & set(imported_names)
        )
        self.assertEqual(BLIND_SOLVER_VERSION, "egv-independent-blind-solver-v1")

    def test_canonical_package_bytes_compile_deterministically_to_pure_return(self) -> None:
        for task in self.corpus.tasks:
            package_bytes = self.package_bytes(task)
            validated = validate_model_visible_package_bytes(package_bytes)
            first = compile_blind_source(package_bytes)
            second = compile_blind_source(bytes(package_bytes))
            self.assertEqual(first, second)
            self.assertEqual(validated["task"]["task_id"], task.task_id)
            self.assertIsNone(validate_candidate_source_contract(first))
            for case in task.hidden_cases:
                self.assertNotIn(case.input_value["case_nonce"], package_bytes.decode("utf-8"))
                self.assertNotIn(case.input_value["case_nonce"], first.decode("utf-8"))

    def test_compiler_accepts_only_serialized_bytes(self) -> None:
        task = self.corpus.tasks[0]
        with self.assertRaisesRegex(TypeError, "must be bytes"):
            compile_blind_source(self.package(task))
        with self.assertRaisesRegex(TypeError, "must be bytes"):
            compile_blind_source(self.package_bytes(task).decode("utf-8"))

    def test_generated_sources_pass_every_frozen_case_via_test_harness(self) -> None:
        sandbox = LocalTestSandbox(self.root / "blind-solver-harness")
        executed = 0
        for task in self.corpus.tasks:
            # The compiler receives the public package before the evaluator-side
            # test harness opens its hidden-case loop.
            source = compile_blind_source(self.package_bytes(task))
            source_digest = digest_bytes(source)
            for case in task.hidden_cases:
                result = sandbox.execute(
                    source,
                    case.input_value,
                    artifact_digest=source_digest,
                    candidate_id="blind-{}-{}".format(task.task_id, case.ordinal),
                )
                self.assertEqual(result.diagnostic_enum, "PASS", case.case_id)
                self.assertEqual(json.loads(result.output_bytes.decode("utf-8")), case.expected_output)
                executed += 1
        self.assertEqual(executed, len(self.corpus.tasks) * 2)

    def test_wrong_runtime_scope_fails_closed_without_using_hidden_material(self) -> None:
        task = self.corpus.tasks[0]
        source = compile_blind_source(self.package_bytes(task))
        sandbox = LocalTestSandbox(self.root / "wrong-scope-harness")
        opaque_input = _detached(task.hidden_cases[0].input_value)
        opaque_input["task_scope"] = "wrong-scope"
        result = sandbox.execute(
            source,
            opaque_input,
            artifact_digest=digest_bytes(source),
            candidate_id="wrong-scope",
        )
        self.assertEqual(result.diagnostic_enum, "PASS")
        self.assertEqual(json.loads(result.output_bytes.decode("utf-8")), {"error": "scope-mismatch"})


class TestBlindSolverAdversarialValidation(BlindSolverTestCase):
    def test_package_and_nested_records_reject_extra_fields(self) -> None:
        task = self.corpus.tasks[0]

        extra_package = self.package(task)
        extra_package["hidden_cases"] = []
        with self.assertRaisesRegex(ValueError, "package is not a closed record"):
            compile_blind_source(_rebind(extra_package))

        extra_task = self.package(task)
        extra_task["task"]["private_seed"] = "forbidden"
        with self.assertRaisesRegex(ValueError, "task is not a closed record"):
            compile_blind_source(_rebind(extra_task))

        extra_spec = self.package(task)
        extra_spec["task"]["public_spec"]["debug"] = True
        with self.assertRaisesRegex(ValueError, "specification is not a closed record"):
            compile_blind_source(_rebind(extra_spec))

        extra_node = self.package(task)
        _field_value(extra_node["task"]["public_spec"], "value")["debug"] = True
        with self.assertRaisesRegex(ValueError, "expression node is not closed"):
            compile_blind_source(_rebind(extra_node))

    def test_unknown_operator_is_rejected_after_full_digest_rebinding(self) -> None:
        task = self.corpus.split("train")[0]
        package = self.package(task)
        _field_value(package["task"]["public_spec"], "value")["op"] = "multiply_integers"
        with self.assertRaisesRegex(ValueError, "unknown operator"):
            compile_blind_source(_rebind(package))

    def test_invalid_input_path_is_rejected_after_full_digest_rebinding(self) -> None:
        task = self.corpus.tasks[0]
        package = self.package(task)
        _field_value(package["task"]["public_spec"], "case_nonce")["path"] = ["evaluator_secret"]
        with self.assertRaisesRegex(ValueError, "path is unavailable"):
            compile_blind_source(_rebind(package))

    def test_unbound_and_invalid_variable_paths_are_rejected(self) -> None:
        task = next(task for task in self.corpus.tasks if task.family_id == "DATA_TRANSFORM")

        unbound = self.package(task)
        rows = _field_value(unbound["task"]["public_spec"], "rows")
        key_value = _field_value(rows["body"], "key")
        key_value["name"] = "not-bound"
        with self.assertRaisesRegex(ValueError, "variable is unavailable"):
            compile_blind_source(_rebind(unbound))

        invalid_path = self.package(task)
        rows = _field_value(invalid_path["task"]["public_spec"], "rows")
        key_value = _field_value(rows["body"], "key")
        key_value["path"] = ["not-a-row-field"]
        with self.assertRaisesRegex(ValueError, "path is unavailable"):
            compile_blind_source(_rebind(invalid_path))

    def test_invalid_input_payload_and_output_contracts_are_rejected(self) -> None:
        task = self.corpus.tasks[0]

        input_contract = self.package(task)
        input_contract["task"]["public_spec"]["input_contract"]["case_nonce"]["length"] = 19
        with self.assertRaisesRegex(ValueError, "input contract differs"):
            compile_blind_source(_rebind(input_contract))

        payload_contract = self.package(task)
        payload_contract["task"]["public_spec"]["payload_contract"] = {"type": "text"}
        with self.assertRaisesRegex(ValueError, "payload contract differs"):
            compile_blind_source(_rebind(payload_contract))

        output_contract = self.package(task)
        output_contract["task"]["public_spec"]["output_contract"] = ["value"]
        with self.assertRaisesRegex(ValueError, "output contract differs"):
            compile_blind_source(_rebind(output_contract))

    def test_scope_identity_and_starter_contract_reject_rebound_substitution(self) -> None:
        task = self.corpus.tasks[0]

        scope = self.package(task)
        scope["task"]["public_spec"]["task_scope"] = "different-task"
        with self.assertRaisesRegex(ValueError, "scope differs"):
            compile_blind_source(_rebind(scope))

        starter = self.package(task)
        starter["task"]["starter_source"] = "def main(value):\n    return {'forged': True}\n"
        with self.assertRaisesRegex(ValueError, "starter source differs"):
            compile_blind_source(_rebind(starter))

    def test_program_output_fields_must_match_the_closed_output_contract(self) -> None:
        task = self.corpus.tasks[0]
        package = self.package(task)
        package["task"]["public_spec"]["program"]["fields"][0]["name"] = "renamed_nonce"
        with self.assertRaisesRegex(ValueError, "program fields differ"):
            compile_blind_source(_rebind(package))

    def test_package_digest_tamper_and_noncanonical_encoding_are_rejected(self) -> None:
        task = self.corpus.tasks[0]
        package = self.package(task)
        package["package_digest"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "package digest mismatch"):
            compile_blind_source(canonical_json(package).encode("utf-8"))

        with self.assertRaisesRegex(ValueError, "not canonical JSON"):
            compile_blind_source(self.package_bytes(task) + b"\n")

    def test_duplicate_json_keys_are_rejected_before_compilation(self) -> None:
        duplicate = b'{"schema_version":"a","schema_version":"b"}'
        with self.assertRaisesRegex(ValueError, "not valid canonical JSON"):
            compile_blind_source(duplicate)


if __name__ == "__main__":
    unittest.main()
