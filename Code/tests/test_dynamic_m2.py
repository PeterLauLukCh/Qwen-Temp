"""Tests for the M2 ANDES transient-stability layer."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from gridmind_mini import (  # noqa: E402
    DynamicSimulationError,
    ToolRegistry,
    ToolRegistryError,
    TransientStabilityRunner,
    ieee118_public_case_source_metadata,
    list_dynamic_cases,
    resolve_ieee118_public_dynamic_files,
    resolve_dynamic_case,
    validate_ieee118_public_dynamic_data,
)


class FakeFrame:
    def __init__(self, index, columns, values) -> None:
        self.index = index
        self.columns = columns
        self.values = values


class FakeVar:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeDynamicModel:
    def __init__(self) -> None:
        self.delta = FakeVar("delta")
        self.omega = FakeVar("omega")


class FakeBus:
    def __init__(self) -> None:
        self.idx = type("Idx", (), {"v": [1, 2, 5]})()
        self.Vn = type("Vn", (), {"v": [138.0, 138.0, 230.0]})()
        self.v = FakeVar("v")


class FakeRoutine:
    def __init__(self, result=True) -> None:
        self.result = result
        self.ran = False

    def run(self):
        self.ran = True
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeTDS(FakeRoutine):
    def __init__(self, result=True) -> None:
        super().__init__(result=result)
        self.config = type("Config", (), {"tf": None})()

    def get_timeseries(self, var):
        if var.name == "delta":
            return FakeFrame(
                [0.0, 1.0, 5.0],
                ["1", "2"],
                [[0.0, 0.0], [0.2, 0.1], [0.3, 0.2]],
            )
        if var.name == "omega":
            return FakeFrame(
                [0.0, 1.0, 5.0],
                ["1", "2"],
                [[1.0, 1.0], [1.02, 0.99], [1.0, 1.0]],
            )
        if var.name == "v":
            return FakeFrame(
                [0.0, 1.0, 5.0],
                ["1", "2"],
                [[1.0, 1.0], [0.8, 0.95], [1.0, 1.0]],
            )
        raise KeyError(var.name)


class FakeToggle:
    def __init__(self) -> None:
        self.idx = type("Idx", (), {"v": [1]})()
        self.set_calls = []

    def set(self, field, idx, value) -> None:
        self.set_calls.append((field, idx, value))


class FakeSystem:
    def __init__(self, *, pflow_result=True, tds_result=True) -> None:
        self.config = type("Config", (), {"mva": 100.0})()
        self.GENROU = FakeDynamicModel()
        self.GENCLS = None
        self.Bus = FakeBus()
        self.Toggle = FakeToggle()
        self.PFlow = FakeRoutine(result=pflow_result)
        self.TDS = FakeTDS(result=tds_result)
        self.add_calls = []
        self.setup_ran = False

    def add(self, model_name, param_dict=None, **kwargs) -> None:
        params = dict(param_dict or {})
        params.update(kwargs)
        if model_name == "Toggle" and "model" in kwargs:
            raise TypeError("add() got multiple values for argument 'model'")
        self.add_calls.append((model_name, params))

    def setup(self) -> None:
        self.setup_ran = True


class FakeAndes:
    def __init__(self, system: FakeSystem | None = None) -> None:
        self.system = system or FakeSystem()
        self.load_calls = []

    def get_case(self, locator):
        return f"/fake/{locator}"

    def load(self, case_file, addfile=None, setup=False):
        self.load_calls.append(
            {"case_file": case_file, "addfile": addfile, "setup": setup}
        )
        return self.system


class DynamicM2Test(unittest.TestCase):
    def test_dynamic_case_aliases_resolve(self) -> None:
        self.assertEqual(resolve_dynamic_case("kundur").case_id, "kundur_full")
        self.assertEqual(resolve_dynamic_case("ieee14").case_id, "ieee14_dynamic")

    def test_list_dynamic_cases_is_metadata_only(self) -> None:
        result = list_dynamic_cases()

        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "list_dynamic_cases")
        self.assertIn("andes_available", result)
        self.assertEqual(
            {case["case_id"] for case in result["cases"]},
            {"kundur_full", "ieee14_dynamic", "ieee118_public_dynamic"},
        )

    def test_missing_andes_dependency_returns_structured_unavailable(self) -> None:
        with patch("gridmind_mini.dynamic.importlib.import_module", side_effect=ImportError):
            result = TransientStabilityRunner().run(
                case_path="kundur",
                disturbance={
                    "type": "bus_fault",
                    "bus": 5,
                    "fault_start_s": 1.0,
                    "clearing_time_s": 1.1,
                },
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_type"], "dependency_unavailable")
        self.assertIn("andes_unavailable", result["reason_codes"])

    def test_bus_fault_runs_pflow_tds_and_extracts_metrics(self) -> None:
        fake = FakeAndes()

        result = TransientStabilityRunner(andes_module=fake).run(
            case_path="kundur_full",
            disturbance={
                "type": "bus_fault",
                "bus": 5,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            simulation_time_s=5.0,
            max_samples=2,
        )

        self.assertTrue(result["ok"])
        self.assertTrue(fake.system.setup_ran)
        self.assertTrue(fake.system.PFlow.ran)
        self.assertTrue(fake.system.TDS.ran)
        self.assertEqual(fake.system.TDS.config.tf, 5.0)
        self.assertEqual(fake.system.add_calls[0][0], "Fault")
        self.assertEqual(fake.system.add_calls[0][1]["bus"], 5)
        self.assertEqual(fake.system.add_calls[0][1]["tf"], 1.0)
        self.assertEqual(fake.system.add_calls[0][1]["tc"], 1.1)
        self.assertEqual(result["stability"]["status"], "pass")
        self.assertEqual(result["metrics"]["max_angle_spread_rad"], 0.1)
        self.assertEqual(result["metrics"]["max_speed_deviation_pu"], 0.02)
        self.assertEqual(result["metrics"]["min_voltage_pu"], 0.8)
        self.assertEqual(len(result["trajectories"]["time_s"]), 2)
        self.assertFalse(result["dynamic_interconnection_modeling"])
        self.assertFalse(result["connection_application"]["requested"])

    def test_static_load_connection_is_added_before_fault(self) -> None:
        fake = FakeAndes()

        result = TransientStabilityRunner(andes_module=fake).run(
            case_path="ieee14_dynamic",
            disturbance={
                "type": "bus_fault",
                "bus": 5,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            connection={
                "bus": 5,
                "p_mw": 50.0,
                "q_mvar": 10.0,
                "connection_type": "load",
                "is_ibr": False,
                "name": "New Load",
            },
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["dynamic_interconnection_modeling"])
        self.assertEqual(fake.system.add_calls[0][0], "PQ")
        self.assertEqual(fake.system.add_calls[1][0], "Fault")
        pq_params = fake.system.add_calls[0][1]
        self.assertEqual(pq_params["bus"], 5)
        self.assertEqual(pq_params["idx"], "New_Load")
        self.assertEqual(pq_params["Vn"], 230.0)
        self.assertEqual(pq_params["p0"], 0.5)
        self.assertEqual(pq_params["q0"], 0.1)
        self.assertEqual(result["connection_application"]["mode"], "static_load")
        self.assertEqual(result["connection_application"]["p0_pu"], 0.5)
        self.assertIn("m2_connection_model_is_static_pq", result["limitations"])

    def test_generation_like_connection_is_negative_pq_injection(self) -> None:
        fake = FakeAndes()

        result = TransientStabilityRunner(andes_module=fake).run(
            case_path="ieee14",
            disturbance={
                "type": "bus_fault",
                "bus": 5,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            connection={
                "bus": 5,
                "p_mw": 25.0,
                "q_mvar": 5.0,
                "connection_type": "solar",
                "is_ibr": True,
            },
        )

        self.assertTrue(result["ok"])
        pq_params = fake.system.add_calls[0][1]
        self.assertEqual(pq_params["p0"], -0.25)
        self.assertEqual(pq_params["q0"], -0.05)
        self.assertEqual(
            result["connection_application"]["mode"],
            "static_generation_as_negative_pq_load",
        )
        self.assertIn("m2_v1_does_not_model_detailed_ibr_controls", result["limitations"])

    def test_invalid_connection_bus_returns_structured_error(self) -> None:
        fake = FakeAndes()

        result = TransientStabilityRunner(andes_module=fake).run(
            case_path="ieee14",
            disturbance={
                "type": "bus_fault",
                "bus": 5,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            connection={
                "bus": 99,
                "p_mw": 5.0,
                "connection_type": "load",
                "is_ibr": False,
            },
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_type"], "connection_application_error")
        self.assertEqual(result["routine"], "connection_modeling")
        self.assertIn("connection_application_error", result["reason_codes"])
        self.assertFalse(fake.system.PFlow.ran)
        self.assertEqual(fake.system.add_calls, [])

    def test_unsupported_connection_type_returns_structured_error(self) -> None:
        result = TransientStabilityRunner(andes_module=FakeAndes()).run(
            case_path="ieee14",
            disturbance={
                "type": "bus_fault",
                "bus": 5,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
            connection={
                "bus": 5,
                "p_mw": 5.0,
                "connection_type": "unknown_resource",
                "is_ibr": False,
            },
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_type"], "connection_input_error")
        self.assertIn("connection_input_error", result["reason_codes"])

    def test_line_trip_uses_toggle_model(self) -> None:
        fake = FakeAndes()

        result = TransientStabilityRunner(andes_module=fake).run(
            case_path="kundur",
            disturbance={
                "type": "line_trip",
                "model": "Line",
                "device": "Line_5",
                "trip_time_s": 1.0,
                "reclose_time_s": 2.0,
            },
        )

        self.assertTrue(result["ok"])
        self.assertEqual(
            fake.system.add_calls[:2],
            [
                ("Toggle", {"model": "Line", "dev": "Line_5", "t": 1.0}),
                ("Toggle", {"model": "Line", "dev": "Line_5", "t": 2.0}),
            ],
        )
        self.assertEqual(result["disturbance"]["type"], "line_trip")

    def test_pflow_failure_returns_structured_error(self) -> None:
        fake = FakeAndes(FakeSystem(pflow_result=False))

        result = TransientStabilityRunner(andes_module=fake).run(
            case_path="kundur",
            disturbance={
                "type": "bus_fault",
                "bus": 5,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["routine"], "PFlow")
        self.assertIn("pflow_not_converged", result["reason_codes"])

    def test_tds_exception_returns_structured_error(self) -> None:
        fake = FakeAndes(FakeSystem(tds_result=RuntimeError("tds failed")))

        result = TransientStabilityRunner(andes_module=fake).run(
            case_path="kundur",
            disturbance={
                "type": "bus_fault",
                "bus": 5,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertIn("tds failed", result["message"])

    def test_ieee118_missing_public_data_returns_case_data_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing_bundle = Path(tmp) / "missing_bundle"
            with patch(
                "gridmind_mini.public_cases.IEEE118_BUNDLED_CASE_DIR",
                missing_bundle,
            ):
                with patch(
                    "gridmind_mini.public_cases.importlib.import_module",
                    side_effect=ImportError,
                ):
                    result = TransientStabilityRunner(andes_module=FakeAndes()).run(
                        case_path="ieee118",
                        disturbance={
                            "type": "bus_fault",
                            "bus": 10,
                            "fault_start_s": 1.0,
                            "clearing_time_s": 1.1,
                        },
                    )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_type"], "case_data_unavailable")
        self.assertIn("case_data_unavailable", result["reason_codes"])

    def test_ieee118_public_dynamic_runs_when_raw_dyr_resolve(self) -> None:
        class FakeCase:
            def __init__(self, raw, dyr) -> None:
                self.raw = raw
                self.dyr = dyr

        class FakePowerfulCases:
            def __init__(self, raw, dyr) -> None:
                self.case = FakeCase(raw, dyr)

            def load(self, name):
                self.loaded_name = name
                return self.case

            def file(self, case, fmt, variant=None):
                if fmt == "psse_dyr" and variant == "genrou":
                    return case.dyr
                return None

        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "ieee118.raw"
            dyr = Path(tmp) / "ieee118.dyr"
            missing_bundle = Path(tmp) / "missing_bundle"
            raw.write_text("raw", encoding="utf-8")
            dyr.write_text("dyr", encoding="utf-8")
            fake_pcase = FakePowerfulCases(str(raw), str(dyr))
            with patch(
                "gridmind_mini.public_cases.IEEE118_BUNDLED_CASE_DIR",
                missing_bundle,
            ):
                with patch(
                    "gridmind_mini.public_cases.importlib.import_module",
                    return_value=fake_pcase,
                ):
                    fake_andes = FakeAndes()
                    result = TransientStabilityRunner(andes_module=fake_andes).run(
                        case_path="ieee118_dynamic",
                        disturbance={
                            "type": "bus_fault",
                            "bus": 5,
                            "fault_start_s": 1.0,
                            "clearing_time_s": 1.1,
                        },
                    )

        self.assertTrue(result["ok"])
        self.assertEqual(result["case_info"]["case_id"], "ieee118_public_dynamic")
        self.assertEqual(result["case_info"]["case_source"], "powerfulcases_ieee118_public_dynamic")
        self.assertEqual(fake_andes.load_calls[0]["case_file"], str(raw))
        self.assertEqual(fake_andes.load_calls[0]["addfile"], str(dyr))

    def test_ieee118_public_dynamic_uses_bundled_data_by_default(self) -> None:
        fake_andes = FakeAndes()

        result = TransientStabilityRunner(andes_module=fake_andes).run(
            case_path="ieee118_dynamic",
            disturbance={
                "type": "bus_fault",
                "bus": 5,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["case_info"]["case_source"], "bundled_public_ieee118_raw_dyr")
        self.assertTrue(fake_andes.load_calls[0]["case_file"].endswith("ieee118.raw"))
        self.assertTrue(fake_andes.load_calls[0]["addfile"].endswith("ieee118.dyr"))

    def test_ieee118_public_dynamic_preflight_reports_available_data(self) -> None:
        class FakeCase:
            def __init__(self, raw, dyr) -> None:
                self.raw = raw
                self.dyr = dyr

        class FakePowerfulCases:
            def __init__(self, raw, dyr) -> None:
                self.case = FakeCase(raw, dyr)

            def load(self, name):
                return self.case

            def file(self, case, fmt, variant=None):
                if fmt == "psse_dyr" and variant == "genrou":
                    return case.dyr
                return None

            def formats(self, case):
                return ["psse_raw", "psse_dyr"]

            def variants(self, case, fmt):
                return ["genrou"] if fmt == "psse_dyr" else []

        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "ieee118.raw"
            dyr = Path(tmp) / "ieee118.dyr"
            raw.write_text("raw", encoding="utf-8")
            dyr.write_text("dyr", encoding="utf-8")

            result = validate_ieee118_public_dynamic_data(
                FakePowerfulCases(str(raw), str(dyr))
            )

        self.assertTrue(result["ok"])
        self.assertTrue(result["raw_available"])
        self.assertTrue(result["dyr_available"])
        self.assertEqual(result["formats"], ["psse_raw", "psse_dyr"])
        self.assertEqual(result["metadata"]["raw_path"], str(raw))
        self.assertEqual(result["metadata"]["dyr_path"], str(dyr))

    def test_ieee118_local_override_preflight_reports_available_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "ieee118.raw"
            dyr = Path(tmp) / "ieee118.dyr"
            raw.write_text("raw", encoding="utf-8")
            dyr.write_text("dyr", encoding="utf-8")

            with patch.dict(
                "os.environ",
                {
                    "GRIDMIND_IEEE118_RAW_PATH": str(raw),
                    "GRIDMIND_IEEE118_DYR_PATH": str(dyr),
                },
                clear=True,
            ):
                with patch(
                    "gridmind_mini.public_cases.importlib.import_module",
                    side_effect=AssertionError("powerfulcases should not be imported"),
                ):
                    result = validate_ieee118_public_dynamic_data()
                    files = resolve_ieee118_public_dynamic_files()

        self.assertTrue(result["ok"])
        self.assertEqual(result["source"], "local_ieee118_raw_dyr_override")
        self.assertTrue(result["raw_available"])
        self.assertTrue(result["dyr_available"])
        self.assertEqual(result["metadata"]["raw_path"], str(raw))
        self.assertEqual(result["metadata"]["dyr_path"], str(dyr))
        self.assertEqual(files.source, "local_ieee118_raw_dyr_override")
        self.assertEqual(files.raw_path, str(raw))
        self.assertEqual(files.dyr_path, str(dyr))

    def test_ieee118_local_override_case_dir_resolves_common_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "IEEE_118_bus.raw"
            dyr = Path(tmp) / "IEEE_118_bus.dyr"
            raw.write_text("raw", encoding="utf-8")
            dyr.write_text("dyr", encoding="utf-8")

            with patch.dict(
                "os.environ",
                {"GRIDMIND_IEEE118_CASE_DIR": str(tmp)},
                clear=True,
            ):
                result = validate_ieee118_public_dynamic_data()

        self.assertTrue(result["ok"])
        self.assertEqual(result["metadata"]["raw_path"], str(raw))
        self.assertEqual(result["metadata"]["dyr_path"], str(dyr))

    def test_ieee118_local_override_missing_dyr_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "ieee118.raw"
            raw.write_text("raw", encoding="utf-8")

            with patch.dict(
                "os.environ",
                {"GRIDMIND_IEEE118_RAW_PATH": str(raw)},
                clear=True,
            ):
                result = validate_ieee118_public_dynamic_data()

        self.assertFalse(result["ok"])
        self.assertTrue(result["raw_available"])
        self.assertFalse(result["dyr_available"])
        self.assertEqual(result["source"], "local_ieee118_raw_dyr_override")
        self.assertEqual(result["error_type"], "dynamic_data_unavailable")

    def test_ieee118_source_metadata_reports_failed_local_override_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "ieee118.raw"
            raw.write_text("raw", encoding="utf-8")

            with patch.dict(
                "os.environ",
                {"GRIDMIND_IEEE118_RAW_PATH": str(raw)},
                clear=True,
            ):
                result = ieee118_public_case_source_metadata()

        self.assertFalse(result["available"])
        self.assertEqual(result["source"], "local_ieee118_raw_dyr_override")
        self.assertEqual(result["error_type"], "dynamic_data_unavailable")

    def test_ieee118_public_dynamic_preflight_reports_missing_dyr(self) -> None:
        class FakeCase:
            def __init__(self, raw) -> None:
                self.raw = raw
                self.dyr = None

        class FakePowerfulCases:
            def __init__(self, raw) -> None:
                self.case = FakeCase(raw)

            def load(self, name):
                return self.case

            def file(self, case, fmt, variant=None):
                return None

            def formats(self, case):
                return ["matpower", "psse_raw"]

            def variants(self, case, fmt):
                return []

        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "ieee118.raw"
            raw.write_text("raw", encoding="utf-8")

            result = validate_ieee118_public_dynamic_data(FakePowerfulCases(str(raw)))

        self.assertFalse(result["ok"])
        self.assertTrue(result["raw_available"])
        self.assertFalse(result["dyr_available"])
        self.assertEqual(result["error_type"], "dynamic_data_unavailable")
        self.assertEqual(result["formats"], ["matpower", "psse_raw"])

    def test_ieee118_public_dynamic_validation_error_is_explicit(self) -> None:
        class FakeCase:
            def __init__(self, raw, dyr) -> None:
                self.raw = raw
                self.dyr = dyr

        class FakePowerfulCases:
            def __init__(self, raw, dyr) -> None:
                self.case = FakeCase(raw, dyr)

            def load(self, name):
                return self.case

            def file(self, case, fmt, variant=None):
                return case.dyr if fmt == "psse_dyr" else None

        class FailingAndes(FakeAndes):
            def load(self, case_file, addfile=None, setup=False):
                raise RuntimeError("dyr parser failed")

        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "ieee118.raw"
            dyr = Path(tmp) / "ieee118.dyr"
            raw.write_text("raw", encoding="utf-8")
            dyr.write_text("dyr", encoding="utf-8")
            with patch(
                "gridmind_mini.public_cases.importlib.import_module",
                return_value=FakePowerfulCases(str(raw), str(dyr)),
            ):
                result = TransientStabilityRunner(andes_module=FailingAndes()).run(
                    case_path="ieee118",
                    disturbance={
                        "type": "bus_fault",
                        "bus": 10,
                        "fault_start_s": 1.0,
                        "clearing_time_s": 1.1,
                    },
                )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_type"], "ieee118_dynamic_validation_failed")
        self.assertIn("dyr parser failed", result["message"])

    def test_unsupported_unknown_case_returns_unsupported_case(self) -> None:
        result = TransientStabilityRunner(andes_module=FakeAndes()).run(
            case_path="ieee999",
            disturbance={
                "type": "bus_fault",
                "bus": 10,
                "fault_start_s": 1.0,
                "clearing_time_s": 1.1,
            },
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_type"], "unsupported_dynamic_case")

    def test_missing_bus_fault_argument_is_rejected(self) -> None:
        with self.assertRaisesRegex(DynamicSimulationError, "clearing_time_s"):
            TransientStabilityRunner(andes_module=FakeAndes()).run(
                case_path="kundur",
                disturbance={
                    "type": "bus_fault",
                    "bus": 5,
                    "fault_start_s": 1.0,
                },
            )

    def test_tool_registry_exposes_dynamic_tools(self) -> None:
        registry = ToolRegistry()
        names = {tool["name"] for tool in registry.list_tools()["tools"]}

        self.assertIn("list_dynamic_cases", names)
        self.assertIn("run_transient_stability", names)

    def test_registry_rejects_invalid_transient_arguments(self) -> None:
        registry = ToolRegistry()

        with self.assertRaisesRegex(ToolRegistryError, "clearing_time_s"):
            registry.call_tool(
                "run_transient_stability",
                {
                    "case_path": "kundur",
                    "disturbance": {
                        "type": "bus_fault",
                        "bus": 5,
                        "fault_start_s": 1.0,
                    },
                },
            )

    def test_registry_returns_ieee118_case_data_error_when_public_data_missing(self) -> None:
        registry = ToolRegistry()

        def fake_import(name):
            if name == "andes":
                return FakeAndes()
            if name == "powerfulcases":
                raise ImportError
            raise ImportError

        with tempfile.TemporaryDirectory() as tmp:
            missing_bundle = Path(tmp) / "missing_bundle"
            with patch(
                "gridmind_mini.public_cases.IEEE118_BUNDLED_CASE_DIR",
                missing_bundle,
            ):
                with patch(
                    "gridmind_mini.dynamic.importlib.import_module",
                    side_effect=fake_import,
                ):
                    result = registry.call_tool(
                        "run_transient_stability",
                        {
                            "case_path": "ieee118",
                            "disturbance": {
                                "type": "bus_fault",
                                "bus": 10,
                                "fault_start_s": 1.0,
                                "clearing_time_s": 1.1,
                            },
                        },
                    )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_type"], "case_data_unavailable")

    def test_registry_accepts_optional_transient_connection(self) -> None:
        registry = ToolRegistry()

        with patch("gridmind_mini.dynamic.importlib.import_module", side_effect=ImportError):
            result = registry.call_tool(
                "run_transient_stability",
                {
                    "case_path": "kundur",
                    "disturbance": {
                        "type": "bus_fault",
                        "bus": 5,
                        "fault_start_s": 1.0,
                        "clearing_time_s": 1.1,
                    },
                    "connection": {
                        "bus": 5,
                        "p_mw": 10.0,
                        "connection_type": "generator",
                        "is_ibr": False,
                    },
                },
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_type"], "dependency_unavailable")
        self.assertEqual(result["connection_model"]["connection_type"], "generator")


if __name__ == "__main__":
    unittest.main()
