"""Solver interfaces for a minimal Grid-Mind reproduction.

Step 1 intentionally implements only the solver spine:
load an IEEE case, run AC power flow, and expose structured results.
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


class SolverDependencyError(RuntimeError):
    """Raised when a solver backend dependency is unavailable."""


class PowerFlowError(RuntimeError):
    """Raised when a power-flow solve fails or has not been run."""


@dataclass(frozen=True)
class CaseInfo:
    """Basic metadata for a loaded grid case."""

    case_name: str
    buses: int
    lines: int
    transformers: int
    loads: int
    generators: int
    static_generators: int
    external_grids: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_name": self.case_name,
            "buses": self.buses,
            "lines": self.lines,
            "transformers": self.transformers,
            "loads": self.loads,
            "generators": self.generators,
            "static_generators": self.static_generators,
            "external_grids": self.external_grids,
        }


class GridSolver(ABC):
    """Backend-neutral interface matching the first Grid-Mind solver layer."""

    @abstractmethod
    def available_cases(self) -> List[str]:
        """Return canonical case names supported by this adapter."""

    @abstractmethod
    def load_case(self, case_name: str) -> CaseInfo:
        """Load a grid case into the backend and return metadata."""

    @abstractmethod
    def run_powerflow(self) -> Dict[str, Any]:
        """Run AC power flow and return a structured solve summary."""

    @abstractmethod
    def bus_results(self) -> List[Dict[str, Any]]:
        """Return per-bus solved voltage results."""

    @abstractmethod
    def branch_results(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return solved line/transformer loading results."""

    @abstractmethod
    def available_contingencies(self) -> List[Dict[str, Any]]:
        """Return single-element line/transformer outage candidates."""

    @abstractmethod
    def apply_contingency(self, element_type: str, element_index: int) -> Dict[str, Any]:
        """Apply one line/transformer outage to the loaded case."""

    @abstractmethod
    def add_connection(
        self,
        bus: int,
        p_mw: float,
        connection_type: str,
        is_ibr: bool,
        q_mvar: float = 0.0,
        name: Optional[str] = None,
        vm_pu: float = 1.0,
    ) -> Dict[str, Any]:
        """Add a proposed interconnection to the loaded case."""

    @abstractmethod
    def network_data(self, max_rows: int = 50) -> Dict[str, Any]:
        """Return read-only topology and equipment data for the loaded case."""

    @abstractmethod
    def case_info(self) -> CaseInfo:
        """Return metadata for the currently loaded case."""


class PandaPowerSolver(GridSolver):
    """pandapower implementation of the minimal GridSolver interface."""

    _MISSING_DEPENDENCY_MESSAGE = (
        "pandapower is required for PandaPowerSolver. Install it in the runtime "
        "environment before executing solver-backed smoke tests."
    )
    _SUPPORTED_CASES = ("ieee14", "ieee30", "ieee57", "ieee118")
    _CASE_ALIASES = {
        "ieee14": "ieee14",
        "case14": "ieee14",
        "14": "ieee14",
        "ieee30": "ieee30",
        "case30": "ieee30",
        "30": "ieee30",
        "ieee57": "ieee57",
        "case57": "ieee57",
        "57": "ieee57",
        "ieee118": "ieee118",
        "case118": "ieee118",
        "118": "ieee118",
    }
    _CASE_BUILDERS = {
        "ieee14": "case14",
        "ieee30": "case30",
        "ieee57": "case57",
        "ieee118": "case118",
    }

    def __init__(self, use_numba: bool = False) -> None:
        self._pp = None
        self._pn = None
        self.net: Any = None
        self._case_name: Optional[str] = None
        self._use_numba = use_numba
        self._ensure_dependency()

    @classmethod
    def supported_cases(cls) -> List[str]:
        """Return canonical case names without importing pandapower."""

        return list(cls._SUPPORTED_CASES)

    @classmethod
    def is_available(cls) -> bool:
        """Return whether the pandapower backend appears importable."""

        return importlib.util.find_spec("pandapower") is not None

    def available_cases(self) -> List[str]:
        return self.supported_cases()

    def load_case(self, case_name: str) -> CaseInfo:
        canonical = self._normalize_case_name(case_name)
        builder_name = self._CASE_BUILDERS[canonical]
        builder = getattr(self._pn, builder_name, None)
        if builder is None:
            raise ValueError(f"pandapower.networks.{builder_name} is unavailable")

        self.net = builder()
        self._case_name = canonical
        return self.case_info()

    def run_powerflow(self) -> Dict[str, Any]:
        self._require_loaded()
        try:
            self._pp.runpp(self.net, numba=self._use_numba)
        except Exception as exc:  # pandapower raises several backend-specific types
            raise PowerFlowError(f"pandapower power flow failed: {exc}") from exc

        if not bool(getattr(self.net, "converged", False)):
            raise PowerFlowError("pandapower did not converge")

        return self._summary()

    def bus_results(self) -> List[Dict[str, Any]]:
        self._require_converged()
        rows: List[Dict[str, Any]] = []
        for bus_idx, result in self.net.res_bus.iterrows():
            bus = self.net.bus.loc[bus_idx]
            rows.append(
                {
                    "bus_index": int(bus_idx),
                    "name": self._safe_scalar(bus.get("name")),
                    "vn_kv": self._safe_float(bus.get("vn_kv")),
                    "vm_pu": self._safe_float(result.get("vm_pu")),
                    "va_degree": self._safe_float(result.get("va_degree")),
                }
            )
        return rows

    def branch_results(self) -> Dict[str, List[Dict[str, Any]]]:
        self._require_converged()
        return {
            "lines": self._line_results(),
            "transformers": self._trafo_results(),
        }

    def available_contingencies(self) -> List[Dict[str, Any]]:
        self._require_loaded()
        contingencies: List[Dict[str, Any]] = []
        for line_idx, row in self.net.line.iterrows():
            if not self._row_in_service(row):
                continue
            contingencies.append(
                {
                    "element_type": "line",
                    "element_index": int(line_idx),
                    "element_name": self._safe_scalar(row.get("name")),
                    "from_bus": self._safe_int(row.get("from_bus")),
                    "to_bus": self._safe_int(row.get("to_bus")),
                }
            )

        for trafo_idx, row in self.net.trafo.iterrows():
            if not self._row_in_service(row):
                continue
            contingencies.append(
                {
                    "element_type": "transformer",
                    "element_index": int(trafo_idx),
                    "element_name": self._safe_scalar(row.get("name")),
                    "hv_bus": self._safe_int(row.get("hv_bus")),
                    "lv_bus": self._safe_int(row.get("lv_bus")),
                }
            )
        return contingencies

    def apply_contingency(self, element_type: str, element_index: int) -> Dict[str, Any]:
        self._require_loaded()
        key = element_type.strip().lower()
        if key in {"line", "lines"}:
            table_name = "line"
            normalized_type = "line"
        elif key in {"transformer", "trafo", "transformers"}:
            table_name = "trafo"
            normalized_type = "transformer"
        else:
            raise ValueError(f"Unsupported contingency element_type '{element_type}'")

        table = getattr(self.net, table_name)
        if element_index not in table.index:
            raise ValueError(f"{normalized_type} contingency index {element_index} was not found")
        if not self._row_in_service(table.loc[element_index]):
            raise ValueError(f"{normalized_type} contingency index {element_index} is already out of service")

        table.at[element_index, "in_service"] = False
        if hasattr(self.net, "converged"):
            self.net.converged = False

        row = table.loc[element_index]
        result = {
            "element_type": normalized_type,
            "element_index": int(element_index),
            "element_name": self._safe_scalar(row.get("name")),
        }
        if normalized_type == "line":
            result.update(
                {
                    "from_bus": self._safe_int(row.get("from_bus")),
                    "to_bus": self._safe_int(row.get("to_bus")),
                }
            )
        else:
            result.update(
                {
                    "hv_bus": self._safe_int(row.get("hv_bus")),
                    "lv_bus": self._safe_int(row.get("lv_bus")),
                }
            )
        return result

    def add_connection(
        self,
        bus: int,
        p_mw: float,
        connection_type: str,
        is_ibr: bool,
        q_mvar: float = 0.0,
        name: Optional[str] = None,
        vm_pu: float = 1.0,
    ) -> Dict[str, Any]:
        self._require_loaded()
        if p_mw < 0:
            raise ValueError("Connection p_mw must be non-negative")

        resource_type = connection_type.strip().lower()
        if resource_type not in {"load", "solar", "wind", "bess", "hybrid", "synchronous"}:
            raise ValueError(f"Unsupported connection_type '{connection_type}'")

        bus_info = self.resolve_bus(bus)
        element_name = name or f"cia_{resource_type}_{p_mw:g}_mw_bus_{bus}"

        if resource_type == "load":
            element_index = self._pp.create_load(
                self.net,
                bus=bus_info["bus_index"],
                p_mw=p_mw,
                q_mvar=q_mvar,
                name=element_name,
            )
            element_table = "load"
        elif resource_type == "synchronous":
            element_index = self._pp.create_gen(
                self.net,
                bus=bus_info["bus_index"],
                p_mw=p_mw,
                vm_pu=vm_pu,
                name=element_name,
            )
            element_table = "gen"
        else:
            element_index = self._pp.create_sgen(
                self.net,
                bus=bus_info["bus_index"],
                p_mw=p_mw,
                q_mvar=q_mvar,
                name=element_name,
                type=resource_type,
            )
            element_table = "sgen"

        # A changed topology/dispatch invalidates any previous solved state.
        if hasattr(self.net, "converged"):
            self.net.converged = False

        return {
            "element_table": element_table,
            "element_index": int(element_index),
            "name": element_name,
            "connection_type": resource_type,
            "is_ibr": bool(is_ibr),
            "bus": bus,
            "resolved_bus": bus_info,
            "p_mw": float(p_mw),
            "q_mvar": float(q_mvar),
        }

    def resolve_bus(self, bus: int) -> Dict[str, Any]:
        self._require_loaded()
        matches = []
        for bus_index, row in self.net.bus.iterrows():
            bus_name = self._safe_scalar(row.get("name"))
            if self._bus_label_matches(bus_name, bus):
                matches.append((bus_index, row, "name"))

        if not matches and bus in self.net.bus.index:
            matches.append((bus, self.net.bus.loc[bus], "index"))

        if not matches:
            raise ValueError(f"Bus '{bus}' was not found in case '{self._case_name}'")
        if len(matches) > 1:
            raise ValueError(f"Bus '{bus}' is ambiguous in case '{self._case_name}'")

        bus_index, row, matched_on = matches[0]
        return {
            "bus_index": int(bus_index),
            "bus_name": self._safe_scalar(row.get("name")),
            "vn_kv": self._safe_float(row.get("vn_kv")),
            "matched_on": matched_on,
        }

    def network_data(self, max_rows: int = 50) -> Dict[str, Any]:
        self._require_loaded()
        return {
            "case": self.case_info().to_dict(),
            "max_rows_per_table": max_rows,
            "tables": {
                "buses": self._table_records(
                    "bus",
                    "bus_index",
                    ["name", "vn_kv", "type", "zone", "in_service"],
                    max_rows,
                ),
                "lines": self._table_records(
                    "line",
                    "line_index",
                    [
                        "name",
                        "from_bus",
                        "to_bus",
                        "length_km",
                        "r_ohm_per_km",
                        "x_ohm_per_km",
                        "c_nf_per_km",
                        "max_i_ka",
                        "in_service",
                    ],
                    max_rows,
                ),
                "transformers": self._table_records(
                    "trafo",
                    "trafo_index",
                    [
                        "name",
                        "hv_bus",
                        "lv_bus",
                        "sn_mva",
                        "vn_hv_kv",
                        "vn_lv_kv",
                        "vk_percent",
                        "vkr_percent",
                        "in_service",
                    ],
                    max_rows,
                ),
                "loads": self._table_records(
                    "load",
                    "load_index",
                    ["name", "bus", "p_mw", "q_mvar", "scaling", "in_service"],
                    max_rows,
                ),
                "generators": self._table_records(
                    "gen",
                    "gen_index",
                    ["name", "bus", "p_mw", "vm_pu", "sn_mva", "min_p_mw", "max_p_mw", "in_service"],
                    max_rows,
                ),
                "static_generators": self._table_records(
                    "sgen",
                    "sgen_index",
                    ["name", "bus", "p_mw", "q_mvar", "sn_mva", "type", "in_service"],
                    max_rows,
                ),
                "external_grids": self._table_records(
                    "ext_grid",
                    "ext_grid_index",
                    ["name", "bus", "vm_pu", "va_degree", "in_service"],
                    max_rows,
                ),
            },
        }

    def case_info(self) -> CaseInfo:
        self._require_loaded()
        return CaseInfo(
            case_name=self._case_name or "unknown",
            buses=self._table_len("bus"),
            lines=self._table_len("line"),
            transformers=self._table_len("trafo"),
            loads=self._table_len("load"),
            generators=self._table_len("gen"),
            static_generators=self._table_len("sgen"),
            external_grids=self._table_len("ext_grid"),
        )

    def _summary(self) -> Dict[str, Any]:
        info = self.case_info().to_dict()
        bus = self.net.res_bus
        line = self.net.res_line
        trafo = self.net.res_trafo

        line_loading = self._series_max(line.get("loading_percent"))
        trafo_loading = self._series_max(trafo.get("loading_percent"))

        return {
            "backend": "pandapower",
            "case": info,
            "converged": bool(self.net.converged),
            "bus_summary": {
                "min_vm_pu": self._safe_float(bus["vm_pu"].min()),
                "max_vm_pu": self._safe_float(bus["vm_pu"].max()),
                "min_va_degree": self._safe_float(bus["va_degree"].min()),
                "max_va_degree": self._safe_float(bus["va_degree"].max()),
            },
            "branch_summary": {
                "max_line_loading_percent": line_loading,
                "max_trafo_loading_percent": trafo_loading,
            },
        }

    def _line_results(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for line_idx, result in self.net.res_line.iterrows():
            line = self.net.line.loc[line_idx]
            rows.append(
                {
                    "line_index": int(line_idx),
                    "name": self._safe_scalar(line.get("name")),
                    "from_bus": self._safe_int(line.get("from_bus")),
                    "to_bus": self._safe_int(line.get("to_bus")),
                    "in_service": self._safe_bool(line.get("in_service")),
                    "loading_percent": self._safe_float(result.get("loading_percent")),
                    "p_from_mw": self._safe_float(result.get("p_from_mw")),
                    "q_from_mvar": self._safe_float(result.get("q_from_mvar")),
                    "p_to_mw": self._safe_float(result.get("p_to_mw")),
                    "q_to_mvar": self._safe_float(result.get("q_to_mvar")),
                }
            )
        return rows

    def _trafo_results(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for trafo_idx, result in self.net.res_trafo.iterrows():
            trafo = self.net.trafo.loc[trafo_idx]
            rows.append(
                {
                    "trafo_index": int(trafo_idx),
                    "name": self._safe_scalar(trafo.get("name")),
                    "hv_bus": self._safe_int(trafo.get("hv_bus")),
                    "lv_bus": self._safe_int(trafo.get("lv_bus")),
                    "in_service": self._safe_bool(trafo.get("in_service")),
                    "loading_percent": self._safe_float(result.get("loading_percent")),
                    "p_hv_mw": self._safe_float(result.get("p_hv_mw")),
                    "q_hv_mvar": self._safe_float(result.get("q_hv_mvar")),
                    "p_lv_mw": self._safe_float(result.get("p_lv_mw")),
                    "q_lv_mvar": self._safe_float(result.get("q_lv_mvar")),
                }
            )
        return rows

    def _ensure_dependency(self) -> None:
        try:
            import pandapower as pp
            import pandapower.networks as pn
        except Exception as exc:
            raise SolverDependencyError(self._MISSING_DEPENDENCY_MESSAGE) from exc
        self._pp = pp
        self._pn = pn

    @classmethod
    def _normalize_case_name(cls, case_name: str) -> str:
        key = case_name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
        if key not in cls._CASE_ALIASES:
            raise ValueError(
                f"Unsupported case '{case_name}'. Available cases: "
                f"{', '.join(cls.supported_cases())}"
            )
        return cls._CASE_ALIASES[key]

    def _table_len(self, table_name: str) -> int:
        table = getattr(self.net, table_name, None)
        return len(table) if table is not None else 0

    def _table_records(
        self,
        table_name: str,
        index_key: str,
        columns: List[str],
        max_rows: int,
    ) -> Dict[str, Any]:
        table = getattr(self.net, table_name, None)
        if table is None:
            return {"rows": [], "total_rows": 0, "truncated_rows": 0}

        total_rows = len(table)
        limited = table if max_rows < 0 else table.head(max_rows)
        rows: List[Dict[str, Any]] = []
        for index, row in limited.iterrows():
            record: Dict[str, Any] = {index_key: self._safe_int(index)}
            for column in columns:
                if column in row:
                    record[column] = self._safe_scalar(row.get(column))
            rows.append(record)

        return {
            "rows": rows,
            "total_rows": total_rows,
            "truncated_rows": max(0, total_rows - len(rows)),
        }

    def _require_loaded(self) -> None:
        if self.net is None:
            raise PowerFlowError("No case is loaded")

    def _require_converged(self) -> None:
        self._require_loaded()
        if not bool(getattr(self.net, "converged", False)):
            raise PowerFlowError("Power flow has not converged yet")

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            # pandas/numpy NaN is not equal to itself.
            if value != value:
                return None
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            if value != value:
                return None
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _safe_bool(value: Any) -> Optional[bool]:
        if value is None:
            return None
        try:
            if value != value:
                return None
        except Exception:
            pass
        return bool(value)

    @staticmethod
    def _safe_scalar(value: Any) -> Optional[Any]:
        if value is None:
            return None
        try:
            if value != value:
                return None
        except Exception:
            pass
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return value

    @staticmethod
    def _row_in_service(row: Any) -> bool:
        value = row.get("in_service", True)
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip().lower() not in {"false", "0", "no", "off"}
        try:
            if value != value:
                return True
        except Exception:
            pass
        return bool(value)

    @staticmethod
    def _bus_label_matches(label: Any, requested_bus: int) -> bool:
        if label == requested_bus:
            return True
        try:
            return int(label) == int(requested_bus)
        except Exception:
            return str(label).strip() == str(requested_bus).strip()

    def _series_max(self, series: Optional[Iterable[Any]]) -> Optional[float]:
        if series is None:
            return None
        try:
            value = series.max()
        except Exception:
            return None
        return self._safe_float(value)
