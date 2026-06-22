"""Frozen TRGC requirement catalog for real M1/M2 routing benchmarks.

The catalog is intentionally small and conservative. It captures the TRGC
requirements needed to build agent-routing testcases without claiming that the
current remote PSS/E worker can execute every TRGC study.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple


TRGC_REQUIREMENT_SCHEMA_VERSION = "trgc_requirement_catalog_v1"

TRGC_LAYERS = (
    "M1_steady_state_psse",
    "M2_dynamic_psse",
    "M1_M2_compliance_psse_pscad",
    "voltage_control_strategy",
    "power_quality",
    "field_validation",
    "data_submittal",
)
TRGC_TECHNOLOGIES = (
    "all",
    "synchronous",
    "ibr_gfl",
    "ibr_gfm",
    "bess",
    "facts",
    "hvdc",
)
TRGC_SUPPORT_STATUSES = (
    "executable_current_remote",
    "unsupported_current_remote",
    "classification_only",
)


@dataclass(frozen=True)
class TRGCRequirement:
    """One benchmark-facing TRGC requirement entry."""

    requirement_id: str
    title: str
    annexure: str
    layer: str
    technology: str
    required_capabilities: Tuple[str, ...]
    current_support_status: str
    current_remote_scenario_type: Optional[str] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": TRGC_REQUIREMENT_SCHEMA_VERSION,
            "requirement_id": self.requirement_id,
            "title": self.title,
            "annexure": self.annexure,
            "layer": self.layer,
            "technology": self.technology,
            "required_capabilities": list(self.required_capabilities),
            "current_support_status": self.current_support_status,
            "current_remote_scenario_type": self.current_remote_scenario_type,
            "description": self.description,
        }


TRGC_REQUIREMENT_CATALOG: Tuple[TRGCRequirement, ...] = (
    TRGCRequirement(
        requirement_id="TRGC_A15_STATIC_LOAD_FLOW",
        title="Steady-state load flow and voltage/thermal screening using PSS/E",
        annexure="TRGC_Annexture-15",
        layer="M1_steady_state_psse",
        technology="all",
        required_capabilities=(
            "psse_load_flow",
            "bus_voltage_range",
            "branch_loading",
            "poc_pqv",
        ),
        current_support_status="executable_current_remote",
        current_remote_scenario_type="static",
        description=(
            "Annexure 15 covers load-flow, voltage, thermal, reactive-power, "
            "and short-circuit steady-state screening. Current remote support "
            "covers the static load-flow subset."
        ),
    ),
    TRGCRequirement(
        requirement_id="TRGC_A16_NO_DISTURBANCE_RMS",
        title="Dynamic PSS/E baseline response without disturbance",
        annexure="TRGC_Annexture-16",
        layer="M2_dynamic_psse",
        technology="all",
        required_capabilities=(
            "psse_rms_dynamic",
            "dynamic_initialization",
            "poc_pqv_channels",
            "frequency_channel",
        ),
        current_support_status="executable_current_remote",
        current_remote_scenario_type="no_disturbance_5s",
        description=(
            "Annexure 16 covers dynamic stability. Current remote support covers "
            "only a 5-second no-disturbance RMS baseline, not fault/CCT studies."
        ),
    ),
    TRGCRequirement(
        requirement_id="TRGC_SMALL_PQ_TARGET_STEP",
        title="Small PPC active/reactive target step reproduction",
        annexure="remote_worker_validated_small_case",
        layer="M2_dynamic_psse",
        technology="ibr_gfl",
        required_capabilities=("ppc_pq_target_step", "poc_pq_channels"),
        current_support_status="executable_current_remote",
        current_remote_scenario_type="pq_target_step",
        description=(
            "Validated small PSS/E case scenario used to reproduce the existing "
            "P/Q target step response."
        ),
    ),
    TRGCRequirement(
        requirement_id="GFL-01",
        title="Reactive capability at connection point",
        annexure="TRGC_Annexture-06&07",
        layer="M1_M2_compliance_psse_pscad",
        technology="ibr_gfl",
        required_capabilities=(
            "pq_capability_curve",
            "dynamic_setpoint_sweep",
            "scr_condition",
            "pscad_benchmark_optional",
        ),
        current_support_status="unsupported_current_remote",
        description=(
            "TRGC requires P/Q capability curve construction and limiter behavior "
            "verification; current remote worker has no dynamic sweep action."
        ),
    ),
    TRGCRequirement(
        requirement_id="GFL-02",
        title="Active power regulation at connection point",
        annexure="TRGC_Annexture-06&07",
        layer="M1_M2_compliance_psse_pscad",
        technology="ibr_gfl",
        required_capabilities=("active_power_step_or_ramp", "poc_p_channel", "controller_response"),
        current_support_status="unsupported_current_remote",
        description="Requires active-power regulation scenarios beyond current allowlist.",
    ),
    TRGCRequirement(
        requirement_id="GFL-03",
        title="High and low voltage ride-through",
        annexure="TRGC_Annexture-06&07",
        layer="M2_dynamic_psse",
        technology="ibr_gfl",
        required_capabilities=("hvrt_lvrt_profile", "fault_or_voltage_playback", "poc_voltage_channel"),
        current_support_status="unsupported_current_remote",
        description="Requires voltage ride-through events unavailable in the current worker.",
    ),
    TRGCRequirement(
        requirement_id="GFL-05",
        title="Voltage droop verification and validation",
        annexure="TRGC_Annexture-06&07",
        layer="M2_dynamic_psse",
        technology="ibr_gfl",
        required_capabilities=("voltage_setpoint_step", "droop_control", "scr_3_xr_10"),
        current_support_status="unsupported_current_remote",
        description="Requires PPC/inverter voltage droop steps unavailable in the current worker.",
    ),
    TRGCRequirement(
        requirement_id="GFL-06",
        title="Frequency droop verification",
        annexure="TRGC_Annexture-06&07",
        layer="M2_dynamic_psse",
        technology="ibr_gfl",
        required_capabilities=("frequency_step_or_ramp", "frequency_droop_control", "active_power_response"),
        current_support_status="unsupported_current_remote",
        description="Requires frequency disturbance and droop-response scenarios unavailable now.",
    ),
    TRGCRequirement(
        requirement_id="GFL-07",
        title="Fault ride-through capability and post-fault recovery",
        annexure="TRGC_Annexture-06&07",
        layer="M2_dynamic_psse",
        technology="ibr_gfl",
        required_capabilities=("balanced_fault", "unbalanced_fault", "post_fault_recovery", "poc_pqv_channels"),
        current_support_status="unsupported_current_remote",
        description="Requires FRT fault scenarios; no-disturbance baseline is not a proxy.",
    ),
    TRGCRequirement(
        requirement_id="GFL-09",
        title="Reactive control operation",
        annexure="TRGC_Annexture-06&07",
        layer="M2_dynamic_psse",
        technology="ibr_gfl",
        required_capabilities=("reactive_mode_switch", "power_factor_control", "voltage_control"),
        current_support_status="unsupported_current_remote",
        description="Requires multiple reactive-control modes and controller edits.",
    ),
    TRGCRequirement(
        requirement_id="GFM-12",
        title="System strength assessment",
        annexure="TRGC_Annexture-06&07",
        layer="M1_M2_compliance_psse_pscad",
        technology="ibr_gfm",
        required_capabilities=("scr_sweep", "short_circuit_strength", "weak_grid_dynamic_response"),
        current_support_status="unsupported_current_remote",
        description="Requires SCR/system-strength studies beyond current live worker scope.",
    ),
    TRGCRequirement(
        requirement_id="TRGC_A12_VOLTAGE_CONTROL_STRATEGY",
        title="Voltage control strategy and PPC/OLTC/STATCOM coordination",
        annexure="TRGC_Annexture-12",
        layer="voltage_control_strategy",
        technology="all",
        required_capabilities=("voltage_control_documentation", "ppc_logic", "reactive_compensation_inventory"),
        current_support_status="classification_only",
        description="Document/submittal requirement; not directly executable by current M1/M2 worker.",
    ),
    TRGCRequirement(
        requirement_id="TRGC_A14_POWER_QUALITY",
        title="Power quality harmonic/flicker/RVC assessment",
        annexure="TRGC_Annexture-14",
        layer="power_quality",
        technology="all",
        required_capabilities=("harmonic_study", "flicker_study", "rapid_voltage_change", "pq_measurements"),
        current_support_status="classification_only",
        description="Power quality is outside current PSS/E M1/M2 remote gym.",
    ),
    TRGCRequirement(
        requirement_id="TRGC_A24_FIELD_MEASUREMENT_CHANNELS",
        title="Field testing POC measurement channels",
        annexure="TRGC_Annexture-24",
        layer="field_validation",
        technology="all",
        required_capabilities=("poc_p", "poc_q", "poc_voltage", "poc_frequency", "field_test_recordings"),
        current_support_status="classification_only",
        description="Field validation defines measurement channels; current worker returns simulation channels only.",
    ),
    TRGCRequirement(
        requirement_id="TRGC_A11_PLANT_DATASHEETS",
        title="Plant datasheets and model/control/protection submittals",
        annexure="TRGC_Annexture-11",
        layer="data_submittal",
        technology="all",
        required_capabilities=("poc_metadata", "pq_capability_data", "converter_data", "protection_settings"),
        current_support_status="classification_only",
        description="Data-submittal requirement used for missing-input testcases.",
    ),
)

_TRGC_BY_ID: Dict[str, TRGCRequirement] = {
    item.requirement_id: item for item in TRGC_REQUIREMENT_CATALOG
}


def list_trgc_requirements(
    *,
    layer: Optional[str] = None,
    technology: Optional[str] = None,
    support_status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List TRGC requirement entries, optionally filtered."""

    entries = []
    for requirement in TRGC_REQUIREMENT_CATALOG:
        if layer and requirement.layer != layer:
            continue
        if technology and requirement.technology != technology:
            continue
        if support_status and requirement.current_support_status != support_status:
            continue
        entries.append(requirement.to_dict())
    return entries


def get_trgc_requirement(requirement_id: str) -> Dict[str, Any]:
    """Return one TRGC requirement entry by id."""

    key = str(requirement_id or "").strip()
    if key not in _TRGC_BY_ID:
        allowed = ", ".join(sorted(_TRGC_BY_ID))
        raise KeyError(f"Unknown TRGC requirement_id {requirement_id!r}. Allowed: {allowed}.")
    return _TRGC_BY_ID[key].to_dict()


def trgc_requirement_from_mapping(payload: Mapping[str, Any]) -> TRGCRequirement:
    """Build a TRGCRequirement from a mapping for internal generator use."""

    return TRGCRequirement(
        requirement_id=str(payload["requirement_id"]),
        title=str(payload["title"]),
        annexure=str(payload["annexure"]),
        layer=str(payload["layer"]),
        technology=str(payload["technology"]),
        required_capabilities=tuple(str(item) for item in payload.get("required_capabilities", [])),
        current_support_status=str(payload["current_support_status"]),
        current_remote_scenario_type=(
            None
            if payload.get("current_remote_scenario_type") in {None, ""}
            else str(payload.get("current_remote_scenario_type"))
        ),
        description=str(payload.get("description") or ""),
    )
