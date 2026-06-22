"""Public benchmark case resolution helpers.

The active codebase keeps customer data out of the repository. For IEEE 118
integrated M1+M2 testing, this module resolves public RAW/DYR files from local
override paths, bundled benchmark data, or ``powerfulcases`` and exposes small
metadata records for reporting.
"""

from __future__ import annotations

import hashlib
import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


IEEE118_PUBLIC_SOURCE = "powerfulcases_ieee118_public_dynamic"
IEEE118_LOCAL_OVERRIDE_SOURCE = "local_ieee118_raw_dyr_override"
IEEE118_BUNDLED_SOURCE = "bundled_public_ieee118_raw_dyr"
IEEE118_PUBLIC_LIMITATION = "ieee118_uses_public_benchmark_dynamic_data_not_customer_validated"
IEEE118_RAW_PATH_ENV = "GRIDMIND_IEEE118_RAW_PATH"
IEEE118_DYR_PATH_ENV = "GRIDMIND_IEEE118_DYR_PATH"
IEEE118_CASE_DIR_ENV = "GRIDMIND_IEEE118_CASE_DIR"
IEEE118_LOCAL_RAW_CANDIDATES = (
    "ieee118.raw",
    "IEEE118.raw",
    "IEEE_118_bus.raw",
    "ieee_118_bus.raw",
)
IEEE118_LOCAL_DYR_CANDIDATES = (
    "ieee118.dyr",
    "IEEE118.dyr",
    "IEEE_118_bus.dyr",
    "ieee_118_bus.dyr",
)
IEEE118_BUNDLED_CASE_DIR = Path(__file__).resolve().parents[1] / "public_data" / "ieee118_dynamic"


class PublicCaseDataError(RuntimeError):
    """Raised when a public benchmark case cannot be resolved."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type


@dataclass(frozen=True)
class PublicDynamicCaseFiles:
    """Resolved public RAW/DYR files for one dynamic benchmark case."""

    case_id: str
    source: str
    raw_path: str
    dyr_path: str
    dyr_variant: str
    raw_sha256: Optional[str] = None
    dyr_sha256: Optional[str] = None

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "source": self.source,
            "raw_path": self.raw_path,
            "dyr_path": self.dyr_path,
            "dyr_variant": self.dyr_variant,
            "raw_sha256": self.raw_sha256,
            "dyr_sha256": self.dyr_sha256,
            "data_label": "public_benchmark_data_not_customer_validated",
        }


def is_ieee118_public_dynamic_alias(case_path: Any) -> bool:
    """Return whether ``case_path`` names the public IEEE118 dynamic case."""

    if not isinstance(case_path, str):
        return False
    key = case_path.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    return key in {
        "ieee118",
        "case118",
        "118",
        "ieee118dynamic",
        "ieee118public",
        "ieee118publicdynamic",
    }


def ieee118_local_override_requested() -> bool:
    """Return whether local IEEE118 RAW/DYR override environment is set."""

    return bool(_ieee118_override_env()["override_requested"])


def ieee118_bundled_data_available() -> bool:
    """Return whether the bundled public IEEE118 RAW/DYR pair is available."""

    try:
        _resolve_ieee118_bundled_files()
    except PublicCaseDataError:
        return False
    return True


def resolve_ieee118_public_dynamic_files(
    powerfulcases_module: Optional[Any] = None,
) -> PublicDynamicCaseFiles:
    """Resolve public IEEE118 RAW+DYR data.

    Local override environment variables are checked first so a runtime node can
    use a manually downloaded public RAW/DYR pair:

    - ``GRIDMIND_IEEE118_RAW_PATH``
    - ``GRIDMIND_IEEE118_DYR_PATH``
    - or ``GRIDMIND_IEEE118_CASE_DIR`` containing recognized RAW/DYR filenames.

    If no local override is requested, the bundled GitHub RAW/DYR pair is used.
    If that pair is unavailable, the fallback is ``powerfulcases``.  The
    preferred DYR there is the ``genrou`` PSS/E DYR variant.  If that variant
    is unavailable but a default DYR exists, the default is used and labeled.
    """

    override = _resolve_ieee118_local_override_files()
    if override is not None:
        return override

    if powerfulcases_module is None:
        bundled = _resolve_ieee118_bundled_files(required=False)
        if bundled is not None:
            return bundled

    pcase = powerfulcases_module
    if pcase is None:
        try:
            pcase = importlib.import_module("powerfulcases")
        except ImportError as exc:
            raise PublicCaseDataError(
                "case_data_unavailable",
                "powerfulcases is required to resolve public IEEE118 RAW/DYR data.",
            ) from exc

    try:
        case = pcase.load("ieee118")
    except Exception as exc:
        raise PublicCaseDataError(
            "case_data_unavailable",
            f"powerfulcases could not load public IEEE118 metadata: {exc}",
        ) from exc

    raw_path = _resolve_raw_path(pcase, case)
    dyr_path, dyr_variant = _resolve_dyr_path(pcase, case)
    return PublicDynamicCaseFiles(
        case_id="ieee118_public_dynamic",
        source=IEEE118_PUBLIC_SOURCE,
        raw_path=raw_path,
        dyr_path=dyr_path,
        dyr_variant=dyr_variant,
        raw_sha256=_file_sha256(raw_path),
        dyr_sha256=_file_sha256(dyr_path),
    )


def ieee118_public_case_source_metadata() -> Dict[str, Any]:
    """Return metadata for public IEEE118 data, or an unavailable record."""

    try:
        return resolve_ieee118_public_dynamic_files().to_metadata()
    except PublicCaseDataError as exc:
        source = _unavailable_source_label()
        return {
            "case_id": "ieee118_public_dynamic",
            "source": source,
            "available": False,
            "error_type": exc.error_type,
            "message": str(exc),
            "data_label": "public_benchmark_data_not_customer_validated",
        }


def validate_ieee118_public_dynamic_data(
    powerfulcases_module: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return a detailed preflight record for public IEEE118 RAW/DYR data."""

    override_record = _preflight_ieee118_local_override()
    if override_record is not None:
        return override_record

    if powerfulcases_module is None:
        bundled_record = _preflight_ieee118_bundled_data()
        if bundled_record is not None:
            return bundled_record

    pcase = powerfulcases_module
    import_error: Optional[Exception] = None
    if pcase is None:
        try:
            pcase = importlib.import_module("powerfulcases")
        except ImportError as exc:
            import_error = exc

    if pcase is None:
        return {
            "ok": False,
            "case_id": "ieee118_public_dynamic",
            "source": IEEE118_PUBLIC_SOURCE,
            "error_type": "case_data_unavailable",
            "message": "powerfulcases is required to resolve public IEEE118 RAW/DYR data.",
            "detail": None if import_error is None else str(import_error),
            "raw_available": False,
            "dyr_available": False,
            "data_label": "public_benchmark_data_not_customer_validated",
        }

    case: Any = None
    case_error: Optional[Exception] = None
    try:
        case = pcase.load("ieee118")
    except Exception as exc:
        case_error = exc

    formats = _safe_call_list(getattr(pcase, "formats", None), case)
    variants = _safe_call_list(getattr(pcase, "variants", None), case, "psse_dyr")

    if case is None:
        return {
            "ok": False,
            "case_id": "ieee118_public_dynamic",
            "source": IEEE118_PUBLIC_SOURCE,
            "error_type": "case_data_unavailable",
            "message": f"powerfulcases could not load public IEEE118 metadata: {case_error}",
            "formats": formats,
            "dyr_variants": variants,
            "raw_available": False,
            "dyr_available": False,
            "data_label": "public_benchmark_data_not_customer_validated",
        }

    raw_record = _preflight_path_record(lambda: _resolve_raw_path(pcase, case))
    dyr_record = _preflight_path_record(lambda: _resolve_dyr_path(pcase, case)[0])
    ok = bool(raw_record.get("available") and dyr_record.get("available"))
    payload: Dict[str, Any] = {
        "ok": ok,
        "case_id": "ieee118_public_dynamic",
        "source": IEEE118_PUBLIC_SOURCE,
        "formats": formats,
        "dyr_variants": variants,
        "raw_available": bool(raw_record.get("available")),
        "dyr_available": bool(dyr_record.get("available")),
        "raw": raw_record,
        "dyr": dyr_record,
        "data_label": "public_benchmark_data_not_customer_validated",
    }
    if ok:
        files = resolve_ieee118_public_dynamic_files(pcase)
        payload["metadata"] = files.to_metadata()
    else:
        payload["error_type"] = (
            "dynamic_data_unavailable"
            if raw_record.get("available") and not dyr_record.get("available")
            else "case_data_unavailable"
        )
        payload["message"] = (
            "Public IEEE118 RAW/DYR preflight failed. A live IEEE118 M2 benchmark "
            "cannot pass until both RAW and DYR files are available."
        )
    return payload


def _resolve_raw_path(pcase: Any, case: Any) -> str:
    raw_path = getattr(case, "raw", None)
    if raw_path is None:
        raw_path = _call_file(pcase, case, "psse_raw")
    if raw_path is None:
        raw_path = _call_file(pcase, case, "raw")
    return _existing_path(raw_path, "case_data_unavailable", "IEEE118 RAW")


def _resolve_ieee118_local_override_files() -> Optional[PublicDynamicCaseFiles]:
    env = _ieee118_override_env()
    if not env["override_requested"]:
        return None

    raw_path = _resolve_local_override_path(
        explicit_path=env["raw_path"],
        case_dir=env["case_dir"],
        candidates=IEEE118_LOCAL_RAW_CANDIDATES,
        label="IEEE118 RAW override",
        error_type="case_data_unavailable",
    )
    dyr_path = _resolve_local_override_path(
        explicit_path=env["dyr_path"],
        case_dir=env["case_dir"],
        candidates=IEEE118_LOCAL_DYR_CANDIDATES,
        label="IEEE118 DYR override",
        error_type="dynamic_data_unavailable",
    )
    return PublicDynamicCaseFiles(
        case_id="ieee118_public_dynamic",
        source=IEEE118_LOCAL_OVERRIDE_SOURCE,
        raw_path=raw_path,
        dyr_path=dyr_path,
        dyr_variant="local_override",
        raw_sha256=_file_sha256(raw_path),
        dyr_sha256=_file_sha256(dyr_path),
    )


def _resolve_ieee118_bundled_files(
    *,
    required: bool = True,
) -> Optional[PublicDynamicCaseFiles]:
    raw_path = IEEE118_BUNDLED_CASE_DIR / "ieee118.raw"
    dyr_path = IEEE118_BUNDLED_CASE_DIR / "ieee118.dyr"
    raw_exists = raw_path.exists()
    dyr_exists = dyr_path.exists()
    if not raw_exists and not dyr_exists:
        if required:
            raise PublicCaseDataError(
                "case_data_unavailable",
                f"Bundled IEEE118 RAW/DYR directory is unavailable: {IEEE118_BUNDLED_CASE_DIR}",
            )
        return None
    if not raw_exists:
        raise PublicCaseDataError(
            "case_data_unavailable",
            f"Bundled IEEE118 RAW file is missing: {raw_path}",
        )
    if not dyr_exists:
        raise PublicCaseDataError(
            "dynamic_data_unavailable",
            f"Bundled IEEE118 DYR file is missing: {dyr_path}",
        )
    raw = _existing_path(str(raw_path), "case_data_unavailable", "Bundled IEEE118 RAW")
    dyr = _existing_path(str(dyr_path), "dynamic_data_unavailable", "Bundled IEEE118 DYR")
    return PublicDynamicCaseFiles(
        case_id="ieee118_public_dynamic",
        source=IEEE118_BUNDLED_SOURCE,
        raw_path=raw,
        dyr_path=dyr,
        dyr_variant="bundled_public",
        raw_sha256=_file_sha256(raw),
        dyr_sha256=_file_sha256(dyr),
    )


def _preflight_ieee118_local_override() -> Optional[Dict[str, Any]]:
    env = _ieee118_override_env()
    if not env["override_requested"]:
        return None

    raw_record = _preflight_path_record(
        lambda: _resolve_local_override_path(
            explicit_path=env["raw_path"],
            case_dir=env["case_dir"],
            candidates=IEEE118_LOCAL_RAW_CANDIDATES,
            label="IEEE118 RAW override",
            error_type="case_data_unavailable",
        )
    )
    dyr_record = _preflight_path_record(
        lambda: _resolve_local_override_path(
            explicit_path=env["dyr_path"],
            case_dir=env["case_dir"],
            candidates=IEEE118_LOCAL_DYR_CANDIDATES,
            label="IEEE118 DYR override",
            error_type="dynamic_data_unavailable",
        )
    )
    ok = bool(raw_record.get("available") and dyr_record.get("available"))
    payload: Dict[str, Any] = {
        "ok": ok,
        "case_id": "ieee118_public_dynamic",
        "source": IEEE118_LOCAL_OVERRIDE_SOURCE,
        "override_requested": True,
        "override_env": {
            IEEE118_RAW_PATH_ENV: env["raw_path"],
            IEEE118_DYR_PATH_ENV: env["dyr_path"],
            IEEE118_CASE_DIR_ENV: env["case_dir"],
        },
        "raw_available": bool(raw_record.get("available")),
        "dyr_available": bool(dyr_record.get("available")),
        "raw": raw_record,
        "dyr": dyr_record,
        "formats": ["psse_raw"],
        "dyr_variants": ["local_override"] if dyr_record.get("available") else [],
        "data_label": "public_benchmark_data_not_customer_validated",
    }
    if ok:
        files = _resolve_ieee118_local_override_files()
        if files is not None:
            payload["metadata"] = files.to_metadata()
    else:
        payload["error_type"] = (
            "dynamic_data_unavailable"
            if raw_record.get("available") and not dyr_record.get("available")
            else "case_data_unavailable"
        )
        payload["message"] = (
            "Local IEEE118 RAW/DYR override preflight failed. Set both "
            f"{IEEE118_RAW_PATH_ENV} and {IEEE118_DYR_PATH_ENV}, or set "
            f"{IEEE118_CASE_DIR_ENV} to a directory containing a recognized "
            "IEEE118 RAW/DYR pair."
        )
    return payload


def _preflight_ieee118_bundled_data() -> Optional[Dict[str, Any]]:
    raw_path = IEEE118_BUNDLED_CASE_DIR / "ieee118.raw"
    dyr_path = IEEE118_BUNDLED_CASE_DIR / "ieee118.dyr"
    if not raw_path.exists() and not dyr_path.exists():
        return None

    raw_record = _preflight_path_record(
        lambda: _existing_path(
            str(raw_path),
            "case_data_unavailable",
            "Bundled IEEE118 RAW",
        )
    )
    dyr_record = _preflight_path_record(
        lambda: _existing_path(
            str(dyr_path),
            "dynamic_data_unavailable",
            "Bundled IEEE118 DYR",
        )
    )
    ok = bool(raw_record.get("available") and dyr_record.get("available"))
    payload: Dict[str, Any] = {
        "ok": ok,
        "case_id": "ieee118_public_dynamic",
        "source": IEEE118_BUNDLED_SOURCE,
        "bundled_case_dir": str(IEEE118_BUNDLED_CASE_DIR),
        "raw_available": bool(raw_record.get("available")),
        "dyr_available": bool(dyr_record.get("available")),
        "raw": raw_record,
        "dyr": dyr_record,
        "formats": ["psse_raw"],
        "dyr_variants": ["bundled_public"] if dyr_record.get("available") else [],
        "data_label": "public_benchmark_data_not_customer_validated",
    }
    if ok:
        files = _resolve_ieee118_bundled_files()
        if files is not None:
            payload["metadata"] = files.to_metadata()
    else:
        payload["error_type"] = (
            "dynamic_data_unavailable"
            if raw_record.get("available") and not dyr_record.get("available")
            else "case_data_unavailable"
        )
        payload["message"] = (
            "Bundled IEEE118 RAW/DYR preflight failed. Restore both "
            f"{raw_path.name} and {dyr_path.name}, or provide override paths with "
            f"{IEEE118_RAW_PATH_ENV}/{IEEE118_DYR_PATH_ENV}."
        )
    return payload


def _ieee118_override_env() -> Dict[str, Any]:
    raw_path = os.environ.get(IEEE118_RAW_PATH_ENV, "").strip()
    dyr_path = os.environ.get(IEEE118_DYR_PATH_ENV, "").strip()
    case_dir = os.environ.get(IEEE118_CASE_DIR_ENV, "").strip()
    return {
        "raw_path": raw_path or None,
        "dyr_path": dyr_path or None,
        "case_dir": case_dir or None,
        "override_requested": bool(raw_path or dyr_path or case_dir),
    }


def _resolve_local_override_path(
    *,
    explicit_path: Optional[str],
    case_dir: Optional[str],
    candidates: tuple[str, ...],
    label: str,
    error_type: str,
) -> str:
    if explicit_path:
        return _existing_path(explicit_path, error_type, label)
    if case_dir:
        root = Path(case_dir).expanduser()
        if not root.exists():
            raise PublicCaseDataError(error_type, f"{label} directory does not exist: {case_dir}")
        if not root.is_dir():
            raise PublicCaseDataError(error_type, f"{label} directory is not a directory: {case_dir}")
        for candidate in candidates:
            path = root / candidate
            if path.exists():
                return _existing_path(str(path), error_type, label)
        raise PublicCaseDataError(
            error_type,
            f"{label} file was not found in {case_dir}. Tried: {', '.join(candidates)}",
        )
    raise PublicCaseDataError(
        error_type,
        f"{label} path is required when using a local IEEE118 override.",
    )


def _unavailable_source_label() -> str:
    if ieee118_local_override_requested():
        return IEEE118_LOCAL_OVERRIDE_SOURCE
    try:
        bundled = _resolve_ieee118_bundled_files(required=False)
    except PublicCaseDataError:
        return IEEE118_BUNDLED_SOURCE
    if bundled is not None:
        return bundled.source
    return IEEE118_PUBLIC_SOURCE


def _safe_call_list(fn: Any, *args: Any) -> list[str]:
    if not callable(fn):
        return []
    try:
        value = fn(*args)
    except Exception:
        return []
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [str(value)]
    try:
        return [str(item) for item in value]
    except TypeError:
        return [str(value)]


def _preflight_path_record(resolver: Any) -> Dict[str, Any]:
    try:
        value = resolver()
        return {"available": True, "path": value}
    except PublicCaseDataError as exc:
        return {
            "available": False,
            "error_type": exc.error_type,
            "message": str(exc),
        }
    except Exception as exc:
        return {
            "available": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


def _resolve_dyr_path(pcase: Any, case: Any) -> tuple[str, str]:
    attempts = [
        ("genrou", lambda: _call_file(pcase, case, "psse_dyr", variant="genrou")),
        ("default", lambda: getattr(case, "dyr", None)),
        ("default", lambda: _call_file(pcase, case, "psse_dyr")),
        ("default", lambda: _call_file(pcase, case, "dyr")),
    ]
    errors = []
    for variant, resolver in attempts:
        try:
            value = resolver()
            if value is None:
                continue
            return (
                _existing_path(value, "dynamic_data_unavailable", "IEEE118 DYR"),
                variant,
            )
        except PublicCaseDataError:
            raise
        except Exception as exc:
            errors.append(str(exc))
            continue
    detail = "; ".join(errors) if errors else "no DYR locator was returned"
    raise PublicCaseDataError(
        "dynamic_data_unavailable",
        f"powerfulcases did not provide a usable IEEE118 DYR file: {detail}",
    )


def _call_file(pcase: Any, case: Any, fmt: str, *, variant: Optional[str] = None) -> Any:
    file_fn = getattr(pcase, "file", None)
    if not callable(file_fn):
        return None
    if variant is None:
        return file_fn(case, fmt)
    return file_fn(case, fmt, variant=variant)


def _existing_path(value: Any, error_type: str, label: str) -> str:
    if value is None:
        raise PublicCaseDataError(error_type, f"{label} path is unavailable.")
    path = str(value)
    if not path.strip():
        raise PublicCaseDataError(error_type, f"{label} path is empty.")
    if "://" in path:
        return path
    expanded_path = Path(path).expanduser()
    if not expanded_path.exists():
        raise PublicCaseDataError(error_type, f"{label} file does not exist: {path}")
    if not expanded_path.is_file():
        raise PublicCaseDataError(error_type, f"{label} path is not a file: {path}")
    return str(expanded_path)


def _file_sha256(path: str) -> Optional[str]:
    if "://" in path:
        return None
    file_path = Path(path).expanduser()
    if not file_path.exists() or not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
