"""
CPET Functional Diagnostics Core
--------------------------------
Decision-support utilities for CPET files exported from systems such as COSMED.

The algorithms are intentionally transparent: they produce method-specific
candidate thresholds and a consensus suggestion, but final thresholds should be
reviewed and confirmed by a qualified practitioner.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable
import json
import math
import re
import warnings

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Small formatting helpers
# -----------------------------------------------------------------------------


def parse_float(value: Any) -> float | None:
    """Parse numbers with either dot or comma decimal separators."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().replace(" ", "")
    if not text:
        return None
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def time_to_seconds(x: Any) -> float:
    """Convert Excel time / Python time / mm:ss / hh:mm:ss / seconds to seconds."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan

    if isinstance(x, (int, float, np.integer, np.floating)):
        # Excel stores time-of-day as day fraction. CPET elapsed time is usually <1 day.
        v = float(x)
        return v * 86400 if abs(v) < 2 else v

    if hasattr(x, "hour") and hasattr(x, "minute") and hasattr(x, "second"):
        return (
            float(x.hour) * 3600
            + float(x.minute) * 60
            + float(x.second)
            + float(getattr(x, "microsecond", 0)) / 1_000_000
        )

    s = str(x).strip()
    if not s:
        return np.nan

    # Examples: 00:12:45, 12:45, 00:04:17
    if re.match(r"^\d{1,2}:\d{1,2}:\d{1,2}(\.\d+)?$", s):
        h, m, sec = s.split(":")
        return float(h) * 3600 + float(m) * 60 + float(sec)
    if re.match(r"^\d{1,2}:\d{1,2}(\.\d+)?$", s):
        m, sec = s.split(":")
        return float(m) * 60 + float(sec)

    v = parse_float(s)
    return np.nan if v is None else v


def seconds_to_mmss(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "—"
    seconds = int(round(float(seconds)))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def normalize_name(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s).strip()).lower()


def compact_name(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_name(s))


def safe_round(value: Any, digits: int = 1) -> float | None:
    try:
        if value is None or not np.isfinite(float(value)):
            return None
        return round(float(value), digits)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# File loading and CPET table extraction
# -----------------------------------------------------------------------------


def _to_bytes_io(file: Any) -> BytesIO | str:
    """Return a BytesIO object for uploaded files, bytes, or a path string."""
    if isinstance(file, (str, Path)):
        return str(file)
    if isinstance(file, bytes):
        return BytesIO(file)
    if hasattr(file, "getvalue"):
        return BytesIO(file.getvalue())
    if hasattr(file, "read"):
        data = file.read()
        return BytesIO(data)
    raise TypeError("Unsupported file input. Use a path, bytes, or Streamlit UploadedFile.")


def cpet_header_score(raw: pd.DataFrame, max_rows: int = 40) -> int:
    """Score how likely a raw sheet contains CPET data."""
    tokens = {
        "t",
        "time",
        "vo2",
        "vco2",
        "ve",
        "ve/vo2",
        "ve/vco2",
        "peto2",
        "petco2",
        "pet o2",
        "pet co2",
        "rq",
        "rer",
        "hr",
        "speed",
        "grade",
    }
    score = 0
    for i in range(min(max_rows, len(raw))):
        vals = {normalize_name(v) for v in raw.iloc[i].tolist() if pd.notna(v)}
        score = max(score, len(tokens.intersection(vals)))
    return score


def auto_select_sheet(excel_file: pd.ExcelFile) -> str:
    """Pick the sheet that looks most like a CPET data sheet."""
    best_sheet = excel_file.sheet_names[0]
    best_score = -1
    for sheet in excel_file.sheet_names:
        try:
            raw = pd.read_excel(excel_file, sheet_name=sheet, header=None, nrows=40)
            score = cpet_header_score(raw)
            if score > best_score:
                best_sheet, best_score = sheet, score
        except Exception:
            continue
    return best_sheet


def read_raw_table(file: Any, sheet_name: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read an Excel/CSV CPET export into a raw DataFrame."""
    source = _to_bytes_io(file)
    name = str(getattr(file, "name", file))
    suffix = Path(name).suffix.lower()

    if suffix in {".csv", ".txt"}:
        raw = pd.read_csv(source, header=None)
        return raw, {"selected_sheet": "CSV", "available_sheets": ["CSV"], "source_name": name}

    xls = pd.ExcelFile(source, engine="openpyxl")
    selected = sheet_name if sheet_name and sheet_name in xls.sheet_names else auto_select_sheet(xls)
    raw = pd.read_excel(xls, sheet_name=selected, header=None)
    return raw, {"selected_sheet": selected, "available_sheets": xls.sheet_names, "source_name": name}


def find_col(columns: Iterable[Any], aliases: Iterable[str]) -> str | None:
    cols = list(columns)
    norm = {normalize_name(c): c for c in cols}
    for a in aliases:
        if normalize_name(a) in norm:
            return norm[normalize_name(a)]
    norm2 = {compact_name(c): c for c in cols}
    for a in aliases:
        if compact_name(a) in norm2:
            return norm2[compact_name(a)]
    return None


def extract_cpet_table(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Extract CPET data and metadata from a raw export sheet.

    Works with COSMED-like files where metadata is stored in the left part of the
    sheet and CPET variables start in the same row with headers such as t, VO2,
    VCO2, VE, VE/VO2, etc.
    """
    header_row = None
    for i in range(min(40, len(raw))):
        vals = [str(v).strip() for v in raw.iloc[i].tolist() if pd.notna(v)]
        score = sum(req in vals for req in ["VO2", "VCO2", "VE", "VE/VO2", "VE/VCO2", "PetO2", "PetCO2", "Speed", "HR", "t"])
        if score >= 4:
            header_row = i
            break
    if header_row is None:
        raise ValueError("Не открих ред с CPET заглавия като VO2, VCO2, VE, VE/VO2, VE/VCO2.")

    row_vals = [str(v).strip() if pd.notna(v) else "" for v in raw.iloc[header_row].tolist()]
    t_indices = [i for i, v in enumerate(row_vals) if normalize_name(v) in {"t", "time"}]
    vo2_indices = [i for i, v in enumerate(row_vals) if normalize_name(v) == "vo2"]
    if not vo2_indices:
        raise ValueError("Открих CPET ред, но не открих колона VO2.")

    start_col = t_indices[0] if t_indices else min(vo2_indices)
    vo2_col = vo2_indices[0]

    first_data_row = None
    for r in range(header_row + 1, min(header_row + 15, len(raw))):
        val = parse_float(raw.iloc[r, vo2_col])
        if val is not None and val > 0:
            first_data_row = r
            break
    if first_data_row is None:
        first_data_row = header_row + 2

    headers = row_vals[start_col:]
    seen: dict[str, int] = {}
    clean_headers: list[str] = []
    for h in headers:
        h = str(h).strip() if h is not None else ""
        if not h or h.lower() == "nan":
            h = f"unnamed_{len(clean_headers)}"
        if h in seen:
            seen[h] += 1
            h = f"{h}.{seen[h]}"
        else:
            seen[h] = 0
        clean_headers.append(h)

    data = raw.iloc[first_data_row:, start_col : start_col + len(clean_headers)].copy()
    data.columns = clean_headers
    data = data.dropna(how="all")

    # Metadata may be interleaved vertically with the first data rows, so inspect
    # the first 60 rows of common metadata column pairs.
    metadata: dict[str, Any] = {}
    for key_col, value_col in [(0, 1), (3, 4), (6, 7)]:
        if value_col >= raw.shape[1]:
            continue
        for r in range(min(len(raw), 60)):
            key = raw.iloc[r, key_col]
            value = raw.iloc[r, value_col]
            if pd.notna(key) and str(key).strip() and pd.notna(value):
                metadata[str(key).strip()] = value

    info = {
        "header_row_1based": header_row + 1,
        "data_start_row_1based": first_data_row + 1,
        "start_col_1based": start_col + 1,
        "n_raw_rows": int(len(data)),
        "n_raw_cols": int(len(data.columns)),
    }
    return data, metadata, info


# -----------------------------------------------------------------------------
# Canonical CPET variables and derived variables
# -----------------------------------------------------------------------------


def get_body_mass(metadata: dict[str, Any]) -> float | None:
    for key in ["Weight (kg)", "Weight", "Body weight", "Body mass", "Mass (kg)"]:
        if key in metadata:
            v = parse_float(metadata[key])
            if v is not None and 20 <= v <= 250:
                return v
    return None


def get_age(metadata: dict[str, Any]) -> float | None:
    for key in ["Age", "Age (years)", "Years"]:
        if key in metadata:
            v = parse_float(metadata[key])
            if v is not None and 5 <= v <= 100:
                return v
    return None


def get_sex(metadata: dict[str, Any]) -> str | None:
    for key in ["Gender", "Sex"]:
        if key in metadata:
            val = str(metadata[key]).strip()
            if val:
                return val
    return None


def canonicalize_cpet(data: pd.DataFrame, metadata: dict[str, Any], smooth_window: int = 5, body_mass_override: float | None = None) -> tuple[pd.DataFrame, dict[str, str], float | None]:
    """Map a CPET export to standard variables and compute derived metrics."""
    raw = data.copy()
    df = data.copy()

    aliases = {
        "time_raw": ["t", "time", "Time", "t Rel", "Elapsed", "Elapsed time"],
        "vo2": ["VO2", "VO₂", "VO2 (mL/min)", "VO2_mL_min"],
        "vco2": ["VCO2", "VCO₂", "VCO2 (mL/min)", "VCO2_mL_min"],
        "ve": ["VE", "VE (L/min)", "VE_L_min"],
        "rer": ["RQ", "RER", "RER/RQ"],
        "vevo2": ["VE/VO2", "VE/VO₂", "VEVO2"],
        "vevco2": ["VE/VCO2", "VE/VCO₂", "VEVCO2"],
        "peto2": ["PetO2", "PETO2", "PET O2", "Pet O2"],
        "petco2": ["PetCO2", "PETCO2", "PET CO2", "Pet CO2"],
        "hr": ["HR", "Heart Rate", "HR_bpm"],
        "speed": ["Speed", "Speed Kmh", "Speed (km/h)", "Speed_kmh", "mark Speed"],
        "power": ["Power", "Power (W)", "Watts", "Work Rate", "WR"],
        "grade": ["Grade", "Incline", "Slope", "Grade %"],
        "vo2kg": ["VO2/kg", "VO₂/kg", "VO2kg"],
        "fat_device": ["Fat", "FAT", "Fat kcal/day"],
        "cho_device": ["CHO", "Carbohydrate"],
        "phase": ["Phase"],
        "marker": ["Marker"],
    }
    colmap: dict[str, str] = {}
    for key, al in aliases.items():
        c = find_col(df.columns, al)
        if c is not None:
            colmap[key] = c

    # Convert numeric columns, but preserve raw time/text columns.
    preserve = {colmap.get("time_raw"), colmap.get("phase"), colmap.get("marker")}
    for c in df.columns:
        if c not in preserve:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    out = pd.DataFrame(index=df.index)
    if "time_raw" in colmap:
        out["time_s"] = raw[colmap["time_raw"]].apply(time_to_seconds)
    else:
        out["time_s"] = np.arange(len(df)) * 15.0

    mapping = [
        ("VO2_mL_min", "vo2"),
        ("VCO2_mL_min", "vco2"),
        ("VE_L_min", "ve"),
        ("RER", "rer"),
        ("VE_VO2", "vevo2"),
        ("VE_VCO2", "vevco2"),
        ("PETO2", "peto2"),
        ("PETCO2", "petco2"),
        ("HR_bpm", "hr"),
        ("speed_kmh", "speed"),
        ("power_w", "power"),
        ("grade_pct", "grade"),
        ("VO2kg", "vo2kg"),
        ("fat_device", "fat_device"),
        ("cho_device", "cho_device"),
    ]
    for standard, key in mapping:
        if key in colmap:
            out[standard] = pd.to_numeric(df[colmap[key]], errors="coerce")

    # Unit normalization for VO2/VCO2.
    if "VO2_mL_min" in out.columns and out["VO2_mL_min"].median(skipna=True) < 100:
        out["VO2_mL_min"] *= 1000
    if "VCO2_mL_min" in out.columns and out["VCO2_mL_min"].median(skipna=True) < 100:
        out["VCO2_mL_min"] *= 1000

    if {"VO2_mL_min", "VCO2_mL_min"}.issubset(out.columns):
        out["RER_calc"] = out["VCO2_mL_min"] / out["VO2_mL_min"].replace(0, np.nan)
        if "RER" not in out.columns or out["RER"].isna().all():
            out["RER"] = out["RER_calc"]

    body_mass = body_mass_override or get_body_mass(metadata)
    if body_mass and "VO2_mL_min" in out.columns:
        if "VO2kg" not in out.columns or out["VO2kg"].isna().all():
            out["VO2kg"] = out["VO2_mL_min"] / body_mass

    # Derived energy metrics. Weir-like equation without urinary nitrogen:
    # kcal/min = 3.941*VO2(L/min) + 1.106*VCO2(L/min)
    if {"VO2_mL_min", "VCO2_mL_min"}.issubset(out.columns):
        out["VO2_L_min"] = out["VO2_mL_min"] / 1000
        out["VCO2_L_min"] = out["VCO2_mL_min"] / 1000
        out["kcal_min_weir"] = 3.941 * out["VO2_L_min"] + 1.106 * out["VCO2_L_min"]
        out["fat_g_min"] = (1.695 * out["VO2_L_min"] - 1.701 * out["VCO2_L_min"]).clip(lower=0)
        out["cho_g_min"] = (4.585 * out["VCO2_L_min"] - 3.226 * out["VO2_L_min"]).clip(lower=0)

    if body_mass and {"kcal_min_weir", "speed_kmh"}.issubset(out.columns):
        speed_km_min = out["speed_kmh"].replace(0, np.nan) / 60.0
        out["kcal_kg_km"] = out["kcal_min_weir"] / (body_mass * speed_km_min)

    if body_mass and {"VO2kg", "speed_kmh"}.issubset(out.columns):
        speed_m_min = out["speed_kmh"].replace(0, np.nan) * 1000 / 60.0
        out["o2_cost_ml_kg_m"] = out["VO2kg"] / speed_m_min

    out = out.dropna(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)
    out = out.replace([np.inf, -np.inf], np.nan)

    if smooth_window and smooth_window > 1:
        if smooth_window % 2 == 0:
            smooth_window += 1
        min_periods = max(1, smooth_window // 2)
        for c in out.select_dtypes(include=[np.number]).columns:
            if c != "time_s" and not c.endswith("_sm"):
                out[f"{c}_sm"] = out[c].rolling(smooth_window, center=True, min_periods=min_periods).median()

    return out, colmap, body_mass


# -----------------------------------------------------------------------------
# Stage endpoint handling
# -----------------------------------------------------------------------------


def aggregate_stages(df: pd.DataFrame, min_duration_s: float = 45, last_window_s: float = 45) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate repeated work stages using last N seconds of each stage.

    Useful for step / interval protocols where a few short transition rows should
    not be used for threshold decisions.
    """
    d = df.sort_values("time_s").copy()
    if "speed_kmh" not in d.columns:
        d["speed_kmh"] = np.nan
    if "grade_pct" not in d.columns:
        d["grade_pct"] = 0.0
    if "power_w" not in d.columns:
        d["power_w"] = np.nan

    # Group by stable external load. For treadmill, speed+grade are the usual
    # identifiers. For cycling, power can be used if speed is missing.
    use_power = d["speed_kmh"].isna().all() and not d["power_w"].isna().all()
    d["_load1"] = d["power_w"].round(1) if use_power else d["speed_kmh"].round(2)
    d["_load2"] = 0.0 if use_power else d["grade_pct"].round(2)
    d["_gap"] = d["time_s"].diff().fillna(0)
    d["_new_stage"] = (
        d["_load1"].diff().abs().fillna(0) > 0.01
    ) | (d["_load2"].diff().abs().fillna(0) > 0.01) | (d["_gap"] > 60)
    d["_stage_id"] = d["_new_stage"].cumsum()

    rows: list[dict[str, Any]] = []
    stage_info: list[dict[str, Any]] = []
    numeric_cols = [c for c in d.select_dtypes(include=[np.number]).columns if not c.startswith("_")]
    median_step = float(d["time_s"].diff().median()) if len(d) > 1 else 0.0
    if not np.isfinite(median_step):
        median_step = 0.0

    for stage_id, g in d.groupby("_stage_id"):
        start = float(g["time_s"].min())
        end = float(g["time_s"].max())
        duration = end - start + (median_step if len(g) > 1 else 0.0)
        speed = safe_round(g["speed_kmh"].median(skipna=True), 2) if "speed_kmh" in g and g["speed_kmh"].notna().any() else None
        grade = safe_round(g["grade_pct"].median(skipna=True), 2) if "grade_pct" in g and g["grade_pct"].notna().any() else None
        power = safe_round(g["power_w"].median(skipna=True), 1) if "power_w" in g and g["power_w"].notna().any() else None
        keep = bool(duration >= min_duration_s)

        if keep:
            w = g[g["time_s"] >= end - last_window_s]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                agg = w[numeric_cols].mean(numeric_only=True).to_dict()
            agg["time_s"] = end
            agg["stage_start_s"] = start
            agg["stage_end_s"] = end
            agg["stage_duration_s"] = duration
            agg["stage_n"] = int(len(g))
            agg["stage_speed_kmh"] = speed
            agg["stage_grade_pct"] = grade
            agg["stage_power_w"] = power
            if power is not None and np.isfinite(power):
                agg["stage_label"] = f"{power:g} W"
            else:
                agg["stage_label"] = f"{speed:g} km/h @ {grade:g}%"
            rows.append(agg)

        stage_info.append(
            {
                "stage_id": int(stage_id),
                "start_s": start,
                "end_s": end,
                "duration_s": duration,
                "n": int(len(g)),
                "speed_kmh": speed,
                "grade_pct": grade,
                "power_w": power,
                "keep_for_stage_analysis": keep,
            }
        )

    stage_df = pd.DataFrame(rows)
    if not stage_df.empty:
        # Ensure canonical external load reflects stage values, not smoothed transition values.
        if "stage_speed_kmh" in stage_df.columns:
            stage_df["speed_kmh"] = stage_df["stage_speed_kmh"]
        if "stage_grade_pct" in stage_df.columns:
            stage_df["grade_pct"] = stage_df["stage_grade_pct"]
        if "stage_power_w" in stage_df.columns:
            stage_df["power_w"] = stage_df["stage_power_w"]
        for c in [
            "VO2_mL_min",
            "VCO2_mL_min",
            "VE_L_min",
            "RER",
            "VE_VO2",
            "VE_VCO2",
            "PETO2",
            "PETCO2",
            "HR_bpm",
            "VO2kg",
            "fat_g_min",
            "cho_g_min",
            "kcal_kg_km",
            "o2_cost_ml_kg_m",
        ]:
            if c in stage_df.columns:
                stage_df[f"{c}_sm"] = stage_df[c].rolling(3, center=True, min_periods=1).median()
        stage_df = stage_df.replace([np.inf, -np.inf], np.nan)

    return stage_df.reset_index(drop=True), pd.DataFrame(stage_info)


def choose_analysis_dataframe(raw_df: pd.DataFrame, stage_df: pd.DataFrame, mode: str = "auto") -> tuple[pd.DataFrame, str]:
    """Choose BxB-smoothed vs stage endpoint analysis."""
    mode_clean = normalize_name(mode)
    if mode_clean.startswith("stage") or "стъп" in mode_clean or "endpoint" in mode_clean:
        return (stage_df if len(stage_df) >= 4 else raw_df), "stage_endpoint"
    if mode_clean.startswith("bxb") or "breath" in mode_clean or "изглад" in mode_clean:
        return raw_df, "bxb_smoothed"
    # Auto: use stage endpoints when enough repeated stages are present.
    if len(stage_df) >= 5 and len(raw_df) >= len(stage_df) * 5:
        return stage_df, "stage_endpoint_auto"
    return raw_df, "bxb_smoothed_auto"


# -----------------------------------------------------------------------------
# Threshold candidates
# -----------------------------------------------------------------------------


@dataclass
class ThresholdCandidate:
    threshold: str
    method: str
    time_s: float | None = None
    speed_kmh: float | None = None
    grade_pct: float | None = None
    power_w: float | None = None
    vo2_ml_min: float | None = None
    vo2kg: float | None = None
    hr_bpm: float | None = None
    rer: float | None = None
    ve_vo2: float | None = None
    ve_vco2: float | None = None
    peto2: float | None = None
    petco2: float | None = None
    confidence: float | None = None
    comment: str = ""
    row_index: int | None = None

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["time"] = seconds_to_mmss(d.get("time_s"))
        return d


def _series(df: pd.DataFrame, base: str, prefer_smoothed: bool = True) -> pd.Series | None:
    if prefer_smoothed and f"{base}_sm" in df.columns:
        return df[f"{base}_sm"]
    if base in df.columns:
        return df[base]
    return None


def _candidate_from_idx(df: pd.DataFrame, idx: int | None, threshold: str, method: str, confidence: float | None = None, comment: str = "") -> ThresholdCandidate | None:
    if idx is None or idx not in df.index:
        return None
    r = df.loc[idx]
    return ThresholdCandidate(
        threshold=threshold,
        method=method,
        time_s=safe_round(r.get("time_s"), 1),
        speed_kmh=safe_round(r.get("speed_kmh"), 2),
        grade_pct=safe_round(r.get("grade_pct"), 2),
        power_w=safe_round(r.get("power_w"), 1),
        vo2_ml_min=safe_round(r.get("VO2_mL_min"), 0),
        vo2kg=safe_round(r.get("VO2kg"), 1),
        hr_bpm=safe_round(r.get("HR_bpm"), 0),
        rer=safe_round(r.get("RER"), 2),
        ve_vo2=safe_round(r.get("VE_VO2"), 1),
        ve_vco2=safe_round(r.get("VE_VCO2"), 1),
        peto2=safe_round(r.get("PETO2"), 1),
        petco2=safe_round(r.get("PETCO2"), 1),
        confidence=safe_round(confidence, 0),
        comment=comment,
        row_index=int(idx),
    )


def _linear_sse(x: np.ndarray, y: np.ndarray) -> tuple[float, tuple[float, float]]:
    if len(x) < 2 or len(y) < 2:
        return np.inf, (np.nan, np.nan)
    try:
        p = np.polyfit(x, y, 1)
        yhat = p[0] * x + p[1]
        return float(np.sum((y - yhat) ** 2)), (float(p[0]), float(p[1]))
    except Exception:
        return np.inf, (np.nan, np.nan)


def piecewise_break_idx(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    min_seg: int = 4,
    start_frac: float = 0.15,
    end_frac: float = 0.85,
    prefer_smoothed: bool = True,
) -> dict[str, Any] | None:
    """Two-segment least-squares breakpoint."""
    xs = _series(df, xcol, prefer_smoothed)
    ys = _series(df, ycol, prefer_smoothed)
    if xs is None or ys is None:
        return None
    tmp = pd.DataFrame({"x": xs, "y": ys, "idx": df.index}).dropna()
    if len(tmp) < 2 * min_seg + 1:
        return None
    x = tmp["x"].to_numpy(float)
    y = tmp["y"].to_numpy(float)
    idx = tmp["idx"].to_numpy()
    lo = max(min_seg, int(len(tmp) * start_frac))
    hi = min(len(tmp) - min_seg, int(math.ceil(len(tmp) * end_frac)))
    if hi <= lo:
        return None

    best = None
    for k in range(lo, hi):
        sse_left, p_left = _linear_sse(x[: k + 1], y[: k + 1])
        sse_right, p_right = _linear_sse(x[k:], y[k:])
        if not np.isfinite(sse_left + sse_right):
            continue
        sse = sse_left + sse_right
        if best is None or sse < best["sse"]:
            best = {
                "idx": int(idx[k]),
                "local_pos": int(k),
                "sse": float(sse),
                "left_slope": float(p_left[0]),
                "right_slope": float(p_right[0]),
                "left_intercept": float(p_left[1]),
                "right_intercept": float(p_right[1]),
            }
    return best


def weighted_median(values: list[float], weights: list[float]) -> float | None:
    vals = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return None
    vals = vals[mask]
    w = w[mask]
    order = np.argsort(vals)
    vals = vals[order]
    w = w[order]
    cum = np.cumsum(w) / np.sum(w)
    return float(vals[np.searchsorted(cum, 0.5)])


def nearest_idx_by_time(df: pd.DataFrame, time_s: float | None) -> int | None:
    if time_s is None or not np.isfinite(time_s) or "time_s" not in df.columns or df.empty:
        return None
    return int((df["time_s"] - time_s).abs().idxmin())


def confidence_from_spread(times: list[float], high_at_s: float = 90, zero_at_s: float = 420) -> float:
    valid = [float(t) for t in times if t is not None and np.isfinite(t)]
    if len(valid) <= 1:
        return 60.0 if len(valid) == 1 else 0.0
    spread = max(valid) - min(valid)
    if spread <= high_at_s:
        return 90.0
    if spread >= zero_at_s:
        return 20.0
    return 90.0 - (spread - high_at_s) * 70.0 / (zero_at_s - high_at_s)


def vt1_ventilatory_equivalent_idx(df: pd.DataFrame) -> int | None:
    vevo2 = _series(df, "VE_VO2")
    vevco2 = _series(df, "VE_VCO2")
    peto2 = _series(df, "PETO2")
    petco2 = _series(df, "PETCO2")
    if vevo2 is None:
        return None
    times = df["time_s"].to_numpy(float)
    search = df[(df["time_s"] >= np.nanquantile(times, 0.10)) & (df["time_s"] <= np.nanquantile(times, 0.80))]
    if search.empty:
        return None
    idx_min = int(vevo2.loc[search.index].idxmin())
    baseline = float(vevo2.loc[idx_min])
    if not np.isfinite(baseline):
        return None
    min_rise = max(1.2, baseline * 0.06)

    for idx in df.loc[idx_min:].index:
        if df.loc[idx, "time_s"] < df.loc[idx_min, "time_s"]:
            continue
        current = float(vevo2.loc[idx])
        if not np.isfinite(current) or current < baseline + min_rise:
            continue
        checks = 0
        total = 0
        if peto2 is not None and np.isfinite(peto2.loc[idx_min]) and np.isfinite(peto2.loc[idx]):
            total += 1
            checks += float(peto2.loc[idx] >= peto2.loc[idx_min] + 1.0)
        if petco2 is not None and np.isfinite(petco2.loc[idx_min]) and np.isfinite(petco2.loc[idx]):
            total += 1
            # VT1: PETCO2 should be stable or only mildly lower, not a clear RCP-like fall.
            checks += float(petco2.loc[idx] >= petco2.loc[idx_min] - 3.0)
        if vevco2 is not None and np.isfinite(vevco2.loc[idx]):
            total += 1
            low_vco2 = float(vevco2.loc[search.index].min())
            checks += float(vevco2.loc[idx] <= low_vco2 + max(2.0, low_vco2 * 0.08))
        if total == 0 or checks >= max(1, math.ceil(total * 0.5)):
            return int(idx)
    return idx_min


def peto2_turn_idx(df: pd.DataFrame) -> int | None:
    peto2 = _series(df, "PETO2")
    petco2 = _series(df, "PETCO2")
    if peto2 is None:
        return None
    times = df["time_s"].to_numpy(float)
    search = df[(df["time_s"] >= np.nanquantile(times, 0.10)) & (df["time_s"] <= np.nanquantile(times, 0.80))]
    if search.empty:
        return None
    idx_min = int(peto2.loc[search.index].idxmin())
    baseline = float(peto2.loc[idx_min])
    for idx in df.loc[idx_min:].index:
        if not np.isfinite(peto2.loc[idx]):
            continue
        if peto2.loc[idx] >= baseline + 2.0:
            if petco2 is None or not np.isfinite(petco2.loc[idx_min]) or petco2.loc[idx] >= petco2.loc[idx_min] - 3.0:
                return int(idx)
    return idx_min


def detect_vt1_gas(df: pd.DataFrame) -> tuple[ThresholdCandidate | None, list[ThresholdCandidate]]:
    """Detect VT1 candidates from V-slope + ventilatory equivalents + PET gases."""
    candidates: list[ThresholdCandidate] = []
    times: list[float] = []
    weights: list[float] = []

    min_seg = max(2, min(8, len(df) // 5))
    vs = piecewise_break_idx(df, "VO2_mL_min", "VCO2_mL_min", min_seg=min_seg, start_frac=0.15, end_frac=0.80)
    if vs is not None:
        conf = 70 if vs["right_slope"] > vs["left_slope"] else 45
        c = _candidate_from_idx(
            df,
            vs["idx"],
            "VT1",
            "V-slope, two-segment regression",
            conf,
            f"left slope={vs['left_slope']:.2f}, right slope={vs['right_slope']:.2f}",
        )
        if c:
            candidates.append(c)
            times.append(c.time_s)
            weights.append(2.0 if vs["right_slope"] > vs["left_slope"] else 1.0)

    idx_vent = vt1_ventilatory_equivalent_idx(df)
    c = _candidate_from_idx(
        df,
        idx_vent,
        "VT1",
        "VE/VO2 rise with VE/VCO2 and PETCO2 control",
        70,
        "First meaningful VE/VO2 rise after nadir while VE/VCO2/PETCO2 are still relatively controlled.",
    )
    if c:
        candidates.append(c)
        times.append(c.time_s)
        weights.append(2.0)

    idx_pet = peto2_turn_idx(df)
    c = _candidate_from_idx(
        df,
        idx_pet,
        "VT1",
        "PETO2 nadir/turn with PETCO2 stability",
        60,
        "PETO2 starts to rise after its nadir without a clear PETCO2 fall.",
    )
    if c:
        candidates.append(c)
        times.append(c.time_s)
        weights.append(1.0)

    t = weighted_median(times, weights)
    idx = nearest_idx_by_time(df, t)
    consensus_conf = confidence_from_spread(times)
    consensus = _candidate_from_idx(
        df,
        idx,
        "VT1",
        "Consensus gas-exchange VT1",
        consensus_conf,
        "Weighted median of method-specific VT1 candidates. Low confidence means expert review is important.",
    )
    return consensus, candidates


def rcp_vevco2_idx(df: pd.DataFrame, after_time_s: float | None = None) -> int | None:
    vevco2 = _series(df, "VE_VCO2")
    if vevco2 is None:
        return None
    times = df["time_s"].to_numpy(float)
    after = after_time_s if after_time_s is not None else np.nanquantile(times, 0.45)
    search = df[(df["time_s"] >= after) & (df["time_s"] <= np.nanquantile(times, 0.98))]
    if search.empty:
        return None
    idx_min = int(vevco2.loc[search.index].idxmin())
    baseline = float(vevco2.loc[idx_min])
    min_rise = max(1.5, baseline * 0.05)
    for idx in df.loc[idx_min:].index:
        val = vevco2.loc[idx]
        if np.isfinite(val) and val >= baseline + min_rise:
            return int(idx)
    return idx_min


def rcp_petco2_drop_idx(df: pd.DataFrame, after_time_s: float | None = None) -> int | None:
    petco2 = _series(df, "PETCO2")
    if petco2 is None:
        return None
    times = df["time_s"].to_numpy(float)
    after = after_time_s if after_time_s is not None else np.nanquantile(times, 0.45)
    search = df[(df["time_s"] >= after) & (df["time_s"] <= np.nanquantile(times, 0.98))]
    if search.empty:
        return None
    idx_peak = int(petco2.loc[search.index].idxmax())
    peak = float(petco2.loc[idx_peak])
    for idx in df.loc[idx_peak:].index:
        val = petco2.loc[idx]
        if np.isfinite(val) and val <= peak - 2.0:
            return int(idx)
    return idx_peak


def first_rer_idx(df: pd.DataFrame, after_time_s: float | None = None, threshold: float = 1.00) -> int | None:
    rer = _series(df, "RER")
    if rer is None:
        return None
    search = df.copy()
    if after_time_s is not None:
        search = search[search["time_s"] >= after_time_s]
    hit = search[rer.loc[search.index] >= threshold]
    return int(hit.index[0]) if not hit.empty else None


def detect_rcp_gas(df: pd.DataFrame, vt1_time_s: float | None = None, rer_support: float = 1.00) -> tuple[ThresholdCandidate | None, list[ThresholdCandidate]]:
    """Detect VT2/RCP candidates."""
    after = vt1_time_s + 120 if vt1_time_s is not None and np.isfinite(vt1_time_s) else None
    candidates: list[ThresholdCandidate] = []
    times: list[float] = []
    weights: list[float] = []

    idx = rcp_vevco2_idx(df, after)
    c = _candidate_from_idx(
        df,
        idx,
        "VT2/RCP",
        "VE/VCO2 nadir plus sustained rise",
        75,
        "RCP pattern: VE/VCO2 begins to rise after its nadir.",
    )
    if c:
        candidates.append(c)
        times.append(c.time_s)
        weights.append(2.0)

    idx = rcp_petco2_drop_idx(df, after)
    c = _candidate_from_idx(
        df,
        idx,
        "VT2/RCP",
        "PETCO2 peak/drop",
        70,
        "RCP support: PETCO2 peaks and then falls as compensation increases.",
    )
    if c:
        candidates.append(c)
        times.append(c.time_s)
        weights.append(1.5)

    # VE/VCO2 slope breakpoint over time after VT1.
    tmp = df[df["time_s"] >= after].copy() if after is not None else df.copy()
    if len(tmp) >= 5 and "VE_VCO2" in tmp.columns:
        min_seg = max(2, min(6, len(tmp) // 4))
        p = piecewise_break_idx(tmp, "time_s", "VE_VCO2", min_seg=min_seg, start_frac=0.05, end_frac=0.95)
        if p is not None:
            conf = 60 if p["right_slope"] > p["left_slope"] else 35
            c = _candidate_from_idx(
                tmp,
                p["idx"],
                "VT2/RCP",
                "VE/VCO2 slope breakpoint",
                conf,
                f"left slope={p['left_slope']:.4f}, right slope={p['right_slope']:.4f}",
            )
            if c:
                candidates.append(c)
                times.append(c.time_s)
                weights.append(1.0 if p["right_slope"] > p["left_slope"] else 0.5)

    idx = first_rer_idx(df, after, threshold=rer_support)
    c = _candidate_from_idx(
        df,
        idx,
        "VT2/RCP",
        f"RER ≥ {rer_support:.2f} support",
        40,
        "Supportive marker only; do not use RER alone as threshold.",
    )
    if c:
        candidates.append(c)
        times.append(c.time_s)
        weights.append(0.6)

    t = weighted_median(times, weights)
    idx = nearest_idx_by_time(df, t)
    consensus_conf = confidence_from_spread(times, high_at_s=120, zero_at_s=480)
    consensus = _candidate_from_idx(
        df,
        idx,
        "VT2/RCP",
        "Consensus gas-exchange VT2/RCP",
        consensus_conf,
        "Weighted median of RCP candidates. Confirm visually with VE/VCO2 and PETCO2.",
    )
    return consensus, candidates


def detect_fatmax(df: pd.DataFrame, pct_of_max: float = 0.90) -> tuple[ThresholdCandidate | None, dict[str, Any]]:
    fat = _series(df, "fat_g_min")
    if fat is None or fat.dropna().empty:
        return None, {}
    idx = int(fat.idxmax())
    fat_max = float(fat.loc[idx])
    c = _candidate_from_idx(
        df,
        idx,
        "FatMax",
        "Maximal calculated fat oxidation",
        70,
        f"Fat oxidation estimated from VO2/VCO2; range = points ≥ {pct_of_max:.0%} of maximum.",
    )
    if c:
        c.comment += f" FatMax≈{fat_max:.2f} g/min."

    eligible = fat >= fat_max * pct_of_max
    # Contiguous block around the maximum.
    indices = list(df.index)
    idx_pos = indices.index(idx)
    left = idx_pos
    right = idx_pos
    while left - 1 >= 0 and bool(eligible.loc[indices[left - 1]]):
        left -= 1
    while right + 1 < len(indices) and bool(eligible.loc[indices[right + 1]]):
        right += 1
    block = df.loc[indices[left] : indices[right]]
    range_info = {
        "fatmax_g_min": safe_round(fat_max, 2),
        "range_low_time_s": safe_round(block["time_s"].min(), 1) if not block.empty else None,
        "range_high_time_s": safe_round(block["time_s"].max(), 1) if not block.empty else None,
        "range_low_speed_kmh": safe_round(block["speed_kmh"].min(), 2) if "speed_kmh" in block else None,
        "range_high_speed_kmh": safe_round(block["speed_kmh"].max(), 2) if "speed_kmh" in block else None,
        "range_low_grade_pct": safe_round(block["grade_pct"].min(), 2) if "grade_pct" in block else None,
        "range_high_grade_pct": safe_round(block["grade_pct"].max(), 2) if "grade_pct" in block else None,
        "pct_of_max": pct_of_max,
    }
    return c, range_info


def detect_all_gas_thresholds(df: pd.DataFrame, rer_support: float = 1.00, fatmax_pct: float = 0.90) -> dict[str, Any]:
    vt1_consensus, vt1_candidates = detect_vt1_gas(df)
    vt1_time = vt1_consensus.time_s if vt1_consensus else None
    rcp_consensus, rcp_candidates = detect_rcp_gas(df, vt1_time, rer_support=rer_support)
    fatmax, fat_range = detect_fatmax(df, pct_of_max=fatmax_pct)

    all_candidates = []
    for group in [vt1_candidates, rcp_candidates, [fatmax] if fatmax else []]:
        all_candidates.extend([c for c in group if c is not None])

    consensus = [c for c in [vt1_consensus, rcp_consensus, fatmax] if c is not None]
    return {
        "consensus": consensus,
        "method_candidates": all_candidates,
        "vt1": vt1_consensus,
        "rcp": rcp_consensus,
        "fatmax": fatmax,
        "fatmax_range": fat_range,
    }


# -----------------------------------------------------------------------------
# Lactate support
# -----------------------------------------------------------------------------


def load_lactate_table(file: Any) -> pd.DataFrame:
    source = _to_bytes_io(file)
    name = str(getattr(file, "name", file))
    suffix = Path(name).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(source, engine="openpyxl")
    else:
        df = pd.read_csv(source)
    return standardize_lactate_table(df)


def standardize_lactate_table(df: pd.DataFrame) -> pd.DataFrame:
    col_time = find_col(df.columns, ["time_s", "time", "t", "min", "time_min", "stage_time"])
    col_speed = find_col(df.columns, ["speed_kmh", "speed", "Speed", "kmh", "km/h"])
    col_power = find_col(df.columns, ["power_w", "power", "watts", "W"])
    col_lac = find_col(df.columns, ["lactate", "lactate_mmol_l", "lactate_mmol", "La", "BLa", "mmol/L"])
    col_hr = find_col(df.columns, ["HR", "hr_bpm", "heart_rate"])

    if col_lac is None:
        raise ValueError("Лактатният файл трябва да съдържа колона lactate / La / BLa / mmol/L.")

    out = pd.DataFrame()
    if col_time:
        # If values are small and header mentions minutes, treat as minutes.
        t = pd.to_numeric(df[col_time].map(time_to_seconds), errors="coerce")
        if "min" in normalize_name(col_time) and t.max(skipna=True) < 100:
            t = pd.to_numeric(df[col_time], errors="coerce") * 60
        out["time_s"] = t
    if col_speed:
        out["speed_kmh"] = pd.to_numeric(df[col_speed], errors="coerce")
    if col_power:
        out["power_w"] = pd.to_numeric(df[col_power], errors="coerce")
    if col_hr:
        out["HR_bpm"] = pd.to_numeric(df[col_hr], errors="coerce")
    out["lactate_mmol_L"] = pd.to_numeric(df[col_lac], errors="coerce")

    out = out.dropna(subset=["lactate_mmol_L"]).reset_index(drop=True)
    return out


def _intensity_column(lactate_df: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred and preferred in lactate_df.columns:
        return preferred
    for c in ["speed_kmh", "power_w", "time_s"]:
        if c in lactate_df.columns and lactate_df[c].notna().sum() >= 3:
            return c
    raise ValueError("Нужна е колона за интензивност: speed_kmh, power_w или time_s.")


def lactate_dmax_idx(lac: pd.DataFrame, xcol: str) -> int | None:
    d = lac[[xcol, "lactate_mmol_L"]].dropna().copy()
    if len(d) < 4:
        return None
    d = d.sort_values(xcol)
    x = d[xcol].to_numpy(float)
    y = d["lactate_mmol_L"].to_numpy(float)
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    line = p2 - p1
    norm = np.linalg.norm(line)
    if norm == 0:
        return None
    distances = np.abs(np.cross(line, np.column_stack([x, y]) - p1)) / norm
    # Avoid endpoints.
    distances[0] = -np.inf
    distances[-1] = -np.inf
    local_pos = int(np.argmax(distances))
    return int(d.index[local_pos])


def lactate_segmented_idx(lac: pd.DataFrame, xcol: str, min_seg: int = 2) -> int | None:
    d = lac[[xcol, "lactate_mmol_L"]].dropna().sort_values(xcol).copy()
    if len(d) < 2 * min_seg + 1:
        return None
    x = d[xcol].to_numpy(float)
    y = d["lactate_mmol_L"].to_numpy(float)
    best = None
    for k in range(min_seg, len(d) - min_seg):
        s1, p1 = _linear_sse(x[: k + 1], y[: k + 1])
        s2, p2 = _linear_sse(x[k:], y[k:])
        # Prefer upper breakpoint where second slope is steeper.
        if p2[0] <= p1[0]:
            continue
        sse = s1 + s2
        if best is None or sse < best["sse"]:
            best = {"idx": int(d.index[k]), "sse": sse, "left_slope": p1[0], "right_slope": p2[0]}
    return None if best is None else best["idx"]


def detect_lactate_thresholds(lactate_df: pd.DataFrame, intensity_col: str | None = None, lt1_delta: float = 0.4) -> tuple[list[ThresholdCandidate], str]:
    lac = standardize_lactate_table(lactate_df) if "lactate_mmol_L" not in lactate_df.columns else lactate_df.copy()
    xcol = _intensity_column(lac, intensity_col)
    lac = lac.sort_values(xcol).reset_index(drop=False).rename(columns={"index": "orig_index"})
    candidates: list[ThresholdCandidate] = []

    # LT1: first sustained rise over early baseline.
    if len(lac) >= 3:
        baseline = float(lac["lactate_mmol_L"].iloc[: min(2, len(lac))].median())
        lt1_pos = None
        for i in range(1, len(lac)):
            current = lac.loc[i, "lactate_mmol_L"]
            next_val = lac.loc[min(i + 1, len(lac) - 1), "lactate_mmol_L"]
            if current >= baseline + lt1_delta and next_val >= current - 0.2:
                lt1_pos = i
                break
        if lt1_pos is not None:
            r = lac.loc[lt1_pos]
            candidates.append(
                ThresholdCandidate(
                    threshold="LT1",
                    method="First sustained lactate rise",
                    time_s=safe_round(r.get("time_s"), 1),
                    speed_kmh=safe_round(r.get("speed_kmh"), 2),
                    power_w=safe_round(r.get("power_w"), 1),
                    hr_bpm=safe_round(r.get("HR_bpm"), 0),
                    confidence=60,
                    comment=f"Baseline≈{baseline:.2f} mmol/L; threshold criterion baseline + {lt1_delta:.2f} mmol/L.",
                    row_index=int(r.get("orig_index", lt1_pos)),
                )
            )

    idx = lactate_dmax_idx(lac, xcol)
    if idx is not None:
        r = lactate_df.loc[idx] if idx in lactate_df.index else lac.loc[idx]
        candidates.append(
            ThresholdCandidate(
                threshold="LT2",
                method="Dmax lactate",
                time_s=safe_round(r.get("time_s"), 1),
                speed_kmh=safe_round(r.get("speed_kmh"), 2),
                power_w=safe_round(r.get("power_w"), 1),
                hr_bpm=safe_round(r.get("HR_bpm"), 0),
                confidence=65,
                comment="Maximum perpendicular distance from first-last lactate line.",
                row_index=int(idx),
            )
        )

    idx = lactate_segmented_idx(lac, xcol)
    if idx is not None:
        r = lactate_df.loc[idx] if idx in lactate_df.index else lac.loc[idx]
        candidates.append(
            ThresholdCandidate(
                threshold="LT2",
                method="Segmented lactate regression",
                time_s=safe_round(r.get("time_s"), 1),
                speed_kmh=safe_round(r.get("speed_kmh"), 2),
                power_w=safe_round(r.get("power_w"), 1),
                hr_bpm=safe_round(r.get("HR_bpm"), 0),
                confidence=65,
                comment="Two-segment least-squares breakpoint where lactate slope increases.",
                row_index=int(idx),
            )
        )

    return candidates, xcol


# -----------------------------------------------------------------------------
# Peak metrics, references, zones, report
# -----------------------------------------------------------------------------


def compute_peak_metrics(df: pd.DataFrame, metadata: dict[str, Any], body_mass: float | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "VO2_mL_min" in df.columns:
        out["VO2peak_mL_min"] = safe_round(df["VO2_mL_min"].max(skipna=True), 0)
    if "VO2kg" in df.columns:
        out["VO2peak_mL_kg_min"] = safe_round(df["VO2kg"].max(skipna=True), 1)
    if "HR_bpm" in df.columns:
        out["HRpeak_bpm"] = safe_round(df["HR_bpm"].max(skipna=True), 0)
    if "RER" in df.columns:
        out["RERpeak"] = safe_round(df["RER"].max(skipna=True), 2)
    if "VE_L_min" in df.columns:
        out["VEmax_L_min"] = safe_round(df["VE_L_min"].max(skipna=True), 1)
    if "speed_kmh" in df.columns:
        out["speed_max_kmh"] = safe_round(df["speed_kmh"].max(skipna=True), 2)
    if "grade_pct" in df.columns:
        out["grade_max_pct"] = safe_round(df["grade_pct"].max(skipna=True), 2)
    if "power_w" in df.columns:
        out["power_max_w"] = safe_round(df["power_w"].max(skipna=True), 0)
    if "fat_g_min" in df.columns:
        valid_fat = df["fat_g_min"].replace([np.inf, -np.inf], np.nan)
        # Exclude early transient rows; otherwise the first breaths can create
        # unrealistic substrate and economy values.
        mask = pd.Series(True, index=df.index)
        if "time_s" in df.columns:
            mask &= df["time_s"] >= 120
        if "VO2kg" in df.columns:
            mask &= df["VO2kg"] >= 20
        out["FatOx_max_g_min"] = safe_round(valid_fat[mask].max(skipna=True), 2)
    if "kcal_kg_km" in df.columns:
        valid = df["kcal_kg_km"].replace([np.inf, -np.inf], np.nan)
        mask = pd.Series(True, index=df.index)
        if "time_s" in df.columns:
            mask &= df["time_s"] >= 120
        if "VO2kg" in df.columns:
            mask &= df["VO2kg"] >= 20
        if "speed_kmh" in df.columns:
            mask &= df["speed_kmh"] >= 5
        valid = valid[mask].dropna()
        out["best_economy_kcal_kg_km"] = safe_round(valid.min(), 3) if not valid.empty else None

    age = get_age(metadata)
    if age is not None:
        hr_pred = 208 - 0.7 * age
        out["HRpred_208_0.7age"] = safe_round(hr_pred, 0)
        if out.get("HRpeak_bpm") is not None:
            out["HRpeak_pct_pred"] = safe_round(out["HRpeak_bpm"] / hr_pred * 100, 0)

    flags = []
    if out.get("RERpeak") is not None:
        if out["RERpeak"] < 1.05:
            flags.append("RERpeak < 1.05: проверете мотивация, протокол и дали тестът е достигнал максимално усилие.")
        else:
            flags.append("RERpeak е съвместим с високо/максимално усилие.")
    if out.get("HRpeak_pct_pred") is not None:
        if out["HRpeak_pct_pred"] < 90:
            flags.append("HRpeak < 90% от предвидения максимум: възможен субмаксимален тест или индивидуално нисък HRmax.")
        else:
            flags.append("HRpeak е близо до предвидения максимум; използвайте индивидуалния контекст.")
    out["quality_flags"] = flags
    return out


def candidates_to_dataframe(candidates: list[ThresholdCandidate]) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame()
    df = pd.DataFrame([c.as_dict() for c in candidates])
    cols = [
        "threshold",
        "method",
        "time",
        "time_s",
        "speed_kmh",
        "grade_pct",
        "power_w",
        "vo2_ml_min",
        "vo2kg",
        "hr_bpm",
        "rer",
        "ve_vo2",
        "ve_vco2",
        "peto2",
        "petco2",
        "confidence",
        "comment",
        "row_index",
    ]
    return df[[c for c in cols if c in df.columns]]


def make_training_zones(vt1: dict[str, Any] | None, final_thr: dict[str, Any] | None, metric: str = "HR_bpm") -> pd.DataFrame:
    """Simple 3-zone model based on VT1 and final threshold.

    The app lets the expert define the final threshold as LT2/RCP/field-test consensus.
    """
    vt1_val = None if not vt1 else vt1.get(metric)
    final_val = None if not final_thr else final_thr.get(metric)
    rows = []
    if vt1_val is not None and np.isfinite(vt1_val):
        rows.append({"zone": "Z1 / below VT1", "lower": None, "upper": safe_round(vt1_val, 1), "purpose": "Extensive aerobic, recovery, low autonomic stress"})
    if vt1_val is not None and final_val is not None and np.isfinite(vt1_val) and np.isfinite(final_val):
        rows.append({"zone": "Z2 / VT1 to final threshold", "lower": safe_round(vt1_val, 1), "upper": safe_round(final_val, 1), "purpose": "Aerobic development / tempo depending on duration"})
        rows.append({"zone": "Z3 / above final threshold", "lower": safe_round(final_val, 1), "upper": None, "purpose": "Severe-intensity work; interval prescription only"})
    elif final_val is not None and np.isfinite(final_val):
        rows.append({"zone": "Below final threshold", "lower": None, "upper": safe_round(final_val, 1), "purpose": "Use when VT1 is not confirmed"})
        rows.append({"zone": "Above final threshold", "lower": safe_round(final_val, 1), "upper": None, "purpose": "High-intensity domain"})
    return pd.DataFrame(rows)


def empty_reference_template() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sport",
            "sex",
            "age_min",
            "age_max",
            "level",
            "metric",
            "unit",
            "higher_is_better",
            "p10",
            "p25",
            "p50",
            "p75",
            "p90",
            "source",
            "note",
        ]
    )


def load_reference_values(file: Any) -> pd.DataFrame:
    if file is None:
        return empty_reference_template()
    source = _to_bytes_io(file)
    name = str(getattr(file, "name", file))
    suffix = Path(name).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(source, engine="openpyxl")
    else:
        df = pd.read_csv(source)
    for c in ["age_min", "age_max", "p10", "p25", "p50", "p75", "p90"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "higher_is_better" in df.columns:
        df["higher_is_better"] = df["higher_is_better"].map(lambda x: str(x).strip().lower() not in {"false", "0", "no", "lower", "ниско"})
    return df


def classify_against_percentiles(value: float, row: pd.Series) -> tuple[str, str]:
    higher = bool(row.get("higher_is_better", True))
    thresholds = {p: row.get(p) for p in ["p10", "p25", "p50", "p75", "p90"]}
    if not all(pd.notna(v) for v in thresholds.values()):
        return "няма достатъчно данни", "Missing percentile columns"
    p10, p25, p50, p75, p90 = [float(thresholds[p]) for p in ["p10", "p25", "p50", "p75", "p90"]]
    v = float(value)
    if higher:
        if v < p10:
            return "много ниско", "<p10"
        if v < p25:
            return "ниско", "p10–p25"
        if v < p75:
            return "типично", "p25–p75"
        if v < p90:
            return "високо", "p75–p90"
        return "много високо", ">p90"
    # lower is better
    if v > p90:
        return "много ниско", ">p90 / worse"
    if v > p75:
        return "ниско", "p75–p90 / worse"
    if v > p25:
        return "типично", "p25–p75"
    if v > p10:
        return "високо", "p10–p25 / better"
    return "много високо", "<p10 / better"


def compare_to_references(metrics: dict[str, Any], references: pd.DataFrame, sport: str | None, sex: str | None, age: float | None) -> pd.DataFrame:
    if references is None or references.empty:
        return pd.DataFrame()
    refs = references.copy()
    for c in ["sport", "sex", "metric"]:
        if c not in refs.columns:
            return pd.DataFrame()
    sport_n = normalize_name(sport or "")
    sex_n = normalize_name(sex or "")
    rows = []
    for _, row in refs.iterrows():
        metric = row.get("metric")
        if metric not in metrics or metrics.get(metric) is None:
            continue
        ref_sport = normalize_name(row.get("sport", ""))
        ref_sex = normalize_name(row.get("sex", ""))
        if ref_sport not in {"", "all", "всички", sport_n}:
            continue
        if ref_sex not in {"", "all", "всички", sex_n}:
            continue
        if age is not None:
            amin = row.get("age_min")
            amax = row.get("age_max")
            if pd.notna(amin) and age < float(amin):
                continue
            if pd.notna(amax) and age > float(amax):
                continue
        cls, band = classify_against_percentiles(float(metrics[metric]), row)
        rows.append(
            {
                "metric": metric,
                "value": metrics[metric],
                "unit": row.get("unit"),
                "sport": row.get("sport"),
                "sex": row.get("sex"),
                "level": row.get("level"),
                "classification": cls,
                "band": band,
                "p10": row.get("p10"),
                "p25": row.get("p25"),
                "p50": row.get("p50"),
                "p75": row.get("p75"),
                "p90": row.get("p90"),
                "source": row.get("source"),
                "note": row.get("note"),
            }
        )
    return pd.DataFrame(rows)


def threshold_to_metric_dict(candidate: ThresholdCandidate | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "time_s": candidate.time_s,
        "speed_kmh": candidate.speed_kmh,
        "grade_pct": candidate.grade_pct,
        "power_w": candidate.power_w,
        "VO2_mL_min": candidate.vo2_ml_min,
        "VO2kg": candidate.vo2kg,
        "HR_bpm": candidate.hr_bpm,
        "RER": candidate.rer,
    }


def generate_rule_based_report(
    metadata: dict[str, Any],
    peak_metrics: dict[str, Any],
    gas_results: dict[str, Any],
    lactate_candidates: list[ThresholdCandidate] | None = None,
    expert_values: dict[str, Any] | None = None,
    reference_comparison: pd.DataFrame | None = None,
) -> str:
    """Bulgarian draft report. This is deterministic, no LLM required."""

    def fmt(v: Any, suffix: str = "") -> str:
        try:
            if v is None or not np.isfinite(float(v)):
                return "—"
            return f"{float(v):g}{suffix}"
        except Exception:
            return "—"

    name = " ".join(str(metadata.get(k, "")).strip() for k in ["First Name", "Last Name"]).strip() or "Състезател"
    sport = metadata.get("ID1") or metadata.get("Protocol") or "—"
    lines = []
    lines.append(f"## Функционален CPET анализ — {name}")
    lines.append("")
    lines.append(f"**Спорт/протокол:** {sport}")
    if metadata.get("Test date"):
        lines.append(f"**Дата:** {metadata.get('Test date')}")
    lines.append("")

    lines.append("### 1) Максимални/пикови показатели")
    peak_bits = []
    if peak_metrics.get("VO2peak_mL_kg_min") is not None:
        peak_bits.append(f"VO₂peak {peak_metrics['VO2peak_mL_kg_min']} ml/kg/min")
    if peak_metrics.get("VO2peak_mL_min") is not None:
        peak_bits.append(f"абсолютен VO₂ {peak_metrics['VO2peak_mL_min']} ml/min")
    if peak_metrics.get("HRpeak_bpm") is not None:
        peak_bits.append(f"HRpeak {peak_metrics['HRpeak_bpm']} bpm")
    if peak_metrics.get("RERpeak") is not None:
        peak_bits.append(f"RERpeak {peak_metrics['RERpeak']}")
    lines.append("; ".join(peak_bits) + "." if peak_bits else "Няма достатъчно данни за пикови показатели.")
    for flag in peak_metrics.get("quality_flags", []):
        lines.append(f"- {flag}")
    lines.append("")

    lines.append("### 2) Газообменни прагове")
    vt1 = gas_results.get("vt1")
    rcp = gas_results.get("rcp")
    fat = gas_results.get("fatmax")
    if vt1:
        lines.append(
            f"- **VT1 газообмен:** около {seconds_to_mmss(vt1.time_s)}; "
            f"HR≈{fmt(vt1.hr_bpm)} bpm, VO₂≈{fmt(vt1.vo2kg)} ml/kg/min, "
            f"скорост≈{fmt(vt1.speed_kmh)} km/h, наклон≈{fmt(vt1.grade_pct)}%. "
            f"Сигурност: {fmt(vt1.confidence)}%."
        )
    if rcp:
        lines.append(
            f"- **VT2/RCP газообмен:** около {seconds_to_mmss(rcp.time_s)}; "
            f"HR≈{fmt(rcp.hr_bpm)} bpm, VO₂≈{fmt(rcp.vo2kg)} ml/kg/min, "
            f"скорост≈{fmt(rcp.speed_kmh)} km/h, наклон≈{fmt(rcp.grade_pct)}%. "
            f"Сигурност: {fmt(rcp.confidence)}%."
        )
    if fat:
        fr = gas_results.get("fatmax_range", {})
        lines.append(
            f"- **FatMax:** около {seconds_to_mmss(fat.time_s)}; FatOx≈{fr.get('fatmax_g_min', '—')} g/min. "
            f"Практически диапазон: {seconds_to_mmss(fr.get('range_low_time_s'))}–{seconds_to_mmss(fr.get('range_high_time_s'))}."
        )
    if vt1 and rcp and vt1.confidence is not None and vt1.confidence < 50:
        lines.append("- VT1 кандидатите са раздалечени; препоръчва се визуална експертна проверка на V-slope, VE/VO₂, PETO₂ и PETCO₂.")
    if rcp and rcp.confidence is not None and rcp.confidence < 50:
        lines.append("- RCP кандидатите са раздалечени; препоръчва се експертна проверка на VE/VCO₂ и PETCO₂.")
    lines.append("")

    if lactate_candidates:
        lines.append("### 3) Лактатни прагове")
        for c in lactate_candidates:
            lines.append(
                f"- **{c.threshold} — {c.method}:** {seconds_to_mmss(c.time_s)}; "
                f"скорост≈{fmt(c.speed_kmh)} km/h, мощност≈{fmt(c.power_w)} W, HR≈{fmt(c.hr_bpm)} bpm."
            )
        lines.append("")

    if expert_values:
        lines.append("### 4) Експертно потвърдени стойности")
        for k, v in expert_values.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    if reference_comparison is not None and not reference_comparison.empty:
        lines.append("### 5) Сравнение с референтна база")
        for _, r in reference_comparison.iterrows():
            lines.append(
                f"- {r['metric']}: {r['value']} {r.get('unit','')}; класификация: **{r['classification']}** ({r['band']})."
            )
        lines.append("")

    lines.append("### 6) Интерпретация")
    lines.append(
        "Това е автоматично генериран проект на анализ. Кодът изчислява кандидат-прагове по отделни критерии, "
        "но финалният тренировъчен праг трябва да бъде експертен консенсус между газообмен, лактат, протокол, "
        "валидност на усилието и полеви тестове."
    )
    return "\n".join(lines)


def make_html_report(markdown_text: str, tables: dict[str, pd.DataFrame] | None = None) -> str:
    """Simple self-contained HTML report."""
    import html

    # Very small Markdown-to-HTML subset to avoid extra dependencies.
    lines = []
    for line in markdown_text.splitlines():
        if line.startswith("### "):
            lines.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("## "):
            lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            txt = html.escape(line[2:])
            txt = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", txt)
            lines.append(f"<li>{txt}</li>")
        elif line.strip() == "":
            lines.append("<br>")
        else:
            txt = html.escape(line)
            txt = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", txt)
            lines.append(f"<p>{txt}</p>")
    table_html = ""
    if tables:
        for name, df in tables.items():
            if df is not None and not df.empty:
                table_html += f"<h3>{html.escape(name)}</h3>" + df.to_html(index=False, border=0)
    css = """
    <style>
    body { font-family: Arial, sans-serif; margin: 36px; line-height: 1.45; color: #172033; }
    h2 { border-bottom: 2px solid #ddd; padding-bottom: 8px; }
    table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; }
    th { background: #f3f5f7; }
    li { margin-bottom: 6px; }
    </style>
    """
    return f"<!doctype html><html><head><meta charset='utf-8'>{css}</head><body>{''.join(lines)}{table_html}</body></html>"


def to_json_download(payload: dict[str, Any]) -> str:
    def default(o: Any):
        if isinstance(o, ThresholdCandidate):
            return o.as_dict()
        if isinstance(o, pd.DataFrame):
            return o.to_dict(orient="records")
        if isinstance(o, (np.integer, np.floating)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    return json.dumps(payload, ensure_ascii=False, indent=2, default=default)
