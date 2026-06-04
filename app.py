from __future__ import annotations

from io import BytesIO
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cpet_core import (
    aggregate_stages,
    candidates_to_dataframe,
    canonicalize_cpet,
    choose_analysis_dataframe,
    compare_to_references,
    compute_peak_metrics,
    detect_all_gas_thresholds,
    detect_lactate_thresholds,
    empty_reference_template,
    extract_cpet_table,
    generate_rule_based_report,
    get_age,
    get_body_mass,
    get_sex,
    load_lactate_table,
    load_reference_values,
    make_html_report,
    make_training_zones,
    nearest_idx_by_time,
    read_raw_table,
    seconds_to_mmss,
    standardize_lactate_table,
    threshold_to_metric_dict,
    to_json_download,
)

st.set_page_config(
    page_title="CPET Functional Diagnostics Lab",
    page_icon="🫁",
    layout="wide",
)


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------


def metric_card(label: str, value, unit: str = ""):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        st.metric(label, "—")
    else:
        st.metric(label, f"{value} {unit}".strip())


def time_axis(df: pd.DataFrame) -> pd.Series:
    return df["time_s"] / 60.0


def add_threshold_lines(fig: go.Figure, threshold_times: dict[str, float | None]):
    for name, t in threshold_times.items():
        if t is None or not np.isfinite(t):
            continue
        fig.add_vline(
            x=float(t) / 60.0,
            line_dash="dash",
            annotation_text=name,
            annotation_position="top left",
        )


def make_time_plot(df: pd.DataFrame, cols: list[str], title: str, y_title: str, threshold_times: dict[str, float | None]):
    fig = go.Figure()
    x = time_axis(df)
    for col in cols:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=x, y=df[col], mode="lines+markers", name=col))
        elif f"{col}_sm" in df.columns:
            fig.add_trace(go.Scatter(x=x, y=df[f"{col}_sm"], mode="lines+markers", name=f"{col}_sm"))
    add_threshold_lines(fig, threshold_times)
    fig.update_layout(title=title, xaxis_title="Време (min)", yaxis_title=y_title, hovermode="x unified", height=430)
    return fig


def make_vslope_plot(df: pd.DataFrame, threshold_times: dict[str, float | None]):
    fig = go.Figure()
    xcol = "VO2_mL_min_sm" if "VO2_mL_min_sm" in df.columns else "VO2_mL_min"
    ycol = "VCO2_mL_min_sm" if "VCO2_mL_min_sm" in df.columns else "VCO2_mL_min"
    fig.add_trace(
        go.Scatter(
            x=df[xcol],
            y=df[ycol],
            mode="markers+lines",
            text=(df["time_s"] / 60).round(1).astype(str) + " min",
            name="VCO2 vs VO2",
        )
    )
    for name, t in threshold_times.items():
        if t is None or not np.isfinite(t):
            continue
        idx = nearest_idx_by_time(df, t)
        if idx is not None and idx in df.index:
            fig.add_trace(
                go.Scatter(
                    x=[df.loc[idx, xcol]],
                    y=[df.loc[idx, ycol]],
                    mode="markers+text",
                    text=[name],
                    textposition="top center",
                    marker=dict(size=12, symbol="x"),
                    name=name,
                )
            )
    fig.update_layout(title="V-slope: VCO₂ спрямо VO₂", xaxis_title="VO₂ (mL/min)", yaxis_title="VCO₂ (mL/min)", height=430)
    return fig


def row_metrics_at_time(df: pd.DataFrame, t: float | None) -> dict:
    idx = nearest_idx_by_time(df, t)
    if idx is None:
        return {}
    row = df.loc[idx]
    return {
        "time_s": row.get("time_s"),
        "time": seconds_to_mmss(row.get("time_s")),
        "speed_kmh": row.get("speed_kmh"),
        "grade_pct": row.get("grade_pct"),
        "power_w": row.get("power_w"),
        "VO2_mL_min": row.get("VO2_mL_min"),
        "VO2kg": row.get("VO2kg"),
        "HR_bpm": row.get("HR_bpm"),
        "RER": row.get("RER"),
        "VE_VO2": row.get("VE_VO2"),
        "VE_VCO2": row.get("VE_VCO2"),
        "PETO2": row.get("PETO2"),
        "PETCO2": row.get("PETCO2"),
        "fat_g_min": row.get("fat_g_min"),
        "kcal_kg_km": row.get("kcal_kg_km"),
    }


def time_slider(label: str, df: pd.DataFrame, default_s: float | None, key: str) -> float | None:
    if df.empty or "time_s" not in df.columns:
        return None
    min_s = float(df["time_s"].min())
    max_s = float(df["time_s"].max())
    if default_s is None or not np.isfinite(default_s):
        default_s = min_s
    default_s = min(max(float(default_s), min_s), max_s)
    val = st.slider(
        label,
        min_value=min_s,
        max_value=max_s,
        value=float(default_s),
        step=15.0,
        format="%.0f s",
        key=key,
    )
    st.caption(f"Избрано време: {seconds_to_mmss(val)}")
    return float(val)


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------

st.title("🫁 CPET Functional Diagnostics Lab")
st.caption("Прототип за стандартизирана обработка на CPET, прагове, FatMax, икономичност, лактат и експертна корекция.")

with st.sidebar:
    st.header("1) Файл")
    uploaded = st.file_uploader("Качи CPET файл", type=["xlsx", "xls", "csv"])

    st.header("2) Обработка")
    analysis_mode = st.selectbox(
        "Режим за прагове",
        ["auto", "stage_endpoint", "bxb_smoothed"],
        help="auto избира stage endpoints, ако разпознае стъпаловиден протокол. bxb_smoothed използва изгладените редови данни.",
    )
    smooth_window = st.slider("Изглаждане на редовите данни — брой точки", 1, 15, 5, step=2)
    min_stage_duration = st.slider("Минимална продължителност на работна стъпка (s)", 15, 180, 45, step=15)
    last_stage_window = st.slider("Stage endpoint: последни N секунди за средно", 15, 120, 45, step=15)
    rer_support = st.slider("RER маркер за RCP support", 0.95, 1.10, 1.00, step=0.01)
    fatmax_pct = st.slider("FatMax диапазон — % от максимума", 0.80, 0.98, 0.90, step=0.01)

    st.header("3) Ръчни настройки")
    body_mass_override = st.number_input("Ръчна телесна маса, kg — 0 = от файла", min_value=0.0, max_value=250.0, value=0.0, step=0.1)
    body_mass_override = body_mass_override if body_mass_override > 0 else None

if uploaded is None:
    st.info("Качи CPET Excel/CSV файл, за да стартира анализът. Приложението е направено за COSMED-подобни файлове, но има автоматично разпознаване на колоните.")

    st.subheader("Шаблони за лабораторна база")
    ref_template = empty_reference_template()
    st.download_button(
        "Изтегли шаблон за референтни стойности CSV",
        data=ref_template.to_csv(index=False).encode("utf-8-sig"),
        file_name="reference_values_template.csv",
        mime="text/csv",
    )
    lactate_template = pd.DataFrame(
        {
            "time_s": [300, 600, 900, 1200],
            "speed_kmh": [10, 12, 14, 16],
            "power_w": [np.nan, np.nan, np.nan, np.nan],
            "lactate_mmol_L": [1.2, 1.6, 3.0, 6.5],
            "HR_bpm": [120, 145, 168, 186],
        }
    )
    st.download_button(
        "Изтегли шаблон за лактат CSV",
        data=lactate_template.to_csv(index=False).encode("utf-8-sig"),
        file_name="lactate_template.csv",
        mime="text/csv",
    )
    st.stop()

try:
    # First pass to expose sheet names; second pass with selected sheet.
    raw_auto, file_info_auto = read_raw_table(uploaded, sheet_name=None)
    selected_sheet = file_info_auto.get("selected_sheet")
    available_sheets = file_info_auto.get("available_sheets", [selected_sheet])
    if len(available_sheets) > 1:
        with st.sidebar:
            selected_sheet = st.selectbox("Лист в Excel", available_sheets, index=available_sheets.index(selected_sheet))
    raw, file_info = read_raw_table(uploaded, sheet_name=selected_sheet)
    raw_data, metadata, extraction_info = extract_cpet_table(raw)
    cpet_df, colmap, body_mass = canonicalize_cpet(raw_data, metadata, smooth_window=smooth_window, body_mass_override=body_mass_override)
    stage_df, stage_info = aggregate_stages(cpet_df, min_duration_s=min_stage_duration, last_window_s=last_stage_window)
    analysis_df, used_mode = choose_analysis_dataframe(cpet_df, stage_df, mode=analysis_mode)

    if analysis_df.empty:
        st.error("Файлът беше прочетен, но не останаха валидни CPET редове за анализ.")
        st.stop()

    gas_results = detect_all_gas_thresholds(analysis_df, rer_support=rer_support, fatmax_pct=fatmax_pct)
    peak_metrics = compute_peak_metrics(cpet_df, metadata, body_mass=body_mass)
    # Economy and substrate metrics are more meaningful on the same analysis
    # dataset used for thresholds, especially with step protocols and transition rows.
    analysis_metrics = compute_peak_metrics(analysis_df, metadata, body_mass=body_mass)
    for k in ["FatOx_max_g_min", "best_economy_kcal_kg_km"]:
        if analysis_metrics.get(k) is not None:
            peak_metrics[k] = analysis_metrics[k]

except Exception as e:
    st.error("Възникна грешка при обработката на файла.")
    st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))
    st.stop()

# Header summary
with st.expander("Разпозната структура на файла", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Избран лист:** {file_info.get('selected_sheet')}")
    c2.write("**Ред със заглавия:**", extraction_info.get("header_row_1based"))
    c3.write("**Първи ред с данни:**", extraction_info.get("data_start_row_1based"))
    st.write("**Разпознати колони:**")
    st.json(colmap)
    st.write("**Метаданни:**")
    st.json({k: str(v) for k, v in metadata.items()})

st.success(f"Файлът е обработен. За праговете се използва режим: **{used_mode}**. Редове: raw={len(cpet_df)}, stage={len(stage_df)}, analysis={len(analysis_df)}.")

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------

tab_qc, tab_thr, tab_graphs, tab_lac, tab_refs, tab_report, tab_data = st.tabs(
    ["QC & пикови", "Прагове + експерт", "Графики", "Лактат", "Референции", "Доклад", "Данни/експорт"]
)

with tab_qc:
    st.subheader("Пикови показатели и валидност")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        metric_card("VO₂peak", peak_metrics.get("VO2peak_mL_kg_min"), "ml/kg/min")
    with m2:
        metric_card("VO₂peak abs", peak_metrics.get("VO2peak_mL_min"), "ml/min")
    with m3:
        metric_card("HRpeak", peak_metrics.get("HRpeak_bpm"), "bpm")
    with m4:
        metric_card("RERpeak", peak_metrics.get("RERpeak"), "")
    with m5:
        metric_card("VEmax", peak_metrics.get("VEmax_L_min"), "L/min")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("FatOx max", peak_metrics.get("FatOx_max_g_min"), "g/min")
    with c2:
        metric_card("Best economy", peak_metrics.get("best_economy_kcal_kg_km"), "kcal/kg/km")
    with c3:
        metric_card("Max speed", peak_metrics.get("speed_max_kmh"), "km/h")
    with c4:
        metric_card("Max grade", peak_metrics.get("grade_max_pct"), "%")

    if peak_metrics.get("quality_flags"):
        st.write("**Автоматични QC бележки:**")
        for flag in peak_metrics["quality_flags"]:
            st.write(f"- {flag}")

    st.write("**Stage detection / кратки преходи**")
    st.dataframe(stage_info, use_container_width=True)

with tab_thr:
    st.subheader("Автоматични кандидат-прагове")
    consensus_df = candidates_to_dataframe(gas_results["consensus"])
    method_df = candidates_to_dataframe(gas_results["method_candidates"])

    st.write("**Консенсус предложения**")
    st.dataframe(consensus_df, use_container_width=True, hide_index=True)

    st.write("**Кандидати по отделни методи**")
    st.dataframe(method_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Експертна корекция")
    st.caption("Плъзгачите не променят суровите изчисления; те записват експертно потвърдените времена за доклада и зоните.")

    vt1_default = gas_results["vt1"].time_s if gas_results.get("vt1") else None
    rcp_default = gas_results["rcp"].time_s if gas_results.get("rcp") else None
    fat_default = gas_results["fatmax"].time_s if gas_results.get("fatmax") else None

    c1, c2, c3 = st.columns(3)
    with c1:
        expert_vt1_t = time_slider("VT1 expert time", analysis_df, vt1_default, "expert_vt1")
    with c2:
        expert_rcp_t = time_slider("VT2/RCP expert time", analysis_df, rcp_default, "expert_rcp")
    with c3:
        expert_fat_t = time_slider("FatMax expert center", analysis_df, fat_default, "expert_fat")

    final_default = expert_rcp_t if expert_rcp_t is not None else rcp_default
    expert_final_t = time_slider("Финален тренировъчен праг — LT2/RCP/field consensus", analysis_df, final_default, "expert_final")

    expert_rows = {
        "VT1_expert": row_metrics_at_time(analysis_df, expert_vt1_t),
        "RCP_expert": row_metrics_at_time(analysis_df, expert_rcp_t),
        "FatMax_expert": row_metrics_at_time(analysis_df, expert_fat_t),
        "Final_training_threshold": row_metrics_at_time(analysis_df, expert_final_t),
    }
    st.write("**Експертно избрани редове**")
    st.dataframe(pd.DataFrame(expert_rows).T, use_container_width=True)

    st.write("**Прости тренировъчни зони — по HR**")
    zones_hr = make_training_zones(expert_rows.get("VT1_expert"), expert_rows.get("Final_training_threshold"), metric="HR_bpm")
    st.dataframe(zones_hr, use_container_width=True, hide_index=True)

    if "speed_kmh" in analysis_df.columns:
        st.write("**Прости тренировъчни зони — по скорост**")
        zones_speed = make_training_zones(expert_rows.get("VT1_expert"), expert_rows.get("Final_training_threshold"), metric="speed_kmh")
        st.dataframe(zones_speed, use_container_width=True, hide_index=True)

with tab_graphs:
    st.subheader("Графики за визуална експертна оценка")
    threshold_times = {
        "VT1": expert_vt1_t if "expert_vt1_t" in globals() else vt1_default,
        "RCP": expert_rcp_t if "expert_rcp_t" in globals() else rcp_default,
        "FatMax": expert_fat_t if "expert_fat_t" in globals() else fat_default,
        "Final": expert_final_t if "expert_final_t" in globals() else rcp_default,
    }

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(make_time_plot(analysis_df, ["VO2_mL_min", "VCO2_mL_min"], "VO₂ и VCO₂", "mL/min", threshold_times), use_container_width=True)
    with g2:
        st.plotly_chart(make_vslope_plot(analysis_df, threshold_times), use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        st.plotly_chart(make_time_plot(analysis_df, ["VE_VO2", "VE_VCO2"], "Вентилаторни еквиваленти", "ratio", threshold_times), use_container_width=True)
    with g4:
        st.plotly_chart(make_time_plot(analysis_df, ["PETO2", "PETCO2"], "PETO₂ и PETCO₂", "mmHg", threshold_times), use_container_width=True)

    g5, g6 = st.columns(2)
    with g5:
        st.plotly_chart(make_time_plot(analysis_df, ["fat_g_min", "cho_g_min"], "Субстратно окисление", "g/min", threshold_times), use_container_width=True)
    with g6:
        st.plotly_chart(make_time_plot(analysis_df, ["kcal_kg_km", "o2_cost_ml_kg_m"], "Икономичност", "cost", threshold_times), use_container_width=True)

with tab_lac:
    st.subheader("Лактат: LT1, Dmax, сегментирана регресия")
    st.caption("Може да качиш отделен lactate CSV/XLSX или да попълниш таблицата ръчно.")
    lactate_file = st.file_uploader("Качи лактатен файл", type=["xlsx", "xls", "csv"], key="lactate_file")

    lactate_df = None
    lactate_candidates = []
    lactate_xcol = None

    if lactate_file is not None:
        try:
            lactate_df = load_lactate_table(lactate_file)
        except Exception as e:
            st.error(f"Неуспешно четене на лактатен файл: {e}")

    if lactate_df is None:
        default_lac = pd.DataFrame(
            {
                "time_s": [np.nan, np.nan, np.nan, np.nan, np.nan],
                "speed_kmh": [np.nan, np.nan, np.nan, np.nan, np.nan],
                "power_w": [np.nan, np.nan, np.nan, np.nan, np.nan],
                "lactate_mmol_L": [np.nan, np.nan, np.nan, np.nan, np.nan],
                "HR_bpm": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        lactate_df = st.data_editor(default_lac, num_rows="dynamic", use_container_width=True, key="manual_lactate")
    else:
        st.dataframe(lactate_df, use_container_width=True)

    if lactate_df is not None and "lactate_mmol_L" in lactate_df.columns and lactate_df["lactate_mmol_L"].notna().sum() >= 3:
        try:
            lactate_candidates, lactate_xcol = detect_lactate_thresholds(lactate_df)
            lac_df = candidates_to_dataframe(lactate_candidates)
            st.write(f"**Използвана ос за интензивност:** {lactate_xcol}")
            st.dataframe(lac_df, use_container_width=True, hide_index=True)

            # Plot lactate curve.
            fig = go.Figure()
            ldf = standardize_lactate_table(lactate_df)
            xcol = lactate_xcol
            fig.add_trace(go.Scatter(x=ldf[xcol], y=ldf["lactate_mmol_L"], mode="markers+lines", name="Lactate"))
            for c in lactate_candidates:
                if c.speed_kmh is not None and xcol == "speed_kmh":
                    x = c.speed_kmh
                elif c.power_w is not None and xcol == "power_w":
                    x = c.power_w
                elif c.time_s is not None and xcol == "time_s":
                    x = c.time_s
                else:
                    continue
                fig.add_vline(x=x, line_dash="dash", annotation_text=f"{c.threshold} {c.method}")
            fig.update_layout(title="Лактатна крива", xaxis_title=xcol, yaxis_title="Lactate (mmol/L)", height=430)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Лактатните прагове не могат да бъдат изчислени: {e}")
    else:
        st.info("Нужни са поне 3 валидни лактатни точки.")

    lactate_template = pd.DataFrame(
        {
            "time_s": [300, 600, 900, 1200],
            "speed_kmh": [10, 12, 14, 16],
            "power_w": [np.nan, np.nan, np.nan, np.nan],
            "lactate_mmol_L": [1.2, 1.6, 3.0, 6.5],
            "HR_bpm": [120, 145, 168, 186],
        }
    )
    st.download_button("Изтегли lactate_template.csv", lactate_template.to_csv(index=False).encode("utf-8-sig"), "lactate_template.csv", "text/csv")

with tab_refs:
    st.subheader("Референтни стойности и спортно-специфична оценка")
    st.caption("Качи собствена база с перцентили. В приложението няма фиктивни спортни норми, за да не създава фалшива точност.")
    ref_file = st.file_uploader("Качи reference_values CSV/XLSX", type=["xlsx", "xls", "csv"], key="ref_file")

    age_value = get_age(metadata)
    sex_value = get_sex(metadata) or "all"
    sport_default = str(metadata.get("ID1") or metadata.get("Protocol") or "all")

    c1, c2, c3 = st.columns(3)
    with c1:
        sport_value = st.text_input("Спорт за сравнение", value=sport_default)
    with c2:
        sex_value = st.text_input("Пол за сравнение", value=str(sex_value))
    with c3:
        age_value = st.number_input("Възраст", min_value=5.0, max_value=100.0, value=float(age_value or 25.0), step=0.1)

    references = load_reference_values(ref_file) if ref_file is not None else empty_reference_template()

    metrics_for_refs = dict(peak_metrics)
    if gas_results.get("vt1"):
        metrics_for_refs["VT1_HR_bpm"] = gas_results["vt1"].hr_bpm
        metrics_for_refs["VT1_VO2kg"] = gas_results["vt1"].vo2kg
        metrics_for_refs["VT1_speed_kmh"] = gas_results["vt1"].speed_kmh
    if gas_results.get("rcp"):
        metrics_for_refs["RCP_HR_bpm"] = gas_results["rcp"].hr_bpm
        metrics_for_refs["RCP_VO2kg"] = gas_results["rcp"].vo2kg
        metrics_for_refs["RCP_speed_kmh"] = gas_results["rcp"].speed_kmh
    if gas_results.get("fatmax_range"):
        metrics_for_refs["FatMax_g_min"] = gas_results["fatmax_range"].get("fatmax_g_min")

    ref_comparison = compare_to_references(metrics_for_refs, references, sport_value, sex_value, age_value)
    if ref_file is not None:
        st.write("**Заредени референции:**")
        st.dataframe(references, use_container_width=True)
        st.write("**Сравнение:**")
        if ref_comparison.empty:
            st.warning("Няма съвпадащи редове в референтната база за текущите показатели/спорт/пол/възраст.")
        else:
            st.dataframe(ref_comparison, use_container_width=True, hide_index=True)
    else:
        st.info("Качи референтна база или изтегли шаблона и попълни свои стойности.")

    st.download_button(
        "Изтегли reference_values_template.csv",
        data=empty_reference_template().to_csv(index=False).encode("utf-8-sig"),
        file_name="reference_values_template.csv",
        mime="text/csv",
    )

with tab_report:
    st.subheader("Автоматичен проект на доклад")
    try:
        lactate_candidates_for_report = lactate_candidates if "lactate_candidates" in globals() else []
        ref_comparison_for_report = ref_comparison if "ref_comparison" in globals() else pd.DataFrame()
    except Exception:
        lactate_candidates_for_report = []
        ref_comparison_for_report = pd.DataFrame()

    expert_values = {}
    try:
        expert_values = {
            "VT1 expert": row_metrics_at_time(analysis_df, expert_vt1_t).get("time"),
            "VT2/RCP expert": row_metrics_at_time(analysis_df, expert_rcp_t).get("time"),
            "FatMax expert": row_metrics_at_time(analysis_df, expert_fat_t).get("time"),
            "Final training threshold": row_metrics_at_time(analysis_df, expert_final_t).get("time"),
        }
    except Exception:
        pass

    report_md = generate_rule_based_report(
        metadata=metadata,
        peak_metrics=peak_metrics,
        gas_results=gas_results,
        lactate_candidates=lactate_candidates_for_report,
        expert_values=expert_values,
        reference_comparison=ref_comparison_for_report,
    )
    edited_report = st.text_area("Редактируем доклад", value=report_md, height=600)

    html_report = make_html_report(
        edited_report,
        tables={
            "Consensus thresholds": consensus_df,
            "Method candidates": method_df,
            "Reference comparison": ref_comparison_for_report,
        },
    )
    st.download_button(
        "Изтегли HTML доклад",
        data=html_report.encode("utf-8"),
        file_name="cpet_report.html",
        mime="text/html",
    )

with tab_data:
    st.subheader("Данни и експорт")
    st.write("**Аналитичен dataset**")
    st.dataframe(analysis_df, use_container_width=True)

    st.write("**Сурови канонични редове**")
    st.dataframe(cpet_df, use_container_width=True)

    payload = {
        "metadata": {k: str(v) for k, v in metadata.items()},
        "file_info": file_info,
        "extraction_info": extraction_info,
        "used_mode": used_mode,
        "peak_metrics": peak_metrics,
        "gas_consensus": gas_results["consensus"],
        "gas_method_candidates": gas_results["method_candidates"],
        "fatmax_range": gas_results.get("fatmax_range"),
        "expert_rows": expert_rows if "expert_rows" in globals() else {},
    }

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Изтегли analysis_data.csv", analysis_df.to_csv(index=False).encode("utf-8-sig"), "analysis_data.csv", "text/csv")
    with c2:
        st.download_button("Изтегли thresholds.csv", pd.concat([consensus_df, method_df], ignore_index=True).to_csv(index=False).encode("utf-8-sig"), "thresholds.csv", "text/csv")
    with c3:
        st.download_button("Изтегли full_results.json", to_json_download(payload).encode("utf-8"), "full_results.json", "application/json")
