"""
Point-and-click interface.

    pip install streamlit
    streamlit run app.py

Nothing here does any calculation: it collects settings, calls the engine, and
shows what comes back. All the logic lives in spchart/.
"""
from __future__ import annotations

import datetime as dt
import io

import pandas as pd
import streamlit as st
import yaml

from spchart.config import ChartConfig, Underlying, Level, Texts, LEVEL_COLORS
from spchart.chart import build_figure
from spchart.datasource import LSEGSource, CsvSource, DemoSource
from spchart.engine import prepare, breach_report, autocall_report
from spchart.export import export_workbook

st.set_page_config(page_title="Structured product chart", layout="wide")

DEFAULT_LEVELS = pd.DataFrame([
    {"Show": True,  "Label": "Autocall trigger 100%", "% of initial": 100.0, "Colour": "green",  "Type": "autocall"},
    {"Show": True,  "Label": "Coupon barrier 70%",    "% of initial": 70.0,  "Colour": "gold",   "Type": "coupon"},
    {"Show": True,  "Label": "Capital barrier 60%",   "% of initial": 60.0,  "Colour": "red",    "Type": "capital"},
    {"Show": False, "Label": "Custom level",          "% of initial": 90.0,  "Colour": "purple", "Type": "generic"},
])
DEFAULT_UND = pd.DataFrame([
    {"RIC": ".STOXX50E", "Legend name": "EURO STOXX 50"},
    {"RIC": ".SPX",      "Legend name": "S&P 500"},
    {"RIC": "",          "Legend name": ""},
    {"RIC": "",          "Legend name": ""},
    {"RIC": "",          "Legend name": ""},
])


# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def fetch(source: str, path: str | None, rics: tuple[str, ...],
          start: dt.date, end: dt.date):
    src = {"Demo (no Reuters)": DemoSource,
           "Reuters / LSEG": LSEGSource,
           "CSV folder or file": lambda: CsvSource(path)}[source]
    return (src() if source != "CSV folder or file" else src()).history(
        list(rics), start, end)


# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("1 · Underlyings")
    source = st.selectbox("Data source",
                          ["Demo (no Reuters)", "Reuters / LSEG", "CSV folder or file"])
    csv_path = st.text_input("CSV path", "./prices") if source == "CSV folder or file" else None
    if source == "Reuters / LSEG":
        st.caption("Requires Workspace/Eikon running and signed in, and the "
                   "lseg-data (or refinitiv-data / eikon) package installed.")

    und_df = st.data_editor(DEFAULT_UND, num_rows="fixed", hide_index=True,
                            use_container_width=True, key="und")

    st.header("2 · Period & scale")
    c1, c2 = st.columns(2)
    start = c1.date_input("Start", dt.date.today() - dt.timedelta(days=365 * 3))
    end = c2.date_input("End", dt.date.today())
    fixing = st.date_input("Initial fixing date", start,
                           help="Strike date of the product: every underlying = 100 here.")
    scale = st.radio("Scale", ["rebased", "absolute"], horizontal=True)
    calendar = st.selectbox("Common calendar", ["union", "weekdays", "intersection"],
                            help="union = every day any market traded (recommended). "
                                 "intersection = only days all markets traded.")
    show_wo = st.checkbox("Show worst-of line", value=True)
    wo_label = st.text_input("Worst-of legend name", "Worst-of basket") if show_wo else "Worst-of basket"

    st.header("3 · Product levels")
    lvl_df = st.data_editor(
        DEFAULT_LEVELS, num_rows="dynamic", hide_index=True, use_container_width=True,
        column_config={
            "Colour": st.column_config.SelectboxColumn(options=list(LEVEL_COLORS)),
            "Type": st.column_config.SelectboxColumn(
                options=["autocall", "coupon", "capital", "generic"]),
            "% of initial": st.column_config.NumberColumn(format="%.1f %%"),
        }, key="lvl")

    ac_raw = st.text_area("Autocall observation dates (one per line, YYYY-MM-DD)", "")

    st.header("4 · Texts")
    title = st.text_input("Title", "Underlying performance")
    subtitle = st.text_input("Subtitle", "Daily closing prices, rebased to 100 at the initial fixing date")
    source_txt = st.text_input("Source line", "Source: LSEG / Refinitiv")
    disclaimer = st.text_area("Disclaimer",
                              "PLACEHOLDER - replace with your compliance-approved wording.")
    annotate = st.checkbox("Label the last value of each line", True)
    in_legend = st.checkbox("Put barrier labels in the legend instead of inline", False)


# --------------------------------------------------------------------------- #
underlyings = [Underlying(str(r["RIC"]).strip(), str(r["Legend name"]).strip())
               for _, r in und_df.iterrows() if str(r["RIC"]).strip()]
levels = [Level(label=str(r["Label"]), pct=float(r["% of initial"]) / 100.0,
                color=str(r["Colour"]), enabled=bool(r["Show"]), kind=str(r["Type"]))
          for _, r in lvl_df.iterrows() if str(r.get("Label", "")).strip()]

ac_dates = []
for line in ac_raw.splitlines():
    line = line.strip()
    if line:
        try:
            ac_dates.append(dt.date.fromisoformat(line))
        except ValueError:
            st.sidebar.warning(f"Ignored autocall date: {line}")

st.title("Structured product — performance chart")

if not underlyings:
    st.info("Enter at least one Reuters code in the sidebar.")
    st.stop()
if end < start:
    st.error("End date is before start date.")
    st.stop()

cfg = ChartConfig(
    underlyings=underlyings, start=start, end=end, fixing_date=fixing,
    scale=scale, calendar=calendar, show_worst_of=show_wo, worst_of_label=wo_label,
    levels=levels, autocall_dates=ac_dates, annotate_last=annotate,
    levels_in_legend=in_legend,
    texts=Texts(title=title, subtitle=subtitle, source=source_txt, disclaimer=disclaimer),
)

try:
    with st.spinner("Fetching prices…"):
        data = fetch(source, csv_path, tuple(cfg.rics), start, end)
except Exception as exc:
    st.error(f"Could not fetch data: {exc}")
    st.stop()

prep = prepare(data, cfg)
for w in prep.warnings:
    st.warning(w)

if prep.empty:
    st.error("Nothing to plot.")
    st.stop()

fig = build_figure(prep, cfg)
st.pyplot(fig, use_container_width=True)

png = io.BytesIO(); fig.savefig(png, format="png", dpi=200, facecolor="white")
pdf = io.BytesIO(); fig.savefig(pdf, format="pdf", facecolor="white")
xlsx_path = export_workbook(prep, cfg, "/tmp/spchart_data.xlsx")

d1, d2, d3, d4 = st.columns(4)
d1.download_button("PNG (slides / email)", png.getvalue(), "chart.png", "image/png")
d2.download_button("PDF (print)", pdf.getvalue(), "chart.pdf", "application/pdf")
with open(xlsx_path, "rb") as fh:
    d3.download_button("Data workbook", fh.read(), "chart_data.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
d4.download_button("Config (YAML)", yaml.safe_dump(cfg.to_dict(), sort_keys=False),
                   "config.yaml", "text/yaml")

t1, t2, t3, t4 = st.tabs(["Data quality", "Barriers", "Autocall", "Chart data"])
with t1:
    st.dataframe(prep.quality, use_container_width=True, hide_index=True)
    st.caption("‘Carried fwd’ = days where that market was closed and the previous "
               "close was reused. High values mean the two markets share few trading days.")
with t2:
    br = breach_report(prep, cfg)
    st.dataframe(br, use_container_width=True, hide_index=True) if not br.empty \
        else st.caption("No downside barrier defined.")
    st.caption("Closing prices only. An American knock-in observed on intraday lows "
               "is not captured here.")
with t3:
    ac = autocall_report(prep, cfg)
    st.dataframe(ac, use_container_width=True, hide_index=True) if not ac.empty \
        else st.caption("Add an autocall level and observation dates to see this.")
with t4:
    st.dataframe(prep.plotted, use_container_width=True)
