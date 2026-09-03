#!/usr/bin/env python3
"""
Build the chart from a YAML config.

    python run.py config.yaml                    # uses the source named in the file
    python run.py config.yaml --source demo      # no Reuters needed
    python run.py config.yaml --source lseg      # needs Workspace running
    python run.py config.yaml --source csv --path ./prices
    python run.py config.yaml --outdir output --stem eurostoxx_vs_spx
"""
from __future__ import annotations

import argparse
import sys

import yaml

from spchart.config import ChartConfig
from spchart.chart import build_figure
from spchart.datasource import LSEGSource, CsvSource, DemoSource
from spchart.engine import prepare, breach_report, autocall_report
from spchart.export import export_images, export_workbook


def make_source(name: str, path: str | None):
    if name == "lseg":
        return LSEGSource()
    if name == "csv":
        if not path:
            sys.exit("--path is required with --source csv")
        return CsvSource(path)
    return DemoSource()


def main() -> int:
    ap = argparse.ArgumentParser(description="Structured product performance chart")
    ap.add_argument("config", help="YAML config file")
    ap.add_argument("--source", choices=["lseg", "csv", "demo"], default=None)
    ap.add_argument("--path", default=None, help="folder or file, for --source csv")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--stem", default="chart")
    ap.add_argument("--formats", default="png,pdf")
    ap.add_argument("--no-excel", action="store_true")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = ChartConfig.from_dict(raw)
    src_name = args.source or raw.get("source", "demo")
    src_path = args.path or raw.get("source_path")

    if not cfg.rics:
        sys.exit("No RICs in the config.")

    print(f"Source: {src_name}   RICs: {', '.join(cfg.rics)}")
    print(f"Period: {cfg.start} -> {cfg.end}   fixing {cfg.fixing_date}   scale {cfg.scale}")

    data = make_source(src_name, src_path).history(cfg.rics, cfg.start, cfg.end)
    prep = prepare(data, cfg)

    print("\n--- data quality ---")
    print(prep.quality.to_string(index=False) if not prep.quality.empty else "(nothing)")
    for w in prep.warnings:
        print(f"  ! {w}")

    br = breach_report(prep, cfg)
    if not br.empty:
        print("\n--- barriers (closing basis) ---")
        print(br.to_string(index=False))
    ac = autocall_report(prep, cfg)
    if not ac.empty:
        print("\n--- autocall observations ---")
        print(ac.to_string(index=False))

    if prep.empty:
        sys.exit("\nNothing to plot.")

    fig = build_figure(prep, cfg)
    imgs = export_images(fig, args.outdir, args.stem,
                         tuple(f.strip() for f in args.formats.split(",") if f.strip()))
    outs = list(imgs)
    if not args.no_excel:
        outs.append(export_workbook(prep, cfg, f"{args.outdir}/{args.stem}_data.xlsx"))

    print("\n--- written ---")
    for p in outs:
        print(" ", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
