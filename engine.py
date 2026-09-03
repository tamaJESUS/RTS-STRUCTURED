"""
Alignment / rebasing engine.

The whole point: several RICs never share the same trading calendar. This module
puts every series on ONE date axis, carries the last close forward where a market
was closed, and rebases everything at the SAME initial fixing date, so a worst-of
basket is actually comparable.

Pure pandas, no I/O, no plotting - which is what makes it testable.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ChartConfig, CalendarMode


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def normalise(series: pd.Series) -> pd.Series:
    """Datetime index -> sorted daily index, no duplicates, no NaN.

    Tolerant on purpose: Reuters may return timestamps (17:35), descending order,
    or two prints on the same day. All three are handled here rather than trusted.
    """
    s = pd.Series(series).dropna()
    if s.empty:
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([], name="date"))
    idx = pd.to_datetime(s.index).normalize()          # strip the time component
    s = pd.Series(np.asarray(s, dtype="float64"), index=idx)
    s = s[~s.index.duplicated(keep="last")]            # one close per day
    s = s.sort_index()                                 # order-agnostic
    s.index.name = "date"
    return s


# --------------------------------------------------------------------------- #
# calendar
# --------------------------------------------------------------------------- #
def build_calendar(series_map: dict[str, pd.Series], start: dt.date, end: dt.date,
                   mode: CalendarMode = "union") -> pd.DatetimeIndex:
    """One date axis shared by every underlying."""
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    if mode == "weekdays":
        return pd.bdate_range(lo, hi, name="date")

    windows = []
    for s in series_map.values():
        if not s.empty:
            windows.append(s.index[(s.index >= lo) & (s.index <= hi)])
    if not windows:
        return pd.DatetimeIndex([], name="date")

    if mode == "intersection":
        idx = windows[0]
        for w in windows[1:]:
            idx = idx.intersection(w)
    else:                                              # union (default)
        idx = windows[0]
        for w in windows[1:]:
            idx = idx.union(w)
    return pd.DatetimeIndex(sorted(idx), name="date")


# --------------------------------------------------------------------------- #
# result container
# --------------------------------------------------------------------------- #
@dataclass
class Prepared:
    prices: pd.DataFrame          # aligned raw prices, forward-filled
    observed: pd.DataFrame        # True where a genuine quote existed that day
    plotted: pd.DataFrame         # rebased to 100, or raw prices in absolute mode
    worst_of: pd.Series           # min across underlyings (NaN if any is missing)
    fixings: pd.Series            # initial price used to rebase each underlying
    quality: pd.DataFrame         # one row per underlying
    base: float                   # 100 when rebased, else the fixing of underlying 1
    calendar: pd.DatetimeIndex
    warnings: list[str]

    @property
    def empty(self) -> bool:
        return self.plotted.empty or self.plotted.dropna(how="all").empty


ROLL_TOLERANCE_DAYS = 7


def _fixing_value(s: pd.Series, fixing: pd.Timestamp) -> tuple[float, str, pd.Timestamp | None]:
    """Resolve the initial fixing.

    Returns (value, how, date_used) where `how` is:
      exact   - a close on or before the fixing date (the normal case)
      rolled  - the fixing fell on a weekend/holiday, so the next close within a
                week was used
      late    - the series only starts well after the fixing date; the first
                available close is used and the caller must warn loudly
      none    - no data at all
    """
    if s.empty:
        return float("nan"), "none", None
    upto = s[s.index <= fixing]
    if len(upto):
        return float(upto.iloc[-1]), "exact", upto.index[-1]
    nxt = s[s.index > fixing]
    if len(nxt):
        gap = (nxt.index[0] - fixing).days
        how = "rolled" if gap <= ROLL_TOLERANCE_DAYS else "late"
        return float(nxt.iloc[0]), how, nxt.index[0]
    return float("nan"), "none", None


def prepare(series_map: dict[str, pd.Series], cfg: ChartConfig) -> Prepared:
    """Turn raw per-RIC histories into everything the chart needs."""
    warnings: list[str] = []
    labels = {u.ric: u.label() for u in cfg.active}
    clean = {ric: normalise(series_map.get(ric, pd.Series(dtype=float)))
             for ric in cfg.rics}

    cal = build_calendar(clean, cfg.start, cfg.end, cfg.calendar)
    if len(cal) == 0:
        empty_df = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
        return Prepared(empty_df, empty_df, empty_df,
                        pd.Series(dtype=float), pd.Series(dtype=float),
                        pd.DataFrame(), 100.0, cal,
                        warnings + ["No data at all - check the RICs and the period."])

    prices, observed = {}, {}
    for ric, s in clean.items():
        label = labels[ric]
        on_cal = s.reindex(cal)
        observed[label] = on_cal.notna()
        # forward-fill only AFTER the first real quote: never invent history
        prices[label] = on_cal.ffill()

    prices = pd.DataFrame(prices, index=cal)
    observed = pd.DataFrame(observed, index=cal)

    # ---- rebasing --------------------------------------------------------- #
    fixing_ts = pd.Timestamp(cfg.fixing_date)
    resolved = {labels[ric]: _fixing_value(s, fixing_ts) for ric, s in clean.items()}
    fixings = pd.Series({k: v[0] for k, v in resolved.items()}, dtype="float64")
    fixing_how = {k: v[1] for k, v in resolved.items()}
    fixing_used = {k: v[2] for k, v in resolved.items()}

    for ric, s in clean.items():
        label = labels[ric]
        how = fixing_how[label]
        if how == "none":
            warnings.append(f"{label} ({ric}): no data returned.")
        elif how == "rolled":
            warnings.append(
                f"{label} ({ric}): {fixing_ts:%d %b %Y} was not a trading day - "
                f"rebased on the next close, {fixing_used[label]:%d %b %Y}.")
        elif how == "late":
            warnings.append(
                f"{label} ({ric}): history only starts {s.index[0]:%d %b %Y}, well "
                f"after the fixing date {fixing_ts:%d %b %Y}. It has been rebased on "
                f"that first close instead - the comparison is NOT on a common "
                f"fixing. Move the start/fixing date.")

    if cfg.scale == "rebased":
        plotted = prices.divide(fixings, axis=1) * 100.0
        base = 100.0
    else:
        plotted = prices.copy()
        first = cfg.active[0].label() if cfg.active else None
        base = float(fixings.get(first, np.nan)) if first else np.nan
        if len(cfg.active) > 1:
            warnings.append(
                "Absolute scale with several underlyings: the barrier levels are a "
                "percentage of underlying 1 only. Use the rebased scale to compare.")
        if np.isnan(base):
            # falling back to 100 here would put the barrier lines at 60/70 on a
            # chart scaled in index points - visibly absurd, but easy to miss
            fallback = prices.iloc[:, 0].dropna() if prices.shape[1] else pd.Series(dtype=float)
            base = float(fallback.iloc[0]) if len(fallback) else 100.0
            warnings.append(
                "Absolute scale: no fixing price for underlying 1, so the barrier "
                f"levels are a percentage of {base:,.2f} (its first available close).")

    # ---- worst-of: undefined unless every underlying has a value ---------- #
    if plotted.shape[1]:
        wo = plotted.min(axis=1)
        wo[plotted.isna().any(axis=1)] = np.nan
    else:
        wo = pd.Series(dtype="float64", index=cal)
    wo.name = cfg.worst_of_label

    # ---- quality report --------------------------------------------------- #
    rows = []
    for ric, s in clean.items():
        label = labels[ric]
        col_obs, col_px = observed[label], prices[label]
        live = col_px.notna()
        n_live = int(live.sum())
        carried = int((live & ~col_obs).sum())
        rows.append({
            "Underlying": label,
            "Fixing used": fixing_used[label].date() if fixing_used[label] is not None else None,
            "RIC": ric,
            "Points": int(len(s)),
            "First": s.index[0].date() if len(s) else None,
            "Last": s.index[-1].date() if len(s) else None,
            "Plotted days": n_live,
            "Carried fwd": carried,
            "% carried": round(100.0 * carried / n_live, 1) if n_live else np.nan,
            "Fixing px": round(float(fixings[label]), 4) if not np.isnan(fixings[label]) else None,
            "Status": _status(s, fixing_how[label], carried, n_live, cfg),
        })
    quality = pd.DataFrame(rows)

    if not quality.empty:
        firsts = [r for r in quality["First"] if r is not None]
        if firsts:
            common = max(firsts)
            if common > cfg.start:
                warnings.append(
                    f"Earliest date common to all underlyings is {common:%d %b %Y}. "
                    f"Do not start the chart before it.")

    return Prepared(prices, observed, plotted, wo, fixings, quality, base, cal, warnings)


def _status(s: pd.Series, how: str, carried: int, n_live: int,
            cfg: ChartConfig) -> str:
    if s.empty or how == "none":
        return "NO DATA"
    if how == "late":
        return "STARTS AFTER FIXING"
    if how == "rolled":
        return "OK (fixing rolled)"
    if s.index[-1].date() < cfg.end - dt.timedelta(days=7):
        return f"STALE (last {s.index[-1]:%d-%b-%y})"
    if n_live and carried / n_live > 0.15:
        return "OK (many carried days)"
    return "OK"


# --------------------------------------------------------------------------- #
# barriers
# --------------------------------------------------------------------------- #
def level_value(pct: float, base: float) -> float:
    return pct * base


def reference_series(prep: Prepared) -> pd.Series:
    """The series a barrier is tested against: the worst-of across underlyings.

    Used whether or not the worst-of line is displayed, because the payoff depends
    on it either way.
    """
    if prep.plotted.shape[1] > 1:
        return prep.worst_of
    if prep.plotted.shape[1] == 1:
        return prep.plotted.iloc[:, 0]
    return pd.Series(dtype="float64")


def breach_report(prep: Prepared, cfg: ChartConfig) -> pd.DataFrame:
    """Did the worst-of ever CLOSE below each downside barrier?

    Autocall triggers are excluded - they are upside conditions tested on fixed
    observation dates, not barriers, and belong in `autocall_report`.

    Closing prices only. An American knock-in observed on intraday lows is NOT
    captured here and must be checked against intraday data.
    """
    ref = reference_series(prep)
    rows = []
    for lv in cfg.levels:
        if not lv.enabled or lv.kind == "autocall":
            continue
        thr = level_value(lv.pct, prep.base)
        below = ref[ref < thr].dropna() if len(ref) else ref
        rows.append({
            "Level": lv.label,
            "Value": round(thr, 4),
            "Breached": bool(len(below)),
            "First breach": below.index[0].date() if len(below) else None,
            "Days below": int(len(below)),
            "Min observed": round(float(ref.min()), 4) if len(ref.dropna()) else None,
        })
    return pd.DataFrame(rows)


def autocall_report(prep: Prepared, cfg: ChartConfig) -> pd.DataFrame:
    """On each observation date, was the worst-of AT OR ABOVE the trigger?

    Observation dates falling on a market holiday are read on the last available
    close, which is the usual convention but not a substitute for the termsheet.
    """
    trigger = next((lv for lv in cfg.levels if lv.enabled and lv.kind == "autocall"), None)
    if trigger is None or not cfg.autocall_dates:
        return pd.DataFrame(columns=["Observation", "Level", "Trigger", "Called"])
    ref = reference_series(prep).dropna()
    thr = level_value(trigger.pct, prep.base)
    rows, called = [], False
    for d in sorted(cfg.autocall_dates):
        ts = pd.Timestamp(d)
        upto = ref[ref.index <= ts]
        val = float(upto.iloc[-1]) if len(upto) else np.nan
        hit = bool(val >= thr) if not np.isnan(val) else False
        rows.append({
            "Observation": d,
            "Level": round(val, 4) if not np.isnan(val) else None,
            "Trigger": round(thr, 4),
            "Called": hit and not called,
            "Status": ("-" if called else ("CALLED" if hit else "not called"))
                      if not np.isnan(val) else "no data",
        })
        called = called or hit
    return pd.DataFrame(rows)
