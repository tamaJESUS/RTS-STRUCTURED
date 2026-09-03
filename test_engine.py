import datetime as dt

import numpy as np
import pandas as pd
import pytest

from spchart.config import ChartConfig, Underlying, Level
from spchart.engine import (normalise, build_calendar, prepare, breach_report,
                            autocall_report, reference_series, level_value)


# --------------------------------------------------------------------------- #
def mkcfg(**kw):
    base = dict(
        underlyings=[Underlying("A.RIC", "Alpha"), Underlying("B.RIC", "Beta")],
        start=dt.date(2024, 1, 1),
        end=dt.date(2024, 3, 29),
    )
    base.update(kw)
    return ChartConfig(**base)


def series(dates, values):
    return pd.Series(values, index=pd.to_datetime(dates))


# --------------------------------------------------------------------------- #
# normalise: the three things Reuters actually does to us
# --------------------------------------------------------------------------- #
def test_normalise_sorts_descending_input():
    s = series(["2024-01-03", "2024-01-02", "2024-01-01"], [3.0, 2.0, 1.0])
    out = normalise(s)
    assert list(out.index) == list(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
    assert list(out.values) == [1.0, 2.0, 3.0]


def test_normalise_strips_time_component():
    s = series(["2024-01-01 17:35", "2024-01-02 17:35"], [1.0, 2.0])
    out = normalise(s)
    assert all(t.hour == 0 and t.minute == 0 for t in out.index)


def test_normalise_keeps_last_of_duplicate_days():
    s = series(["2024-01-01 09:00", "2024-01-01 17:35"], [1.0, 9.9])
    out = normalise(s)
    assert len(out) == 1 and out.iloc[0] == 9.9


def test_normalise_drops_nan():
    s = series(["2024-01-01", "2024-01-02"], [np.nan, 2.0])
    assert len(normalise(s)) == 1


# --------------------------------------------------------------------------- #
# calendar
# --------------------------------------------------------------------------- #
def test_union_calendar_covers_both_holiday_calendars():
    a = series(["2024-01-01", "2024-01-02"], [1.0, 1.0])          # B closed the 1st
    b = series(["2024-01-02", "2024-01-03"], [1.0, 1.0])          # A closed the 3rd
    cal = build_calendar({"a": a, "b": b}, dt.date(2024, 1, 1), dt.date(2024, 1, 3))
    assert len(cal) == 3


def test_intersection_calendar_keeps_common_days_only():
    a = series(["2024-01-01", "2024-01-02"], [1.0, 1.0])
    b = series(["2024-01-02", "2024-01-03"], [1.0, 1.0])
    cal = build_calendar({"a": a, "b": b}, dt.date(2024, 1, 1), dt.date(2024, 1, 3),
                         mode="intersection")
    assert len(cal) == 1 and cal[0] == pd.Timestamp("2024-01-02")


def test_weekday_calendar_excludes_weekends():
    cal = build_calendar({}, dt.date(2024, 1, 1), dt.date(2024, 1, 7), mode="weekdays")
    assert len(cal) == 5
    assert all(d.weekday() < 5 for d in cal)


# --------------------------------------------------------------------------- #
# the core promise: coherence
# --------------------------------------------------------------------------- #
def test_every_underlying_rebases_to_exactly_100_at_the_fixing():
    days = pd.bdate_range("2024-01-01", "2024-03-29")
    a = pd.Series(np.linspace(4000, 4500, len(days)), index=days)
    b = pd.Series(np.linspace(150, 90, len(days)), index=days)
    prep = prepare({"A.RIC": a, "B.RIC": b}, mkcfg())
    first = prep.plotted.iloc[0]
    assert first["Alpha"] == pytest.approx(100.0)
    assert first["Beta"] == pytest.approx(100.0)


def test_missing_quote_is_carried_forward_not_zeroed():
    a = series(["2024-01-01", "2024-01-02", "2024-01-03"], [100.0, 110.0, 120.0])
    b = series(["2024-01-01", "2024-01-03"], [50.0, 60.0])        # closed on the 2nd
    cfg = mkcfg(start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 3))
    prep = prepare({"A.RIC": a, "B.RIC": b}, cfg)
    assert prep.prices.loc["2024-01-02", "Beta"] == 50.0          # carried, not 0/NaN
    assert prep.observed.loc["2024-01-02", "Beta"] == False       # and flagged as such
    assert prep.quality.set_index("Underlying").loc["Beta", "Carried fwd"] == 1


def test_history_before_the_first_quote_is_not_back_filled():
    a = series(["2024-01-01", "2024-01-02"], [100.0, 110.0])
    b = series(["2024-01-02"], [50.0])                            # starts a day late
    cfg = mkcfg(start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 2))
    prep = prepare({"A.RIC": a, "B.RIC": b}, cfg)
    assert np.isnan(prep.prices.loc["2024-01-01", "Beta"])


def test_descending_and_timestamped_input_still_aligns():
    days = pd.bdate_range("2024-01-01", "2024-02-29")
    a = pd.Series(np.linspace(100, 120, len(days)), index=days)
    b_idx = pd.to_datetime([d.strftime("%Y-%m-%d") + " 17:35" for d in days])[::-1]
    b = pd.Series(np.linspace(100, 120, len(days))[::-1], index=b_idx)
    cfg = mkcfg(end=dt.date(2024, 2, 29))
    prep = prepare({"A.RIC": a, "B.RIC": b}, cfg)
    # identical economics fed in opposite order must produce identical curves
    np.testing.assert_allclose(prep.plotted["Alpha"].values,
                               prep.plotted["Beta"].values, rtol=1e-9)


def test_fixing_uses_last_close_on_or_before_the_fixing_date():
    a = series(["2024-01-01", "2024-01-05", "2024-01-10"], [100.0, 200.0, 400.0])
    cfg = mkcfg(underlyings=[Underlying("A.RIC", "Alpha")],
                start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 10),
                fixing_date=dt.date(2024, 1, 7))          # not a trading day
    prep = prepare({"A.RIC": a}, cfg)
    assert prep.fixings["Alpha"] == 200.0                 # the 5th, not the 10th
    assert prep.plotted.loc["2024-01-10", "Alpha"] == pytest.approx(200.0)


def test_history_starting_long_after_the_fixing_is_flagged_loudly():
    a = series(["2024-02-01", "2024-02-02"], [100.0, 110.0])
    cfg = mkcfg(underlyings=[Underlying("A.RIC", "Alpha")],
                start=dt.date(2024, 1, 1), end=dt.date(2024, 2, 2))
    prep = prepare({"A.RIC": a}, cfg)
    # still plots, on its own first close, but says so in the clearest terms
    assert prep.fixings["Alpha"] == 100.0
    assert any("NOT on a common fixing" in w for w in prep.warnings)
    assert prep.quality.set_index("Underlying").loc["Alpha", "Status"] == "STARTS AFTER FIXING"


def test_fixing_on_a_weekend_rolls_to_the_next_close():
    # 2024-09-01 is a Sunday - the trap that silently broke the absolute scale
    days = pd.bdate_range("2024-09-02", "2024-10-31")
    a = pd.Series(np.linspace(4000, 4400, len(days)), index=days)
    cfg = mkcfg(underlyings=[Underlying("A.RIC", "Alpha")],
                start=dt.date(2024, 9, 1), end=dt.date(2024, 10, 31),
                fixing_date=dt.date(2024, 9, 1))
    prep = prepare({"A.RIC": a}, cfg)
    assert prep.fixings["Alpha"] == 4000.0
    assert prep.plotted.iloc[0, 0] == pytest.approx(100.0)
    assert any("not a trading day" in w for w in prep.warnings)
    assert prep.quality.set_index("Underlying").loc["Alpha", "Status"] == "OK (fixing rolled)"


def test_absolute_scale_never_falls_back_to_a_base_of_100():
    # a barrier at 60% of 100 on a chart scaled in index points is nonsense
    days = pd.bdate_range("2024-09-02", "2024-10-31")
    a = pd.Series(np.full(len(days), 4200.0), index=days)
    cfg = mkcfg(underlyings=[Underlying("A.RIC", "Alpha")], scale="absolute",
                start=dt.date(2024, 9, 1), end=dt.date(2024, 10, 31),
                fixing_date=dt.date(2024, 9, 1))
    prep = prepare({"A.RIC": a}, cfg)
    assert prep.base == 4200.0
    assert level_value(0.60, prep.base) == 2520.0


# --------------------------------------------------------------------------- #
# worst-of
# --------------------------------------------------------------------------- #
def test_worst_of_is_the_minimum_and_undefined_when_a_leg_is_missing():
    a = series(["2024-01-01", "2024-01-02"], [100.0, 120.0])
    b = series(["2024-01-01", "2024-01-02"], [100.0, 80.0])
    cfg = mkcfg(start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 2))
    prep = prepare({"A.RIC": a, "B.RIC": b}, cfg)
    assert prep.worst_of.loc["2024-01-02"] == pytest.approx(80.0)

    b2 = series(["2024-01-02"], [80.0])                   # Beta has no 1 Jan print
    prep2 = prepare({"A.RIC": a, "B.RIC": b2}, cfg)
    assert np.isnan(prep2.worst_of.loc["2024-01-01"])


# --------------------------------------------------------------------------- #
# barriers
# --------------------------------------------------------------------------- #
def test_breach_detected_on_closing_basis():
    days = pd.bdate_range("2024-01-01", "2024-01-19")
    v = np.full(len(days), 100.0)
    v[5] = 55.0                                            # one close below 60%
    a = pd.Series(v, index=days)
    cfg = mkcfg(underlyings=[Underlying("A.RIC", "Alpha")],
                start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 19),
                levels=[Level("Capital barrier 60%", 0.60, "red", kind="capital"),
                        Level("Coupon barrier 70%", 0.70, "gold", kind="coupon")])
    prep = prepare({"A.RIC": a}, cfg)
    rep = breach_report(prep, cfg).set_index("Level")
    assert rep.loc["Capital barrier 60%", "Breached"]
    assert rep.loc["Capital barrier 60%", "First breach"] == days[5].date()
    assert rep.loc["Capital barrier 60%", "Days below"] == 1
    assert rep.loc["Coupon barrier 70%", "Breached"]


def test_no_breach_reported_when_never_below():
    days = pd.bdate_range("2024-01-01", "2024-01-19")
    a = pd.Series(np.full(len(days), 100.0), index=days)
    cfg = mkcfg(underlyings=[Underlying("A.RIC", "Alpha")],
                start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 19),
                levels=[Level("Capital barrier 60%", 0.60, "red", kind="capital")])
    rep = breach_report(prepare({"A.RIC": a}, cfg), cfg)
    assert not rep.iloc[0]["Breached"]
    assert rep.iloc[0]["First breach"] is None


def test_disabled_levels_are_ignored():
    days = pd.bdate_range("2024-01-01", "2024-01-19")
    a = pd.Series(np.full(len(days), 100.0), index=days)
    cfg = mkcfg(underlyings=[Underlying("A.RIC", "Alpha")],
                start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 19),
                levels=[Level("off", 0.60, "red", enabled=False)])
    assert breach_report(prepare({"A.RIC": a}, cfg), cfg).empty


# --------------------------------------------------------------------------- #
# absolute scale + edge cases
# --------------------------------------------------------------------------- #
def test_absolute_scale_keeps_raw_prices_and_bases_levels_on_underlying_one():
    days = pd.bdate_range("2024-01-01", "2024-01-19")
    a = pd.Series(np.full(len(days), 4000.0), index=days)
    cfg = mkcfg(underlyings=[Underlying("A.RIC", "Alpha")], scale="absolute",
                start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 19))
    prep = prepare({"A.RIC": a}, cfg)
    assert prep.plotted.iloc[0, 0] == 4000.0
    assert prep.base == 4000.0


def test_empty_input_does_not_raise():
    cfg = mkcfg()
    prep = prepare({}, cfg)
    assert prep.empty
    assert prep.warnings


def test_end_before_start_is_rejected():
    with pytest.raises(ValueError):
        mkcfg(start=dt.date(2024, 3, 1), end=dt.date(2024, 1, 1))


# --------------------------------------------------------------------------- #
# autocall: an upside condition on fixed dates, not a barrier
# --------------------------------------------------------------------------- #
def test_autocall_excluded_from_breach_report():
    days = pd.bdate_range("2024-01-01", "2024-01-19")
    a = pd.Series(np.full(len(days), 90.0), index=days)   # always below 100
    cfg = mkcfg(underlyings=[Underlying("A.RIC", "Alpha")],
                start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 19),
                levels=[Level("Autocall 100%", 1.00, "green", kind="autocall")])
    assert breach_report(prepare({"A.RIC": a}, cfg), cfg).empty


def test_autocall_called_on_first_observation_at_or_above_trigger():
    # rebasing happens BEFORE the comparison: fixing = 100 => trigger sits at 100
    days = pd.bdate_range("2024-01-01", "2024-03-29")
    a = pd.Series(np.full(len(days), 100.0), index=days)
    a.loc["2024-01-15":"2024-02-28"] = 90.0               # below trigger at obs 1
    a.loc["2024-03-01":] = 105.0                          # recovers before obs 2
    cfg = mkcfg(underlyings=[Underlying("A.RIC", "Alpha")],
                start=dt.date(2024, 1, 1), end=dt.date(2024, 3, 29),
                levels=[Level("Autocall 100%", 1.00, "green", kind="autocall")],
                autocall_dates=[dt.date(2024, 2, 1), dt.date(2024, 3, 15)])
    rep = autocall_report(prepare({"A.RIC": a}, cfg), cfg)
    assert list(rep["Called"]) == [False, True]
    assert list(rep["Status"]) == ["not called", "CALLED"]


def test_autocall_only_calls_once():
    days = pd.bdate_range("2024-01-01", "2024-03-29")
    a = pd.Series(np.full(len(days), 100.0), index=days)
    a.loc["2024-01-15":] = 120.0                           # above trigger at both obs
    cfg = mkcfg(underlyings=[Underlying("A.RIC", "Alpha")],
                start=dt.date(2024, 1, 1), end=dt.date(2024, 3, 29),
                levels=[Level("Autocall 100%", 1.00, "green", kind="autocall")],
                autocall_dates=[dt.date(2024, 2, 1), dt.date(2024, 3, 15)])
    rep = autocall_report(prepare({"A.RIC": a}, cfg), cfg)
    assert list(rep["Called"]) == [True, False]            # first date only
    assert list(rep["Status"]) == ["CALLED", "-"]


def test_autocall_observation_on_a_holiday_reads_the_last_close_not_the_next():
    days = pd.bdate_range("2024-01-01", "2024-03-29")
    a = pd.Series(np.full(len(days), 100.0), index=days)
    a.loc["2024-03-18":] = 130.0            # jumps only AFTER the observation
    cfg = mkcfg(underlyings=[Underlying("A.RIC", "Alpha")],
                start=dt.date(2024, 1, 1), end=dt.date(2024, 3, 29),
                levels=[Level("Autocall 110%", 1.10, "green", kind="autocall")],
                autocall_dates=[dt.date(2024, 3, 17)])     # a Sunday
    rep = autocall_report(prepare({"A.RIC": a}, cfg), cfg)
    # must read Friday 15 March (100), not Monday 18 March (130)
    assert rep.iloc[0]["Level"] == 100.0
    assert not rep.iloc[0]["Called"]


def test_autocall_report_empty_without_observation_dates():
    days = pd.bdate_range("2024-01-01", "2024-01-19")
    a = pd.Series(np.full(len(days), 105.0), index=days)
    cfg = mkcfg(underlyings=[Underlying("A.RIC", "Alpha")],
                start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 19),
                levels=[Level("Autocall 100%", 1.00, "green", kind="autocall")])
    assert autocall_report(prepare({"A.RIC": a}, cfg), cfg).empty


def test_breach_uses_worst_of_not_the_average():
    days = pd.bdate_range("2024-01-01", "2024-01-19")
    a = pd.Series(np.full(len(days), 100.0), index=days)
    b = pd.Series(np.full(len(days), 100.0), index=days)
    b.iloc[5] = 50.0                                       # one leg dives
    cfg = mkcfg(start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 19),
                levels=[Level("Capital 60%", 0.60, "red", kind="capital")])
    rep = breach_report(prepare({"A.RIC": a, "B.RIC": b}, cfg), cfg)
    assert rep.iloc[0]["Breached"]
