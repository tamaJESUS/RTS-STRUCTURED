"""
Where the prices come from.

Every source returns the same thing: {ric: pd.Series indexed by date}. The engine
never knows which one was used, so you can develop against `DemoSource` and switch
to `LSEGSource` on a Workspace machine without touching anything else.
"""
from __future__ import annotations

import datetime as dt
import glob
import os
from typing import Protocol

import numpy as np
import pandas as pd


class PriceSource(Protocol):
    def history(self, rics: list[str], start: dt.date, end: dt.date) -> dict[str, pd.Series]:
        ...


def _pick_price_column(df: pd.DataFrame, preferred: list[str]) -> pd.Series | None:
    """Be forgiving about column naming - it varies between add-in versions."""
    if df is None or len(df) == 0:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["/".join(str(x) for x in c if x is not None) for c in df.columns]
    for want in preferred:
        for col in df.columns:
            if str(col).strip().upper() == want.upper():
                return pd.to_numeric(df[col], errors="coerce")
    for want in preferred:                       # then a loose contains-match
        for col in df.columns:
            if want.upper() in str(col).strip().upper():
                return pd.to_numeric(df[col], errors="coerce")
    numeric = df.select_dtypes("number")         # last resort: first numeric column
    return numeric.iloc[:, 0] if numeric.shape[1] else None


# --------------------------------------------------------------------------- #
# LSEG / Refinitiv / Eikon
# --------------------------------------------------------------------------- #
class LSEGSource:
    """Wraps whichever LSEG Python library is installed on the machine.

    Tries, in order:
      1. lseg.data        (current)          -> ld.get_history(...)
      2. refinitiv.data   (previous name)    -> rd.get_history(...)
      3. eikon            (legacy Eikon API) -> ek.get_timeseries(...)

    A Workspace/Eikon desktop session must be running and signed in, or an app key
    supplied. NOT exercised by the test suite - no terminal in the build sandbox.
    """

    def __init__(self, fields: list[str] | None = None, app_key: str | None = None):
        self.fields = fields or ["TRDPRC_1", "TR.PriceClose", "CLOSE"]
        self.app_key = app_key or os.environ.get("LSEG_APP_KEY") or os.environ.get("EIKON_APP_KEY")
        self._lib = None
        self._mod = None

    # -- session ------------------------------------------------------------ #
    def _open(self):
        if self._lib:
            return
        try:
            import lseg.data as ld
            ld.open_session()
            self._lib, self._mod = "lseg", ld
            return
        except Exception:
            pass
        try:
            import refinitiv.data as rd
            rd.open_session()
            self._lib, self._mod = "refinitiv", rd
            return
        except Exception:
            pass
        try:
            import eikon as ek
            if self.app_key:
                ek.set_app_key(self.app_key)
            self._lib, self._mod = "eikon", ek
            return
        except Exception as exc:
            raise RuntimeError(
                "No LSEG library available. Install one of:\n"
                "    pip install lseg-data          (current)\n"
                "    pip install refinitiv-data     (previous name)\n"
                "    pip install eikon              (legacy)\n"
                "and make sure Workspace/Eikon is running and signed in.\n"
                f"Last error: {exc}") from exc

    def close(self):
        try:
            if self._lib in ("lseg", "refinitiv"):
                self._mod.close_session()
        except Exception:
            pass

    # -- fetch -------------------------------------------------------------- #
    def history(self, rics, start, end) -> dict[str, pd.Series]:
        self._open()
        out: dict[str, pd.Series] = {}
        for ric in rics:
            try:
                out[ric] = self._one(ric, start, end)
            except Exception as exc:                 # one bad RIC must not kill the rest
                print(f"  ! {ric}: {exc}")
                out[ric] = pd.Series(dtype="float64")
        return out

    def _one(self, ric: str, start: dt.date, end: dt.date) -> pd.Series:
        s0, s1 = str(start), str(end)
        if self._lib in ("lseg", "refinitiv"):
            df = self._mod.get_history(universe=ric, fields=self.fields,
                                       interval="1D", start=s0, end=s1)
        else:
            df = self._mod.get_timeseries(ric, fields="CLOSE", interval="daily",
                                          start_date=s0, end_date=s1)
        col = _pick_price_column(df, self.fields)
        if col is None:
            raise ValueError(f"no numeric price column in {list(getattr(df, 'columns', []))}")
        col.index = pd.to_datetime(df.index)
        return col.dropna()


# --------------------------------------------------------------------------- #
# files
# --------------------------------------------------------------------------- #
class CsvSource:
    """A folder of `<RIC>.csv` files, or one wide CSV (first column = date).

    Per-RIC files need a date column and a price column; the names are guessed.
    """

    def __init__(self, path: str, date_col: str | None = None, price_col: str | None = None):
        self.path, self.date_col, self.price_col = path, date_col, price_col

    def history(self, rics, start, end) -> dict[str, pd.Series]:
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        out = {}
        if os.path.isdir(self.path):
            for ric in rics:
                hits = [p for p in glob.glob(os.path.join(self.path, "*"))
                        if os.path.splitext(os.path.basename(p))[0].upper() == ric.upper()]
                out[ric] = self._read_one(hits[0], lo, hi) if hits else pd.Series(dtype="float64")
        else:
            wide = pd.read_csv(self.path, index_col=0, parse_dates=True)
            for ric in rics:
                match = [c for c in wide.columns if str(c).upper() == ric.upper()]
                s = pd.to_numeric(wide[match[0]], errors="coerce").dropna() if match \
                    else pd.Series(dtype="float64")
                out[ric] = s[(s.index >= lo) & (s.index <= hi)] if len(s) else s
        return out

    def _read_one(self, path, lo, hi) -> pd.Series:
        df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
        dcol = self.date_col or next(
            (c for c in df.columns if "date" in str(c).lower()), df.columns[0])
        s = _pick_price_column(df.drop(columns=[dcol]),
                               [self.price_col] if self.price_col else
                               ["close", "px_last", "price", "trdprc_1"])
        if s is None:
            return pd.Series(dtype="float64")
        s.index = pd.to_datetime(df[dcol], errors="coerce")
        s = s.dropna()
        return s[(s.index >= lo) & (s.index <= hi)]


# --------------------------------------------------------------------------- #
# demo
# --------------------------------------------------------------------------- #
class DemoSource:
    """Synthetic geometric-Brownian prices so the app runs with zero setup.

    Deliberately gives each RIC a different holiday calendar and a different
    starting price, so the alignment work is visible. Not market data.
    """

    def __init__(self, seed: int = 7, vol: float = 0.18, drift: float = 0.05):
        self.seed, self.vol, self.drift = seed, vol, drift

    def history(self, rics, start, end) -> dict[str, pd.Series]:
        rng = np.random.default_rng(self.seed)
        days = pd.bdate_range(start, end)
        out = {}
        for k, ric in enumerate(rics):
            n = len(days)
            dt_ = 1 / 252
            shocks = rng.normal((self.drift - 0.5 * self.vol ** 2) * dt_,
                                self.vol * np.sqrt(dt_), n)
            px = float([4200, 5100, 7800, 320, 95][k % 5]) * np.exp(np.cumsum(shocks))
            keep = rng.random(n) > (0.01 + 0.02 * k)      # each RIC loses different days
            idx = days[keep]
            out[ric] = pd.Series(px[keep], index=idx)
        return out


def get_source(name: str, **kw) -> PriceSource:
    return {"lseg": LSEGSource, "csv": CsvSource, "demo": DemoSource}[name.lower()](**kw)
