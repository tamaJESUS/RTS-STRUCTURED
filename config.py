"""Configuration objects. Everything the user can change lives here."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, asdict
from typing import Literal

import yaml

Scale = Literal["rebased", "absolute"]
CalendarMode = Literal["union", "weekdays", "intersection"]

# BNPP-style palette. Chosen so the barrier lines stay distinguishable in greyscale.
UNDERLYING_COLORS = ["#00915A", "#0F2B46", "#8C9BA5", "#C8A45C", "#5B8FA8"]
WORST_OF_COLOR = "#E0651A"
LEVEL_COLORS = {
    "black": "#1F1F1F", "green": "#00915A", "gold": "#C8A45C",
    "red": "#C0392B", "purple": "#6B4E9B", "grey": "#7F8C8D",
}


@dataclass
class Underlying:
    ric: str
    name: str = ""

    def label(self) -> str:
        return self.name or self.ric


@dataclass
class Level:
    """A horizontal line drawn at `pct` of the initial fixing level."""
    label: str
    pct: float                      # 0.60 => 60% of the initial fixing
    color: str = "black"            # key of LEVEL_COLORS, or any matplotlib colour
    enabled: bool = True
    kind: str = "generic"           # generic | autocall | coupon | capital

    def hex_color(self) -> str:
        return LEVEL_COLORS.get(self.color, self.color)


@dataclass
class Texts:
    title: str = "Underlying performance"
    subtitle: str = ""
    footnote: str = ""
    disclaimer: str = ""
    source: str = "Source: LSEG / Refinitiv"


@dataclass
class ChartConfig:
    underlyings: list[Underlying] = field(default_factory=list)
    start: dt.date = dt.date.today() - dt.timedelta(days=365 * 3)
    end: dt.date = dt.date.today()
    fixing_date: dt.date | None = None      # defaults to `start`
    scale: Scale = "rebased"
    calendar: CalendarMode = "union"
    show_worst_of: bool = False
    worst_of_label: str = "Worst-of basket"
    levels: list[Level] = field(default_factory=list)
    autocall_dates: list[dt.date] = field(default_factory=list)
    annotate_last: bool = True
    levels_in_legend: bool = False           # False => labelled inline at the right
    mark_fixing: bool = True
    texts: Texts = field(default_factory=Texts)

    def __post_init__(self):
        if self.fixing_date is None:
            self.fixing_date = self.start
        if self.end < self.start:
            raise ValueError("end date is before start date")

    @property
    def rics(self) -> list[str]:
        return [u.ric for u in self.underlyings if u.ric.strip()]

    @property
    def active(self) -> list[Underlying]:
        return [u for u in self.underlyings if u.ric.strip()]

    # ---------- serialisation ----------
    @classmethod
    def from_yaml(cls, path: str) -> "ChartConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "ChartConfig":
        def as_date(v, default=None):
            if v in (None, ""):
                return default
            if isinstance(v, dt.datetime):
                return v.date()
            if isinstance(v, dt.date):
                return v
            return dt.date.fromisoformat(str(v))

        kwargs: dict = {}
        kwargs["underlyings"] = [Underlying(**u) if isinstance(u, dict) else Underlying(str(u))
                                 for u in raw.get("underlyings", [])]
        kwargs["levels"] = [Level(**l) for l in raw.get("levels", [])]
        for key in ("scale", "calendar", "show_worst_of", "worst_of_label",
                    "annotate_last", "levels_in_legend", "mark_fixing"):
            if key in raw:
                kwargs[key] = raw[key]
        today = dt.date.today()
        kwargs["start"] = as_date(raw.get("start"), today - dt.timedelta(days=365 * 3))
        kwargs["end"] = as_date(raw.get("end"), today)
        kwargs["fixing_date"] = as_date(raw.get("fixing_date"))
        kwargs["autocall_dates"] = [as_date(d) for d in raw.get("autocall_dates", [])]
        kwargs["texts"] = Texts(**raw.get("texts", {}))
        return cls(**kwargs)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("start", "end", "fixing_date"):
            d[k] = d[k].isoformat() if d[k] else None
        d["autocall_dates"] = [x.isoformat() for x in self.autocall_dates]
        return d
