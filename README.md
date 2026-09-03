# Structured product performance chart

Python version of the Excel tool. Same job — put Reuters tickers in, get a
client-ready graphic out — but with a tested engine, barrier-breach detection and
autocall observation logic that the spreadsheet could not do cleanly.

```
spchart/
  config.py       settings objects (underlyings, levels, texts) + YAML loading
  datasource.py   where prices come from: LSEG | CSV | demo
  engine.py       alignment, carry-forward, rebasing, barriers, autocall
  chart.py        the matplotlib graphic
  export.py       PNG / PDF + the data workbook
app.py            Streamlit point-and-click interface
run.py            command line, driven by a YAML file
tests/            26 tests covering the engine
```

## Install

```bash
pip install -r requirements.txt
```

For live Reuters data you also need one of the LSEG libraries and a signed-in
Workspace/Eikon desktop session:

```bash
pip install lseg-data        # current
# pip install refinitiv-data # previous name
# pip install eikon          # legacy
```

## Run it

Nothing to configure to see it work — the demo source generates synthetic prices:

```bash
python run.py config.example.yaml --source demo
streamlit run app.py
```

Against the terminal:

```bash
python run.py config.example.yaml --source lseg
```

From files, when you have no terminal on the machine:

```bash
python run.py config.example.yaml --source csv --path ./prices
```

`./prices` is either a folder of `<RIC>.csv` files (a date column and a price
column, names are guessed) or one wide CSV with dates in the first column and one
column per RIC.

## Outputs

| File | What it is |
|---|---|
| `chart.png` | 200 dpi, for slides and email |
| `chart.pdf` | vector, for print |
| `chart_data_data.xlsx` | every number behind the picture |

The workbook has seven sheets: **Chart data** (what is plotted, including the
barrier levels as columns), **Prices** (aligned raw prices), **Observed**
(`TRUE` = a real quote that day, `FALSE` = carried forward), **Quality**,
**Barriers**, **Autocall**, **Config**.

## How the data is made coherent

This is the part that matters and the part that is tested.

- **One calendar.** Every RIC is reindexed onto a single date axis. Default
  `union` = every day on which any of the markets traded. `intersection` keeps
  only days where all of them traded; `weekdays` uses a plain Mon–Fri calendar.
- **Carry-forward, never zero.** A market closed for a local holiday reuses its
  previous close. The `Observed` sheet flags every carried day and the quality
  table reports the percentage.
- **No back-fill.** A series that starts late stays empty before its first quote
  rather than inventing history.
- **One fixing.** Everything is rebased at the same initial fixing date, using
  the last close on or before it. If a series has no history at that date it is
  flagged `STARTS AFTER FIXING` rather than silently rebased on something else.
- **Order and timestamps.** Input can arrive descending, with a time component,
  or with two prints on the same day. All three are normalised.

## Barriers and autocall

`breach_report` tests the **worst-of** — the minimum across underlyings, whether
or not the worst-of line is displayed, because that is what the payoff depends on.

`autocall_report` is separate on purpose: an autocall trigger is an upside
condition tested on fixed observation dates, not a barrier. It reports the level
on each observation date and which one called first.

**Both use closing prices.** An American knock-in observed on intraday lows is
*not* captured. Do not use this to confirm whether a barrier was touched.

## Limitations — read before relying on it

- **The LSEG adapter is not covered by the tests.** There was no terminal on the
  machine this was written on, so `LSEGSource` is written from the documented API
  but never executed. Check the first pull against Excel before trusting it. The
  field defaults are `TRDPRC_1`, then `TR.PriceClose`, then `CLOSE`, and the
  column matching is deliberately loose because naming varies between versions.
- **Price close, not total return.** Dividends are excluded. For a fair
  comparison of equity underlyings over several years, switch the field to
  `TR.TotalReturn` in `LSEGSource(fields=[...])`.
- **No FX conversion.** Underlyings in different currencies are rebased
  individually, which is usually what you want for a chart, but is not a
  currency-hedged basket.
- **The disclaimer is placeholder text** and there is no logo. Your compliance
  wording, past-performance language and any PRIIPs/MiFID requirements come from
  your own templates.

## Tests

```bash
python -m pytest tests/ -q
```

They cover the awkward cases specifically: descending input, timestamps,
duplicate prints, mismatched holiday calendars, a fixing date on a non-trading
day, a series starting after the fixing, worst-of with a missing leg, breach
detection, and autocall observations falling on a weekend.
