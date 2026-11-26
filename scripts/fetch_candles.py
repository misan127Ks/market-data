#!/usr/bin/env python3
"""Fetch OHLC candles and save them to data/*.csv.

- BTC / ETH / ADA : Kraken public API (daily = 1440 min, 4h = 240 min)
- USDJPY / GOLD   : yfinance 1h bars (90 days) aggregated to UTC-midnight daily
                    and UTC-aligned 4h bars (00,04,08,12,16,20 UTC)

Columns: time_utc, open, high, low, close, volume, closed
  time_utc = candle open time (UTC, ISO 8601)
  closed   = true if the candle period has fully ended at fetch time
Each CSV keeps the latest 120 candles. data/status.json records the run.
If a source fails, the previous CSV is left untouched and the error is logged.
"""
from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KEEP = 120
COLUMNS = ["time_utc", "open", "high", "low", "close", "volume", "closed"]
JST = timezone(timedelta(hours=9))

KRAKEN_PAIRS = {"BTC": "XBTUSD", "ETH": "ETHUSD", "ADA": "ADAUSD"}
KRAKEN_INTERVALS = {"1d": 1440, "4h": 240}

YF_TICKERS = {"USDJPY": "USDJPY=X", "GOLD": "GC=F"}
YF_PERIOD = "90d"
YF_RULES = {"1d": ("24h", timedelta(days=1)), "4h": ("4h", timedelta(hours=4))}


def iso(ts: pd.Timestamp | datetime) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def write_csv(name: str, df: pd.DataFrame) -> dict:
    df = df[COLUMNS].tail(KEEP).reset_index(drop=True)
    if df.empty:
        raise ValueError("no rows")
    path = DATA_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    closed = df[df["closed"]]
    return {
        "ok": True,
        "rows": int(len(df)),
        "first_time_utc": df["time_utc"].iloc[0],
        "last_time_utc": df["time_utc"].iloc[-1],
        "last_close": float(df["close"].iloc[-1]),
        "last_closed_time_utc": closed["time_utc"].iloc[-1] if len(closed) else None,
    }


# ---------------------------------------------------------------- Kraken
def kraken_ohlc(pair: str, interval: int, now: datetime) -> pd.DataFrame:
    url = "https://api.kraken.com/0/public/OHLC"
    last_err = None
    for attempt in range(4):
        try:
            r = requests.get(url, params={"pair": pair, "interval": interval}, timeout=30)
            r.raise_for_status()
            js = r.json()
            if js.get("error"):
                raise RuntimeError(f"Kraken error: {js['error']}")
            res = js["result"]
            key = next(k for k in res if k != "last")
            rows = res[key]
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(3 * (attempt + 1))
    else:
        raise RuntimeError(f"Kraken {pair} {interval}: {last_err}")

    # row: [time, open, high, low, close, vwap, volume, count]
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "vwap", "volume", "count"])
    df["t"] = df["t"].astype(int)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df.sort_values("t").drop_duplicates("t")
    now_s = now.timestamp()
    df["closed"] = (df["t"] + interval * 60) <= now_s
    df["time_utc"] = pd.to_datetime(df["t"], unit="s", utc=True).map(iso)
    return df


# --------------------------------------------------------------- yfinance
def yf_hourly(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    last_err = None
    for attempt in range(4):
        try:
            h = yf.Ticker(ticker).history(period=YF_PERIOD, interval="1h",
                                          auto_adjust=False, actions=False)
            if h is None or h.empty:
                raise RuntimeError("empty response")
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"yfinance {ticker}: {last_err}")

    h = h[["Open", "High", "Low", "Close", "Volume"]].copy()
    h.columns = ["open", "high", "low", "close", "volume"]
    idx = h.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    h.index = idx.tz_convert("UTC")
    h = h[~h.index.duplicated(keep="last")].sort_index()
    h = h.dropna(subset=["open", "high", "low", "close"])
    h["volume"] = h["volume"].fillna(0)
    return h


def aggregate(h: pd.DataFrame, rule: str, span: timedelta, now: datetime) -> pd.DataFrame:
    # origin="epoch" -> bins start at 00:00 UTC (daily) / 00,04,...,20 UTC (4h)
    agg = h.resample(rule, origin="epoch", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    agg = agg.dropna(subset=["open"])  # drop weekend / holiday bins without data
    agg["closed"] = (agg.index + span) <= pd.Timestamp(now)
    agg["time_utc"] = agg.index.map(iso)
    for c in ["open", "high", "low", "close"]:
        agg[c] = agg[c].round(3)  # float32 noise from yfinance; USDJPY quotes 3dp
    agg["volume"] = agg["volume"].round(0).astype("int64")
    return agg.reset_index(drop=True)


# ------------------------------------------------------------------- main
def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    files: dict[str, dict] = {}
    errors: list[dict] = []

    def record_error(name: str, e: Exception):
        msg = f"{type(e).__name__}: {e}"
        files[name] = {"ok": False, "error": msg}
        errors.append({"file": name, "error": msg})
        traceback.print_exc()

    for sym, pair in KRAKEN_PAIRS.items():
        for tf, interval in KRAKEN_INTERVALS.items():
            name = f"{sym}_{tf}"
            try:
                files[name] = write_csv(name, kraken_ohlc(pair, interval, now))
                files[name]["source"] = f"kraken:{pair}:{interval}"
            except Exception as e:  # noqa: BLE001
                record_error(name, e)
            time.sleep(1.5)  # be polite to the public API

    for sym, ticker in YF_TICKERS.items():
        try:
            h = yf_hourly(ticker)
            hourly_info = {"hourly_rows": int(len(h)), "hourly_last_utc": iso(h.index[-1])}
        except Exception as e:  # noqa: BLE001
            for tf in YF_RULES:
                record_error(f"{sym}_{tf}", e)
            continue
        for tf, (rule, span) in YF_RULES.items():
            name = f"{sym}_{tf}"
            try:
                files[name] = write_csv(name, aggregate(h, rule, span, now))
                files[name]["source"] = f"yfinance:{ticker}:1h->{tf}"
                files[name].update(hourly_info)
            except Exception as e:  # noqa: BLE001
                record_error(name, e)

    status = {
        "updated_utc": iso(now),
        "updated_jst": now.astimezone(JST).strftime("%Y-%m-%d %H:%M:%S JST"),
        "ok": not errors,
        "error_count": len(errors),
        "errors": errors,
        "files": files,
    }
    (DATA_DIR / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: status[k] for k in ("updated_utc", "ok", "error_count", "errors")}, indent=2))
    # Exit 0 even on partial failure so successful files still get committed;
    # errors are visible in status.json.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
