#!/usr/bin/env python3
"""Build a full-market one-shot broker branch dashboard.

This local-first dashboard uses the 2024-2025 broker price-level aggregates and
keeps only high-win net-buy branch events:

    net buy >= threshold -> buy-side event, success means future price is up
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
BROKER_PL_DIR = ROOT / "data" / "broker_pl_agg"
BRANCHES_DIR = ROOT / "data" / "branches"
PV_DIR = ROOT / "data" / "tquant_lab_pv"
TRADER_INFO_PATH = ROOT / "data" / "reference" / "securities_trader_info.parquet"
STOCK_INFO_PATH = ROOT / "data" / "reference" / "taiwan_stock_info.parquet"
DATA_OUTPUT_DIR = ROOT / "data" / "one_shot_full_market"
BRANCH_SUMMARY_PATH = DATA_OUTPUT_DIR / "branch_direction_summary.csv"
SEED_EVENTS_PATH = DATA_OUTPUT_DIR / "one_shot_full_market_events.parquet"
DASHBOARD_OUTPUT_DIR = ROOT / "dashboard" / "one_shot"

DEFAULT_THRESHOLD = 100_000_000.0
DEFAULT_MIN_EVENTS = 5
DEFAULT_MAX_EVENTS = 400
DEFAULT_HIGH_WIN_RATE = 0.55
DEFAULT_RECENT_DAYS = 20
DEFAULT_BRANCH_START_DATE = "2026-01-01"
DEFAULT_SOURCE_BUY_EVENT_COUNT = 40_544
FORWARD_HORIZONS = [1, 3, 5, 10, 20]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full-market one-shot branch dashboard.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--min-events", type=int, default=DEFAULT_MIN_EVENTS)
    parser.add_argument("--max-events", type=int, default=DEFAULT_MAX_EVENTS)
    parser.add_argument("--high-win-rate", type=float, default=DEFAULT_HIGH_WIN_RATE)
    parser.add_argument("--recent-days", type=int, default=DEFAULT_RECENT_DAYS)
    parser.add_argument("--branch-start-date", default=DEFAULT_BRANCH_START_DATE)
    parser.add_argument("--branch-end-date")
    parser.add_argument("--rebuild-baseline", action="store_true")
    parser.add_argument("--source-buy-event-count", type=int, default=DEFAULT_SOURCE_BUY_EVENT_COUNT)
    parser.add_argument("--data-output-dir", default=str(DATA_OUTPUT_DIR))
    parser.add_argument("--output-dir", default=str(DASHBOARD_OUTPUT_DIR))
    return parser.parse_args()


def output_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num = np.asarray(num, dtype="float64")
    den = np.asarray(den, dtype="float64")
    out = np.full(len(num), np.nan, dtype="float64")
    mask = np.isfinite(den) & (den != 0)
    out[mask] = num[mask] / den[mask]
    return out


def round_or_none(value: Any, digits: int = 4) -> Any:
    if value is None or pd.isna(value):
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return round(value, digits)


def load_name_maps() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    if TRADER_INFO_PATH.exists():
        trader = pd.read_parquet(TRADER_INFO_PATH)
        broker_names = dict(
            zip(trader["securities_trader_id"].astype(str), trader["securities_trader"].astype(str))
        )
    else:
        broker_names = {}

    if STOCK_INFO_PATH.exists():
        stocks = pd.read_parquet(STOCK_INFO_PATH)
        stocks["stock_id"] = stocks["stock_id"].astype(str)
        if "date" in stocks.columns:
            stocks = stocks.sort_values(["stock_id", "date"]).drop_duplicates("stock_id", keep="last")
        stock_names = dict(zip(stocks["stock_id"], stocks["stock_name"].astype(str)))
        stock_industries = dict(zip(stocks["stock_id"], stocks["industry_category"].astype(str)))
    else:
        stock_names = {}
        stock_industries = {}
    return broker_names, stock_names, stock_industries


def read_close_matrix() -> pd.DataFrame:
    path = PV_DIR / "收盤價.parquet"
    if not path.exists():
        path = ROOT / "Close.parquet"
    close = pd.read_parquet(path)
    close.index = pd.to_datetime(close.index)
    close.columns = close.columns.astype(str)
    return close.sort_index()


def gather_matrix_values(mat: pd.DataFrame, row_idx: np.ndarray, stock_ids: pd.Series) -> np.ndarray:
    arr = mat.to_numpy()
    col_map = {str(col): i for i, col in enumerate(mat.columns)}
    col_idx = stock_ids.astype(str).map(col_map).fillna(-1).astype(int).to_numpy()
    out = np.full(len(stock_ids), np.nan, dtype="float64")
    mask = (row_idx >= 0) & (row_idx < arr.shape[0]) & (col_idx >= 0)
    out[mask] = arr[row_idx[mask], col_idx[mask]]
    return out


def build_events(threshold: float) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    cols = ["date", "stock_id", "broker_id", "buy_qty", "sell_qty", "buy_vwap", "sell_vwap"]

    for path in sorted(BROKER_PL_DIR.glob("broker_pl_*.parquet")):
        df = pd.read_parquet(path, columns=cols)
        df["date"] = pd.to_datetime(df["date"])
        df["stock_id"] = df["stock_id"].astype(str)
        df["broker_id"] = df["broker_id"].astype(str)
        df["buy_amt"] = df["buy_qty"].fillna(0.0) * df["buy_vwap"].fillna(0.0) * 1000.0
        df["sell_amt"] = df["sell_qty"].fillna(0.0) * df["sell_vwap"].fillna(0.0) * 1000.0
        df["net_amount"] = df["buy_amt"] - df["sell_amt"]
        gross = df["buy_amt"] + df["sell_amt"]
        df["side_purity"] = safe_div(df["buy_amt"].to_numpy(), gross.to_numpy())

        buy = df[df["net_amount"] >= threshold].copy()
        buy["side"] = "buy"
        buy["entry_price"] = buy["buy_vwap"]
        buy["rank_net_amount_stock_day"] = buy.groupby(["date", "stock_id"])["net_amount"].rank(
            ascending=False, method="min"
        )

        if not buy.empty:
            parts.append(buy)
        print(f"{path.name}: buy={len(buy):,}", flush=True)

    if not parts:
        return pd.DataFrame()

    events = pd.concat(parts, ignore_index=True)
    events["abs_net_amount"] = events["net_amount"].abs()
    events["event_id"] = np.arange(len(events), dtype=np.int64)
    return events.sort_values(["date", "abs_net_amount"], ascending=[True, False]).reset_index(drop=True)


def normalize_summary(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["broker_id"] = summary["broker_id"].astype(str)
    if "side" not in summary.columns:
        summary["side"] = "buy"
    if "side_label" not in summary.columns:
        summary["side_label"] = "買超做多"
    if "is_high_win" not in summary.columns:
        summary["is_high_win"] = True
    else:
        summary["is_high_win"] = summary["is_high_win"].astype(str).str.lower().isin(["true", "1", "yes"])
    numeric_cols = [
        "rank",
        "n_events",
        "valid_5d",
        "n_stocks",
        "n_dates",
        "median_abs_net_M",
        "mean_purity",
        "win_rate_5d",
        "mean_direction_ret_5d",
        "win_rate_10d",
        "mean_direction_ret_10d",
    ]
    for col in numeric_cols:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")
    summary = summary[summary["is_high_win"] & summary["side"].eq("buy")].copy()
    if "score" in summary.columns:
        summary = summary.drop(columns=["score"])
    summary = summary.sort_values(
        ["win_rate_5d", "valid_5d", "mean_direction_ret_5d", "n_events"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    return summary


def load_branch_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return normalize_summary(pd.read_csv(path, dtype={"broker_id": str}))


def read_seed_events(path: Path, branch_start_date: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    events = pd.read_parquet(path)
    if events.empty:
        return events
    events = events.copy()
    events["date"] = pd.to_datetime(events["date"])
    events["stock_id"] = events["stock_id"].astype(str)
    events["broker_id"] = events["broker_id"].astype(str)
    events = events[events["date"] < pd.Timestamp(branch_start_date)].copy()
    return events


def branch_id_from_dir(branch_dir: Path) -> str:
    return branch_dir.name.split("_", 1)[0]


def build_events_from_branch_aggregates(
    branch_summary: pd.DataFrame,
    threshold: float,
    start_date: str,
    end_date: str | None,
) -> pd.DataFrame:
    if branch_summary.empty or not BRANCHES_DIR.exists():
        return pd.DataFrame()

    wanted_ids = set(branch_summary["broker_id"].astype(str))
    branch_name_map = dict(zip(branch_summary["broker_id"].astype(str), branch_summary["broker_name"].astype(str)))
    parts: list[pd.DataFrame] = []
    agg_cols = [
        "date",
        "stock_id",
        "securities_trader_id",
        "securities_trader",
        "buy_qty",
        "sell_qty",
        "buy_amount_est",
        "sell_amount_est",
        "avg_buy_price_est",
    ]

    for branch_dir in sorted(BRANCHES_DIR.glob("*")):
        if not branch_dir.is_dir():
            continue
        branch_id = branch_id_from_dir(branch_dir)
        if branch_id not in wanted_ids:
            continue
        path = branch_dir / "derived" / "stock_daily_agg.parquet"
        if not path.exists():
            continue

        try:
            df = pd.read_parquet(path, columns=agg_cols)
        except (KeyError, ValueError):
            df = pd.read_parquet(path)
            for col in agg_cols:
                if col not in df.columns:
                    df[col] = None
            df = df[agg_cols]
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] >= pd.Timestamp(start_date)].copy()
        if end_date:
            df = df[df["date"] <= pd.Timestamp(end_date)].copy()
        if df.empty:
            continue

        df["stock_id"] = df["stock_id"].astype(str)
        df["broker_id"] = branch_id
        if "securities_trader" in df.columns:
            df["broker_name"] = df["securities_trader"].fillna(branch_name_map.get(branch_id, branch_id)).astype(str)
        else:
            df["broker_name"] = branch_name_map.get(branch_id, branch_id)
        df["buy_amt"] = pd.to_numeric(df.get("buy_amount_est"), errors="coerce").fillna(0.0)
        df["sell_amt"] = pd.to_numeric(df.get("sell_amount_est"), errors="coerce").fillna(0.0)
        df["net_amount"] = df["buy_amt"] - df["sell_amt"]
        gross = df["buy_amt"] + df["sell_amt"]
        df["side_purity"] = safe_div(df["buy_amt"].to_numpy(), gross.to_numpy())

        buy = df[df["net_amount"] >= threshold].copy()
        if buy.empty:
            continue
        buy["side"] = "buy"
        buy["entry_price"] = pd.to_numeric(buy.get("avg_buy_price_est"), errors="coerce")
        buy["rank_net_amount_stock_day"] = buy.groupby(["date", "stock_id"])["net_amount"].rank(
            ascending=False, method="min"
        )
        parts.append(buy)

    if not parts:
        return pd.DataFrame()

    events = pd.concat(parts, ignore_index=True)
    events["abs_net_amount"] = events["net_amount"].abs()
    return events.sort_values(["date", "abs_net_amount"], ascending=[True, False]).reset_index(drop=True)


def attach_forward_returns(events: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    cal = close.index
    date_idx = cal.get_indexer(pd.to_datetime(events["date"]))
    events["t_close"] = gather_matrix_values(close, date_idx, events["stock_id"])
    entry = events["entry_price"].to_numpy(dtype="float64")

    for horizon in FORWARD_HORIZONS:
        target_idx = date_idx + horizon
        target_idx[(date_idx < 0) | (target_idx >= len(cal))] = -1
        close_h = gather_matrix_values(close, target_idx, events["stock_id"])
        raw_ret = safe_div(close_h - entry, entry)
        events[f"raw_ret_{horizon}d"] = raw_ret
        events[f"direction_ret_{horizon}d"] = raw_ret
        labels = np.full(len(events), None, dtype=object)
        valid = target_idx >= 0
        labels[valid] = cal[target_idx[valid]].strftime("%Y-%m-%d")
        events[f"target_date_{horizon}d"] = labels

    return events


def summarize_by_branch(
    events: pd.DataFrame,
    min_events: int,
    max_events: int,
    high_win_rate: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for broker_id, group in events.groupby("broker_id", sort=False):
        ret5 = group["direction_ret_5d"].dropna()
        ret10 = group["direction_ret_10d"].dropna()
        n = len(ret5)
        wins5 = int((ret5 > 0).sum()) if n else 0
        win5 = wins5 / n if n else np.nan
        mean5 = float(ret5.mean()) if n else np.nan
        is_high_win = bool(min_events <= n <= max_events and win5 >= high_win_rate and mean5 > 0)
        rows.append(
            {
                "broker_id": broker_id,
                "side": "buy",
                "side_label": "買超做多",
                "n_events": int(len(group)),
                "valid_5d": int(n),
                "n_stocks": int(group["stock_id"].nunique()),
                "n_dates": int(group["date"].nunique()),
                "total_abs_net_B": float(group["abs_net_amount"].sum() / 1e9),
                "median_abs_net_M": float(group["abs_net_amount"].median() / 1e6),
                "mean_purity": float(group["side_purity"].mean()),
                "win_rate_5d": float(win5) if np.isfinite(win5) else np.nan,
                "mean_direction_ret_5d": mean5,
                "median_direction_ret_5d": float(ret5.median()) if n else np.nan,
                "win_rate_10d": float((ret10 > 0).mean()) if len(ret10) else np.nan,
                "mean_direction_ret_10d": float(ret10.mean()) if len(ret10) else np.nan,
                "is_high_win": is_high_win,
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    broker_names, _, _ = load_name_maps()
    summary["broker_name"] = summary["broker_id"].map(broker_names).fillna(summary["broker_id"])
    summary = summary[summary["is_high_win"]].copy()
    summary = summary.sort_values(
        ["win_rate_5d", "valid_5d", "mean_direction_ret_5d", "n_events"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    return summary


def enrich_events(events: pd.DataFrame, branch_summary: pd.DataFrame) -> pd.DataFrame:
    broker_names, stock_names, stock_industries = load_name_maps()
    events = events.copy()
    stale_stats = [
        "branch_win_rate_5d",
        "branch_mean_direction_ret_5d",
        "branch_valid_5d",
        "is_high_win",
    ]
    events = events.drop(columns=[col for col in stale_stats if col in events.columns])
    summary_names = dict(zip(branch_summary["broker_id"].astype(str), branch_summary["broker_name"].astype(str)))
    events["broker_name"] = (
        events["broker_id"].astype(str).map(broker_names).fillna(events["broker_id"].astype(str).map(summary_names)).fillna(events["broker_id"])
    )
    events["stock_name"] = events["stock_id"].map(stock_names)
    events["industry"] = events["stock_id"].map(stock_industries)
    stats_cols = [
        "broker_id",
        "side",
        "win_rate_5d",
        "mean_direction_ret_5d",
        "is_high_win",
        "valid_5d",
    ]
    stats = branch_summary[stats_cols].rename(
        columns={
            "win_rate_5d": "branch_win_rate_5d",
            "mean_direction_ret_5d": "branch_mean_direction_ret_5d",
            "valid_5d": "branch_valid_5d",
        }
    )
    return events.merge(stats, on=["broker_id", "side"], how="left")


def frame_to_records(frame: pd.DataFrame, columns: list[str], float_digits: int = 4) -> list[dict[str, Any]]:
    records = []
    for row in frame[columns].to_dict(orient="records"):
        record: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                record[key] = value.strftime("%Y-%m-%d")
            elif isinstance(value, (np.integer,)):
                record[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                record[key] = round_or_none(value, float_digits)
            elif pd.isna(value):
                record[key] = None
            else:
                record[key] = value
        records.append(record)
    return records


def scalar_for_json(value: Any, float_digits: int = 4) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round_or_none(value, float_digits)
    if pd.isna(value):
        return None
    return value


def frame_to_rows(frame: pd.DataFrame, columns: list[str], float_digits: int = 4) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for values in frame[columns].itertuples(index=False, name=None):
        rows.append([scalar_for_json(value, float_digits) for value in values])
    return rows


def build_payload(
    events: pd.DataFrame,
    branch_summary: pd.DataFrame,
    threshold: float,
    min_events: int,
    max_events: int,
    high_win_rate: float,
    recent_days: int,
    source_buy_event_count: int,
    source_broker_count: int,
) -> dict[str, Any]:
    latest_dates = sorted(events["date"].dropna().unique())
    recent_cut = latest_dates[-recent_days] if len(latest_dates) >= recent_days else latest_dates[0]
    recent = events[events["date"] >= recent_cut].copy()
    recent = recent.sort_values(["date", "branch_win_rate_5d", "abs_net_amount"], ascending=[False, False, False])

    ranking_cols = [
        "rank",
        "broker_id",
        "broker_name",
        "n_events",
        "valid_5d",
        "n_stocks",
        "n_dates",
        "median_abs_net_M",
        "mean_purity",
        "win_rate_5d",
        "mean_direction_ret_5d",
        "win_rate_10d",
        "mean_direction_ret_10d",
    ]
    event_cols = [
        "event_id",
        "date",
        "stock_id",
        "stock_name",
        "industry",
        "broker_id",
        "broker_name",
        "net_amount",
        "abs_net_amount",
        "entry_price",
        "t_close",
        "side_purity",
        "rank_net_amount_stock_day",
        "direction_ret_1d",
        "direction_ret_3d",
        "direction_ret_5d",
        "direction_ret_10d",
        "direction_ret_20d",
        "branch_win_rate_5d",
        "branch_mean_direction_ret_5d",
        "branch_valid_5d",
    ]
    return {
        "meta": {
            "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "threshold": threshold,
            "min_events": min_events,
            "max_events": max_events,
            "high_win_rate": high_win_rate,
            "recent_days": recent_days,
            "date_min": events["date"].min().strftime("%Y-%m-%d"),
            "date_max": events["date"].max().strftime("%Y-%m-%d"),
            "event_count": int(len(events)),
            "source_buy_event_count": int(source_buy_event_count),
            "branch_count": int(len(branch_summary)),
            "source_broker_count": int(source_broker_count),
            "broker_count": int(events["broker_id"].nunique()),
        },
        "columns": {
            "rankings": ranking_cols,
            "recent": event_cols,
            "events": event_cols,
        },
        "rankings": frame_to_rows(branch_summary, ranking_cols),
        "recent": frame_to_rows(recent, event_cols),
        "events": frame_to_rows(
            events.sort_values(["date", "abs_net_amount"], ascending=[False, False]),
            event_cols,
        ),
    }


def render_html() -> str:
    return """<!doctype html>
<html lang="zh-Hant">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>一槍分點勝率工作台</title>
    <style>
      :root {
        --bg: #f7f8fa;
        --panel: #ffffff;
        --text: #18202a;
        --muted: #657080;
        --line: #dfe3e8;
        --buy: #0b7a53;
        --accent: #2459a6;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        background: var(--bg);
        color: var(--text);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        letter-spacing: 0;
      }
      header {
        padding: 22px 24px 14px;
        border-bottom: 1px solid var(--line);
        background: var(--panel);
      }
      h1 { margin: 0 0 12px; font-size: 24px; font-weight: 750; }
      h2 { margin: 0 0 12px; font-size: 18px; }
      .metrics {
        display: grid;
        grid-template-columns: repeat(6, minmax(120px, 1fr));
        gap: 10px;
      }
      .metric {
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 10px 12px;
        background: #fbfcfd;
      }
      .metric .label { color: var(--muted); font-size: 12px; }
      .metric .value { margin-top: 4px; font-size: 18px; font-weight: 720; }
      main { padding: 18px 24px 30px; display: grid; gap: 18px; }
      section {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 14px;
      }
      .bar {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
        margin-bottom: 12px;
      }
      button, select, input {
        height: 34px;
        border: 1px solid var(--line);
        border-radius: 6px;
        background: white;
        color: var(--text);
        padding: 0 10px;
        font: inherit;
      }
      button.active { border-color: var(--accent); background: #eaf1fb; color: var(--accent); }
      input[type="search"] { min-width: min(360px, 100%); }
      .table-wrap { overflow: auto; border: 1px solid var(--line); border-radius: 6px; }
      table { border-collapse: collapse; width: 100%; min-width: 960px; font-size: 13px; }
      th, td { border-bottom: 1px solid var(--line); padding: 9px 10px; text-align: right; white-space: nowrap; }
      th:first-child, td:first-child, .left { text-align: left; }
      th { position: sticky; top: 0; background: #f1f4f7; color: #3d4652; font-size: 12px; z-index: 1; }
      tr:hover td { background: #f8fafc; }
      .buy { color: var(--buy); font-weight: 680; }
      .muted { color: var(--muted); }
      .empty { padding: 18px; color: var(--muted); text-align: center; }
      .pager {
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        align-items: center;
        gap: 6px;
        margin-top: 10px;
      }
      .pager button {
        width: 34px;
        padding: 0;
      }
      .pager .gap {
        color: var(--muted);
        min-width: 18px;
        text-align: center;
      }
      @media (max-width: 900px) {
        header, main { padding-left: 12px; padding-right: 12px; }
        .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        table { min-width: 820px; }
      }
    </style>
  </head>
  <body>
    <header>
      <h1>一槍分點勝率工作台</h1>
      <div id="metrics" class="metrics"></div>
    </header>
    <main>
      <section>
        <div class="bar">
          <h2 style="margin-right:auto">近期高勝率買超分點出手</h2>
        </div>
        <div id="recentTable" class="table-wrap"></div>
        <div id="recentPager" class="pager"></div>
      </section>
      <section>
        <div class="bar">
          <h2 style="margin-right:auto">高勝率分點績效排行</h2>
        </div>
        <div id="rankTable" class="table-wrap"></div>
        <div id="rankPager" class="pager"></div>
      </section>
      <section>
        <div class="bar">
          <h2 style="margin-right:auto">事件查詢</h2>
          <input id="search" type="search" placeholder="搜尋股票、分點、產業" />
        </div>
        <div id="eventTable" class="table-wrap"></div>
        <div id="eventPager" class="pager"></div>
      </section>
    </main>
    <script src="./one-shot-full-market-data.js"></script>
    <script>
      const DATA = window.ONE_SHOT_FULL_MARKET;
      const inflate = (cols, rows) => rows.map(row => Object.fromEntries(cols.map((col, idx) => [col, row[idx]])));
      const RANKINGS = inflate(DATA.columns.rankings, DATA.rankings);
      const RECENT = inflate(DATA.columns.recent, DATA.recent);
      const EVENTS = inflate(DATA.columns.events, DATA.events);
      const fmt = (v, d = 2) => v == null || Number.isNaN(v) ? "N/A" : Number(v).toLocaleString("zh-TW", { maximumFractionDigits: d, minimumFractionDigits: d });
      const int = (v) => v == null ? "N/A" : Number(v).toLocaleString("zh-TW");
      const pct = (v) => v == null ? "N/A" : `${fmt(v * 100, 1)}%`;
      const amount = (v) => v == null ? "N/A" : `${fmt(v / 100000000, 2)} 億`;
      const PAGE_SIZE = 25;
      let rankPage = 1;
      let recentPage = 1;
      let eventPage = 1;

      function renderMetrics() {
        const m = DATA.meta;
        const rows = [
          ["資料期間", `${m.date_min} ~ ${m.date_max}`],
          ["高勝率事件", int(m.event_count)],
          ["高勝率分點", int(m.branch_count)],
          ["全市場買超事件", int(m.source_buy_event_count)],
          ["樣本範圍", `${int(m.min_events)} ~ ${int(m.max_events)}`],
          ["門檻", amount(m.threshold)],
        ];
        document.getElementById("metrics").innerHTML = rows.map(([label, value]) => `<div class="metric"><div class="label">${label}</div><div class="value">${value}</div></div>`).join("");
      }

      function table(headers, rows, emptyText) {
        if (!rows.length) return `<div class="empty">${emptyText}</div>`;
        return `<table><thead><tr>${headers.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table>`;
      }

      function pageItems(items, page) {
        const totalPages = Math.max(1, Math.ceil(items.length / PAGE_SIZE));
        const safePage = Math.min(Math.max(1, page), totalPages);
        const start = (safePage - 1) * PAGE_SIZE;
        return { rows: items.slice(start, start + PAGE_SIZE), page: safePage, totalPages };
      }

      function pagesAround(page, totalPages) {
        const wanted = new Set([1, totalPages, page - 2, page - 1, page, page + 1, page + 2]);
        const pages = [...wanted].filter(p => p >= 1 && p <= totalPages).sort((a, b) => a - b);
        return pages.reduce((out, p, idx) => {
          if (idx && p - pages[idx - 1] > 1) out.push("gap");
          out.push(p);
          return out;
        }, []);
      }

      function renderPager(id, page, totalPages, fnName) {
        const el = document.getElementById(id);
        if (totalPages <= 1) {
          el.innerHTML = "";
          return;
        }
        el.innerHTML = pagesAround(page, totalPages).map(p => {
          if (p === "gap") return `<span class="gap">...</span>`;
          return `<button class="${p === page ? "active" : ""}" onclick="${fnName}(${p})">${p}</button>`;
        }).join("");
      }

      function renderRankings() {
        const paged = pageItems(RANKINGS, rankPage);
        rankPage = paged.page;
        const rows = paged.rows
          .map(r => `<tr>
            <td class="left"><b>${r.broker_id}</b> ${r.broker_name}</td>
            <td>${int(r.valid_5d)}</td>
            <td>${int(r.n_stocks)}</td>
            <td>${int(r.n_dates)}</td>
            <td>${pct(r.win_rate_5d)}</td>
            <td class="${r.mean_direction_ret_5d >= 0 ? "buy" : ""}">${pct(r.mean_direction_ret_5d)}</td>
            <td>${pct(r.win_rate_10d)}</td>
            <td>${amount(r.median_abs_net_M * 1000000)}</td>
          </tr>`);
        document.getElementById("rankTable").innerHTML = table(
          ["分點", "樣本", "股票", "日期", "5日勝率", "5日均報酬", "10日勝率", "中位淨額"],
          rows,
          "沒有符合條件的分點"
        );
        renderPager("rankPager", paged.page, paged.totalPages, "setRankPage");
      }

      function renderRecent() {
        const paged = pageItems(RECENT, recentPage);
        recentPage = paged.page;
        const rows = paged.rows
          .map(r => `<tr>
            <td class="left">${r.date}</td>
            <td class="left"><b>${r.broker_id}</b> ${r.broker_name}</td>
            <td class="left"><b>${r.stock_id}</b> ${r.stock_name || ""}</td>
            <td>${amount(r.abs_net_amount)}</td>
            <td>${pct(r.side_purity)}</td>
            <td>${pct(r.branch_win_rate_5d)}</td>
            <td class="${r.direction_ret_5d >= 0 ? "buy" : ""}">${pct(r.direction_ret_5d)}</td>
            <td class="${r.direction_ret_10d >= 0 ? "buy" : ""}">${pct(r.direction_ret_10d)}</td>
          </tr>`);
        document.getElementById("recentTable").innerHTML = table(
          ["日期", "分點", "股票", "淨額", "純度", "分點5日勝率", "事件5日", "事件10日"],
          rows,
          "近期沒有高勝率分點事件"
        );
        renderPager("recentPager", paged.page, paged.totalPages, "setRecentPage");
      }

      function renderEvents() {
        const q = document.getElementById("search").value.trim().toLowerCase();
        const filtered = EVENTS
          .filter(r => !q || `${r.stock_id} ${r.stock_name || ""} ${r.industry || ""} ${r.broker_id} ${r.broker_name}`.toLowerCase().includes(q))
        const paged = pageItems(filtered, eventPage);
        eventPage = paged.page;
        const rows = paged.rows
          .map(r => `<tr>
            <td class="left">${r.date}</td>
            <td class="left"><b>${r.stock_id}</b> ${r.stock_name || ""}</td>
            <td class="left"><b>${r.broker_id}</b> ${r.broker_name}</td>
            <td>${amount(r.abs_net_amount)}</td>
            <td>${pct(r.side_purity)}</td>
            <td>${pct(r.branch_win_rate_5d)}</td>
            <td class="${r.direction_ret_1d >= 0 ? "buy" : ""}">${pct(r.direction_ret_1d)}</td>
            <td class="${r.direction_ret_5d >= 0 ? "buy" : ""}">${pct(r.direction_ret_5d)}</td>
            <td class="${r.direction_ret_20d >= 0 ? "buy" : ""}">${pct(r.direction_ret_20d)}</td>
          </tr>`);
        document.getElementById("eventTable").innerHTML = table(
          ["日期", "股票", "分點", "淨額", "純度", "分點5日勝率", "1日", "5日", "20日"],
          rows,
          "沒有符合條件的事件"
        );
        renderPager("eventPager", paged.page, paged.totalPages, "setEventPage");
      }

      function setRankPage(page) {
        rankPage = page;
        renderRankings();
      }

      function setRecentPage(page) {
        recentPage = page;
        renderRecent();
      }

      function setEventPage(page) {
        eventPage = page;
        renderEvents();
      }

      window.setRankPage = setRankPage;
      window.setRecentPage = setRecentPage;
      window.setEventPage = setEventPage;

      document.getElementById("search").addEventListener("input", () => {
        eventPage = 1;
        renderEvents();
      });

      renderMetrics();
      renderRecent();
      renderRankings();
      renderEvents();
    </script>
  </body>
</html>
"""


def main() -> int:
    args = parse_args()
    data_dir = output_path(args.data_output_dir)
    dashboard_dir = output_path(args.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    close = read_close_matrix()
    branch_summary_path = data_dir / "branch_direction_summary.csv"
    seed_events_path = data_dir / "one_shot_full_market_events.parquet"

    if args.rebuild_baseline or not branch_summary_path.exists() or not seed_events_path.exists():
        events = build_events(args.threshold)
        if events.empty:
            raise SystemExit("No baseline events found.")
        events = attach_forward_returns(events, close)
        source_buy_event_count = len(events)
        source_broker_count = events["broker_id"].nunique()
        branch_summary = summarize_by_branch(events, args.min_events, args.max_events, args.high_win_rate)
        events = enrich_events(events, branch_summary)
        seed_events = events[events["is_high_win"].fillna(False)].copy()
    else:
        branch_summary = load_branch_summary(branch_summary_path)
        seed_events = read_seed_events(seed_events_path, args.branch_start_date)
        source_buy_event_count = args.source_buy_event_count
        source_broker_count = int(branch_summary["broker_id"].nunique()) if not branch_summary.empty else 0

    branch_events = build_events_from_branch_aggregates(
        branch_summary,
        args.threshold,
        args.branch_start_date,
        args.branch_end_date,
    )

    pieces = [frame for frame in [seed_events, branch_events] if frame is not None and not frame.empty]
    if not pieces:
        raise SystemExit("No high-win buy events found.")

    combined_events = pd.concat(pieces, ignore_index=True)
    combined_events["date"] = pd.to_datetime(combined_events["date"])
    combined_events["stock_id"] = combined_events["stock_id"].astype(str)
    combined_events["broker_id"] = combined_events["broker_id"].astype(str)
    combined_events = combined_events.drop_duplicates(
        subset=["date", "stock_id", "broker_id", "net_amount"],
        keep="last",
    ).sort_values(["date", "abs_net_amount"], ascending=[True, False]).reset_index(drop=True)
    combined_events["event_id"] = np.arange(len(combined_events), dtype=np.int64)
    combined_events = attach_forward_returns(combined_events, close)
    combined_events = enrich_events(combined_events, branch_summary)
    high_win_events = combined_events[combined_events["is_high_win"].fillna(False)].copy()
    if high_win_events.empty:
        raise SystemExit("No high-win buy events found.")
    payload = build_payload(
        high_win_events,
        branch_summary,
        args.threshold,
        args.min_events,
        args.max_events,
        args.high_win_rate,
        args.recent_days,
        source_buy_event_count,
        int(source_broker_count),
    )

    high_win_events.to_parquet(data_dir / "one_shot_full_market_events.parquet", index=False)
    branch_summary.to_csv(data_dir / "branch_direction_summary.csv", index=False)
    (data_dir / "one_shot_full_market_events_sample.json").write_text(
        json.dumps(payload["events"][:1000], ensure_ascii=False),
        encoding="utf-8",
    )
    (dashboard_dir / "one-shot-full-market-data.js").write_text(
        "window.ONE_SHOT_FULL_MARKET = "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    (dashboard_dir / "index.html").write_text(render_html(), encoding="utf-8")

    print(
        json.dumps(
            {
                "events": payload["meta"]["event_count"],
                "source_buy_events": payload["meta"]["source_buy_event_count"],
                "broker_count": payload["meta"]["broker_count"],
                "high_win_branch_count": payload["meta"]["branch_count"],
                "dashboard": str(dashboard_dir / "index.html"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
