#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
BRANCHES_DIR = ROOT / "data" / "branches"
REFERENCE_DIR = ROOT / "data" / "reference"
STOCK_INFO_PATH = REFERENCE_DIR / "taiwan_stock_info.parquet"
CLOSE_PATH = ROOT / "Close.parquet"

DATA_OUTPUT_DIR = ROOT / "data" / "one_shot"
DASHBOARD_OUTPUT_DIR = ROOT / "dashboard" / "one_shot"
EVENTS_PARQUET = DATA_OUTPUT_DIR / "one_shot_events.parquet"
EVENTS_JSON = DATA_OUTPUT_DIR / "one_shot_events.json"
OUTPUT_HTML = DASHBOARD_OUTPUT_DIR / "index.html"
OUTPUT_JS = DASHBOARD_OUTPUT_DIR / "one-shot-data.js"

DEFAULT_THRESHOLD = 200_000_000
DEFAULT_MIN_RETAINED_RATIO = 0.5
DEFAULT_LIMIT_UP_GAP = 0.01
EXCLUDED_BRANCH_IDS = {"9268"}
OFFICIAL_BENCHMARK_COLUMN = "TAIEX"
BENCHMARK_COLUMN = "__MARKET_BENCHMARK__"
BENCHMARK_LABEL = "TAIEX / local equal-weight proxy"
FORWARD_WINDOWS = [10, 20, 60, 120]
RETENTION_DAYS = 5
CHART_FORWARD_DAYS = 130

BRANCH_COLUMNS = [
    "date",
    "stock_id",
    "buy_qty",
    "sell_qty",
    "net_qty",
    "buy_amount_est",
    "sell_amount_est",
    "avg_buy_price_est",
    "avg_sell_price_est",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one-shot branch chip dashboard.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--min-retained-ratio", type=float, default=DEFAULT_MIN_RETAINED_RATIO)
    parser.add_argument("--limit-up-gap", type=float, default=DEFAULT_LIMIT_UP_GAP, help="Exclude buy events whose entry cost is within this absolute gap of estimated limit-up price.")
    parser.add_argument("--include-sell", action="store_true", help="Also include net-sell events for ad hoc diagnostics.")
    parser.add_argument("--output-dir", default=str(DASHBOARD_OUTPUT_DIR))
    parser.add_argument("--data-output-dir", default=str(DATA_OUTPUT_DIR))
    return parser.parse_args()


def finite_number(value: Any) -> bool:
    return value is not None and not pd.isna(value) and np.isfinite(float(value))


def round_or_none(value: Any, digits: int = 4) -> Optional[float]:
    if not finite_number(value):
        return None
    return round(float(value), digits)


def scalar_or_none(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def to_records(frame: pd.DataFrame, float_columns: Optional[list[str]] = None) -> list[dict[str, Any]]:
    float_columns = float_columns or []
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        record: dict[str, Any] = {}
        for key, value in row.items():
            if key in float_columns:
                record[key] = round_or_none(value, 4)
            else:
                record[key] = scalar_or_none(value)
        records.append(record)
    return records


def load_stock_info() -> pd.DataFrame:
    if not STOCK_INFO_PATH.exists():
        return pd.DataFrame(columns=["stock_id", "stock_name", "industry_category", "type"])
    frame = pd.read_parquet(STOCK_INFO_PATH)
    frame["stock_id"] = frame["stock_id"].astype("string")
    if "date" in frame.columns:
        frame = frame.sort_values(["stock_id", "date"]).drop_duplicates("stock_id", keep="last")
    needed = ["stock_id", "stock_name", "industry_category", "type"]
    for column in needed:
        if column not in frame.columns:
            frame[column] = None
    return frame[needed].copy()


def is_strategy_stock(stock_id: str, stock_meta: dict[str, Any]) -> bool:
    stock_id = str(stock_id)
    if not (stock_id.isdigit() and len(stock_id) == 4):
        return False
    if stock_id.startswith("00"):
        return False
    industry = str(stock_meta.get("industry_category") or "").upper()
    stock_name = str(stock_meta.get("stock_name") or "").upper()
    blocked_markers = ["ETF", "ETN", "指數投資證券", "受益證券"]
    return not any(marker in industry or marker in stock_name for marker in blocked_markers)


def load_branch_manifest(branch_dir: Path) -> dict[str, Any]:
    manifest_path = branch_dir / "meta" / "manifest.json"
    if not manifest_path.exists():
        return {
            "securities_trader_id": branch_dir.name.split("_", 1)[0],
            "securities_trader": branch_dir.name,
        }
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def build_market_benchmark(close_df: pd.DataFrame) -> pd.Series:
    stock_columns = [
        column
        for column in close_df.columns
        if str(column).isdigit() and len(str(column)) == 4 and not str(column).startswith("00")
    ]
    if not stock_columns:
        if OFFICIAL_BENCHMARK_COLUMN not in close_df.columns:
            raise SystemExit("No benchmark or stock columns are available.")
        return close_df[OFFICIAL_BENCHMARK_COLUMN]

    stock_returns = close_df[stock_columns].pct_change(fill_method=None)
    proxy_returns = stock_returns.median(axis=1, skipna=True).fillna(0.0)
    proxy = (1.0 + proxy_returns).cumprod() * 100.0

    if OFFICIAL_BENCHMARK_COLUMN not in close_df.columns:
        return proxy

    official = close_df[OFFICIAL_BENCHMARK_COLUMN]
    first_official_date = official.first_valid_index()
    if first_official_date is None:
        return proxy

    scale_base = proxy.loc[first_official_date]
    if not finite_number(scale_base) or float(scale_base) == 0:
        return official.combine_first(proxy)

    scaled_proxy = proxy * (float(official.loc[first_official_date]) / float(scale_base))
    return official.combine_first(scaled_proxy)


def trading_window_dates(trading_dates: pd.DatetimeIndex, event_date: pd.Timestamp, days: int) -> list[pd.Timestamp]:
    if event_date not in trading_dates:
        return []
    start = trading_dates.get_loc(event_date)
    end = min(len(trading_dates), int(start) + days + 1)
    return list(trading_dates[int(start) + 1 : end])


def estimate_limit_up_from_previous_close(
    close_df: pd.DataFrame,
    stock_id: str,
    event_date: pd.Timestamp,
) -> tuple[Optional[float], Optional[float]]:
    if stock_id not in close_df.columns or event_date not in close_df.index:
        return None, None
    start_index = int(close_df.index.get_loc(event_date))
    if start_index <= 0:
        return None, None
    previous_close = close_df.iloc[start_index - 1][stock_id]
    if not finite_number(previous_close) or float(previous_close) <= 0:
        return None, None
    return float(previous_close), float(previous_close) * 1.1


def calc_forward_metrics(
    close_df: pd.DataFrame,
    stock_id: str,
    event_date: pd.Timestamp,
    entry_price: float,
    direction: int,
    horizons: list[int],
) -> dict[str, Optional[float]]:
    metrics: dict[str, Optional[float]] = {}
    if stock_id not in close_df.columns or event_date not in close_df.index:
        for horizon in horizons:
            metrics[f"stock_return_{horizon}d"] = None
            metrics[f"market_return_{horizon}d"] = None
            metrics[f"direction_return_{horizon}d"] = None
            metrics[f"alpha_{horizon}d"] = None
        return metrics

    start_index = int(close_df.index.get_loc(event_date))
    market_start = close_df.at[event_date, BENCHMARK_COLUMN]
    for horizon in horizons:
        stock_return: Optional[float] = None
        market_return: Optional[float] = None
        direction_return: Optional[float] = None
        alpha_return: Optional[float] = None

        future_index = start_index + horizon
        if future_index < len(close_df.index):
            future_date = close_df.index[future_index]
            future_close = close_df.at[future_date, stock_id]
            market_future = close_df.at[future_date, BENCHMARK_COLUMN]
            if finite_number(entry_price) and finite_number(future_close) and float(entry_price) > 0:
                stock_return = float(future_close) / float(entry_price) - 1.0
                direction_return = stock_return * direction
            if finite_number(market_start) and finite_number(market_future) and float(market_start) > 0:
                market_return = float(market_future) / float(market_start) - 1.0
            if stock_return is not None and market_return is not None:
                alpha_return = (stock_return - market_return) * direction

        metrics[f"stock_return_{horizon}d"] = stock_return
        metrics[f"market_return_{horizon}d"] = market_return
        metrics[f"direction_return_{horizon}d"] = direction_return
        metrics[f"alpha_{horizon}d"] = alpha_return

    return metrics


def build_chart_rows(
    close_df: pd.DataFrame,
    stock_id: str,
    event_date: pd.Timestamp,
    chart_forward_days: int,
) -> list[dict[str, Any]]:
    if stock_id not in close_df.columns or event_date not in close_df.index:
        return []
    start_index = int(close_df.index.get_loc(event_date))
    end_index = min(len(close_df.index) - 1, start_index + chart_forward_days)
    subset = close_df.iloc[start_index : end_index + 1][[stock_id, BENCHMARK_COLUMN]].copy()
    subset = subset.rename(columns={stock_id: "stock_close", BENCHMARK_COLUMN: "market_close"})
    start_stock = subset["stock_close"].dropna().iloc[0] if subset["stock_close"].notna().any() else np.nan
    start_market = subset["market_close"].dropna().iloc[0] if subset["market_close"].notna().any() else np.nan
    subset["date"] = subset.index
    subset["stock_norm"] = subset["stock_close"] / start_stock * 100 if finite_number(start_stock) else np.nan
    subset["market_norm"] = subset["market_close"] / start_market * 100 if finite_number(start_market) else np.nan
    return to_records(
        subset[["date", "stock_close", "market_close", "stock_norm", "market_norm"]],
        float_columns=["stock_close", "market_close", "stock_norm", "market_norm"],
    )


def build_followup_rows(
    stock_flows: pd.DataFrame,
    close_df: pd.DataFrame,
    stock_id: str,
    event_date: pd.Timestamp,
    follow_dates: list[pd.Timestamp],
) -> list[dict[str, Any]]:
    flow_map = stock_flows.set_index("date")
    rows: list[dict[str, Any]] = []
    for offset, date in enumerate([event_date, *follow_dates]):
        if date in flow_map.index:
            flow = flow_map.loc[date]
            buy_qty = float(flow["buy_qty"])
            sell_qty = float(flow["sell_qty"])
            net_qty = float(flow["net_qty"])
            buy_amount = float(flow["buy_amount_est"])
            sell_amount = float(flow["sell_amount_est"])
        else:
            buy_qty = sell_qty = net_qty = buy_amount = sell_amount = 0.0
        close = close_df.at[date, stock_id] if stock_id in close_df.columns and date in close_df.index else np.nan
        rows.append(
            {
                "offset": int(offset),
                "date": date.strftime("%Y-%m-%d"),
                "close": round_or_none(close, 4),
                "buy_lots": round_or_none(buy_qty / 1000, 4),
                "sell_lots": round_or_none(sell_qty / 1000, 4),
                "net_lots": round_or_none(net_qty / 1000, 4),
                "net_amount_billion": round_or_none((buy_amount - sell_amount) / 100_000_000, 4),
            }
        )
    return rows


def build_one_branch_events(
    branch_dir: Path,
    close_df: pd.DataFrame,
    stock_lookup: dict[str, dict[str, Any]],
    allowed_stock_ids: set[str],
    threshold: float,
    include_sell: bool,
    limit_up_gap: float,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    manifest = load_branch_manifest(branch_dir)
    branch_id = str(manifest.get("securities_trader_id", branch_dir.name.split("_", 1)[0]))
    branch_name = str(manifest.get("securities_trader", branch_dir.name))
    agg_path = branch_dir / "derived" / "stock_daily_agg.parquet"
    if not agg_path.exists():
        return [], {}, {}

    frame = pd.read_parquet(agg_path, columns=BRANCH_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["stock_id"] = frame["stock_id"].astype(str)
    frame["net_amount_est"] = frame["buy_amount_est"] - frame["sell_amount_est"]
    frame = frame[frame["stock_id"].isin(allowed_stock_ids)].copy()
    frame = frame[frame["stock_id"].isin(close_df.columns)].copy()

    buy_candidates = frame[frame["net_amount_est"] >= threshold].copy()
    if include_sell:
        sell_candidates = frame[frame["net_amount_est"] <= -threshold].copy()
        candidates = pd.concat([buy_candidates, sell_candidates], ignore_index=True)
    else:
        candidates = buy_candidates
    if candidates.empty:
        return [], {}, {}

    trading_dates = close_df.index
    stock_groups = {stock_id: group.sort_values("date").copy() for stock_id, group in frame.groupby("stock_id")}
    events: list[dict[str, Any]] = []
    event_windows: dict[str, list[dict[str, Any]]] = {}
    followups: dict[str, list[dict[str, Any]]] = {}

    for candidate in candidates.sort_values(["date", "stock_id"]).itertuples(index=False):
        event_date = pd.Timestamp(candidate.date).normalize()
        stock_id = str(candidate.stock_id)
        if event_date not in trading_dates or stock_id not in close_df.columns:
            continue

        direction = 1 if float(candidate.net_amount_est) > 0 else -1
        side = "buy" if direction == 1 else "sell"
        entry_price_raw = candidate.avg_buy_price_est if side == "buy" else candidate.avg_sell_price_est
        close_at_event = close_df.at[event_date, stock_id]
        entry_price_source = "avg_buy_price_est" if side == "buy" else "avg_sell_price_est"
        if not finite_number(entry_price_raw) or float(entry_price_raw) <= 0:
            entry_price_raw = close_at_event
            entry_price_source = "close"
        if not finite_number(entry_price_raw) or float(entry_price_raw) <= 0:
            continue

        previous_close, estimated_limit_up_price = estimate_limit_up_from_previous_close(close_df, stock_id, event_date)
        limit_up_gap_value = None
        if side == "buy" and finite_number(estimated_limit_up_price) and float(estimated_limit_up_price) > 0:
            limit_up_gap_value = float(entry_price_raw) / float(estimated_limit_up_price) - 1.0
            if abs(limit_up_gap_value) <= limit_up_gap:
                continue

        follow_dates = trading_window_dates(trading_dates, event_date, RETENTION_DAYS)
        stock_flows = stock_groups[stock_id]
        follow_flow = stock_flows[stock_flows["date"].isin(follow_dates)]
        initial_direction_qty = abs(float(candidate.net_qty))
        if initial_direction_qty <= 0:
            continue
        retained_qty = initial_direction_qty + float((follow_flow["net_qty"] * direction).sum())
        retained_ratio = retained_qty / initial_direction_qty
        opposite_qty_5d = float(follow_flow.loc[follow_flow["net_qty"] * direction < 0, "net_qty"].abs().sum())
        same_side_qty_5d = float(follow_flow.loc[follow_flow["net_qty"] * direction > 0, "net_qty"].abs().sum())

        metrics = calc_forward_metrics(
            close_df=close_df,
            stock_id=stock_id,
            event_date=event_date,
            entry_price=float(entry_price_raw),
            direction=direction,
            horizons=FORWARD_WINDOWS,
        )
        stock_meta = stock_lookup.get(stock_id, {})
        event_id = f"{branch_id}_{stock_id}_{event_date.strftime('%Y%m%d')}_{side}"
        event = {
            "event_id": event_id,
            "branch_id": branch_id,
            "branch_name": branch_name,
            "branch_folder": branch_dir.name,
            "stock_id": stock_id,
            "stock_name": stock_meta.get("stock_name"),
            "industry_category": stock_meta.get("industry_category"),
            "stock_type": stock_meta.get("type"),
            "side": side,
            "direction": direction,
            "date": event_date,
            "net_amount": float(candidate.net_amount_est),
            "abs_net_amount": abs(float(candidate.net_amount_est)),
            "net_amount_billion": float(candidate.net_amount_est) / 100_000_000,
            "buy_amount_billion": float(candidate.buy_amount_est) / 100_000_000,
            "sell_amount_billion": float(candidate.sell_amount_est) / 100_000_000,
            "buy_lots": float(candidate.buy_qty) / 1000,
            "sell_lots": float(candidate.sell_qty) / 1000,
            "net_lots": float(candidate.net_qty) / 1000,
            "entry_price": float(entry_price_raw),
            "entry_price_source": entry_price_source,
            "event_close": float(close_at_event) if finite_number(close_at_event) else None,
            "previous_close": previous_close,
            "estimated_limit_up_price": estimated_limit_up_price,
            "limit_up_gap": limit_up_gap_value,
            "retained_qty_5d": retained_qty,
            "retained_ratio_5d": retained_ratio,
            "opposite_lots_5d": opposite_qty_5d / 1000,
            "same_side_lots_5d": same_side_qty_5d / 1000,
            **metrics,
        }
        events.append(event)
        event_windows[event_id] = build_chart_rows(close_df, stock_id, event_date, CHART_FORWARD_DAYS)
        followups[event_id] = build_followup_rows(stock_flows, close_df, stock_id, event_date, follow_dates)

    return events, event_windows, followups


def build_summary(events: pd.DataFrame, threshold: float, min_retained_ratio: float) -> dict[str, Any]:
    def median_or_none(series: pd.Series) -> Optional[float]:
        valid = series.dropna()
        if valid.empty:
            return None
        return round_or_none(valid.median(), 4)

    kept = events[(events["side"] == "buy") & (events["retained_ratio_5d"] >= min_retained_ratio)]
    by_branch = []
    if not events.empty:
        for (branch_id, branch_name), group in events.groupby(["branch_id", "branch_name"], sort=False):
            buy_group = group[group["side"] == "buy"]
            kept_group = buy_group[buy_group["retained_ratio_5d"] >= min_retained_ratio]
            by_branch.append(
                {
                    "branch_id": branch_id,
                    "branch_name": branch_name,
                    "events": int(len(group)),
                    "buy_events": int(len(buy_group)),
                    "kept_buy_events": int(len(kept_group)),
                    "median_alpha_20d": median_or_none(kept_group["alpha_20d"]),
                    "median_alpha_60d": median_or_none(kept_group["alpha_60d"]),
                }
            )

    return {
        "threshold": threshold,
        "min_retained_ratio": min_retained_ratio,
        "retention_days": RETENTION_DAYS,
        "benchmark": BENCHMARK_LABEL,
        "forward_windows": FORWARD_WINDOWS,
        "total_events": int(len(events)),
        "buy_events": int((events["side"] == "buy").sum()) if not events.empty else 0,
        "sell_events": int((events["side"] == "sell").sum()) if not events.empty else 0,
        "default_buy_events": int(len(kept)),
        "branch_count": int(events["branch_id"].nunique()) if not events.empty else 0,
        "median_alpha_20d": median_or_none(kept["alpha_20d"]) if not kept.empty else None,
        "median_alpha_60d": median_or_none(kept["alpha_60d"]) if not kept.empty else None,
        "win_rate_alpha_20d": round_or_none((kept["alpha_20d"] > 0).mean(), 4) if kept["alpha_20d"].notna().any() else None,
        "win_rate_alpha_60d": round_or_none((kept["alpha_60d"] > 0).mean(), 4) if kept["alpha_60d"].notna().any() else None,
        "by_branch": by_branch,
    }


def build_branch_rankings(events: pd.DataFrame, min_retained_ratio: float) -> list[dict[str, Any]]:
    def median_or_none(series: pd.Series) -> Optional[float]:
        valid = series.dropna()
        if valid.empty:
            return None
        return round_or_none(valid.median(), 4)

    def win_rate_or_none(series: pd.Series) -> Optional[float]:
        valid = series.dropna()
        if valid.empty:
            return None
        return round_or_none((valid > 0).mean(), 4)

    qualified = events[(events["side"] == "buy") & (events["retained_ratio_5d"] >= min_retained_ratio)].copy()
    rankings: list[dict[str, Any]] = []
    for (branch_id, branch_name), group in qualified.groupby(["branch_id", "branch_name"], sort=False):
        valid20 = group["alpha_20d"].dropna()
        median20 = median_or_none(group["alpha_20d"])
        median60 = median_or_none(group["alpha_60d"])
        median120 = median_or_none(group["alpha_120d"])
        win20 = win_rate_or_none(group["alpha_20d"])
        avg_amount = round_or_none(group["abs_net_amount"].mean(), 4)

        score = None
        if median20 is not None:
            sample_bonus = min(np.log1p(len(valid20)) / 10.0, 0.2)
            win_bonus = ((win20 or 0.0) - 0.5) * 0.08
            score = float(median20) + sample_bonus + win_bonus

        rankings.append(
            {
                "branch_id": branch_id,
                "branch_name": branch_name,
                "sample_count": int(len(group)),
                "valid_20d_count": int(len(valid20)),
                "avg_amount": avg_amount,
                "median_alpha_20d": median20,
                "median_alpha_60d": median60,
                "median_alpha_120d": median120,
                "win_rate_alpha_20d": win20,
                "score": round_or_none(score, 4),
            }
        )

    return sorted(
        rankings,
        key=lambda row: (
            row["score"] is not None,
            row["score"] if row["score"] is not None else -999,
            row["valid_20d_count"],
            row["sample_count"],
        ),
        reverse=True,
    )


def render_html() -> str:
    return """<!doctype html>
<html lang="zh-Hant">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>一槍策略工作台</title>
    <style>
      :root {
        --bg: #f6f7f8;
        --panel: #ffffff;
        --ink: #15191f;
        --muted: #667085;
        --line: #d6dbe1;
        --line-soft: #e7eaee;
        --accent: #146c5f;
        --accent-soft: #e6f3f0;
        --red: #a53b35;
        --red-soft: #fae9e7;
        --blue: #265f9f;
        --shadow: 0 10px 24px rgba(25, 33, 44, 0.08);
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        color: var(--ink);
        background: var(--bg);
        font-family: "Noto Sans TC", "PingFang TC", -apple-system, BlinkMacSystemFont, sans-serif;
      }
      .shell {
        width: min(1680px, calc(100% - 28px));
        margin: 18px auto 36px;
      }
      .topbar {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: end;
        padding: 8px 0 14px;
        border-bottom: 1px solid var(--line);
      }
      h1 {
        margin: 0;
        font-size: 26px;
        line-height: 1.2;
      }
      .meta {
        margin-top: 5px;
        color: var(--muted);
        font-size: 13px;
      }
      .summary {
        display: grid;
        grid-template-columns: repeat(6, minmax(130px, 1fr));
        gap: 10px;
        margin: 14px 0;
      }
      .metric {
        background: var(--panel);
        border: 1px solid var(--line-soft);
        border-radius: 8px;
        padding: 12px;
        box-shadow: var(--shadow);
      }
      .metric .label {
        color: var(--muted);
        font-size: 12px;
      }
      .metric .value {
        margin-top: 6px;
        font-size: 22px;
        font-weight: 750;
      }
      .ranking-panel {
        margin-bottom: 14px;
      }
      .recent-panel {
        margin-bottom: 14px;
      }
      .ranking-wrap {
        overflow: auto;
      }
      .ranking-row, .recent-row {
        cursor: pointer;
      }
      .ranking-row:hover, .recent-row:hover {
        background: #f5faf8;
      }
      .rank-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 26px;
        height: 26px;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        font-weight: 800;
      }
      .toolbar {
        display: grid;
        grid-template-columns: 1.25fr repeat(5, minmax(130px, 1fr));
        gap: 10px;
        margin-bottom: 14px;
      }
      input, select {
        width: 100%;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: #fff;
        color: var(--ink);
        font-size: 14px;
        padding: 10px 11px;
      }
      .layout {
        display: grid;
        grid-template-columns: minmax(0, 1.25fr) minmax(430px, 0.75fr);
        gap: 14px;
        align-items: start;
      }
      .panel {
        background: var(--panel);
        border: 1px solid var(--line-soft);
        border-radius: 8px;
        box-shadow: var(--shadow);
      }
      .panel-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        padding: 12px 14px;
        border-bottom: 1px solid var(--line-soft);
      }
      .panel-head h2 {
        margin: 0;
        font-size: 16px;
      }
      .count {
        color: var(--muted);
        font-size: 13px;
      }
      .table-wrap {
        max-height: calc(100vh - 250px);
        overflow: auto;
      }
      table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
      }
      th, td {
        border-bottom: 1px solid var(--line-soft);
        padding: 9px 8px;
        text-align: right;
        white-space: nowrap;
      }
      th {
        position: sticky;
        top: 0;
        z-index: 1;
        background: #f8fafb;
        color: #475467;
        font-weight: 700;
      }
      th:first-child, td:first-child,
      th:nth-child(2), td:nth-child(2) { text-align: left; }
      tr.event-row { cursor: pointer; }
      tr.event-row:hover { background: #f5faf8; }
      tr.event-row.active { background: var(--accent-soft); }
      .tag {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 44px;
        border-radius: 999px;
        padding: 4px 8px;
        font-size: 12px;
        font-weight: 750;
      }
      .tag.buy { background: var(--accent-soft); color: var(--accent); }
      .pos { color: var(--accent); font-weight: 700; }
      .neg { color: var(--red); font-weight: 700; }
      .detail {
        position: sticky;
        top: 14px;
      }
      .detail-body {
        padding: 14px;
      }
      .detail-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
      }
      .detail-title h2 {
        margin: 0;
        font-size: 22px;
      }
      .subline {
        margin-top: 5px;
        color: var(--muted);
        font-size: 13px;
      }
      .detail-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 8px;
        margin-top: 12px;
      }
      .box {
        border: 1px solid var(--line-soft);
        border-radius: 8px;
        padding: 10px;
        background: #fbfcfd;
      }
      .box .label {
        color: var(--muted);
        font-size: 12px;
      }
      .box .value {
        margin-top: 5px;
        font-size: 18px;
        font-weight: 750;
      }
      canvas {
        width: 100%;
        height: auto;
        display: block;
        margin-top: 12px;
        border: 1px solid var(--line-soft);
        border-radius: 8px;
        background: #fff;
      }
      .mini-table {
        margin-top: 12px;
        max-height: 260px;
        overflow: auto;
        border: 1px solid var(--line-soft);
        border-radius: 8px;
      }
      .mini-table table { font-size: 12px; }
      .empty {
        padding: 20px;
        color: var(--muted);
      }
      @media (max-width: 1240px) {
        .summary { grid-template-columns: repeat(3, minmax(130px, 1fr)); }
        .toolbar { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .layout { grid-template-columns: 1fr; }
        .detail { position: static; }
      }
      @media (max-width: 720px) {
        .summary { grid-template-columns: repeat(2, minmax(130px, 1fr)); }
        .toolbar { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <header class="topbar">
        <div>
          <h1>一槍策略工作台</h1>
          <div class="meta" id="meta"></div>
        </div>
      </header>

      <section class="summary" id="summary"></section>

      <section class="panel ranking-panel">
        <div class="panel-head">
          <h2>分點一槍績效排行</h2>
          <div class="count">買超一槍且 T+5 留半，依 20D Alpha 與樣本穩定度排序</div>
        </div>
        <div class="ranking-wrap">
          <table>
            <thead>
              <tr>
                <th>排行</th>
                <th>分點</th>
                <th>樣本</th>
                <th>20D 勝率</th>
                <th>20D Alpha 中位</th>
                <th>60D Alpha 中位</th>
                <th>120D Alpha 中位</th>
                <th>平均一槍金額</th>
              </tr>
            </thead>
            <tbody id="branch-ranking-body"></tbody>
          </table>
        </div>
      </section>

      <section class="panel recent-panel">
        <div class="panel-head">
          <h2>近期買超一槍</h2>
          <div class="count">最新事件直接顯示；N/A 代表還沒有足夠未來交易日，不納入排行統計</div>
        </div>
        <div class="ranking-wrap">
          <table>
            <thead>
              <tr>
                <th>日期</th>
                <th>分點 / 股票</th>
                <th>一槍金額</th>
                <th>T+5 保留</th>
                <th>20D Alpha</th>
                <th>60D Alpha</th>
                <th>120D Alpha</th>
              </tr>
            </thead>
            <tbody id="recent-body"></tbody>
          </table>
        </div>
      </section>

      <section class="toolbar">
        <input id="search" type="search" placeholder="搜尋分點、股票、產業" />
        <select id="branch-filter"></select>
        <select id="retention-filter">
          <option value="-999">不篩保留率</option>
          <option value="0.5">T+5 保留 >= 50%</option>
          <option value="0">T+5 未完全出掉</option>
          <option value="0.8">T+5 保留 >= 80%</option>
        </select>
        <select id="sort-filter">
          <option value="date">日期</option>
          <option value="alpha_20d">20D Alpha</option>
          <option value="alpha_60d">60D Alpha</option>
          <option value="alpha_120d">120D Alpha</option>
          <option value="abs_net_amount">一槍金額</option>
        </select>
        <select id="industry-filter"></select>
      </section>

      <main class="layout">
        <section class="panel">
          <div class="panel-head">
            <h2>事件清單</h2>
            <div class="count" id="count"></div>
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>日期</th>
                  <th>分點 / 股票</th>
                  <th>一槍金額</th>
                  <th>T+5 保留</th>
                  <th>10D</th>
                  <th>10D Alpha</th>
                  <th>20D</th>
                  <th>20D Alpha</th>
                  <th>60D Alpha</th>
                  <th>120D Alpha</th>
                </tr>
              </thead>
              <tbody id="event-body"></tbody>
            </table>
          </div>
        </section>

        <aside class="panel detail">
          <div class="detail-body" id="detail"></div>
        </aside>
      </main>
    </div>

    <script src="./one-shot-data.js"></script>
    <script>
      const payload = window.ONE_SHOT_DASHBOARD;
      const events = payload.events;
      const branchRankings = payload.branch_rankings || [];
      const windows = payload.event_windows;
      const followups = payload.followups;
      let activeEventId = events[0]?.event_id || null;
      let filters = {
        query: "",
        branch: "all",
        retention: -999,
        sort: "date",
        industry: "all",
      };

      const fmt = (value, digits = 2) => value == null ? "N/A" : Number(value).toLocaleString("zh-Hant-TW", { maximumFractionDigits: digits, minimumFractionDigits: digits });
      const pct = (value) => value == null ? "N/A" : `${fmt(value * 100, 1)}%`;
      const cls = (value) => value == null ? "" : Number(value) >= 0 ? "pos" : "neg";
      const signedPct = (value) => value == null ? "N/A" : `${Number(value) >= 0 ? "+" : ""}${fmt(value * 100, 1)}%`;
      const amount = (value) => value == null ? "N/A" : `${fmt(value / 100000000, 2)} 億`;

      function renderMeta() {
        const s = payload.summary;
        document.getElementById("meta").textContent = `門檻 ${fmt(s.threshold / 100000000, 0)} 億 ｜ T+${s.retention_days} 保留率 ｜ Benchmark ${s.benchmark}`;
      }

      function renderSummary(list = events) {
        const kept = list.filter((event) => event.retained_ratio_5d >= 0.5);
        const valid20 = kept.filter((event) => event.alpha_20d != null);
        const valid60 = kept.filter((event) => event.alpha_60d != null);
        const median = (items, key) => {
          const values = items.map((event) => event[key]).filter((value) => value != null).sort((a, b) => a - b);
          if (!values.length) return null;
          const mid = Math.floor(values.length / 2);
          return values.length % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
        };
        const cards = [
          ["顯示買超", list.length],
          ["T+5 留半", kept.length],
          ["20D 勝率", valid20.length ? (valid20.filter((event) => event.alpha_20d > 0).length / valid20.length) : null, "pct"],
          ["20D Alpha 中位", median(kept, "alpha_20d"), "pct"],
          ["60D Alpha 中位", median(valid60, "alpha_60d"), "pct"],
        ];
        document.getElementById("summary").innerHTML = cards.map(([label, value, type]) => `
          <div class="metric">
            <div class="label">${label}</div>
            <div class="value ${type === "pct" ? cls(value) : ""}">${type === "pct" ? pct(value) : fmt(value, 0)}</div>
          </div>
        `).join("");
      }

      function renderBranchRankings() {
        const tbody = document.getElementById("branch-ranking-body");
        if (!branchRankings.length) {
          tbody.innerHTML = `<tr><td colspan="8" class="empty">目前沒有符合預設策略的分點樣本。</td></tr>`;
          return;
        }
        tbody.innerHTML = branchRankings.map((row, index) => `
          <tr class="ranking-row" data-branch-id="${row.branch_id}">
            <td><span class="rank-num">${index + 1}</span></td>
            <td>${row.branch_name}<br>${row.branch_id}</td>
            <td>${fmt(row.sample_count, 0)} / 有效 ${fmt(row.valid_20d_count, 0)}</td>
            <td class="${cls(row.win_rate_alpha_20d)}">${pct(row.win_rate_alpha_20d)}</td>
            <td class="${cls(row.median_alpha_20d)}">${signedPct(row.median_alpha_20d)}</td>
            <td class="${cls(row.median_alpha_60d)}">${signedPct(row.median_alpha_60d)}</td>
            <td class="${cls(row.median_alpha_120d)}">${signedPct(row.median_alpha_120d)}</td>
            <td>${amount(row.avg_amount)}</td>
          </tr>
        `).join("");

        Array.from(tbody.querySelectorAll(".ranking-row")).forEach((row) => {
          row.addEventListener("click", () => {
            filters.branch = row.dataset.branchId;
            filters.retention = 0.5;
            document.getElementById("branch-filter").value = filters.branch;
            document.getElementById("retention-filter").value = String(filters.retention);
            render();
          });
        });
      }

      function renderRecentEvents() {
        const tbody = document.getElementById("recent-body");
        const recent = [...events].sort((a, b) => b.date.localeCompare(a.date)).slice(0, 20);
        if (!recent.length) {
          tbody.innerHTML = `<tr><td colspan="7" class="empty">目前沒有近期事件。</td></tr>`;
          return;
        }
        tbody.innerHTML = recent.map((event) => `
          <tr class="recent-row" data-event-id="${event.event_id}">
            <td>${event.date}</td>
            <td>${event.branch_name}<br>${event.stock_id} ${event.stock_name || ""}</td>
            <td>${amount(event.abs_net_amount)}</td>
            <td class="${cls(event.retained_ratio_5d)}">${pct(event.retained_ratio_5d)}</td>
            <td class="${cls(event.alpha_20d)}">${signedPct(event.alpha_20d)}</td>
            <td class="${cls(event.alpha_60d)}">${signedPct(event.alpha_60d)}</td>
            <td class="${cls(event.alpha_120d)}">${signedPct(event.alpha_120d)}</td>
          </tr>
        `).join("");

        Array.from(tbody.querySelectorAll(".recent-row")).forEach((row) => {
          row.addEventListener("click", () => {
            activeEventId = row.dataset.eventId;
            renderDetail();
            renderEventTable(filteredEvents());
          });
        });
      }

      function setupFilters() {
        const branchSelect = document.getElementById("branch-filter");
        const branches = Array.from(new Map(events.map((event) => [event.branch_id, `${event.branch_name} (${event.branch_id})`])).entries())
          .sort((a, b) => a[1].localeCompare(b[1], "zh-Hant"));
        branchSelect.innerHTML = `<option value="all">全部分點</option>` + branches.map(([id, name]) => `<option value="${id}">${name}</option>`).join("");

        const industrySelect = document.getElementById("industry-filter");
        const industries = Array.from(new Set(events.map((event) => event.industry_category || "未分類"))).sort((a, b) => a.localeCompare(b, "zh-Hant"));
        industrySelect.innerHTML = `<option value="all">全部產業</option>` + industries.map((name) => `<option value="${name}">${name}</option>`).join("");

        document.getElementById("search").addEventListener("input", (event) => {
          filters.query = event.target.value.trim().toLowerCase();
          render();
        });
        branchSelect.addEventListener("change", (event) => {
          filters.branch = event.target.value;
          render();
        });
        document.getElementById("retention-filter").addEventListener("change", (event) => {
          filters.retention = Number(event.target.value);
          render();
        });
        document.getElementById("sort-filter").addEventListener("change", (event) => {
          filters.sort = event.target.value;
          render();
        });
        industrySelect.addEventListener("change", (event) => {
          filters.industry = event.target.value;
          render();
        });
      }

      function filteredEvents() {
        const query = filters.query;
        return events.filter((event) => {
          if (filters.branch !== "all" && event.branch_id !== filters.branch) return false;
          if (filters.industry !== "all" && (event.industry_category || "未分類") !== filters.industry) return false;
          if (filters.retention > -900 && event.retained_ratio_5d < filters.retention) return false;
          if (query) {
            const haystack = `${event.branch_id} ${event.branch_name} ${event.stock_id} ${event.stock_name || ""} ${event.industry_category || ""}`.toLowerCase();
            if (!haystack.includes(query)) return false;
          }
          return true;
        }).sort((a, b) => {
          if (filters.sort === "date") return b.date.localeCompare(a.date);
          const av = a[filters.sort] == null ? -Infinity : a[filters.sort];
          const bv = b[filters.sort] == null ? -Infinity : b[filters.sort];
          return bv - av;
        });
      }

      function renderEventTable(list) {
        if (!list.find((event) => event.event_id === activeEventId)) {
          activeEventId = list[0]?.event_id || null;
        }
        document.getElementById("count").textContent = `顯示 ${list.length} 筆`;
        const tbody = document.getElementById("event-body");
        if (!list.length) {
          tbody.innerHTML = `<tr><td colspan="10" class="empty">沒有符合條件的事件。</td></tr>`;
          return;
        }
        tbody.innerHTML = list.map((event) => `
          <tr class="event-row ${event.event_id === activeEventId ? "active" : ""}" data-event-id="${event.event_id}">
            <td>${event.date}</td>
            <td>${event.branch_name}<br>${event.stock_id} ${event.stock_name || ""}</td>
            <td>${amount(event.abs_net_amount)}</td>
            <td class="${cls(event.retained_ratio_5d)}">${pct(event.retained_ratio_5d)}</td>
            <td class="${cls(event.direction_return_10d)}">${signedPct(event.direction_return_10d)}</td>
            <td class="${cls(event.alpha_10d)}">${signedPct(event.alpha_10d)}</td>
            <td class="${cls(event.direction_return_20d)}">${signedPct(event.direction_return_20d)}</td>
            <td class="${cls(event.alpha_20d)}">${signedPct(event.alpha_20d)}</td>
            <td class="${cls(event.alpha_60d)}">${signedPct(event.alpha_60d)}</td>
            <td class="${cls(event.alpha_120d)}">${signedPct(event.alpha_120d)}</td>
          </tr>
        `).join("");
        Array.from(tbody.querySelectorAll(".event-row")).forEach((row) => {
          row.addEventListener("click", () => {
            activeEventId = row.dataset.eventId;
            render();
          });
        });
      }

      function drawChart(event) {
        const canvas = document.getElementById("chart");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const rows = windows[event.event_id] || [];
        const w = canvas.width;
        const h = canvas.height;
        const pad = { left: 48, right: 18, top: 18, bottom: 30 };
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, w, h);
        if (!rows.length) return;
        const values = rows.flatMap((row) => [row.stock_norm, row.market_norm]).filter((value) => value != null);
        const min = Math.min(...values, 92);
        const max = Math.max(...values, 108);
        const range = Math.max(max - min, 1);
        const x = (i) => pad.left + (w - pad.left - pad.right) * (i / Math.max(rows.length - 1, 1));
        const y = (value) => pad.top + (max - value) / range * (h - pad.top - pad.bottom);

        ctx.strokeStyle = "#e2e6ea";
        ctx.lineWidth = 1;
        for (let i = 0; i < 5; i += 1) {
          const gy = pad.top + (h - pad.top - pad.bottom) * (i / 4);
          ctx.beginPath();
          ctx.moveTo(pad.left, gy);
          ctx.lineTo(w - pad.right, gy);
          ctx.stroke();
        }

        const drawLine = (key, color) => {
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.beginPath();
          let started = false;
          rows.forEach((row, i) => {
            if (row[key] == null) return;
            const px = x(i);
            const py = y(row[key]);
            if (!started) {
              ctx.moveTo(px, py);
              started = true;
            } else {
              ctx.lineTo(px, py);
            }
          });
          ctx.stroke();
        };
        drawLine("market_norm", "#8a96a3");
        drawLine("stock_norm", "#146c5f");

        const fiveIndex = Math.min(5, rows.length - 1);
        ctx.strokeStyle = "#265f9f";
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(x(fiveIndex), pad.top);
        ctx.lineTo(x(fiveIndex), h - pad.bottom);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = "#667085";
        ctx.font = "12px sans-serif";
        ctx.fillText(`${max.toFixed(0)}`, 8, y(max) + 4);
        ctx.fillText(`${min.toFixed(0)}`, 8, y(min) + 4);
        ctx.fillText(rows[0].date.slice(5), pad.left, h - 8);
        ctx.fillText(rows[rows.length - 1].date.slice(5), w - 58, h - 8);
      }

      function renderDetail() {
        const event = events.find((item) => item.event_id === activeEventId);
        const root = document.getElementById("detail");
        if (!event) {
          root.innerHTML = `<div class="empty">請選擇一筆事件。</div>`;
          return;
        }
        root.innerHTML = `
          <div class="detail-title">
            <div>
              <h2>${event.stock_id} ${event.stock_name || ""}</h2>
              <div class="subline">${event.branch_name} ｜ ${event.date} ｜ ${event.industry_category || "未分類"}</div>
            </div>
            <span class="tag buy">買超</span>
          </div>
          <div class="detail-grid">
            <div class="box"><div class="label">一槍金額</div><div class="value">${amount(event.abs_net_amount)}</div></div>
            <div class="box"><div class="label">進場價</div><div class="value">${fmt(event.entry_price, 2)}</div></div>
            <div class="box"><div class="label">T+5 保留</div><div class="value ${cls(event.retained_ratio_5d)}">${pct(event.retained_ratio_5d)}</div></div>
            <div class="box"><div class="label">10D Alpha</div><div class="value ${cls(event.alpha_10d)}">${signedPct(event.alpha_10d)}</div></div>
            <div class="box"><div class="label">20D Alpha</div><div class="value ${cls(event.alpha_20d)}">${signedPct(event.alpha_20d)}</div></div>
            <div class="box"><div class="label">60D Alpha</div><div class="value ${cls(event.alpha_60d)}">${signedPct(event.alpha_60d)}</div></div>
            <div class="box"><div class="label">120D Alpha</div><div class="value ${cls(event.alpha_120d)}">${signedPct(event.alpha_120d)}</div></div>
            <div class="box"><div class="label">20D 大盤</div><div class="value ${cls(event.market_return_20d)}">${signedPct(event.market_return_20d)}</div></div>
          </div>
          <canvas id="chart" width="640" height="320"></canvas>
          <div class="mini-table">
            <table>
              <thead>
                <tr><th>T</th><th>日期</th><th>收盤</th><th>買張</th><th>賣張</th><th>淨張</th><th>淨額</th></tr>
              </thead>
              <tbody>
                ${(followups[event.event_id] || []).map((row) => `
                  <tr>
                    <td>T+${row.offset}</td>
                    <td>${row.date}</td>
                    <td>${fmt(row.close, 2)}</td>
                    <td>${fmt(row.buy_lots, 0)}</td>
                    <td>${fmt(row.sell_lots, 0)}</td>
                    <td class="${cls(row.net_lots)}">${fmt(row.net_lots, 0)}</td>
                    <td class="${cls(row.net_amount_billion)}">${fmt(row.net_amount_billion, 2)} 億</td>
                  </tr>
                `).join("")}
              </tbody>
            </table>
          </div>
        `;
        drawChart(event);
      }

      function render() {
        const list = filteredEvents();
        renderSummary(list);
        renderEventTable(list);
        renderDetail();
      }

      renderMeta();
      setupFilters();
      renderBranchRankings();
      renderRecentEvents();
      render();
    </script>
  </body>
</html>
"""


def main() -> int:
    args = parse_args()
    threshold = float(args.threshold)
    min_retained_ratio = float(args.min_retained_ratio)
    limit_up_gap = float(args.limit_up_gap)
    dashboard_dir = Path(args.output_dir)
    data_dir = Path(args.data_output_dir)

    close_df = pd.read_parquet(CLOSE_PATH)
    close_df.index = pd.to_datetime(close_df.index).normalize()
    close_df[BENCHMARK_COLUMN] = build_market_benchmark(close_df)

    stock_info = load_stock_info()
    stock_lookup = {
        str(row.stock_id): {
            "stock_name": scalar_or_none(row.stock_name),
            "industry_category": scalar_or_none(row.industry_category),
            "type": scalar_or_none(row.type),
        }
        for row in stock_info.itertuples(index=False)
    }
    allowed_stock_ids = {
        stock_id
        for stock_id, stock_meta in stock_lookup.items()
        if is_strategy_stock(stock_id, stock_meta)
    }

    all_events: list[dict[str, Any]] = []
    event_windows: dict[str, list[dict[str, Any]]] = {}
    followups: dict[str, list[dict[str, Any]]] = {}
    for branch_dir in sorted(BRANCHES_DIR.glob("*")):
        if not branch_dir.is_dir():
            continue
        if branch_dir.name.split("_", 1)[0] in EXCLUDED_BRANCH_IDS:
            continue
        events, windows, branch_followups = build_one_branch_events(
            branch_dir=branch_dir,
            close_df=close_df,
            stock_lookup=stock_lookup,
            allowed_stock_ids=allowed_stock_ids,
            threshold=threshold,
            include_sell=bool(args.include_sell),
            limit_up_gap=limit_up_gap,
        )
        all_events.extend(events)
        event_windows.update(windows)
        followups.update(branch_followups)

    events_df = pd.DataFrame(all_events)
    if events_df.empty:
        raise SystemExit("No one-shot events found.")
    events_df = events_df.sort_values(["abs_net_amount", "date"], ascending=[False, False]).reset_index(drop=True)
    summary = build_summary(events_df, threshold=threshold, min_retained_ratio=min_retained_ratio)
    branch_rankings = build_branch_rankings(events_df, min_retained_ratio=min_retained_ratio)

    float_columns = [
        "net_amount",
        "abs_net_amount",
        "net_amount_billion",
        "buy_amount_billion",
        "sell_amount_billion",
        "buy_lots",
        "sell_lots",
        "net_lots",
        "entry_price",
        "event_close",
        "retained_qty_5d",
        "retained_ratio_5d",
        "opposite_lots_5d",
        "same_side_lots_5d",
        *[f"stock_return_{horizon}d" for horizon in FORWARD_WINDOWS],
        *[f"market_return_{horizon}d" for horizon in FORWARD_WINDOWS],
        *[f"direction_return_{horizon}d" for horizon in FORWARD_WINDOWS],
        *[f"alpha_{horizon}d" for horizon in FORWARD_WINDOWS],
    ]

    data_dir.mkdir(parents=True, exist_ok=True)
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    events_df.to_parquet(data_dir / EVENTS_PARQUET.name, index=False)
    export_events_df = events_df.drop(
        columns=["previous_close", "estimated_limit_up_price", "limit_up_gap"],
        errors="ignore",
    )
    events_json = to_records(export_events_df, float_columns=float_columns)
    (data_dir / EVENTS_JSON.name).write_text(json.dumps(events_json, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {
        "summary": summary,
        "branch_rankings": branch_rankings,
        "events": events_json,
        "event_windows": event_windows,
        "followups": followups,
    }
    (dashboard_dir / OUTPUT_JS.name).write_text(
        "window.ONE_SHOT_DASHBOARD = " + json.dumps(payload, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    (dashboard_dir / OUTPUT_HTML.name).write_text(render_html(), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "events": int(len(events_df)),
                "buy_events": int((events_df["side"] == "buy").sum()),
                "sell_events": int((events_df["side"] == "sell").sum()),
                "default_buy_events": summary["default_buy_events"],
                "ranked_branches": len(branch_rankings),
                "html": str(dashboard_dir / OUTPUT_HTML.name),
                "events_parquet": str(data_dir / EVENTS_PARQUET.name),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
