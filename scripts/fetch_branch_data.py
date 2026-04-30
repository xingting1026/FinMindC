#!/usr/bin/env python3

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


API_TOKEN = os.environ.get("FINMIND_API_TOKEN", "").strip()

DETAIL_API_URL = "https://api.finmindtrade.com/api/v4/taiwan_stock_trading_daily_report"
GENERIC_API_URL = "https://api.finmindtrade.com/api/v4/data"

DEFAULT_START_DATE = "2021-07-01"
DEFAULT_END_DATE = datetime.now().date().isoformat()
DEFAULT_OUTPUT_ROOT = Path("data/branches")
DEFAULT_MAX_WORKERS = 6
DEFAULT_BATCH_SIZE = 20
DEFAULT_RETRIES = 5
DEFAULT_SLEEP_SECONDS = 0.08

RAW_COLUMNS = [
    "securities_trader",
    "price",
    "buy",
    "sell",
    "securities_trader_id",
    "stock_id",
    "date",
]
RAW_UNIQUE_KEYS = ["date", "stock_id", "securities_trader_id", "price"]
AGG_COLUMNS = [
    "date",
    "stock_id",
    "securities_trader_id",
    "securities_trader",
    "buy_qty",
    "sell_qty",
    "net_qty",
    "buy_amount_est",
    "sell_amount_est",
    "avg_buy_price_est",
    "avg_sell_price_est",
    "trade_price_count",
    "is_active",
]


def slugify(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch FinMind TaiwanStockTradingDailyReport by securities_trader_id.")
    parser.add_argument("--branch-id", required=True)
    parser.add_argument("--branch-name", help="Optional display name. If omitted, fetch from TaiwanSecuritiesTraderInfo.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", help="Optional absolute/relative output dir override.")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def request_json(url: str, params: dict) -> dict:
    if not API_TOKEN:
        raise RuntimeError("FINMIND_API_TOKEN is not set.")
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"Authorization": f"Bearer {API_TOKEN}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_branch_profile(branch_id: str) -> dict:
    payload = request_json(GENERIC_API_URL, {"dataset": "TaiwanSecuritiesTraderInfo"})
    if payload.get("status") != 200:
        raise RuntimeError(f"Failed to fetch TaiwanSecuritiesTraderInfo: {payload}")
    rows = payload.get("data", [])
    matches = [row for row in rows if str(row.get("securities_trader_id")) == branch_id]
    if not matches:
        raise RuntimeError(f"Could not find branch profile for securities_trader_id={branch_id}")
    return matches[0]


def fetch_trading_dates(start_date: str, end_date: str) -> list[str]:
    payload = request_json(GENERIC_API_URL, {"dataset": "TaiwanStockTradingDate"})
    if payload.get("status") != 200:
        raise RuntimeError(f"Failed to fetch TaiwanStockTradingDate: {payload}")
    dates = [row["date"] for row in payload.get("data", [])]
    return [date for date in dates if start_date <= date <= end_date]


def fetch_rows_for_date(branch_id: str, date: str, max_retries: int, sleep_seconds: float) -> tuple[str, list[dict]]:
    for attempt in range(1, max_retries + 1):
        try:
            time.sleep(sleep_seconds + random.uniform(0, sleep_seconds))
            payload = request_json(DETAIL_API_URL, {"securities_trader_id": branch_id, "date": date})
            status = payload.get("status", 200)
            if status != 200:
                raise RuntimeError(json.dumps(payload, ensure_ascii=False))
            rows = payload.get("data", [])
            if not isinstance(rows, list):
                raise RuntimeError(f"Unexpected payload shape: {payload}")
            return date, rows
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code in {402, 429, 500, 502, 503, 504}
            if attempt == max_retries or not retryable:
                raise RuntimeError(f"FinMind HTTP {exc.code}: {detail}") from exc
            time.sleep(min(90.0, 2 ** attempt))
        except urllib.error.URLError as exc:
            if attempt == max_retries:
                raise RuntimeError(f"Network error for {date}: {exc}") from exc
            time.sleep(min(90.0, 2 ** attempt))
    raise RuntimeError(f"Unreachable retry state for date={date}")


def normalize_raw_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        frame = pd.DataFrame(columns=RAW_COLUMNS)
    for column in RAW_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object")
    frame = frame[RAW_COLUMNS].copy()
    frame["securities_trader"] = frame["securities_trader"].astype("string")
    frame["securities_trader_id"] = frame["securities_trader_id"].astype("string")
    frame["stock_id"] = frame["stock_id"].astype("string")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["buy"] = pd.to_numeric(frame["buy"], errors="coerce").fillna(0).astype("int64")
    frame["sell"] = pd.to_numeric(frame["sell"], errors="coerce").fillna(0).astype("int64")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["date", "price"]).reset_index(drop=True)
    return frame


def merge_raw_rows(output_path: Path, rows: list[dict]) -> int:
    incoming = normalize_raw_frame(pd.DataFrame(rows))
    if output_path.exists():
        existing = normalize_raw_frame(pd.read_parquet(output_path))
        frame = pd.concat([existing, incoming], ignore_index=True)
    else:
        frame = incoming
    frame = frame.drop_duplicates(subset=RAW_UNIQUE_KEYS, keep="last")
    frame = frame.sort_values(by=["date", "stock_id", "price"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return len(incoming)


def build_daily_agg(raw_path: Path, output_path: Path) -> dict:
    raw = normalize_raw_frame(pd.read_parquet(raw_path))
    raw["buy_amount_component"] = raw["buy"] * raw["price"]
    raw["sell_amount_component"] = raw["sell"] * raw["price"]
    grouped = (
        raw.groupby(["date", "stock_id", "securities_trader_id", "securities_trader"], dropna=False)
        .agg(
            buy_qty=("buy", "sum"),
            sell_qty=("sell", "sum"),
            buy_amount_est=("buy_amount_component", "sum"),
            sell_amount_est=("sell_amount_component", "sum"),
            trade_price_count=("price", "count"),
        )
        .reset_index()
    )
    grouped["net_qty"] = grouped["buy_qty"] - grouped["sell_qty"]
    grouped["avg_buy_price_est"] = grouped["buy_amount_est"].div(grouped["buy_qty"].where(grouped["buy_qty"] != 0))
    grouped["avg_sell_price_est"] = grouped["sell_amount_est"].div(grouped["sell_qty"].where(grouped["sell_qty"] != 0))
    grouped["is_active"] = (grouped["buy_qty"] > 0) | (grouped["sell_qty"] > 0)
    grouped = grouped[AGG_COLUMNS].sort_values(by=["date", "stock_id"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_parquet(output_path, index=False)
    return {
        "row_count": int(len(grouped)),
        "min_date": None if grouped.empty else str(grouped["date"].min().date()),
        "max_date": None if grouped.empty else str(grouped["date"].max().date()),
        "distinct_stock_count": int(grouped["stock_id"].nunique()),
    }


def checkpoint_path(output_dir: Path, branch_id: str) -> Path:
    return output_dir / "meta" / "_checkpoints" / f"{branch_id}.json"


def raw_path(output_dir: Path) -> Path:
    return output_dir / "raw" / "trading_daily_detail.parquet"


def agg_path(output_dir: Path) -> Path:
    return output_dir / "derived" / "stock_daily_agg.parquet"


def branch_profile_path(output_dir: Path) -> Path:
    return output_dir / "meta" / "branch_profile.json"


def make_checkpoint(branch_id: str, branch_name: str, start_date: str, end_date: str) -> dict:
    return {
        "securities_trader_id": branch_id,
        "securities_trader": branch_name,
        "start_date": start_date,
        "end_date": end_date,
        "next_date": start_date,
        "last_completed_date": None,
        "last_attempted_date": None,
        "completed_trading_days": 0,
        "rows_written": 0,
        "status": "pending",
        "updated_at": None,
    }


def load_checkpoint(output_dir: Path, branch_id: str, branch_name: str, start_date: str, end_date: str) -> dict:
    path = checkpoint_path(output_dir, branch_id)
    if path.exists():
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        checkpoint["start_date"] = min(str(checkpoint.get("start_date") or start_date), start_date)
        if str(checkpoint.get("end_date") or "") < end_date:
            checkpoint["end_date"] = end_date
            if checkpoint.get("status") == "complete":
                checkpoint["status"] = "pending"
                checkpoint["next_date"] = checkpoint.get("last_completed_date") or start_date
        else:
            checkpoint["end_date"] = end_date
        return checkpoint
    return make_checkpoint(branch_id, branch_name, start_date, end_date)


def save_checkpoint(output_dir: Path, branch_id: str, checkpoint: dict) -> None:
    path = checkpoint_path(output_dir, branch_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")


def save_manifest(output_dir: Path, manifest: dict) -> None:
    path = output_dir / "meta" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def save_branch_profile(output_dir: Path, profile: dict) -> None:
    path = branch_profile_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")


def compute_manifest(output_dir: Path, branch_id: str, branch_name: str, checkpoint: dict, agg_meta: dict) -> dict:
    raw_file = raw_path(output_dir)
    raw_rows = 0
    if raw_file.exists():
        raw_rows = int(len(pd.read_parquet(raw_file)))
    return {
        "securities_trader_id": branch_id,
        "securities_trader": branch_name,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "date_range": {
            "start": checkpoint["start_date"],
            "end": checkpoint["end_date"],
            "min_observed_date": agg_meta.get("min_date"),
            "max_observed_date": agg_meta.get("max_date"),
        },
        "raw": {"path": str(raw_file), "row_count": raw_rows},
        "derived": {"path": str(agg_path(output_dir)), **agg_meta},
        "checkpoint": checkpoint,
    }


def resolve_output_dir(args: argparse.Namespace, branch_name: str) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    slug = slugify(branch_name)
    return Path(args.output_root) / f"{args.branch_id}_{slug}"


def main() -> int:
    args = parse_args()
    branch_profile = fetch_branch_profile(args.branch_id)
    branch_name = args.branch_name or str(branch_profile["securities_trader"])
    output_dir = resolve_output_dir(args, branch_name)

    if args.reset and output_dir.exists():
        for path in [raw_path(output_dir), agg_path(output_dir), checkpoint_path(output_dir, args.branch_id)]:
            if path.exists():
                path.unlink()

    save_branch_profile(output_dir, branch_profile)
    trading_dates = fetch_trading_dates(args.start_date, args.end_date)
    checkpoint = load_checkpoint(output_dir, args.branch_id, branch_name, args.start_date, args.end_date)
    remaining_dates = [date for date in trading_dates if checkpoint["last_completed_date"] is None or date > checkpoint["last_completed_date"]]

    if not remaining_dates:
        agg_meta = build_daily_agg(raw_path(output_dir), agg_path(output_dir))
        save_manifest(output_dir, compute_manifest(output_dir, args.branch_id, branch_name, checkpoint, agg_meta))
        print(json.dumps({"status": "already_complete", "output_dir": str(output_dir)}, ensure_ascii=False))
        return 0

    checkpoint["status"] = "running"
    checkpoint["updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    save_checkpoint(output_dir, args.branch_id, checkpoint)

    for offset in range(0, len(remaining_dates), args.batch_size):
        batch = remaining_dates[offset : offset + args.batch_size]
        results = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_map = {
                executor.submit(fetch_rows_for_date, args.branch_id, date, args.max_retries, args.sleep_seconds): date
                for date in batch
            }
            for future in as_completed(future_map):
                date, rows = future.result()
                results.append((date, rows))
                checkpoint["last_attempted_date"] = date

        sorted_results = sorted(results, key=lambda item: item[0])
        batch_rows = [row for _, rows in sorted_results for row in rows]
        merge_raw_rows(raw_path(output_dir), batch_rows)

        for date, rows in sorted_results:
            checkpoint["last_completed_date"] = date
            checkpoint["next_date"] = date
            checkpoint["completed_trading_days"] += 1
            checkpoint["rows_written"] += len(rows)
            checkpoint["updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            save_checkpoint(output_dir, args.branch_id, checkpoint)

    checkpoint["status"] = "complete"
    checkpoint["next_date"] = None
    checkpoint["updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    save_checkpoint(output_dir, args.branch_id, checkpoint)
    agg_meta = build_daily_agg(raw_path(output_dir), agg_path(output_dir))
    save_manifest(output_dir, compute_manifest(output_dir, args.branch_id, branch_name, checkpoint, agg_meta))
    print(json.dumps({"status": "ok", "output_dir": str(output_dir), "agg_meta": agg_meta}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
