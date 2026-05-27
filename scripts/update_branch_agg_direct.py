#!/usr/bin/env python3

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from fetch_branch_data import (
    AGG_COLUMNS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_END_DATE,
    DEFAULT_MAX_WORKERS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RETRIES,
    DEFAULT_SLEEP_SECONDS,
    DEFAULT_START_DATE,
    compute_manifest,
    fetch_rows_for_date,
    fetch_trading_dates,
    load_checkpoint,
    normalize_raw_frame,
    save_checkpoint,
    save_manifest,
    slugify,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXCLUDED_BRANCH_IDS = {"9268"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh branch daily aggregates without storing raw detail rows.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--branch-id", action="append", help="Optional branch id to refresh. Repeatable.")
    parser.add_argument("--branch-list-csv", help="CSV with broker_id and broker_name columns for missing target folders.")
    parser.add_argument("--exclude-branch-id", action="append", default=[], help="Branch id to skip. Repeatable.")
    return parser.parse_args()


def output_root_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def agg_path(branch_dir: Path) -> Path:
    return branch_dir / "derived" / "stock_daily_agg.parquet"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(branch_dir: Path) -> dict[str, Any]:
    manifest_path = branch_dir / "meta" / "manifest.json"
    profile_path = branch_dir / "meta" / "branch_profile.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if profile_path.exists():
            profile = load_json(profile_path)
            if profile.get("securities_trader"):
                manifest["securities_trader"] = profile["securities_trader"]
        return manifest

    branch_id, branch_name = branch_dir.name.split("_", 1)
    return {"securities_trader_id": branch_id, "securities_trader": branch_name}


def load_branch_targets(path: Optional[str]) -> dict[str, str]:
    if not path:
        return {}
    csv_path = output_root_path(path)
    if not csv_path.exists():
        return {}
    frame = pd.read_csv(csv_path, dtype={"broker_id": str})
    if "broker_id" not in frame.columns:
        return {}
    name_col = "broker_name" if "broker_name" in frame.columns else None
    targets: dict[str, str] = {}
    for row in frame.to_dict(orient="records"):
        branch_id = str(row.get("broker_id") or "").strip()
        if not branch_id:
            continue
        branch_name = str(row.get(name_col) or branch_id) if name_col else branch_id
        targets[branch_id] = branch_name
    return targets


def local_branches(
    root: Path,
    selected_ids: Optional[set[str]],
    excluded_ids: set[str],
    target_names: Optional[dict[str, str]] = None,
) -> list[tuple[Path, str, str]]:
    target_names = target_names or {}
    branches: list[tuple[Path, str, str]] = []
    seen_ids: set[str] = set()
    for branch_dir in sorted(root.glob("*")):
        if not branch_dir.is_dir():
            continue
        manifest = load_manifest(branch_dir)
        branch_id = str(manifest.get("securities_trader_id", branch_dir.name.split("_", 1)[0]))
        branch_name = str(manifest.get("securities_trader", branch_dir.name.split("_", 1)[-1]))
        if selected_ids and branch_id not in selected_ids:
            continue
        if branch_id in excluded_ids:
            continue
        seen_ids.add(branch_id)
        branches.append((branch_dir, branch_id, branch_name))
    for branch_id, branch_name in sorted(target_names.items()):
        if branch_id in seen_ids or branch_id in excluded_ids:
            continue
        if selected_ids and branch_id not in selected_ids:
            continue
        branch_dir = root / f"{branch_id}_{slugify(branch_name)}"
        branches.append((branch_dir, branch_id, branch_name))
    return branches


def normalize_agg_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        frame = pd.DataFrame(columns=AGG_COLUMNS)
    for column in AGG_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object")
    frame = frame[AGG_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["stock_id"] = frame["stock_id"].astype("string")
    frame["securities_trader_id"] = frame["securities_trader_id"].astype("string")
    frame["securities_trader"] = frame["securities_trader"].astype("string")
    numeric_columns = [
        "buy_qty",
        "sell_qty",
        "net_qty",
        "buy_amount_est",
        "sell_amount_est",
        "avg_buy_price_est",
        "avg_sell_price_est",
        "trade_price_count",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["buy_qty"] = frame["buy_qty"].fillna(0).astype("int64")
    frame["sell_qty"] = frame["sell_qty"].fillna(0).astype("int64")
    frame["net_qty"] = frame["net_qty"].fillna(0).astype("int64")
    frame["trade_price_count"] = frame["trade_price_count"].fillna(0).astype("int64")
    frame["is_active"] = frame["is_active"].fillna(False).astype("bool")
    return frame.dropna(subset=["date", "stock_id", "securities_trader_id"]).reset_index(drop=True)


def aggregate_rows(rows: list[dict]) -> pd.DataFrame:
    raw = normalize_raw_frame(pd.DataFrame(rows))
    if raw.empty:
        return pd.DataFrame(columns=AGG_COLUMNS)

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
    return normalize_agg_frame(grouped)


def merge_agg_rows(path: Path, incoming_rows: list[dict]) -> dict[str, Any]:
    incoming = aggregate_rows(incoming_rows)
    if path.exists():
        existing = normalize_agg_frame(pd.read_parquet(path))
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        combined = incoming

    combined = combined.drop_duplicates(subset=["date", "stock_id", "securities_trader_id"], keep="last")
    combined = combined.sort_values(["date", "stock_id", "securities_trader_id"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)

    return {
        "row_count": int(len(combined)),
        "min_date": None if combined.empty else str(combined["date"].min().date()),
        "max_date": None if combined.empty else str(combined["date"].max().date()),
        "distinct_stock_count": int(combined["stock_id"].nunique()),
    }


def refresh_branch(
    branch_dir: Path,
    branch_id: str,
    branch_name: str,
    trading_dates: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(branch_dir, branch_id, branch_name, args.start_date, args.end_date)
    remaining_dates = [
        date
        for date in trading_dates
        if checkpoint["last_completed_date"] is None or date > checkpoint["last_completed_date"]
    ]

    if not remaining_dates:
        existing = normalize_agg_frame(pd.read_parquet(agg_path(branch_dir)))
        agg_meta = {
            "row_count": int(len(existing)),
            "min_date": None if existing.empty else str(existing["date"].min().date()),
            "max_date": None if existing.empty else str(existing["date"].max().date()),
            "distinct_stock_count": int(existing["stock_id"].nunique()),
        }
        save_manifest(branch_dir, compute_manifest(branch_dir, branch_id, branch_name, checkpoint, agg_meta))
        return {"status": "already_complete", "branch_id": branch_id, "branch_name": branch_name}

    checkpoint["status"] = "running"
    checkpoint["updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    save_checkpoint(branch_dir, branch_id, checkpoint)

    total_rows = 0
    for offset in range(0, len(remaining_dates), args.batch_size):
        batch = remaining_dates[offset : offset + args.batch_size]
        results: list[tuple[str, list[dict]]] = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_map = {
                executor.submit(fetch_rows_for_date, branch_id, date, args.max_retries, args.sleep_seconds): date
                for date in batch
            }
            for future in as_completed(future_map):
                date, rows = future.result()
                results.append((date, rows))
                checkpoint["last_attempted_date"] = date

        sorted_results = sorted(results, key=lambda item: item[0])
        batch_rows = [row for _, rows in sorted_results for row in rows]
        total_rows += len(batch_rows)
        agg_meta = merge_agg_rows(agg_path(branch_dir), batch_rows)

        for date, rows in sorted_results:
            checkpoint["last_completed_date"] = date
            checkpoint["next_date"] = date
            checkpoint["completed_trading_days"] += 1
            checkpoint["rows_written"] += len(rows)
            checkpoint["updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            save_checkpoint(branch_dir, branch_id, checkpoint)

    checkpoint["status"] = "complete"
    checkpoint["next_date"] = None
    checkpoint["updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    save_checkpoint(branch_dir, branch_id, checkpoint)
    save_manifest(branch_dir, compute_manifest(branch_dir, branch_id, branch_name, checkpoint, agg_meta))
    return {
        "status": "ok",
        "branch_id": branch_id,
        "branch_name": branch_name,
        "dates": len(remaining_dates),
        "rows": total_rows,
        "agg_meta": agg_meta,
    }


def main() -> int:
    args = parse_args()
    root = output_root_path(args.output_root)
    target_names = load_branch_targets(args.branch_list_csv)
    selected_ids = set(args.branch_id or target_names.keys()) or None
    excluded_ids = DEFAULT_EXCLUDED_BRANCH_IDS | set(args.exclude_branch_id or [])
    branches = local_branches(root, selected_ids, excluded_ids, target_names)
    if not branches:
        raise SystemExit("No local branch folders matched.")

    trading_dates = [date for date in fetch_trading_dates(args.start_date, args.end_date) if args.start_date <= date <= args.end_date]
    failures = []
    for index, (branch_dir, branch_id, branch_name) in enumerate(branches, start=1):
        print(
            json.dumps(
                {
                    "status": "starting_branch",
                    "index": index,
                    "total": len(branches),
                    "branch_id": branch_id,
                    "branch_name": branch_name,
                    "end_date": args.end_date,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        try:
            print(
                json.dumps(
                    refresh_branch(branch_dir, branch_id, branch_name, trading_dates, args),
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            print(f"[error] branch_id={branch_id} branch_name={branch_name} error={exc}", file=sys.stderr, flush=True)
            failures.append({"branch_id": branch_id, "branch_name": branch_name, "error": str(exc)})

    print(
        json.dumps(
            {
                "status": "ok" if not failures else "partial_failure",
                "refreshed": len(branches) - len(failures),
                "failed": failures,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
