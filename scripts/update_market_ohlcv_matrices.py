#!/usr/bin/env python3

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd


API_TOKEN = os.environ.get("FINMIND_API_TOKEN", "").strip()
BASE_URL = "https://api.finmindtrade.com/api/v4/data"
FILES = {
    "Open": "open",
    "High": "max",
    "Low": "min",
    "Close": "close",
    "Volume": "Trading_Volume",
}
DEFAULT_SLEEP_SECONDS = 0.15
DEFAULT_RETRIES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update Open/High/Low/Close/Volume parquet matrices from FinMind.")
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_RETRIES)
    return parser.parse_args()


def request_json(params: dict) -> dict:
    if not API_TOKEN:
        raise RuntimeError("FINMIND_API_TOKEN is not set.")
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{BASE_URL}?{query}",
        headers={"Authorization": f"Bearer {API_TOKEN}"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_stock_info() -> pd.DataFrame:
    payload = request_json({"dataset": "TaiwanStockInfo"})
    if payload.get("status") != 200:
        raise RuntimeError(f"Failed to fetch TaiwanStockInfo: {payload}")
    frame = pd.DataFrame(payload.get("data", []))
    frame = frame.sort_values(["stock_id", "date"]).drop_duplicates(subset=["stock_id"], keep="last")
    frame["stock_id"] = frame["stock_id"].astype("string")
    return frame


def fetch_trading_dates() -> list[str]:
    payload = request_json({"dataset": "TaiwanStockTradingDate"})
    if payload.get("status") != 200:
        raise RuntimeError(f"Failed to fetch TaiwanStockTradingDate: {payload}")
    return [row["date"] for row in payload.get("data", [])]


def fetch_price_rows(date: str, max_retries: int, sleep_seconds: float) -> list[dict]:
    for attempt in range(1, max_retries + 1):
        try:
            payload = request_json(
                {
                    "dataset": "TaiwanStockPrice",
                    "start_date": date,
                    "end_date": date,
                }
            )
            status = payload.get("status", 200)
            if status != 200:
                raise RuntimeError(json.dumps(payload, ensure_ascii=False))
            rows = payload.get("data", [])
            if not isinstance(rows, list):
                raise RuntimeError(f"Unexpected payload shape: {payload}")
            time.sleep(sleep_seconds)
            return rows
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code in {402, 429, 500, 502, 503, 504}
            if attempt == max_retries or not retryable:
                raise RuntimeError(f"FinMind HTTP {exc.code}: {detail}") from exc
            wait_seconds = min(90.0, 2 ** attempt)
            print(f"[retry] date={date} attempt={attempt}/{max_retries} http_status={exc.code} wait={wait_seconds:.1f}s")
            time.sleep(wait_seconds)
        except urllib.error.URLError as exc:
            if attempt == max_retries:
                raise RuntimeError(f"Network error for {date}: {exc}") from exc
            wait_seconds = min(90.0, 2 ** attempt)
            print(f"[retry] date={date} attempt={attempt}/{max_retries} network_error={exc} wait={wait_seconds:.1f}s")
            time.sleep(wait_seconds)

    raise RuntimeError(f"Unreachable retry state for date={date}")


def load_matrix(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "mdate"
    frame.columns = frame.columns.astype("string")
    frame = frame.sort_index()
    return frame


def main() -> int:
    args = parse_args()
    matrices = {name: load_matrix(Path(f"{name}.parquet")) for name in FILES}
    last_date = max(frame.index.max() for frame in matrices.values())
    stock_info = fetch_stock_info()
    trading_dates = fetch_trading_dates()

    today = datetime.now().date().isoformat()
    pending_dates = [date for date in trading_dates if str(last_date.date()) < date <= today]
    if not pending_dates:
        print(json.dumps({"status": "ok", "message": "already up to date", "last_date": str(last_date.date())}, ensure_ascii=False))
        return 0

    # Keep only normal stock ids from TaiwanStockInfo to avoid warrant / derivative rows in TaiwanStockPrice.
    allowed_ids = set(stock_info["stock_id"].astype(str))

    fetched_frames = []
    for date in pending_dates:
        rows = fetch_price_rows(date=date, max_retries=args.max_retries, sleep_seconds=args.sleep_seconds)
        frame = pd.DataFrame(rows)
        if frame.empty:
            print(f"[skip] date={date} no rows")
            continue
        frame["stock_id"] = frame["stock_id"].astype("string")
        frame = frame[frame["stock_id"].astype(str).isin(allowed_ids)].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        fetched_frames.append(frame)
        print(f"[progress] date={date} rows={len(frame)}")

    if not fetched_frames:
        print(json.dumps({"status": "ok", "message": "no new market rows", "last_date": str(last_date.date())}, ensure_ascii=False))
        return 0

    fetched = pd.concat(fetched_frames, ignore_index=True)
    all_stock_ids = sorted(set().union(*[set(map(str, frame.columns)) for frame in matrices.values()]).union(set(fetched["stock_id"].astype(str))))

    for name, field in FILES.items():
        matrix = matrices[name].copy()
        pivot = (
            fetched.pivot_table(index="date", columns="stock_id", values=field, aggfunc="last")
            .sort_index()
        )
        pivot.columns = pivot.columns.astype("string")
        matrix.columns = matrix.columns.astype("string")
        combined = pd.concat([matrix, pivot], axis=0)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        combined = combined.reindex(columns=pd.Index(all_stock_ids, dtype="string"))
        combined.index.name = "mdate"
        combined.to_parquet(Path(f"{name}.parquet"))
        print(
            json.dumps(
                {
                    "file": f"{name}.parquet",
                    "rows": int(len(combined)),
                    "cols": int(len(combined.columns)),
                    "min_date": str(combined.index.min().date()),
                    "max_date": str(combined.index.max().date()),
                },
                ensure_ascii=False,
            )
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "updated_dates": pending_dates,
                "last_previous_date": str(last_date.date()),
                "last_new_date": str(pd.to_datetime(fetched["date"]).max().date()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
