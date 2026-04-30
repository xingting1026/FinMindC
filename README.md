# FinMind One-Shot Branch Dashboard

This repo builds a static dashboard for the broker-branch "one-shot" strategy.

## Strategy Rules

- Keep normal 4-digit listed stocks only, excluding ETF/ETN-like symbols.
- Keep net-buy events whose estimated buy amount is at least NT$200,000,000.
- Exclude branches listed in `EXCLUDED_BRANCH_IDS` in `scripts/build_one_shot_dashboard.py`.
- Exclude events where the estimated entry cost is within 1% of the estimated limit-up price.
- Rank branches only with matured events. Recent events with unavailable returns still appear in the event list.

## Local Update

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

export FINMIND_API_TOKEN="YOUR_TOKEN"
python scripts/update_market_ohlcv_matrices.py
python scripts/update_branch_agg_direct.py --exclude-branch-id 9268
python scripts/build_one_shot_dashboard.py
```

Open `dashboard/one_shot/index.html` after the build finishes.

## GitHub Pages

`.github/workflows/update-one-shot-dashboard.yml` refreshes data every day at 21:07 Asia/Taipei, rebuilds `dashboard/one_shot`, commits refreshed data, and deploys the static dashboard with GitHub Pages.

Required repository secret:

- `FINMIND_API_TOKEN`
