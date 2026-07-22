"""RESEARCH SCREENING ONLY - deep-history sanity anchor on histdata.com bars.

Does the feature family + LightGBM show the same (lack of / presence of)
edge over YEARS of spot 1-minute data, not just the broker's ~60-day window?
Downloads histdata.com 1-minute bars (free, EST timestamps converted to
UTC), runs the identical walk-forward machinery, and reports the same
independent-trade / margin-sweep metrics as research_pooled.

Spot-market anchor only: histdata prices are not IQ Option's feed and know
nothing about OTC. A signal here supports plausibility; it proves nothing
about the broker's book. No run bundles; never feeds execution.
"""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss

from features import EXTRA_VOL_COLUMNS, FEATURE_COLUMNS, build_features
from research_pooled import evaluate_margin
from train import ChronoCalibratedModel

EST = timezone(timedelta(hours=-5))  # histdata timestamps: fixed UTC-5, no DST


def download_years(pair: str, years: range, dest: Path) -> list[Path]:
    from histdata import download_hist_data
    from histdata.api import Platform, TimeFrame

    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for year in years:
        existing = list(dest.glob(f"*{pair.upper()}*{year}*.zip"))
        if existing:
            paths.extend(existing)
            continue
        try:
            out = download_hist_data(
                year=str(year), pair=pair, platform=Platform.GENERIC_ASCII,
                time_frame=TimeFrame.ONE_MINUTE, output_directory=str(dest),
            )
            paths.append(Path(out))
            print(f"downloaded {year}", flush=True)
        except Exception as exc:  # a missing year must not kill the sweep
            print(f"SKIP {year}: {exc}", flush=True)
    return paths


def load_candles(zips: list[Path], interval: int = 60) -> pd.DataFrame:
    import zipfile

    frames = []
    for zp in sorted(zips):
        with zipfile.ZipFile(zp) as zf:
            # Each histdata zip holds the data .csv plus a status .txt.
            member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            with zf.open(member) as fh:
                raw = pd.read_csv(
                    fh, sep=";", header=None,
                    names=["dt", "open", "high", "low", "close", "volume"],
                )
        ts = pd.to_datetime(raw["dt"], format="%Y%m%d %H%M%S")
        from_ts = ts.map(lambda t: int(t.replace(tzinfo=EST).timestamp()))
        frames.append(pd.DataFrame({
            "from_ts": from_ts,
            "to_ts": from_ts + interval,
            "open": raw["open"], "high": raw["high"],
            "low": raw["low"], "close": raw["close"],
            "volume": raw["volume"].astype(float),
        }))
    df = pd.concat(frames, ignore_index=True)
    return (
        df.drop_duplicates(subset=["from_ts"], keep="last")
        .sort_values("from_ts")
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default="eurusd")
    parser.add_argument("--from-year", type=int, default=2016)
    parser.add_argument("--to-year", type=int, default=2025)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--splits", type=int, default=8)
    parser.add_argument("--payout", type=float, default=0.87)
    parser.add_argument("--ev-margins", default="0.02,0.03,0.04")
    parser.add_argument("--entry-next-open", action="store_true")
    parser.add_argument("--extra-vol", action="store_true",
                        help="add range-based volatility + Corwin-Schultz spread features")
    parser.add_argument("--cache-dir", default="histdata_cache")
    parser.add_argument("--dump-preds", default=None,
                        help="write all (pair, ts, p_up, label) test predictions to this JSON path")
    args = parser.parse_args()

    zips = download_years(
        args.pair, range(args.from_year, args.to_year + 1), Path(args.cache_dir)
    )
    candles = load_candles(zips)
    print(f"candles={len(candles)} "
          f"{datetime.fromtimestamp(candles['from_ts'].min(), timezone.utc).date()}"
          f"..{datetime.fromtimestamp(candles['from_ts'].max(), timezone.utc).date()}",
          flush=True)

    ff = build_features(
        candles, interval=60, horizon=args.horizon,
        entry_next_open=args.entry_next_open, extra_vol=args.extra_vol,
    ).dropna(subset=["label_up"]).reset_index(drop=True)
    print(f"labeled rows={len(ff)}", flush=True)

    purge_s = args.horizon * 60
    feature_cols = list(FEATURE_COLUMNS) + (EXTRA_VOL_COLUMNS if args.extra_vol else [])
    X, y = ff[feature_cols], ff["label_up"]
    ts = ff["to_ts"].to_numpy()
    edges = np.linspace(ts[0], ts[-1], args.splits + 2)
    briers, preds = [], []
    for k in range(1, args.splits + 1):
        te = np.where((ts > edges[k]) & (ts <= edges[k + 1]))[0]
        tr = np.where(ts <= edges[k] - purge_s)[0]
        if not len(te) or len(tr) < 10000:
            continue
        model = ChronoCalibratedModel(n_folds=3, gap=args.horizon, model_kind="lgbm")
        model.fit(X.iloc[tr], y.iloc[tr])
        p_up = model.predict_proba_up(X.iloc[te])
        brier = float(brier_score_loss(y.iloc[te].to_numpy(), p_up))
        briers.append(brier)
        rows = ff.iloc[te]
        preds.extend(zip(
            [args.pair.upper()] * len(te), rows["to_ts"].astype(int),
            map(float, p_up), rows["label_up"],
        ))
        print(f"fold {k - 1}: brier={brier:.5f} test_rows={len(te)}", flush=True)

    print("mean brier:", round(float(np.mean(briers)), 5))
    if args.dump_preds:
        with open(args.dump_preds, "w") as fh:
            json.dump(preds, fh)
    results = [
        evaluate_margin(preds, float(m), args.payout, purge_s)
        for m in args.ev_margins.split(",")
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
