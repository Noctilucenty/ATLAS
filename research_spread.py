"""RESEARCH - does the edge survive the bid/ask spread?

The single biggest offline unknown: every label so far is spread-free
(mid-ish candles). Dukascopy publishes separate BID and ASK m1 candles;
this runs the standard pipeline on their MID and then rescores the same
trades under three settlement rules:

  mid       entry mid open(t+1), exercise mid close(t+15)  - baseline,
            should reproduce the familiar numbers
  half      mid entry/exit but the move must clear half the entry spread
  adverse   calls enter at ASK open(t+1) and settle against BID
            close(t+15); puts mirrored - the worst-case both-sides cost

The mid-vs-adverse win-rate delta is the spread haircut. IQ's real strike
mechanics sit between mid and adverse (one broker-quoted rate on both
legs), so the truth lies inside this bracket.

Era split for the meta: trained on trades <= 2024-06-30, scored after -
data starts 2023 so the usual 2023 split is unusable.
"""

import glob
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from features import FEATURE_COLUMNS, build_features
from research_wr import META_CONTEXT, independent
from train import ChronoCalibratedModel, decide_action

PAYOUT = 0.87
BREAKEVEN = 1.0 / (1.0 + PAYOUT)
PURGE_S = 15 * 60
HORIZON = 15
ERA_SPLIT = int(datetime(2024, 7, 1, tzinfo=timezone.utc).timestamp())
DUKA = Path("research_logs/duka")


def load_pair(pair: str):
    def read(pt):
        f = next(DUKA.glob(f"{pair}-{pt}*.csv"))
        df = pd.read_csv(f)
        df["from_ts"] = (df["timestamp"] // 1000).astype(int)
        return df.set_index("from_ts")[["open", "high", "low", "close"]]
    bid, ask = read("bid"), read("ask")
    both = bid.join(ask, lsuffix="_bid", rsuffix="_ask", how="inner")
    mid = pd.DataFrame({
        "from_ts": both.index,
        "to_ts": both.index + 60,
        "open": (both["open_bid"] + both["open_ask"]) / 2,
        "high": (both["high_bid"] + both["high_ask"]) / 2,
        "low": (both["low_bid"] + both["low_ask"]) / 2,
        "close": (both["close_bid"] + both["close_ask"]) / 2,
        "volume": 0.0,
    }).reset_index(drop=True)
    return mid, both


def walk_forward_preds(mid: pd.DataFrame, pair: str) -> pd.DataFrame:
    ff = build_features(mid, interval=60, horizon=HORIZON, entry_next_open=True)
    ff = ff.dropna(subset=["label_up"]).reset_index(drop=True)
    ts = ff["to_ts"].to_numpy()
    edges = np.linspace(ts[0], ts[-1], 8)
    preds = []
    for k in range(1, 7):
        te = np.where((ts > edges[k]) & (ts <= edges[k + 1]))[0]
        tr = np.where(ts <= edges[k] - PURGE_S)[0]
        if not len(te) or len(tr) < 10000:
            continue
        m = ChronoCalibratedModel(n_folds=3, gap=HORIZON, model_kind="lgbm")
        m.fit(ff[FEATURE_COLUMNS].iloc[tr], ff["label_up"].iloc[tr])
        sub = ff.iloc[te].copy()
        sub["p_up"] = m.predict_proba_up(ff[FEATURE_COLUMNS].iloc[te])
        preds.append(sub)
        print(f"  {pair} fold {k}", flush=True)
    out = pd.concat(preds, ignore_index=True)
    out["asset"] = pair.upper()
    return out


def main() -> None:
    from research_wr import fit_meta_on_selection

    trades_all = []
    for pair in ("eurusd", "gbpusd", "usdjpy"):
        mid, both = load_pair(pair)
        p = walk_forward_preds(mid, pair)
        p["action"] = [decide_action(float(x), PAYOUT, 0.0) for x in p["p_up"]]
        p = p[p["action"] != "no_trade"].reset_index(drop=True)

        # Settlement legs from the bid/ask frames.
        entry_ts = p["to_ts"].astype(int)            # bar starting at signal close
        exit_ts = entry_ts + (HORIZON - 1) * 60      # bar whose close is t+15min
        for col, src, key in (
            ("entry_bid", "open_bid", entry_ts), ("entry_ask", "open_ask", entry_ts),
            ("exit_bid", "close_bid", exit_ts), ("exit_ask", "close_ask", exit_ts),
        ):
            p[col] = both[src].reindex(key).to_numpy()
        p = p.dropna(subset=["entry_bid", "entry_ask", "exit_bid", "exit_ask"])
        p["entry_mid"] = (p["entry_bid"] + p["entry_ask"]) / 2
        p["exit_mid"] = (p["exit_bid"] + p["exit_ask"]) / 2
        p["spread"] = p["entry_ask"] - p["entry_bid"]
        call = p["action"] == "binary_call"
        p["won_mid"] = np.where(call, p["exit_mid"] > p["entry_mid"],
                                p["exit_mid"] < p["entry_mid"]).astype(float)
        half = p["spread"] / 2
        p["won_half"] = np.where(call, p["exit_mid"] > p["entry_mid"] + half,
                                 p["exit_mid"] < p["entry_mid"] - half).astype(float)
        p["won_adverse"] = np.where(call, p["exit_bid"] > p["entry_ask"],
                                    p["exit_ask"] < p["entry_bid"]).astype(float)
        trades_all.append(p)

    base = pd.concat(trades_all, ignore_index=True)
    base["is_call"] = (base["action"] == "binary_call").astype(float)
    base["conf"] = (base["p_up"] - 0.5).abs()
    ce = base["p_up"] * PAYOUT - (1 - base["p_up"])
    pe = (1 - base["p_up"]) * PAYOUT - base["p_up"]
    base["ev"] = np.maximum(ce, pe)
    base["ts"] = base["to_ts"].astype(int)
    base["won"] = base["won_mid"]  # meta trains on the baseline outcome
    sel = (base["to_ts"] < ERA_SPLIT).to_numpy()
    base["meta_p"] = fit_meta_on_selection(base, sel)
    hold = base[~sel]

    report = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "window": "duka 2023-2025, meta trained <=2024-06, scored after",
              "median_spread_pips_at_trades": round(
                  float((hold["spread"] / np.where(hold["asset"].str.contains("JPY"),
                                                   0.01, 0.0001)).median()), 2),
              "results": {}}
    for mt in (0.0, 0.60, 0.65, 0.70):
        g = independent(hold[(hold["ev"] > 0.03) & (hold["meta_p"] >= mt)], PURGE_S)
        n = len(g)
        row = {"independent_trades": n}
        for variant in ("mid", "half", "adverse"):
            wr = float(g[f"won_{variant}"].mean()) if n else None
            row[f"wr_{variant}"] = round(wr, 4) if wr is not None else None
        if n >= 100:
            wins = int(g["won_adverse"].sum())
            row["p_adverse_beats_breakeven"] = float(f"{stats.binomtest(wins, n, BREAKEVEN, alternative='greater').pvalue:.3e}")
        report["results"][f"meta_{mt}"] = row

    Path("research_logs/spread_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
