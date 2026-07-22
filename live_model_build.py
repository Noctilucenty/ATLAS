"""Train the hypothesis-#2 pooled model on ALL stored history and freeze it.

Writes models/h2-<utcdate>.pkl containing the fitted ChronoCalibratedModel,
the exact feature column list, and provenance (feature version, row count,
data end timestamp). live_h2_runner.py loads the newest pickle; it never
trains. Retrain deliberately (e.g. weekly) by re-running this script.
"""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

from features import FEATURE_COLUMNS, FEATURE_VERSION
from research_pooled import XS_COLUMNS, add_cross_asset, load_pooled
from storage import open_db
from train import ChronoCalibratedModel

HORIZON = 15  # bars; must match FORWARD_TEST.md hypothesis #2


def main() -> None:
    pooled = load_pooled(open_db(), interval=60, horizon=HORIZON, entry_next_open=True)
    pooled = add_cross_asset(pooled)
    feature_cols = list(FEATURE_COLUMNS) + XS_COLUMNS

    model = ChronoCalibratedModel(
        n_folds=3, gap=HORIZON * pooled["asset"].nunique(), model_kind="lgbm"
    )
    model.fit(pooled[feature_cols], pooled["label_up"])

    meta = {
        "hypothesis": "h2",
        "horizon_bars": HORIZON,
        "feature_version": FEATURE_VERSION,
        "feature_columns": feature_cols,
        "rows": len(pooled),
        "assets": sorted(pooled["asset"].unique()),
        "data_end_ts": int(pooled["to_ts"].max()),
        "built_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_dir = Path(__file__).resolve().parent / "models"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = out_dir / f"h2-{stamp}.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"model": model, "meta": meta}, fh)
    print(json.dumps({"path": str(path), **{k: v for k, v in meta.items() if k != "feature_columns"}}, indent=2))


if __name__ == "__main__":
    main()
