"""Versioned feature/label pipeline for binary-option prediction.

Every feature at row t uses only information available at t's close - no
future rows, and higher-timeframe features only use higher-TF bars that have
fully completed by t's close. Labels look forward by construction (that is
what a label is); leakage control between train and test is the walk-forward
orchestrator's job.

The feature set is deliberately small and curated (reviewer decision: do not
auto-generate indicator zoo). Any change to the set or its parameters
requires a FEATURE_VERSION bump so old model results stay reproducible.
"""

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

FEATURE_VERSION = "1.1.0"  # 1.1.0: gap-aware - windows and labels never span missing bars

EMA_FAST = 8
EMA_SLOW = 21
RSI_PERIOD = 14
ATR_PERIOD = 14
VOLUME_WINDOW = 20
REGIME_WINDOW = 480  # trailing bars for the volatility-percentile regime
SLOPE_LAG = 3

FEATURE_COLUMNS = [
    "ret_1",
    "ret_5",
    "ret_15",
    "ema_spread_atr",
    "ema_fast_slope",
    "rsi",
    "atr_norm",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "vol_rel",
    "mtf_align",
    "hour_sin",
    "hour_cos",
    "session_asia",
    "session_europe",
    "session_us",
    "vol_regime",
]


def _mtf_trend(df: pd.DataFrame, interval: int, factor: int) -> pd.Series:
    """Higher-timeframe EMA trend (+1/-1/0) mapped leak-safely onto base bars.

    A higher-TF bar's trend value becomes available only at rows whose close
    time is >= that bar's close time; incomplete bars are never used."""
    group = df["from_ts"] // (interval * factor)
    agg = (
        df.assign(_group=group)
        .groupby("_group")
        .agg(close=("close", "last"), to_ts=("to_ts", "max"), bars=("close", "size"))
        .reset_index(drop=True)
    )
    complete = agg[agg["bars"] == factor].reset_index(drop=True)
    if len(complete) < EMA_SLOW:
        return pd.Series(np.nan, index=df.index)
    fast = EMAIndicator(complete["close"], EMA_FAST).ema_indicator()
    slow = EMAIndicator(complete["close"], EMA_SLOW).ema_indicator()
    trend = pd.DataFrame(
        {"to_ts": complete["to_ts"], "trend": np.sign(fast - slow)}
    ).dropna()
    merged = pd.merge_asof(
        df[["to_ts"]].reset_index(),
        trend,
        on="to_ts",
        direction="backward",
    ).set_index("index")
    return merged["trend"]


def split_contiguous(df: pd.DataFrame, interval: int) -> list[pd.DataFrame]:
    """Split a candle frame into maximal contiguous segments at gap boundaries."""
    df = df.reset_index(drop=True)
    if df.empty:
        return []
    breaks = df.index[df["from_ts"].diff() != interval].tolist()
    breaks.append(len(df))
    return [
        df.iloc[start:end].reset_index(drop=True)
        for start, end in zip(breaks, breaks[1:])
        if end > start
    ]


def build_features(df: pd.DataFrame, interval: int = 60, horizon: int = 5) -> pd.DataFrame:
    """Compute the curated feature set plus the forward label, gap-aware.

    The frame is split into contiguous segments at every missing-bar boundary
    and each segment is processed independently, so no rolling window, EMA
    state, multi-timeframe trend, or forward label ever crosses a gap.
    Segments too short to survive indicator warmup contribute nothing.

    Input: a validated candle frame (validation.validate_candles), ascending.
    Output columns: from_ts, to_ts, every FEATURE_COLUMNS entry, label_up
    (1.0 price rose over `horizon` bars, 0.0 fell, NaN tie/end-of-data), and
    feature_version. Warmup rows with undefined features are dropped."""
    segments = split_contiguous(df, interval)
    parts = [
        _build_features_segment(segment, interval, horizon)
        for segment in segments
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(
            columns=["from_ts", "to_ts", *FEATURE_COLUMNS, "label_up", "feature_version"]
        )
    return pd.concat(parts, ignore_index=True)


def _build_features_segment(df: pd.DataFrame, interval: int, horizon: int) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    out = pd.DataFrame({"from_ts": df["from_ts"], "to_ts": df["to_ts"]})

    close, high, low, open_ = df["close"], df["high"], df["low"], df["open"]

    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    out["ret_15"] = close.pct_change(15)

    ema_fast = EMAIndicator(close, EMA_FAST).ema_indicator()
    ema_slow = EMAIndicator(close, EMA_SLOW).ema_indicator()
    atr = AverageTrueRange(high, low, close, ATR_PERIOD).average_true_range()
    atr_safe = atr.replace(0.0, np.nan)
    out["ema_spread_atr"] = (ema_fast - ema_slow) / atr_safe
    out["ema_fast_slope"] = (ema_fast - ema_fast.shift(SLOPE_LAG)) / close
    out["rsi"] = RSIIndicator(close, RSI_PERIOD).rsi()
    out["atr_norm"] = atr / close

    bar_range = (high - low).replace(0.0, np.nan)
    out["body_ratio"] = ((close - open_).abs() / bar_range).fillna(0.0)
    out["upper_wick_ratio"] = ((high - np.maximum(close, open_)) / bar_range).fillna(0.0)
    out["lower_wick_ratio"] = ((np.minimum(close, open_) - low) / bar_range).fillna(0.0)

    out["vol_rel"] = df["volume"] / df["volume"].rolling(VOLUME_WINDOW).mean()

    trend_5 = _mtf_trend(df, interval, 5)
    trend_15 = _mtf_trend(df, interval, 15)
    both = pd.concat([trend_5, trend_15], axis=1)
    out["mtf_align"] = np.where(
        both.isna().any(axis=1),
        np.nan,
        np.where(
            (trend_5 == trend_15) & (trend_5 != 0), trend_5, 0.0
        ),
    )

    seconds_of_day = (out["to_ts"] % 86400).astype(float)
    angle = 2 * np.pi * seconds_of_day / 86400.0
    out["hour_sin"] = np.sin(angle)
    out["hour_cos"] = np.cos(angle)
    hour = (seconds_of_day // 3600).astype(int)
    out["session_asia"] = ((hour >= 0) & (hour < 7)).astype(float)
    out["session_europe"] = ((hour >= 7) & (hour < 13)).astype(float)
    out["session_us"] = ((hour >= 13) & (hour < 21)).astype(float)

    out["vol_regime"] = (
        out["atr_norm"].rolling(REGIME_WINDOW, min_periods=REGIME_WINDOW // 4).rank(pct=True)
    )

    future_close = close.shift(-horizon)
    out["label_up"] = np.where(
        future_close.isna() | (future_close == close),
        np.nan,
        (future_close > close).astype(float),
    )

    out["feature_version"] = FEATURE_VERSION
    return out.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
