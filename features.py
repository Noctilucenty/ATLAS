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
from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands

FEATURE_VERSION = "1.3.0"  # 1.3.0: + adx, bb_pctb, macd_hist_atr, ret_60

EMA_FAST = 8
EMA_SLOW = 21
RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14
BB_WINDOW = 20
VOLUME_WINDOW = 20
REGIME_WINDOW = 480  # trailing bars for the volatility-percentile regime
SLOPE_LAG = 3

FEATURE_COLUMNS = [
    "ret_1",
    "ret_5",
    "ret_15",
    "ret_60",
    "adx",
    "bb_pctb",
    "macd_hist_atr",
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


def build_features(
    df: pd.DataFrame,
    interval: int = 60,
    horizon: int = 5,
    entry_next_open: bool = False,
) -> pd.DataFrame:
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
        _build_features_segment(segment, interval, horizon, entry_next_open)
        for segment in segments
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(
            columns=["from_ts", "to_ts", *FEATURE_COLUMNS, "label_up", "feature_version"]
        )
    return pd.concat(parts, ignore_index=True)


def _build_features_segment(
    df: pd.DataFrame, interval: int, horizon: int, entry_next_open: bool = False
) -> pd.DataFrame:
    # Segments shorter than indicator warmup can't yield any feature row and
    # crash ta's ATR/ADX outright (ADX needs ~2x its window); they
    # contribute nothing.
    if len(df) <= max(ATR_PERIOD, RSI_PERIOD, VOLUME_WINDOW, EMA_SLOW, 2 * ADX_PERIOD):
        return pd.DataFrame()
    df = df.reset_index(drop=True)
    out = pd.DataFrame({"from_ts": df["from_ts"], "to_ts": df["to_ts"]})

    close, high, low, open_ = df["close"], df["high"], df["low"], df["open"]

    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    out["ret_15"] = close.pct_change(15)
    out["ret_60"] = close.pct_change(60)

    ema_fast = EMAIndicator(close, EMA_FAST).ema_indicator()
    ema_slow = EMAIndicator(close, EMA_SLOW).ema_indicator()
    atr = AverageTrueRange(high, low, close, ATR_PERIOD).average_true_range()
    atr_safe = atr.replace(0.0, np.nan)
    out["ema_spread_atr"] = (ema_fast - ema_slow) / atr_safe
    out["ema_fast_slope"] = (ema_fast - ema_fast.shift(SLOPE_LAG)) / close
    out["rsi"] = RSIIndicator(close, RSI_PERIOD).rsi()
    out["atr_norm"] = atr / close

    # Trend strength (regime discriminator: trending vs ranging), scaled to ~[0,1].
    out["adx"] = ADXIndicator(high, low, close, ADX_PERIOD).adx() / 100.0
    # Position inside the Bollinger channel (mean-reversion pressure).
    out["bb_pctb"] = BollingerBands(close, BB_WINDOW).bollinger_pband()
    # Momentum acceleration, volatility-normalized like ema_spread_atr.
    out["macd_hist_atr"] = MACD(close).macd_diff() / atr_safe

    bar_range = (high - low).replace(0.0, np.nan)
    out["body_ratio"] = ((close - open_).abs() / bar_range).fillna(0.0)
    out["upper_wick_ratio"] = ((high - np.maximum(close, open_)) / bar_range).fillna(0.0)
    out["lower_wick_ratio"] = ((np.minimum(close, open_) - low) / bar_range).fillna(0.0)

    # Some feeds (IQ Option OTC markets) report volume=0 on every bar; a 0/0
    # vol_rel would NaN out the whole segment, so fall back to a neutral 1.0.
    # A zero rolling mean on an otherwise real volume feed is also mapped to
    # NaN (dropped) rather than division-by-zero inf.
    if df["volume"].gt(0).any():
        vol_mean = df["volume"].rolling(VOLUME_WINDOW).mean().replace(0.0, np.nan)
        out["vol_rel"] = df["volume"] / vol_mean
    else:
        out["vol_rel"] = 1.0

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
    # entry_next_open: realistic-execution label. The signal fires at t's
    # close but a real order fills at the NEXT bar's open, so the option's
    # strike is open[t+1], not close[t]. Exercise stays at close[t+horizon].
    entry = open_.shift(-1) if entry_next_open else close
    out["label_up"] = np.where(
        future_close.isna() | entry.isna() | (future_close == entry),
        np.nan,
        (future_close > entry).astype(float),
    )

    out["feature_version"] = FEATURE_VERSION
    return out.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
