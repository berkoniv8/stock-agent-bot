"""
Technical Analysis Module — EMAs, SMA, pattern detection, Fibonacci levels.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TechnicalSignals:
    """Container for all technical signals detected on a ticker."""
    ticker: str = ""
    ema_cross_bullish: bool = False
    ema_cross_bearish: bool = False
    price_above_200sma: bool = False
    breakout_with_volume: bool = False
    double_bottom: bool = False
    head_and_shoulders: bool = False
    inverse_head_shoulders: bool = False
    fib_bounce_382: bool = False
    fib_bounce_618: bool = False
    fib_levels: dict = field(default_factory=dict)
    current_price: float = 0.0
    ema9: float = 0.0
    ema21: float = 0.0
    sma200: float = 0.0
    support_level: float = 0.0
    resistance_level: float = 0.0
    pattern_details: list = field(default_factory=list)
    # Additional indicators
    rsi: float = 0.0
    rsi_oversold: bool = False       # RSI < 30
    rsi_overbought: bool = False     # RSI > 70
    macd_bullish_cross: bool = False
    macd_bearish_cross: bool = False
    macd_value: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    bb_squeeze: bool = False          # Bandwidth < 20th percentile
    bb_breakout_upper: bool = False   # Close above upper band
    bb_breakout_lower: bool = False   # Close below lower band
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    bb_middle: float = 0.0
    # ATR & trailing stop
    atr: float = 0.0
    atr_stop_long: float = 0.0   # entry - 2*ATR
    atr_stop_short: float = 0.0  # entry + 2*ATR
    # Divergence
    rsi_bullish_divergence: bool = False   # price lower low, RSI higher low
    rsi_bearish_divergence: bool = False   # price higher high, RSI lower high
    macd_bullish_divergence: bool = False
    macd_bearish_divergence: bool = False
    # VWAP
    vwap: float = 0.0
    price_above_vwap: bool = False
    # Money flow indicators
    mfi: float = 0.0                    # Money Flow Index (0-100)
    mfi_oversold: bool = False          # MFI < 20
    mfi_overbought: bool = False        # MFI > 80
    ad_line: float = 0.0               # Accumulation/Distribution line value
    ad_trend_bullish: bool = False     # A/D line rising (bullish accumulation)
    ad_trend_bearish: bool = False     # A/D line falling (distribution)
    obv: float = 0.0                   # On-Balance Volume
    obv_trend_bullish: bool = False    # OBV rising (volume confirms price)
    obv_trend_bearish: bool = False    # OBV falling (volume diverges)
    obv_divergence_bullish: bool = False  # Price down but OBV up
    obv_divergence_bearish: bool = False  # Price up but OBV down
    # Ichimoku Cloud
    ichimoku_above_cloud: bool = False
    ichimoku_below_cloud: bool = False
    ichimoku_bullish_cross: bool = False
    ichimoku_bearish_cross: bool = False
    tenkan_sen: float = 0.0
    kijun_sen: float = 0.0
    senkou_a: float = 0.0
    senkou_b: float = 0.0
    # Stochastic RSI
    stoch_rsi: float = 0.0
    stoch_rsi_k: float = 0.0
    stoch_rsi_d: float = 0.0
    stoch_rsi_oversold: bool = False
    stoch_rsi_overbought: bool = False
    stoch_rsi_bullish_cross: bool = False
    # ADX — trend strength
    adx: float = 0.0
    adx_strong_trend: bool = False
    adx_plus_di: float = 0.0
    adx_minus_di: float = 0.0
    adx_bullish: bool = False
    adx_bearish: bool = False
    # Keltner Channels / TTM Squeeze
    keltner_upper: float = 0.0
    keltner_lower: float = 0.0
    ttm_squeeze: bool = False
    ttm_squeeze_fired: bool = False
    # Relative Strength vs SPY
    rs_vs_spy: float = 0.0
    rs_trending_up: bool = False
    # Pivot Points
    pivot: float = 0.0
    pivot_r1: float = 0.0
    pivot_r2: float = 0.0
    pivot_s1: float = 0.0
    pivot_s2: float = 0.0
    near_pivot_support: bool = False
    near_pivot_resistance: bool = False
    # Gap detection
    gap_up: bool = False
    gap_down: bool = False
    gap_up_pct: float = 0.0
    gap_down_pct: float = 0.0
    # Additional patterns
    cup_and_handle: bool = False
    ascending_triangle: bool = False
    descending_triangle: bool = False
    bull_flag: bool = False
    # EMA Ribbon
    ema_ribbon_bullish: bool = False
    ema_ribbon_bearish: bool = False
    # 52-week context
    pct_from_52w_high: float = 0.0
    pct_from_52w_low: float = 0.0
    near_52w_high: bool = False
    near_52w_low: bool = False


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA-9, EMA-21, SMA-200, RSI, MACD, Bollinger Bands to the DataFrame."""
    df = df.copy()

    # Trend EMAs/SMA
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["SMA200"] = df["Close"].rolling(window=200).mean()
    df["AvgVol20"] = df["Volume"].rolling(window=20).mean()

    # EMA Ribbon
    df["EMA8"] = df["Close"].ewm(span=8, adjust=False).mean()
    df["EMA13"] = df["Close"].ewm(span=13, adjust=False).mean()
    df["EMA34"] = df["Close"].ewm(span=34, adjust=False).mean()
    df["EMA55"] = df["Close"].ewm(span=55, adjust=False).mean()

    # RSI (14-period)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger Bands (20-period, 2 std dev)
    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + 2 * bb_std
    df["BB_Lower"] = df["BB_Middle"] - 2 * bb_std
    df["BB_Bandwidth"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"] * 100

    # ATR (14-period Average True Range)
    high_low = df["High"] - df["Low"]
    high_prev_close = (df["High"] - df["Close"].shift(1)).abs()
    low_prev_close = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df["ATR"] = true_range.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    # VWAP (cumulative for the visible window — resets daily in intraday,
    # but for daily bars we compute a rolling 20-bar VWAP)
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_tp_vol = (typical_price * df["Volume"]).rolling(window=20).sum()
    cum_vol = df["Volume"].rolling(window=20).sum()
    df["VWAP"] = cum_tp_vol / cum_vol.replace(0, np.nan)

    # Money Flow Index (14-period)
    raw_mf = typical_price * df["Volume"]
    mf_direction = typical_price.diff()
    pos_mf = raw_mf.where(mf_direction > 0, 0.0)
    neg_mf = raw_mf.where(mf_direction < 0, 0.0)
    pos_mf_sum = pos_mf.rolling(window=14).sum()
    neg_mf_sum = neg_mf.rolling(window=14).sum()
    mf_ratio = pos_mf_sum / neg_mf_sum.replace(0, np.nan)
    df["MFI"] = 100 - (100 / (1 + mf_ratio))

    # Accumulation/Distribution Line
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / \
          (df["High"] - df["Low"]).replace(0, np.nan)
    clv = clv.fillna(0)
    df["AD"] = (clv * df["Volume"]).cumsum()

    # On-Balance Volume
    obv_sign = pd.Series(0, index=df.index)
    obv_sign[df["Close"].diff() > 0] = 1
    obv_sign[df["Close"].diff() < 0] = -1
    df["OBV"] = (obv_sign * df["Volume"]).cumsum()

    return df


# ---------------------------------------------------------------------------
# RSI detection
# ---------------------------------------------------------------------------

def detect_rsi(df: pd.DataFrame) -> tuple:
    """Detect RSI oversold/overbought conditions.

    Returns (rsi_value, oversold, overbought).
    """
    if "RSI" not in df.columns or pd.isna(df["RSI"].iloc[-1]):
        return 0.0, False, False

    rsi = float(df["RSI"].iloc[-1])
    return rsi, rsi < 30, rsi > 70


# ---------------------------------------------------------------------------
# MACD crossover detection
# ---------------------------------------------------------------------------

def detect_macd_cross(df: pd.DataFrame) -> tuple:
    """Detect MACD / Signal line crossovers on the last 2 bars.

    Returns (macd, signal, histogram, bullish_cross, bearish_cross).
    """
    if len(df) < 2 or "MACD" not in df.columns:
        return 0.0, 0.0, 0.0, False, False

    if pd.isna(df["MACD"].iloc[-1]) or pd.isna(df["MACD_Signal"].iloc[-1]):
        return 0.0, 0.0, 0.0, False, False

    prev_macd = df["MACD"].iloc[-2]
    prev_sig = df["MACD_Signal"].iloc[-2]
    curr_macd = df["MACD"].iloc[-1]
    curr_sig = df["MACD_Signal"].iloc[-1]
    hist = df["MACD_Hist"].iloc[-1]

    bullish = prev_macd <= prev_sig and curr_macd > curr_sig
    bearish = prev_macd >= prev_sig and curr_macd < curr_sig

    return float(curr_macd), float(curr_sig), float(hist), bullish, bearish


# ---------------------------------------------------------------------------
# Bollinger Band detection
# ---------------------------------------------------------------------------

def detect_bollinger(df: pd.DataFrame) -> tuple:
    """Detect Bollinger Band squeeze and breakouts.

    Returns (upper, middle, lower, squeeze, breakout_upper, breakout_lower).
    """
    if "BB_Upper" not in df.columns or pd.isna(df["BB_Upper"].iloc[-1]):
        return 0.0, 0.0, 0.0, False, False, False

    upper = float(df["BB_Upper"].iloc[-1])
    middle = float(df["BB_Middle"].iloc[-1])
    lower = float(df["BB_Lower"].iloc[-1])
    close = float(df["Close"].iloc[-1])

    # Squeeze: bandwidth is in lowest 20th percentile of last 120 bars
    bw = df["BB_Bandwidth"].dropna()
    squeeze = False
    if len(bw) >= 20:
        pct20 = bw.tail(120).quantile(0.20)
        squeeze = float(bw.iloc[-1]) < pct20

    breakout_upper = close > upper
    breakout_lower = close < lower

    return upper, middle, lower, squeeze, breakout_upper, breakout_lower


# ---------------------------------------------------------------------------
# ATR-based stop levels
# ---------------------------------------------------------------------------

def compute_atr_stops(df: pd.DataFrame, multiplier: float = 2.0) -> tuple:
    """Compute ATR-based stop levels.

    Returns (atr_value, stop_long, stop_short).
    - stop_long:  entry - multiplier * ATR  (for buy positions)
    - stop_short: entry + multiplier * ATR  (for short positions)
    """
    if "ATR" not in df.columns or pd.isna(df["ATR"].iloc[-1]):
        return 0.0, 0.0, 0.0

    atr = float(df["ATR"].iloc[-1])
    close = float(df["Close"].iloc[-1])

    stop_long = close - multiplier * atr
    stop_short = close + multiplier * atr

    return atr, stop_long, stop_short


# ---------------------------------------------------------------------------
# RSI / MACD divergence detection
# ---------------------------------------------------------------------------

def _find_price_swing_lows(df: pd.DataFrame, lookback: int = 30, window: int = 5) -> list:
    """Find swing lows in price (Low column) over the last `lookback` bars."""
    lows = df["Low"].tail(lookback).values
    indices = list(range(len(df) - lookback, len(df)))
    swings = []
    for i in range(window, len(lows) - window):
        if lows[i] == min(lows[max(0, i - window):i + window + 1]):
            swings.append((indices[i], lows[i]))
    return swings


def _find_price_swing_highs(df: pd.DataFrame, lookback: int = 30, window: int = 5) -> list:
    """Find swing highs in price (High column) over the last `lookback` bars."""
    highs = df["High"].tail(lookback).values
    indices = list(range(len(df) - lookback, len(df)))
    swings = []
    for i in range(window, len(highs) - window):
        if highs[i] == max(highs[max(0, i - window):i + window + 1]):
            swings.append((indices[i], highs[i]))
    return swings


def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 30) -> tuple:
    """Detect bullish and bearish RSI divergence.

    Bullish divergence: price makes a lower low but RSI makes a higher low.
    Bearish divergence: price makes a higher high but RSI makes a lower high.

    Returns (bullish_div, bearish_div).
    """
    if "RSI" not in df.columns or len(df) < lookback:
        return False, False

    rsi_vals = df["RSI"].values

    # Bullish: lower low in price, higher low in RSI
    bullish = False
    swing_lows = _find_price_swing_lows(df, lookback)
    if len(swing_lows) >= 2:
        prev_idx, prev_price = swing_lows[-2]
        curr_idx, curr_price = swing_lows[-1]
        if curr_price < prev_price:  # price lower low
            prev_rsi = rsi_vals[prev_idx] if not pd.isna(rsi_vals[prev_idx]) else 0
            curr_rsi = rsi_vals[curr_idx] if not pd.isna(rsi_vals[curr_idx]) else 0
            if curr_rsi > prev_rsi:  # RSI higher low
                bullish = True

    # Bearish: higher high in price, lower high in RSI
    bearish = False
    swing_highs = _find_price_swing_highs(df, lookback)
    if len(swing_highs) >= 2:
        prev_idx, prev_price = swing_highs[-2]
        curr_idx, curr_price = swing_highs[-1]
        if curr_price > prev_price:  # price higher high
            prev_rsi = rsi_vals[prev_idx] if not pd.isna(rsi_vals[prev_idx]) else 0
            curr_rsi = rsi_vals[curr_idx] if not pd.isna(rsi_vals[curr_idx]) else 0
            if curr_rsi < prev_rsi:  # RSI lower high
                bearish = True

    return bullish, bearish


def detect_macd_divergence(df: pd.DataFrame, lookback: int = 30) -> tuple:
    """Detect bullish and bearish MACD histogram divergence.

    Same logic as RSI divergence but using MACD histogram values.

    Returns (bullish_div, bearish_div).
    """
    if "MACD_Hist" not in df.columns or len(df) < lookback:
        return False, False

    hist_vals = df["MACD_Hist"].values

    # Bullish: lower low in price, higher low in MACD histogram
    bullish = False
    swing_lows = _find_price_swing_lows(df, lookback)
    if len(swing_lows) >= 2:
        prev_idx, prev_price = swing_lows[-2]
        curr_idx, curr_price = swing_lows[-1]
        if curr_price < prev_price:
            prev_hist = hist_vals[prev_idx] if not pd.isna(hist_vals[prev_idx]) else 0
            curr_hist = hist_vals[curr_idx] if not pd.isna(hist_vals[curr_idx]) else 0
            if curr_hist > prev_hist:
                bullish = True

    # Bearish: higher high in price, lower high in MACD histogram
    bearish = False
    swing_highs = _find_price_swing_highs(df, lookback)
    if len(swing_highs) >= 2:
        prev_idx, prev_price = swing_highs[-2]
        curr_idx, curr_price = swing_highs[-1]
        if curr_price > prev_price:
            prev_hist = hist_vals[prev_idx] if not pd.isna(hist_vals[prev_idx]) else 0
            curr_hist = hist_vals[curr_idx] if not pd.isna(hist_vals[curr_idx]) else 0
            if curr_hist < prev_hist:
                bearish = True

    return bullish, bearish


# ---------------------------------------------------------------------------
# VWAP detection
# ---------------------------------------------------------------------------

def detect_vwap(df: pd.DataFrame) -> tuple:
    """Get VWAP value and price relationship.

    Returns (vwap_value, price_above_vwap).
    """
    if "VWAP" not in df.columns or pd.isna(df["VWAP"].iloc[-1]):
        return 0.0, False

    vwap = float(df["VWAP"].iloc[-1])
    close = float(df["Close"].iloc[-1])
    return vwap, close > vwap


# ---------------------------------------------------------------------------
# Money Flow Index detection
# ---------------------------------------------------------------------------

def detect_mfi(df: pd.DataFrame) -> tuple:
    """Detect MFI oversold/overbought conditions.

    Returns (mfi_value, oversold, overbought).
    """
    if "MFI" not in df.columns or pd.isna(df["MFI"].iloc[-1]):
        return 0.0, False, False

    mfi = float(df["MFI"].iloc[-1])
    return mfi, mfi < 20, mfi > 80


# ---------------------------------------------------------------------------
# Accumulation/Distribution detection
# ---------------------------------------------------------------------------

def detect_ad_trend(df: pd.DataFrame, lookback: int = 10) -> tuple:
    """Detect A/D line trend direction over the last `lookback` bars.

    Returns (ad_value, bullish_trend, bearish_trend).
    """
    if "AD" not in df.columns or len(df) < lookback:
        return 0.0, False, False

    ad = df["AD"].tail(lookback).values
    ad_current = float(ad[-1])

    # Simple linear regression slope
    x = np.arange(lookback)
    slope = np.polyfit(x, ad, 1)[0]

    return ad_current, slope > 0, slope < 0


# ---------------------------------------------------------------------------
# On-Balance Volume detection
# ---------------------------------------------------------------------------

def detect_obv(df: pd.DataFrame, lookback: int = 10) -> tuple:
    """Detect OBV trend and price-volume divergence.

    Returns (obv_value, bullish_trend, bearish_trend, bullish_div, bearish_div).
    """
    if "OBV" not in df.columns or len(df) < lookback:
        return 0.0, False, False, False, False

    obv = df["OBV"].tail(lookback).values
    close = df["Close"].tail(lookback).values
    obv_current = float(obv[-1])

    x = np.arange(lookback)
    obv_slope = np.polyfit(x, obv, 1)[0]
    price_slope = np.polyfit(x, close, 1)[0]

    bullish_trend = obv_slope > 0
    bearish_trend = obv_slope < 0

    # Divergence: price and OBV moving in opposite directions
    bullish_div = price_slope < 0 and obv_slope > 0  # price down, volume accumulating
    bearish_div = price_slope > 0 and obv_slope < 0  # price up, volume distributing

    return obv_current, bullish_trend, bearish_trend, bullish_div, bearish_div


# ---------------------------------------------------------------------------
# EMA crossover detection
# ---------------------------------------------------------------------------

def detect_ema_cross(df: pd.DataFrame) -> tuple[bool, bool]:
    """Detect bullish/bearish EMA-9 / EMA-21 crossovers on the last 2 bars.

    Returns (bullish_cross, bearish_cross).
    """
    if len(df) < 2 or df["EMA9"].isna().iloc[-1] or df["EMA21"].isna().iloc[-1]:
        return False, False

    prev_ema9 = df["EMA9"].iloc[-2]
    prev_ema21 = df["EMA21"].iloc[-2]
    curr_ema9 = df["EMA9"].iloc[-1]
    curr_ema21 = df["EMA21"].iloc[-1]

    bullish = prev_ema9 <= prev_ema21 and curr_ema9 > curr_ema21
    bearish = prev_ema9 >= prev_ema21 and curr_ema9 < curr_ema21
    return bullish, bearish


# ---------------------------------------------------------------------------
# Breakout detection
# ---------------------------------------------------------------------------

def detect_breakout(df: pd.DataFrame, lookback: int = 20, vol_mult: float = 1.5) -> bool:
    """Price closes above prior 20-bar high with volume > 1.5x average."""
    if len(df) < lookback + 1:
        return False

    prior_high = df["High"].iloc[-(lookback + 1):-1].max()
    curr_close = df["Close"].iloc[-1]
    curr_vol = df["Volume"].iloc[-1]
    avg_vol = df["AvgVol20"].iloc[-1]

    if pd.isna(avg_vol) or avg_vol == 0:
        return False

    return curr_close > prior_high and curr_vol > vol_mult * avg_vol


# ---------------------------------------------------------------------------
# Double-bottom detection
# ---------------------------------------------------------------------------

def detect_double_bottom(df: pd.DataFrame, tolerance: float = 0.03, min_gap: int = 10) -> bool:
    """Two troughs within `tolerance` of each other, separated by ≥ `min_gap` bars,
    with a neckline break on the latest bar.
    """
    if len(df) < 30:
        return False

    window = df.tail(60).copy()
    closes = window["Close"].values
    lows = window["Low"].values

    # Find local minima (simple: lower than both neighbors)
    troughs = []
    for i in range(1, len(lows) - 1):
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            troughs.append((i, lows[i]))

    if len(troughs) < 2:
        return False

    # Check pairs from most recent backward
    for j in range(len(troughs) - 1, 0, -1):
        for k in range(j - 1, -1, -1):
            idx_a, val_a = troughs[k]
            idx_b, val_b = troughs[j]
            gap = idx_b - idx_a

            if gap < min_gap:
                continue

            pct_diff = abs(val_a - val_b) / ((val_a + val_b) / 2)
            if pct_diff > tolerance:
                continue

            # Neckline = max close between the two troughs
            neckline = closes[idx_a:idx_b + 1].max()
            if closes[-1] > neckline:
                return True

    return False


# ---------------------------------------------------------------------------
# Head & Shoulders / Inverse H&S
# ---------------------------------------------------------------------------

def _find_peaks(arr: np.ndarray) -> list:
    """Return list of (index, value) for local maxima."""
    peaks = []
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            peaks.append((i, arr[i]))
    return peaks


def _find_troughs(arr: np.ndarray) -> list:
    """Return list of (index, value) for local minima."""
    troughs = []
    for i in range(1, len(arr) - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            troughs.append((i, arr[i]))
    return troughs


def detect_head_and_shoulders(df: pd.DataFrame, tolerance: float = 0.05) -> bool:
    """Detect bearish Head & Shoulders pattern in the last 60 bars.

    Left shoulder, head (higher high), right shoulder (lower than head,
    similar height to left), neckline violation.
    """
    if len(df) < 30:
        return False

    highs = df["High"].tail(60).values
    closes = df["Close"].tail(60).values
    peaks = _find_peaks(highs)

    if len(peaks) < 3:
        return False

    # Try combinations of 3 consecutive peaks
    for i in range(len(peaks) - 2):
        ls_idx, ls_val = peaks[i]
        h_idx, h_val = peaks[i + 1]
        rs_idx, rs_val = peaks[i + 2]

        # Head must be the highest
        if h_val <= ls_val or h_val <= rs_val:
            continue

        # Shoulders should be similar height
        shoulder_diff = abs(ls_val - rs_val) / ((ls_val + rs_val) / 2)
        if shoulder_diff > tolerance:
            continue

        # Neckline: min of troughs between shoulders
        trough_between = min(closes[ls_idx:rs_idx + 1])

        # Neckline break
        if closes[-1] < trough_between:
            return True

    return False


def detect_inverse_head_shoulders(df: pd.DataFrame, tolerance: float = 0.05) -> bool:
    """Detect bullish Inverse Head & Shoulders in the last 60 bars."""
    if len(df) < 30:
        return False

    lows = df["Low"].tail(60).values
    closes = df["Close"].tail(60).values
    troughs = _find_troughs(lows)

    if len(troughs) < 3:
        return False

    for i in range(len(troughs) - 2):
        ls_idx, ls_val = troughs[i]
        h_idx, h_val = troughs[i + 1]
        rs_idx, rs_val = troughs[i + 2]

        # Head (inverse) must be the lowest
        if h_val >= ls_val or h_val >= rs_val:
            continue

        # Shoulders similar depth
        shoulder_diff = abs(ls_val - rs_val) / ((ls_val + rs_val) / 2)
        if shoulder_diff > tolerance:
            continue

        # Neckline: max of peaks between shoulders
        peak_between = max(closes[ls_idx:rs_idx + 1])

        # Neckline break upward
        if closes[-1] > peak_between:
            return True

    return False


# ---------------------------------------------------------------------------
# Fibonacci retracement
# ---------------------------------------------------------------------------

def compute_fibonacci_levels(df: pd.DataFrame, lookback: int = 60) -> dict:
    """Identify recent swing high/low and compute Fib retracement levels.

    Returns dict with keys: swing_high, swing_low, and the retracement
    percentages (23.6, 38.2, 50.0, 61.8, 78.6) mapped to price levels.
    """
    if len(df) < lookback:
        lookback = len(df)

    window = df.tail(lookback)
    swing_high = window["High"].max()
    swing_low = window["Low"].min()
    diff = swing_high - swing_low

    if diff == 0:
        return {}

    levels = {
        "swing_high": swing_high,
        "swing_low": swing_low,
        23.6: swing_high - 0.236 * diff,
        38.2: swing_high - 0.382 * diff,
        50.0: swing_high - 0.500 * diff,
        61.8: swing_high - 0.618 * diff,
        78.6: swing_high - 0.786 * diff,
        # Extension levels for take-profit targets
        127.2: swing_high + 0.272 * diff,
        161.8: swing_high + 0.618 * diff,
    }
    return levels


def detect_fib_bounce(df: pd.DataFrame, fib_levels: dict, tolerance: float = 0.015) -> tuple[bool, bool]:
    """Check if price bounced off the 38.2% or 61.8% Fib level.

    A bounce = price touched the level (within tolerance) and the latest bar
    closed higher (confirming candle).

    Returns (bounce_382, bounce_618).
    """
    if not fib_levels or len(df) < 3:
        return False, False

    bounce_382 = False
    bounce_618 = False

    recent = df.tail(3)
    low_vals = recent["Low"].values
    close_vals = recent["Close"].values

    for level_key, is_382 in [(38.2, True), (61.8, False)]:
        level = fib_levels.get(level_key)
        if level is None:
            continue

        # Check if any of the last 3 bars' lows touched the level
        for i in range(len(low_vals) - 1):
            pct_from_level = abs(low_vals[i] - level) / level
            if pct_from_level <= tolerance:
                # Confirming candle: next close is higher than the touch bar's close
                if close_vals[i + 1] > close_vals[i]:
                    if is_382:
                        bounce_382 = True
                    else:
                        bounce_618 = True
                    break

    return bounce_382, bounce_618


# ---------------------------------------------------------------------------
# Ichimoku Cloud
# ---------------------------------------------------------------------------

def compute_ichimoku(df: pd.DataFrame) -> tuple:
    """Compute Ichimoku Cloud components.

    Returns (tenkan, kijun, senkou_a, senkou_b, above_cloud, below_cloud,
             bullish_cross, bearish_cross).
    """
    if len(df) < 52:
        return 0.0, 0.0, 0.0, 0.0, False, False, False, False

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
    # Senkou Span A: (Tenkan + Kijun) / 2, shifted 26 periods ahead (we use current)
    senkou_a = (tenkan + kijun) / 2
    # Senkou Span B: (52-period high + 52-period low) / 2, shifted 26 periods ahead
    senkou_b = (high.rolling(window=52).max() + low.rolling(window=52).min()) / 2

    tenkan_val = float(tenkan.iloc[-1]) if not pd.isna(tenkan.iloc[-1]) else 0.0
    kijun_val = float(kijun.iloc[-1]) if not pd.isna(kijun.iloc[-1]) else 0.0
    sa_val = float(senkou_a.iloc[-1]) if not pd.isna(senkou_a.iloc[-1]) else 0.0
    sb_val = float(senkou_b.iloc[-1]) if not pd.isna(senkou_b.iloc[-1]) else 0.0

    current_close = float(close.iloc[-1])
    cloud_top = max(sa_val, sb_val)
    cloud_bottom = min(sa_val, sb_val)

    above_cloud = current_close > cloud_top
    below_cloud = current_close < cloud_bottom

    # TK Cross
    bullish_cross = False
    bearish_cross = False
    if len(df) >= 2 and not pd.isna(tenkan.iloc[-2]) and not pd.isna(kijun.iloc[-2]):
        prev_tenkan = float(tenkan.iloc[-2])
        prev_kijun = float(kijun.iloc[-2])
        bullish_cross = prev_tenkan <= prev_kijun and tenkan_val > kijun_val
        bearish_cross = prev_tenkan >= prev_kijun and tenkan_val < kijun_val

    return (tenkan_val, kijun_val, sa_val, sb_val,
            above_cloud, below_cloud, bullish_cross, bearish_cross)


# ---------------------------------------------------------------------------
# Stochastic RSI
# ---------------------------------------------------------------------------

def compute_stochastic_rsi(df: pd.DataFrame, period: int = 14,
                           k_period: int = 3, d_period: int = 3) -> tuple:
    """Compute Stochastic RSI.

    Returns (stoch_rsi, k, d, oversold, overbought, bullish_cross).
    """
    if "RSI" not in df.columns or len(df) < period + k_period + d_period:
        return 0.0, 0.0, 0.0, False, False, False

    rsi = df["RSI"].copy()
    rsi_min = rsi.rolling(window=period).min()
    rsi_max = rsi.rolling(window=period).max()
    rsi_range = rsi_max - rsi_min
    stoch_rsi = ((rsi - rsi_min) / rsi_range.replace(0, np.nan)) * 100

    k = stoch_rsi.rolling(window=k_period).mean()
    d = k.rolling(window=d_period).mean()

    stoch_val = float(stoch_rsi.iloc[-1]) if not pd.isna(stoch_rsi.iloc[-1]) else 0.0
    k_val = float(k.iloc[-1]) if not pd.isna(k.iloc[-1]) else 0.0
    d_val = float(d.iloc[-1]) if not pd.isna(d.iloc[-1]) else 0.0

    oversold = k_val < 20
    overbought = k_val > 80

    # Bullish cross: K crosses above D while both are below 20
    bullish_cross = False
    if len(df) >= 2 and not pd.isna(k.iloc[-2]) and not pd.isna(d.iloc[-2]):
        prev_k = float(k.iloc[-2])
        prev_d = float(d.iloc[-2])
        bullish_cross = prev_k <= prev_d and k_val > d_val and k_val < 20

    return stoch_val, k_val, d_val, oversold, overbought, bullish_cross


# ---------------------------------------------------------------------------
# ADX — Average Directional Index
# ---------------------------------------------------------------------------

def compute_adx(df: pd.DataFrame, period: int = 14) -> tuple:
    """Compute ADX, +DI, -DI.

    Returns (adx, plus_di, minus_di, strong_trend, bullish, bearish).
    """
    if len(df) < period * 2:
        return 0.0, 0.0, 0.0, False, False, False

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # +DM and -DM
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)

    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Smoothed averages
    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    plus_di = 100 * plus_dm_smooth / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm_smooth / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    adx_val = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0
    pdi_val = float(plus_di.iloc[-1]) if not pd.isna(plus_di.iloc[-1]) else 0.0
    mdi_val = float(minus_di.iloc[-1]) if not pd.isna(minus_di.iloc[-1]) else 0.0

    strong = adx_val > 25
    bullish = pdi_val > mdi_val and strong
    bearish = mdi_val > pdi_val and strong

    return adx_val, pdi_val, mdi_val, strong, bullish, bearish


# ---------------------------------------------------------------------------
# Keltner Channels / TTM Squeeze
# ---------------------------------------------------------------------------

def compute_keltner(df: pd.DataFrame, period: int = 20,
                    atr_mult: float = 1.5) -> tuple:
    """Compute Keltner Channels and TTM Squeeze.

    Returns (upper, lower, squeeze, squeeze_fired).
    """
    if "ATR" not in df.columns or "BB_Upper" not in df.columns or len(df) < period:
        return 0.0, 0.0, False, False

    ema = df["Close"].ewm(span=period, adjust=False).mean()
    atr = df["ATR"]

    kelt_upper = ema + atr_mult * atr
    kelt_lower = ema - atr_mult * atr

    upper_val = float(kelt_upper.iloc[-1]) if not pd.isna(kelt_upper.iloc[-1]) else 0.0
    lower_val = float(kelt_lower.iloc[-1]) if not pd.isna(kelt_lower.iloc[-1]) else 0.0

    # TTM Squeeze: BB inside Keltner
    bb_upper = df["BB_Upper"]
    bb_lower = df["BB_Lower"]

    squeeze = (bb_upper.iloc[-1] < kelt_upper.iloc[-1] and
               bb_lower.iloc[-1] > kelt_lower.iloc[-1])

    # Squeeze fired: was in squeeze on previous bar but not now
    squeeze_fired = False
    if len(df) >= 2:
        prev_squeeze = (bb_upper.iloc[-2] < kelt_upper.iloc[-2] and
                        bb_lower.iloc[-2] > kelt_lower.iloc[-2])
        squeeze_fired = prev_squeeze and not squeeze

    return upper_val, lower_val, bool(squeeze), bool(squeeze_fired)


# ---------------------------------------------------------------------------
# Relative Strength vs SPY
# ---------------------------------------------------------------------------

def compute_relative_strength_vs_spy(df: pd.DataFrame, spy_df=None) -> tuple:
    """Compute relative strength ratio vs SPY over 20 bars.

    Returns (rs_ratio, trending_up).
    """
    lookback = 20
    if len(df) < lookback:
        return 0.0, False

    if spy_df is None:
        try:
            import yfinance as yf
            spy_df = yf.download("SPY", period="6mo", progress=False)
            if spy_df.empty:
                return 0.0, False
            # Handle multi-level columns from yfinance
            if isinstance(spy_df.columns, pd.MultiIndex):
                spy_df.columns = spy_df.columns.get_level_values(0)
        except Exception:
            return 0.0, False

    if len(spy_df) < lookback:
        return 0.0, False

    stock_ret = float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-lookback]) - 1
    spy_ret = float(spy_df["Close"].iloc[-1]) / float(spy_df["Close"].iloc[-lookback]) - 1

    if spy_ret == 0:
        return 0.0, False

    rs_ratio = (1 + stock_ret) / (1 + spy_ret)

    # Check if RS is trending up: compare current RS to RS 10 bars ago
    trending_up = False
    if len(df) >= lookback + 10 and len(spy_df) >= lookback + 10:
        stock_ret_prev = float(df["Close"].iloc[-11]) / float(df["Close"].iloc[-lookback - 10]) - 1
        spy_ret_prev = float(spy_df["Close"].iloc[-11]) / float(spy_df["Close"].iloc[-lookback - 10]) - 1
        if (1 + spy_ret_prev) != 0:
            rs_prev = (1 + stock_ret_prev) / (1 + spy_ret_prev)
            trending_up = rs_ratio > rs_prev

    return rs_ratio, trending_up


# ---------------------------------------------------------------------------
# Pivot Points
# ---------------------------------------------------------------------------

def compute_pivot_points(df: pd.DataFrame) -> tuple:
    """Compute standard floor pivot points from previous bar's HLC.

    Returns (pivot, r1, r2, s1, s2, near_support, near_resistance).
    """
    if len(df) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0, False, False

    prev_high = float(df["High"].iloc[-2])
    prev_low = float(df["Low"].iloc[-2])
    prev_close = float(df["Close"].iloc[-2])
    current_close = float(df["Close"].iloc[-1])

    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)

    # Near support: within 1% of S1 or S2
    near_support = False
    if s1 > 0:
        near_support = abs(current_close - s1) / s1 < 0.01
    if s2 > 0 and not near_support:
        near_support = abs(current_close - s2) / s2 < 0.01

    # Near resistance: within 1% of R1 or R2
    near_resistance = False
    if r1 > 0:
        near_resistance = abs(current_close - r1) / r1 < 0.01
    if r2 > 0 and not near_resistance:
        near_resistance = abs(current_close - r2) / r2 < 0.01

    return pivot, r1, r2, s1, s2, near_support, near_resistance


# ---------------------------------------------------------------------------
# Gap Detection
# ---------------------------------------------------------------------------

def detect_gaps(df: pd.DataFrame) -> tuple:
    """Detect gap up/down between last two bars.

    Returns (gap_up, gap_down, gap_up_pct, gap_down_pct).
    """
    if len(df) < 2:
        return False, False, 0.0, 0.0

    prev_high = float(df["High"].iloc[-2])
    prev_low = float(df["Low"].iloc[-2])
    curr_open = float(df["Open"].iloc[-1])

    gap_up = curr_open > prev_high
    gap_down = curr_open < prev_low

    gap_up_pct = 0.0
    gap_down_pct = 0.0
    if gap_up and prev_high > 0:
        gap_up_pct = (curr_open - prev_high) / prev_high * 100
    if gap_down and prev_low > 0:
        gap_down_pct = (prev_low - curr_open) / prev_low * 100

    return gap_up, gap_down, gap_up_pct, gap_down_pct


# ---------------------------------------------------------------------------
# Cup and Handle
# ---------------------------------------------------------------------------

def detect_cup_and_handle(df: pd.DataFrame, lookback: int = 120) -> bool:
    """Detect cup-and-handle pattern (bullish continuation).

    Simple version: U-shaped base in first 2/3 of lookback, recovery to
    prior high, small pullback (handle), then breakout.
    """
    if len(df) < lookback:
        return False

    window = df.tail(lookback)
    closes = window["Close"].values
    highs = window["High"].values

    cup_end = int(lookback * 2 / 3)
    handle_start = cup_end

    # Left rim: high in first 10% of the lookback
    left_rim_zone = highs[:int(lookback * 0.15)]
    if len(left_rim_zone) == 0:
        return False
    left_rim = float(left_rim_zone.max())

    # Cup bottom: lowest low in the cup zone
    cup_zone = closes[:cup_end]
    cup_bottom_idx = int(np.argmin(cup_zone))
    cup_bottom = float(cup_zone[cup_bottom_idx])

    # Cup must be meaningful (at least 10% deep)
    if left_rim == 0 or (left_rim - cup_bottom) / left_rim < 0.10:
        return False

    # Right rim: price should recover close to the left rim level
    right_rim_zone = closes[handle_start:]
    if len(right_rim_zone) == 0:
        return False
    right_rim = float(right_rim_zone.max())

    # Right rim should be within 5% of left rim
    if left_rim > 0 and abs(right_rim - left_rim) / left_rim > 0.05:
        return False

    # Handle: small pullback after right rim (last 15 bars), max 5% drop
    handle_zone = closes[-15:]
    handle_low = float(handle_zone.min())
    if right_rim > 0 and (right_rim - handle_low) / right_rim > 0.05:
        return False

    # Breakout: current close above both rims
    current_close = float(closes[-1])
    return current_close > left_rim and current_close > right_rim


# ---------------------------------------------------------------------------
# Triangle Patterns
# ---------------------------------------------------------------------------

def detect_triangles(df: pd.DataFrame, lookback: int = 40) -> tuple:
    """Detect ascending and descending triangle patterns.

    Returns (ascending, descending).
    """
    if len(df) < lookback:
        return False, False

    window = df.tail(lookback)
    highs = window["High"].values
    lows = window["Low"].values

    # Find swing highs and swing lows
    swing_highs = []
    swing_lows = []
    for i in range(2, len(highs) - 2):
        if highs[i] >= highs[i - 1] and highs[i] >= highs[i + 1]:
            swing_highs.append((i, highs[i]))
        if lows[i] <= lows[i - 1] and lows[i] <= lows[i + 1]:
            swing_lows.append((i, lows[i]))

    ascending = False
    descending = False

    if len(swing_highs) >= 3 and len(swing_lows) >= 3:
        # Ascending triangle: flat resistance (highs similar), rising lows
        high_vals = [h[1] for h in swing_highs[-3:]]
        low_vals = [l[1] for l in swing_lows[-3:]]

        high_range = (max(high_vals) - min(high_vals))
        avg_high = np.mean(high_vals)
        flat_resistance = avg_high > 0 and high_range / avg_high < 0.02

        rising_lows = low_vals[-1] > low_vals[0]

        if flat_resistance and rising_lows:
            ascending = True

        # Descending triangle: flat support (lows similar), falling highs
        low_range = (max(low_vals) - min(low_vals))
        avg_low = np.mean(low_vals)
        flat_support = avg_low > 0 and low_range / avg_low < 0.02

        falling_highs = high_vals[-1] < high_vals[0]

        if flat_support and falling_highs:
            descending = True

    return ascending, descending


# ---------------------------------------------------------------------------
# Bull Flag
# ---------------------------------------------------------------------------

def detect_bull_flag(df: pd.DataFrame, lookback: int = 30) -> bool:
    """Detect bull flag pattern: strong move up followed by tight consolidation.

    Strong move: >5% gain in 5 bars. Consolidation: low ATR relative range.
    """
    if len(df) < lookback or "ATR" not in df.columns:
        return False

    window = df.tail(lookback)
    closes = window["Close"].values

    # Look for a strong pole in the first half
    pole_zone = closes[:15]
    best_gain = 0.0
    for i in range(len(pole_zone) - 5):
        gain = (pole_zone[i + 5] - pole_zone[i]) / pole_zone[i] if pole_zone[i] > 0 else 0
        if gain > best_gain:
            best_gain = gain

    if best_gain < 0.05:
        return False

    # Consolidation in last 10 bars: range < 3% and slight downward or flat drift
    consol = closes[-10:]
    consol_range = (consol.max() - consol.min()) / consol.mean() if consol.mean() > 0 else 1.0
    if consol_range > 0.03:
        return False

    # Breakout: last close above consolidation high
    return float(closes[-1]) >= float(consol.max())


# ---------------------------------------------------------------------------
# EMA Ribbon
# ---------------------------------------------------------------------------

def compute_ema_ribbon(df: pd.DataFrame) -> tuple:
    """Check if EMA 8, 13, 21, 34, 55 are in bullish or bearish order.

    Returns (bullish, bearish).
    """
    needed = ["EMA8", "EMA13", "EMA21", "EMA34", "EMA55"]
    for col in needed:
        if col not in df.columns or pd.isna(df[col].iloc[-1]):
            return False, False

    e8 = float(df["EMA8"].iloc[-1])
    e13 = float(df["EMA13"].iloc[-1])
    e21 = float(df["EMA21"].iloc[-1])
    e34 = float(df["EMA34"].iloc[-1])
    e55 = float(df["EMA55"].iloc[-1])

    bullish = e8 > e13 > e21 > e34 > e55
    bearish = e8 < e13 < e21 < e34 < e55

    return bullish, bearish


# ---------------------------------------------------------------------------
# 52-Week Context
# ---------------------------------------------------------------------------

def compute_52w_context(df: pd.DataFrame) -> tuple:
    """Compute 52-week (252 bars) high/low context.

    Returns (pct_from_high, pct_from_low, near_high, near_low).
    """
    bars = min(252, len(df))
    if bars < 20:
        return 0.0, 0.0, False, False

    window = df.tail(bars)
    high_52w = float(window["High"].max())
    low_52w = float(window["Low"].min())
    current = float(df["Close"].iloc[-1])

    pct_from_high = 0.0
    pct_from_low = 0.0

    if high_52w > 0:
        pct_from_high = (high_52w - current) / high_52w * 100
    if low_52w > 0:
        pct_from_low = (current - low_52w) / low_52w * 100

    near_high = pct_from_high <= 5.0
    near_low = pct_from_low <= 10.0 if low_52w > 0 else False

    return pct_from_high, pct_from_low, near_high, near_low


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze(ticker: str, df: pd.DataFrame) -> TechnicalSignals:
    """Run the full technical analysis suite on the given OHLCV data.

    Returns a populated TechnicalSignals object.
    """
    signals = TechnicalSignals(ticker=ticker)

    if df.empty or len(df) < 30:
        logger.warning("Insufficient data for %s (%d bars)", ticker, len(df))
        return signals

    # Compute indicators
    df = compute_indicators(df)
    signals.current_price = df["Close"].iloc[-1]
    signals.ema9 = df["EMA9"].iloc[-1] if not pd.isna(df["EMA9"].iloc[-1]) else 0.0
    signals.ema21 = df["EMA21"].iloc[-1] if not pd.isna(df["EMA21"].iloc[-1]) else 0.0
    signals.sma200 = df["SMA200"].iloc[-1] if not pd.isna(df["SMA200"].iloc[-1]) else 0.0

    # Trend regime
    signals.price_above_200sma = (
        signals.current_price > signals.sma200 and signals.sma200 > 0
    )

    # EMA cross
    bullish, bearish = detect_ema_cross(df)
    signals.ema_cross_bullish = bullish and signals.price_above_200sma
    signals.ema_cross_bearish = bearish

    # Breakout
    signals.breakout_with_volume = detect_breakout(df)

    # Patterns
    signals.double_bottom = detect_double_bottom(df)
    signals.head_and_shoulders = detect_head_and_shoulders(df)
    signals.inverse_head_shoulders = detect_inverse_head_shoulders(df)

    # Fibonacci
    fib = compute_fibonacci_levels(df)
    signals.fib_levels = fib
    signals.fib_bounce_382, signals.fib_bounce_618 = detect_fib_bounce(df, fib)

    # RSI
    signals.rsi, signals.rsi_oversold, signals.rsi_overbought = detect_rsi(df)

    # MACD
    (signals.macd_value, signals.macd_signal, signals.macd_histogram,
     signals.macd_bullish_cross, signals.macd_bearish_cross) = detect_macd_cross(df)

    # Bollinger Bands
    (signals.bb_upper, signals.bb_middle, signals.bb_lower,
     signals.bb_squeeze, signals.bb_breakout_upper, signals.bb_breakout_lower) = detect_bollinger(df)

    # ATR-based stops
    signals.atr, signals.atr_stop_long, signals.atr_stop_short = compute_atr_stops(df)

    # Divergence detection
    signals.rsi_bullish_divergence, signals.rsi_bearish_divergence = detect_rsi_divergence(df)
    signals.macd_bullish_divergence, signals.macd_bearish_divergence = detect_macd_divergence(df)

    # VWAP
    signals.vwap, signals.price_above_vwap = detect_vwap(df)

    # Money Flow Index
    signals.mfi, signals.mfi_oversold, signals.mfi_overbought = detect_mfi(df)

    # Accumulation/Distribution
    signals.ad_line, signals.ad_trend_bullish, signals.ad_trend_bearish = detect_ad_trend(df)

    # On-Balance Volume
    (signals.obv, signals.obv_trend_bullish, signals.obv_trend_bearish,
     signals.obv_divergence_bullish, signals.obv_divergence_bearish) = detect_obv(df)

    # Ichimoku Cloud
    (signals.tenkan_sen, signals.kijun_sen, signals.senkou_a, signals.senkou_b,
     signals.ichimoku_above_cloud, signals.ichimoku_below_cloud,
     signals.ichimoku_bullish_cross, signals.ichimoku_bearish_cross) = compute_ichimoku(df)

    # Stochastic RSI
    (signals.stoch_rsi, signals.stoch_rsi_k, signals.stoch_rsi_d,
     signals.stoch_rsi_oversold, signals.stoch_rsi_overbought,
     signals.stoch_rsi_bullish_cross) = compute_stochastic_rsi(df)

    # ADX
    (signals.adx, signals.adx_plus_di, signals.adx_minus_di,
     signals.adx_strong_trend, signals.adx_bullish, signals.adx_bearish) = compute_adx(df)

    # Keltner Channels / TTM Squeeze
    (signals.keltner_upper, signals.keltner_lower,
     signals.ttm_squeeze, signals.ttm_squeeze_fired) = compute_keltner(df)

    # Relative Strength vs SPY
    signals.rs_vs_spy, signals.rs_trending_up = compute_relative_strength_vs_spy(df)

    # Pivot Points
    (signals.pivot, signals.pivot_r1, signals.pivot_r2,
     signals.pivot_s1, signals.pivot_s2,
     signals.near_pivot_support, signals.near_pivot_resistance) = compute_pivot_points(df)

    # Gaps
    (signals.gap_up, signals.gap_down,
     signals.gap_up_pct, signals.gap_down_pct) = detect_gaps(df)

    # Cup and Handle
    signals.cup_and_handle = detect_cup_and_handle(df)

    # Triangles
    signals.ascending_triangle, signals.descending_triangle = detect_triangles(df)

    # Bull Flag
    signals.bull_flag = detect_bull_flag(df)

    # EMA Ribbon
    signals.ema_ribbon_bullish, signals.ema_ribbon_bearish = compute_ema_ribbon(df)

    # 52-Week Context
    (signals.pct_from_52w_high, signals.pct_from_52w_low,
     signals.near_52w_high, signals.near_52w_low) = compute_52w_context(df)

    # Support / resistance for position sizing (now enhanced with ATR)
    signals.resistance_level = df["High"].tail(20).max()
    atr_support = signals.atr_stop_long if signals.atr_stop_long > 0 else 0
    signals.support_level = max(
        signals.ema21,
        df["Low"].tail(20).min(),
        atr_support,
    )

    # Compile pattern details for logging
    details = []
    if signals.ema_cross_bullish:
        details.append("Bullish EMA 9/21 cross (above 200 SMA)")
    if signals.ema_cross_bearish:
        details.append("Bearish EMA 9/21 cross")
    if signals.breakout_with_volume:
        details.append("Breakout above 20-bar high with volume surge")
    if signals.double_bottom:
        details.append("Double bottom pattern detected")
    if signals.head_and_shoulders:
        details.append("Head & Shoulders (bearish) detected")
    if signals.inverse_head_shoulders:
        details.append("Inverse Head & Shoulders (bullish) detected")
    if signals.fib_bounce_382:
        details.append("Fib bounce at 38.2% level")
    if signals.fib_bounce_618:
        details.append("Fib bounce at 61.8% level")
    if signals.rsi_oversold:
        details.append(f"RSI oversold ({signals.rsi:.1f})")
    if signals.rsi_overbought:
        details.append(f"RSI overbought ({signals.rsi:.1f})")
    if signals.macd_bullish_cross:
        details.append("MACD bullish crossover")
    if signals.macd_bearish_cross:
        details.append("MACD bearish crossover")
    if signals.bb_squeeze:
        details.append("Bollinger Band squeeze (low volatility)")
    if signals.bb_breakout_upper:
        details.append("BB breakout above upper band")
    if signals.bb_breakout_lower:
        details.append("BB breakdown below lower band")
    if signals.rsi_bullish_divergence:
        details.append("RSI bullish divergence (reversal signal)")
    if signals.rsi_bearish_divergence:
        details.append("RSI bearish divergence (reversal signal)")
    if signals.macd_bullish_divergence:
        details.append("MACD bullish divergence")
    if signals.macd_bearish_divergence:
        details.append("MACD bearish divergence")
    if signals.price_above_vwap:
        details.append(f"Price above VWAP ({signals.vwap:.2f})")
    if signals.atr > 0:
        details.append(f"ATR: {signals.atr:.2f} | stops: L={signals.atr_stop_long:.2f} S={signals.atr_stop_short:.2f}")
    if signals.mfi_oversold:
        details.append(f"MFI oversold ({signals.mfi:.0f})")
    if signals.mfi_overbought:
        details.append(f"MFI overbought ({signals.mfi:.0f})")
    if signals.ad_trend_bullish:
        details.append("A/D line bullish (accumulation)")
    if signals.ad_trend_bearish:
        details.append("A/D line bearish (distribution)")
    if signals.obv_divergence_bullish:
        details.append("OBV bullish divergence (price down, volume accumulating)")
    if signals.obv_divergence_bearish:
        details.append("OBV bearish divergence (price up, volume distributing)")
    if signals.ichimoku_above_cloud:
        details.append("Price above Ichimoku cloud (bullish)")
    if signals.ichimoku_below_cloud:
        details.append("Price below Ichimoku cloud (bearish)")
    if signals.ichimoku_bullish_cross:
        details.append("Ichimoku TK bullish cross (Tenkan > Kijun)")
    if signals.ichimoku_bearish_cross:
        details.append("Ichimoku TK bearish cross (Tenkan < Kijun)")
    if signals.stoch_rsi_oversold:
        details.append(f"StochRSI oversold (K={signals.stoch_rsi_k:.1f})")
    if signals.stoch_rsi_overbought:
        details.append(f"StochRSI overbought (K={signals.stoch_rsi_k:.1f})")
    if signals.stoch_rsi_bullish_cross:
        details.append("StochRSI bullish cross in oversold zone")
    if signals.adx_strong_trend:
        direction = "bullish" if signals.adx_bullish else "bearish" if signals.adx_bearish else "neutral"
        details.append(f"ADX strong trend ({signals.adx:.1f}) — {direction}")
    if signals.ttm_squeeze:
        details.append("TTM Squeeze active (volatility compression)")
    if signals.ttm_squeeze_fired:
        details.append("TTM Squeeze FIRED (momentum expansion starting)")
    if signals.rs_vs_spy > 0 and signals.rs_trending_up:
        details.append(f"Outperforming SPY (RS ratio: {signals.rs_vs_spy:.2f}, trending up)")
    if signals.near_pivot_support:
        details.append(f"Near pivot support (S1={signals.pivot_s1:.2f}, S2={signals.pivot_s2:.2f})")
    if signals.near_pivot_resistance:
        details.append(f"Near pivot resistance (R1={signals.pivot_r1:.2f}, R2={signals.pivot_r2:.2f})")
    if signals.gap_up:
        details.append(f"Gap up {signals.gap_up_pct:.1f}%")
    if signals.gap_down:
        details.append(f"Gap down {signals.gap_down_pct:.1f}%")
    if signals.cup_and_handle:
        details.append("Cup & handle pattern (bullish continuation)")
    if signals.ascending_triangle:
        details.append("Ascending triangle (bullish breakout setup)")
    if signals.descending_triangle:
        details.append("Descending triangle (bearish breakdown setup)")
    if signals.bull_flag:
        details.append("Bull flag pattern (bullish continuation)")
    if signals.ema_ribbon_bullish:
        details.append("EMA ribbon bullish (8>13>21>34>55 — strong uptrend)")
    if signals.ema_ribbon_bearish:
        details.append("EMA ribbon bearish (8<13<21<34<55 — strong downtrend)")
    if signals.near_52w_high:
        details.append(f"Near 52-week high ({signals.pct_from_52w_high:.1f}% away)")
    if signals.near_52w_low:
        details.append(f"Near 52-week low ({signals.pct_from_52w_low:.1f}% from low)")
    signals.pattern_details = details

    return signals
