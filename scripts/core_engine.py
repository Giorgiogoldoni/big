"""Motore segnale di core (BUY1/BUY2/BUY3/EXIT1/EXIT2/MEAN REV/WATCH).
Codice estratto da github.com/Giorgiogoldoni/core/scripts/generate_charts.py."""
import math


def calc_kama(close, n=10, fast=2, slow=30):
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    kama = [None] * len(close)
    if len(close) <= n:
        return kama
    kama[n] = close[n]
    for i in range(n + 1, len(close)):
        direction = abs(close[i] - close[i - n])
        volatility = sum(abs(close[j] - close[j - 1]) for j in range(i - n + 1, i + 1))
        er = direction / volatility if volatility != 0 else 0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    return kama


def calc_sar_array(high, low, af0=0.02, af_max=0.20):
    n = len(high)
    sar_arr = [None] * n
    bull_arr = [None] * n
    if n < 5:
        return sar_arr, bull_arr
    sar = low[0]
    ep = high[0]
    af = af0
    bull = True
    sar_arr[0] = round(sar, 4)
    bull_arr[0] = bull
    for i in range(1, n):
        if bull:
            new_sar = sar + af * (ep - sar)
            new_sar = min(new_sar, low[max(0, i - 1)], low[max(0, i - 2)])
            if low[i] < new_sar:
                bull = False
                new_sar = ep
                ep = low[i]
                af = af0
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af0, af_max)
        else:
            new_sar = sar + af * (ep - sar)
            new_sar = max(new_sar, high[max(0, i - 1)], high[max(0, i - 2)])
            if high[i] > new_sar:
                bull = True
                new_sar = ep
                ep = high[i]
                af = af0
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af0, af_max)
        sar = new_sar
        sar_arr[i] = round(sar, 4)
        bull_arr[i] = bull
    return sar_arr, bull_arr


def calc_ao_array(high, low):
    mid = [(h + l) / 2 for h, l in zip(high, low)]
    result = [None] * len(mid)
    for i in range(33, len(mid)):
        sma5 = sum(mid[i - 4:i + 1]) / 5
        sma34 = sum(mid[i - 33:i + 1]) / 34
        result[i] = round(sma5 - sma34, 4)
    return result


def calc_rsi_array(close, n=14):
    result = [None] * len(close)
    if len(close) < n + 2:
        return result
    for i in range(n, len(close)):
        gains = 0.0
        losses = 0.0
        for j in range(i - n + 1, i + 1):
            d = close[j] - close[j - 1]
            if d > 0:
                gains += d
            else:
                losses += -d
        ag = gains / n
        al = losses / n
        result[i] = round(100 - 100 / (1 + ag / al), 2) if al > 0 else 100.0
    return result


def calc_er_array(close, n=10):
    result = [0] * len(close)
    for i in range(n, len(close)):
        direction = abs(close[i] - close[i - n])
        volatility = sum(abs(close[j] - close[j - 1]) for j in range(i - n + 1, i + 1))
        result[i] = round(direction / volatility, 4) if volatility != 0 else 0
    return result


def calc_baffetti_array(high, low):
    """Barre consecutive con mid-price in salita."""
    mid = [(h + l) / 2 for h, l in zip(high, low)]
    result = [0] * len(mid)
    streak = 0
    for i in range(1, len(mid)):
        streak = streak + 1 if mid[i] > mid[i - 1] else 0
        result[i] = streak
    return result


def calc_mm_align_array(close):
    n = len(close)
    result = [False] * n
    cum = [0.0] * (n + 1)
    for i in range(n):
        cum[i + 1] = cum[i] + close[i]

    def avg(i, w):
        return (cum[i + 1] - cum[i + 1 - w]) / w if i + 1 >= w else None

    for i in range(n):
        mm20, mm50, mm100 = avg(i, 20), avg(i, 50), avg(i, 100)
        if mm20 is not None and mm50 is not None and mm100 is not None:
            result[i] = close[i] > mm20 > mm50 > mm100
    return result


def calc_cross_days_array(close, kama):
    n = len(close)
    result = [999] * n
    last_flip = None
    prev_above = None
    for i in range(n):
        if kama[i] is None:
            continue
        above = close[i] > kama[i]
        if prev_above is None:
            prev_above = above
            last_flip = i
            result[i] = 0
            continue
        if above != prev_above:
            last_flip = i
            prev_above = above
        result[i] = i - last_flip
    return result


def calc_ao_improving_array(ao):
    n = len(ao)
    result = [False] * n
    for i in range(1, n):
        if ao[i] is not None and ao[i - 1] is not None and ao[i] > ao[i - 1]:
            result[i] = True
    return result


def calc_segnale_array(close, kama, er_arr, baff_arr, ao_imp_arr, sar_bull_arr, cross_arr, mm_arr, rsi_arr):
    """Motore di segnale nativo del grafico (BUY1/BUY2/BUY3/EXIT1/EXIT2/MEAN REV/WATCH)."""
    n = len(close)
    result = [None] * n
    for i in range(n):
        if kama[i] is None or sar_bull_arr[i] is None:
            continue
        lk = kama[i]
        lc = close[i]
        above_kama = lc > lk if lk else False
        sar_bull = sar_bull_arr[i]
        cross = cross_arr[i]
        ao_imp = ao_imp_arr[i]
        baff = baff_arr[i]
        er = er_arr[i]
        mm_align = mm_arr[i]
        rsi = rsi_arr[i] if rsi_arr[i] is not None else 50
        if sar_bull and cross <= 3 and ao_imp:
            result[i] = "BUY1"
        elif above_kama and baff >= 2:
            result[i] = "BUY2"
        elif above_kama and er >= 0.50 and baff >= 3 and mm_align:
            result[i] = "BUY3"
        elif not above_kama and not sar_bull:
            result[i] = "EXIT2"
        elif not sar_bull:
            result[i] = "EXIT1"
        else:
            near_kama = abs(lc - lk) / lk < 0.03 if lk and lk > 0 else False
            if er < 0.30 and rsi < 30 and ao_imp and (near_kama or not above_kama):
                result[i] = "MEAN REV"
            else:
                result[i] = "WATCH"
    return result
