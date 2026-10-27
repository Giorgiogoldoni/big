"""Motori di azionario (BUY2/BUY3, Inversione, Super Best Buy, Mean Reversion, Super Mean Reversion).
Codice estratto da github.com/Giorgiogoldoni/azionario/fetch_azionario.py: stesse regole e stesse soglie.
Unica differenza: i segnali sono letti su ogni barra storica, non solo sull'ultima."""
import math
import numpy as np
import pandas as pd

KAMA_FAST_N = 10
KAMA_SLOW_N = 20
KAMA_FAST_SC = 2
KAMA_SLOW_SC = 30
ER_N = 10
AO_FAST = 3
AO_SLOW = 13
RVI_N = 4
RSI_FAST_N = 5
RSI_SLOW_N = 14
ADX_N = 14
SAR_STEP = 0.02
SAR_MAX = 0.2
VOL_AVG_N = 20

# Uscita: Chandelier Exit (trailing) + hard stop fisso di sicurezza
CHANDELIER_ATR_MULT = 3.0   # Massimo da entrata - 3xATR14
HARD_STOP_PCT = 0.08        # -8% dall'entrata: rete di sicurezza contro i gap
TREND_HURST_THRESHOLD = 0.55  # BUY2/BUY3 entrano solo se Hurst60 >= soglia (titolo che trenda davvero)

# Motore Mean-Reversion (parallelo/separato da BUY2/BUY3/Chandelier) — "compra i minimi, vendi i massimi"
MR_BB_N = 20                 # periodo Bollinger Bands
MR_BB_STD = 2.0              # deviazioni standard
MR_RSI_OVERSOLD = 30         # RSI14 sotto questa soglia = ipervenduto
MR_HURST_THRESHOLD = 0.45    # sotto questa soglia: regime mean-reverting confermato (filtro bloccante)

# Motore SUPER MEAN REVERSION — "compra dopo un forte calo", separato dalla MR normale
# (che anzi blocca volutamente gli ingressi durante un vero crollo). Due fasi:
# 1) IN OSSERVAZIONE quando c'è un calo forte (dall'ultimo pivot ZigZag, non da una finestra
#    fissa di barre — vedi calc_zigzag) + ipervenduto profondo
# 2) INGRESSO solo quando arriva una conferma di inversione (non si compra "a scatola chiusa")
SMR_DRAWDOWN_THRESHOLD = -0.25   # calo minimo dal massimo per accendere "in osservazione"
SMR_RSI14_MAX = 35           # ipervenduto profondo richiesto per "in osservazione"
SMR_WATCH_EXPIRY = 15        # barre entro cui deve arrivare la conferma, altrimenti si annulla
SMR_STOP_PCT = -0.15         # stop di invalidazione se il rimbalzo fallisce dopo l'ingresso
ZIGZAG_ATR_MULT = 4.0        # soglia ZigZag = ATR14 x questo moltiplicatore (si adatta da solo alla volatilità)
HURST_STRIDE = 5             # ricalcolo Hurst ogni N barre (statistica lenta, approssimazione lecita


def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    change = (close - close.shift(n)).abs()
    volatility = close.diff().abs().rolling(n).sum()
    er = change / volatility.replace(0, np.nan)
    return er.fillna(0)


def kama(close: pd.Series, n: int, fast_sc: int, slow_sc: int) -> pd.Series:
    er = efficiency_ratio(close, n)
    fast_alpha = 2 / (fast_sc + 1)
    slow_alpha = 2 / (slow_sc + 1)
    sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2

    out = np.full(len(close), np.nan)
    first_valid = n
    if len(close) <= first_valid:
        return pd.Series(out, index=close.index)
    out[first_valid] = close.iloc[first_valid]
    for i in range(first_valid + 1, len(close)):
        prev = out[i - 1]
        if np.isnan(prev):
            prev = close.iloc[i - 1]
        out[i] = prev + sc.iloc[i] * (close.iloc[i] - prev)
    return pd.Series(out, index=close.index)


def awesome_oscillator(close: pd.Series, fast=AO_FAST, slow=AO_SLOW) -> pd.Series:
    return close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()


def rvi(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, n=RVI_N) -> pd.Series:
    num = (close - open_).rolling(n).mean()
    den = (high - low).rolling(n).mean()
    return (num / den.replace(0, np.nan)).fillna(0)


def rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def calc_obv(close: pd.Series, vol: pd.Series) -> pd.Series:
    """On-Balance Volume (standard min-finder)."""
    direction = np.sign(close.diff().fillna(0))
    return (direction * vol.fillna(0)).cumsum()


def bollinger_bands(close: pd.Series, n=MR_BB_N, num_std=MR_BB_STD):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std()
    return mid + num_std * std, mid, mid - num_std * std


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n=ADX_N):
    """Restituisce (adx, plus_di, minus_di). plus_di/minus_di servono a classify_regime."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / n, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / n, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_series = dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0)
    return adx_series, plus_di.fillna(0), minus_di.fillna(0)


def calc_hurst(prices_list, min_points=30) -> float:
    """
    Hurst Exponent con metodo varianza.
    H > 0.55 -> trending | H ~ 0.5 -> random walk | H < 0.45 -> mean reverting
    """
    if len(prices_list) < min_points:
        return 0.5
    lags = [l for l in [2, 4, 8, 16, 32] if l < len(prices_list) // 2]
    if len(lags) < 3:
        return 0.5
    try:
        log_p = [math.log(p) for p in prices_list if p > 0]
        if len(log_p) < max(lags) + 1:
            return 0.5
        vars_ = []
        for lag in lags:
            diffs = [log_p[i] - log_p[i - lag] for i in range(lag, len(log_p))]
            mean_d = sum(diffs) / len(diffs)
            var = sum((d - mean_d) ** 2 for d in diffs) / len(diffs)
            vars_.append(var if var > 0 else 1e-10)
        log_lags = [math.log(l) for l in lags]
        log_vars = [math.log(v) for v in vars_]
        n = len(lags)
        mean_x = sum(log_lags) / n
        mean_y = sum(log_vars) / n
        num = sum((log_lags[i] - mean_x) * (log_vars[i] - mean_y) for i in range(n))
        den = sum((log_lags[i] - mean_x) ** 2 for i in range(n))
        if den == 0:
            return 0.5
        return round(max(0.1, min(0.9, num / den / 2)), 3)
    except Exception:
        return 0.5


def calc_zigzag(high: list, low: list, atr_series: list, atr_mult: float = 4.0):
    """ZigZag con soglia dinamica = ATR14 x atr_mult (invece di una % fissa): si adatta
    da solo alla volatilità reale di ciascun titolo (un'azione tranquilla e un ETF a
    leva 5x su materie prime avranno soglie diverse senza bisogno di tabelle manuali).
    Ritorna la lista di pivot [(indice, prezzo, 'H'|'L'), ...] in ordine cronologico.
    """
    n = len(high)
    if n < 2:
        return []
    pivots = []
    up_zig = True
    tmp_max, tmp_max_idx = high[0], 0
    tmp_min, tmp_min_idx = low[0], 0
    for i in range(1, n):
        atr = atr_series[i]
        if atr is None:
            continue
        base_price = tmp_max if up_zig else tmp_min
        pct = (atr * atr_mult) / base_price if base_price else 0
        if up_zig:
            if high[i] > tmp_max:
                tmp_max, tmp_max_idx = high[i], i
            if pct and (low[i] - tmp_max) / tmp_max <= -pct:
                pivots.append((tmp_max_idx, tmp_max, "H"))
                up_zig = False
                tmp_min, tmp_min_idx = low[i], i
        else:
            if low[i] < tmp_min:
                tmp_min, tmp_min_idx = low[i], i
            if pct and (high[i] - tmp_min) / tmp_min >= pct:
                pivots.append((tmp_min_idx, tmp_min, "L"))
                up_zig = True
                tmp_max, tmp_max_idx = high[i], i
    return pivots


def calc_atr_series(high: list, low: list, close: list, n: int = 14) -> list:
    """ATR (Average True Range) allineato per indice con high/low/close: il risultato ha
    la STESSA lunghezza dell'input, con None nelle prime n barre (dati insufficienti).
    ATTENZIONE: prima di questa correzione la versione restituiva un array più corto di
    (n-1) elementi ma SENZA aggiustare l'indice — chi la usava indicizzandola con lo
    stesso indice di close (es. il Chandelier Exit) leggeva quindi l'ATR di (n-1) barre
    nel futuro rispetto a quella corrente. Corretto qui una volta per tutte."""
    length = len(close)
    if length < 2:
        return [None] * length
    tr = [None] * length  # tr[0] non definito (manca il close precedente)
    for i in range(1, length):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    result = [None] * length
    if length <= n:
        return result
    av = sum(tr[1:n + 1]) / n
    result[n] = round(av, 5)
    for i in range(n + 1, length):
        av = (av * (n - 1) + tr[i]) / n
        result[i] = round(av, 5)
    return result


def parabolic_sar(high: pd.Series, low: pd.Series, close: pd.Series, step=SAR_STEP, max_af=SAR_MAX):
    n = len(close)
    sar = np.zeros(n)
    trend = np.zeros(n, dtype=int)   # 1 = rialzista, -1 = ribassista
    flip = np.zeros(n, dtype=bool)
    if n < 2:
        return pd.Series(sar, index=close.index), pd.Series(trend, index=close.index), pd.Series(flip, index=close.index)

    trend[0] = 1 if close.iloc[1] >= close.iloc[0] else -1
    sar[0] = low.iloc[0] if trend[0] == 1 else high.iloc[0]
    af = step
    ep = high.iloc[0] if trend[0] == 1 else low.iloc[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]
        if trend[i - 1] == 1:
            new_sar = prev_sar + af * (ep - prev_sar)
            new_sar = min(new_sar, low.iloc[i - 1], low.iloc[i - 2] if i >= 2 else low.iloc[i - 1])
            if low.iloc[i] < new_sar:
                trend[i] = -1
                flip[i] = True
                sar[i] = ep
                ep = low.iloc[i]
                af = step
            else:
                trend[i] = 1
                sar[i] = new_sar
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + step, max_af)
        else:
            new_sar = prev_sar + af * (ep - prev_sar)
            new_sar = max(new_sar, high.iloc[i - 1], high.iloc[i - 2] if i >= 2 else high.iloc[i - 1])
            if high.iloc[i] > new_sar:
                trend[i] = 1
                flip[i] = True
                sar[i] = ep
                ep = high.iloc[i]
                af = step
            else:
                trend[i] = -1
                sar[i] = new_sar
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + step, max_af)
    return (pd.Series(sar, index=close.index),
            pd.Series(trend, index=close.index),
            pd.Series(flip, index=close.index))


def inversione_flags(c, kf, atr_series, obv, rsi14, ao):
    """Segnale Inversione di azionario (calc_inversion_signal) calcolato su ogni barra:
    stessa logica, ma letta alla barra i invece che sull'ultima. flag = almeno 2 trigger su 4."""
    n = len(c); out = [False] * n
    for i in range(n):
        m = i + 1  # lunghezza della serie vista alla barra i
        cross = False
        for ba in range(1, 6):
            j = m - ba
            if j < 1 or math.isnan(kf[j]) or math.isnan(kf[j - 1]):
                continue
            if c[j] > kf[j] and c[j - 1] <= kf[j - 1]:
                cross = True; break
        atr_rising = False
        if m >= 6:
            now = atr_series[i] if i < len(atr_series) and atr_series[i] else None
            ago = None
            for j in range(m - 2, max(m - 7, -1), -1):
                if 0 <= j < len(atr_series) and atr_series[j]:
                    ago = atr_series[j]; break
            if now and ago:
                atr_rising = bool(now > ago and c[i] > c[i - 4])
        obv_div = False
        if m > 20:
            raw = bool((c[i] - c[i - 19]) < 0 and (obv[i] - obv[i - 19]) > 0)
            flat = False
            if m >= 11 and not math.isnan(kf[i]) and not math.isnan(kf[i - 10]) and kf[i - 10] != 0:
                flat = bool(abs(kf[i] - kf[i - 10]) / abs(kf[i - 10]) < 0.02)
            obv_div = bool(raw and flat)
        rsi_imp = bool(m >= 2 and rsi14[i] < 35 and ao[i] > ao[i - 1])
        out[i] = (cross + atr_rising + obv_div + rsi_imp) >= 2
    return out


def az_serie(df):
    if df is None or len(df) == 0:
        return None

    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(df) < KAMA_SLOW_N + 5:
        return None
    close, open_, high, low, vol = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]

    er = efficiency_ratio(close, ER_N)
    kama_fast = kama(close, KAMA_FAST_N, KAMA_FAST_SC, KAMA_SLOW_N * 1)
    kama_slow = kama(close, KAMA_SLOW_N, KAMA_FAST_SC, 30)
    ao = awesome_oscillator(close)
    rvi_v = rvi(open_, high, low, close)
    rsi5 = rsi(close, RSI_FAST_N)
    rsi14 = rsi(close, RSI_SLOW_N)
    adx_v, plus_di, minus_di = adx(high, low, close)
    sar, sar_trend, sar_flip = parabolic_sar(high, low, close)
    vol_avg = vol.rolling(VOL_AVG_N).mean()
    vol_ratio = (vol / vol_avg.replace(0, np.nan)).fillna(0)
    obv = calc_obv(close, vol)

    price_above_kf = close > kama_fast

    # ---- Storico vettorizzato zona/segnale (per grafico + tabella storia segnali) ----
    grp = (price_above_kf != price_above_kf.shift()).cumsum()
    baff_series = price_above_kf.groupby(grp).cumcount() + 1
    baff_series = baff_series.where(price_above_kf, -baff_series)

    gap_pct_series = (kama_fast - kama_slow) / kama_slow.replace(0, np.nan) * 100
    sar_bullish_series = close > sar
    d1_ao = ao.diff()
    ao_improving_series = (d1_ao > 0) & (d1_ao.shift(1) > 0)

    zona_series = pd.Series(np.select(
        [
            (close > kama_fast) & (kama_fast > kama_slow),
            (close > kama_fast) & (close <= kama_slow),
            (close < kama_slow * 0.98),
            (close < kama_slow),
        ],
        ["LONG_CONF", "LONG_EARLY", "STOP", "USCITA"],
        default="NEUTRA",
    ), index=close.index)

    # ---- Uscita a Chandelier Exit (sostituisce l'uscita basata su KAMA slow) ----
    # SELL = trailing stop (Massimo da entrata - CHANDELIER_ATR_MULT x ATR14): segue il prezzo,
    #        molto meno in ritardo della vecchia regola "prezzo < KAMA slow".
    # STOP = rete di sicurezza fissa (-8% dall'entrata) per gap improvvisi: controllata PRIMA
    #        del chandelier, ha priorità se il prezzo crolla di colpo.
    # BUY2 accetta ora anche "AO in miglioramento (3 barre)" oltre a "AO>0" — BUY3 resta rigido.
    atr_series_full = calc_atr_series(high.tolist(), low.tolist(), close.tolist())
    high_list = high.tolist()
    low_list = low.tolist()
    zigzag_pivots = calc_zigzag(high_list, low_list, atr_series_full, ZIGZAG_ATR_MULT)
    # ultimo pivot "H" (massimo) confermato PRIMA (o a) ciascuna barra — per il calcolo del
    # calo reale in Super Mean Reversion, sostituisce la vecchia finestra fissa di 60 barre.
    ultimo_pivot_h_per_barra = [None] * len(high_list)
    _ultimo_h = None
    _pivot_iter = iter(zigzag_pivots)
    _prossimo = next(_pivot_iter, None)
    for idx in range(len(high_list)):
        while _prossimo is not None and _prossimo[0] <= idx:
            if _prossimo[2] == "H":
                _ultimo_h = _prossimo[1]
            _prossimo = next(_pivot_iter, None)
        ultimo_pivot_h_per_barra[idx] = _ultimo_h
    close_list_full = close.tolist()
    ao_list = ao.tolist()
    baff_list = baff_series.tolist()
    er_list = er.tolist()
    gap_list = gap_pct_series.tolist()
    sarb_list = sar_bullish_series.tolist()
    ao_impr_list = ao_improving_series.tolist()
    zona_list = zona_series.tolist()

    # Hurst rolling (60gg, ricalcolato ogni HURST_STRIDE barre) — usato sia come filtro di
    # regime per BUY2/BUY3 (trend-following: entra solo se il titolo sta davvero trendando)
    # sia dal motore Mean-Reversion più sotto (soglia opposta: solo se sta oscillando)
    rolling_hurst = [None] * len(close_list_full)
    last_h = 0.5
    for idx in range(59, len(close_list_full)):
        if (idx - 59) % HURST_STRIDE == 0:
            last_h = calc_hurst(close_list_full[max(0, idx - 59):idx + 1])
        rolling_hurst[idx] = last_h

    seg_vals = []
    chandelier_stop_series = []
    kf_list = kama_fast.tolist()
    state = "FLAT"
    entry_price = None
    highest_high = None
    for idx in range(len(close_list_full)):
        c = close_list_full[idx]
        if state == "FLAT":
            chandelier_stop_series.append(None)
            h = rolling_hurst[idx]
            regime_trend_ok = h is not None and h >= TREND_HURST_THRESHOLD
            buy3_ok = (regime_trend_ok and zona_list[idx] == "LONG_CONF" and ao_list[idx] > 0
                       and baff_list[idx] >= 3 and er_list[idx] >= 0.35 and gap_list[idx] >= 0.3 and sarb_list[idx])
            # BUY2 ridefinito: condizione diretta, non più legata a zona/baff/ER/Hurst —
            # prezzo sopra KAMA fast + SAR rialzista + AO in miglioramento, punto.
            kf_v = kf_list[idx]
            buy2_ok = (kf_v is not None and not math.isnan(kf_v) and c > kf_v
                       and sarb_list[idx] and ao_impr_list[idx])
            if buy3_ok:
                state = "LONG"; entry_price = c; highest_high = high_list[idx]
                seg_vals.append("BUY3")
            elif buy2_ok:
                state = "LONG"; entry_price = c; highest_high = high_list[idx]
                seg_vals.append("BUY2")
            else:
                seg_vals.append("HOLD")
        else:  # in posizione (LONG)
            highest_high = max(highest_high, high_list[idx])
            atr_v = atr_series_full[idx] if idx < len(atr_series_full) else None
            chand_stop = (highest_high - CHANDELIER_ATR_MULT * atr_v) if atr_v else None
            chandelier_stop_series.append(round(chand_stop, 4) if chand_stop is not None else None)
            hard_stop = entry_price * (1 - HARD_STOP_PCT)
            if c < hard_stop:
                seg_vals.append("STOP"); state = "FLAT"; entry_price = None; highest_high = None
            elif chand_stop is not None and c < chand_stop:
                seg_vals.append("SELL"); state = "FLAT"; entry_price = None; highest_high = None
            else:
                seg_vals.append("HOLD")

    segnale_series = pd.Series(seg_vals, index=close.index)

    # ---- Motore Mean-Reversion (parallelo, separato) — "compra i minimi, vendi i massimi" ----
    # Entrata: prezzo <= banda inferiore Bollinger + RSI14 ipervenduto + Hurst60 < soglia
    #          (filtro di regime BLOCCANTE: niente segnale se il titolo non sta davvero
    #          oscillando in un range, per evitare di comprare durante un vero crollo)
    # Uscita:  prezzo >= banda superiore Bollinger (massimo guadagno, come richiesto)
    bb_upper, bb_mid, bb_lower = bollinger_bands(close)
    bb_upper_l, bb_lower_l = bb_upper.tolist(), bb_lower.tolist()
    rsi14_l = rsi14.tolist()

    mr_state = "FLAT"
    mr_entry_price = None
    mr_entry_date = None
    mr_seg = []
    for idx in range(len(close_list_full)):
        c = close_list_full[idx]
        if mr_state == "FLAT":
            h, bl, r = rolling_hurst[idx], bb_lower_l[idx], rsi14_l[idx]
            if (h is not None and bl is not None and not math.isnan(bl)
                    and h < MR_HURST_THRESHOLD and c <= bl and r < MR_RSI_OVERSOLD):
                mr_state = "LONG"; mr_entry_price = c; mr_entry_date = str(df.index[idx].date())
                mr_seg.append("MR_BUY")
            else:
                mr_seg.append("MR_FLAT")
        else:
            bu = bb_upper_l[idx]
            if bu is not None and not math.isnan(bu) and c >= bu:
                mr_state = "FLAT"; mr_entry_price = None; mr_entry_date = None
                mr_seg.append("MR_SELL")
            else:
                mr_seg.append("MR_HOLD")

    mean_reversion = {
        "segnale": mr_seg[-1],
        "in_posizione": mr_state == "LONG",
        "entry_date": mr_entry_date,
        "hurst_60_rolling": rolling_hurst[-1],
        "bb_lower": round(bb_lower.iloc[-1], 4) if not math.isnan(bb_lower.iloc[-1]) else None,
        "bb_upper": round(bb_upper.iloc[-1], 4) if not math.isnan(bb_upper.iloc[-1]) else None,
        "entry_price": round(mr_entry_price, 4) if mr_entry_price is not None else None,
    }

    # ---- Motore SUPER MEAN REVERSION (parallelo, separato) — "compra dopo un forte calo" ----
    rsi5_l = rsi5.tolist()
    smr_entries = []
    smr_state = "FLAT"   # FLAT -> WATCHING -> LONG
    smr_watch_since = None
    smr_entry_price = None
    smr_entry_date = None
    smr_closed_deltas = []  # Δ% di ogni trade CHIUSO, in ordine cronologico
    for idx in range(len(close_list_full)):
        c = close_list_full[idx]
        if smr_state == "FLAT":
            ref_high = ultimo_pivot_h_per_barra[idx]
            if ref_high:
                dd = c / ref_high - 1
                r14 = rsi14_l[idx]
                if dd <= SMR_DRAWDOWN_THRESHOLD and r14 is not None and not math.isnan(r14) and r14 < SMR_RSI14_MAX:
                    smr_state = "WATCHING"; smr_watch_since = idx
        elif smr_state == "WATCHING":
            if idx - smr_watch_since > SMR_WATCH_EXPIRY:
                smr_state = "FLAT"; smr_watch_since = None
            else:
                r5_today, r5_yday = rsi5_l[idx], rsi5_l[idx - 1] if idx > 0 else None
                c_yday = close_list_full[idx - 1] if idx > 0 else None
                if (r5_today is not None and r5_yday is not None and not math.isnan(r5_today)
                        and not math.isnan(r5_yday) and r5_today > r5_yday
                        and c_yday is not None and c > c_yday):
                    smr_state = "LONG"; smr_entry_price = c; smr_entries.append(idx)
                    smr_entry_date = str(df.index[idx].date())
        else:  # LONG
            bu = bb_upper_l[idx]
            delta = c / smr_entry_price - 1 if smr_entry_price else 0
            if (bu is not None and not math.isnan(bu) and c >= bu) or delta <= SMR_STOP_PCT:
                smr_closed_deltas.append(round(delta * 100, 2))
                smr_state = "FLAT"; smr_entry_price = None; smr_entry_date = None; smr_watch_since = None

    smr_trade_chiusi = len(smr_closed_deltas)
    smr_pct_vincenti = round(sum(1 for x in smr_closed_deltas if x > 0) / smr_trade_chiusi * 100, 1) if smr_trade_chiusi else None
    smr_delta_medio_pct = round(sum(smr_closed_deltas) / smr_trade_chiusi, 2) if smr_trade_chiusi else None
    smr_rendimento_cumulato = None
    if smr_trade_chiusi:
        cum = 1.0
        for x in smr_closed_deltas:
            cum *= (1 + x / 100)
        smr_rendimento_cumulato = round((cum - 1) * 100, 2)

    super_mean_reversion = {
        "segnale": smr_state,  # "FLAT" | "WATCHING" | "LONG"
        "in_posizione": smr_state == "LONG",
        "in_osservazione": smr_state == "WATCHING",
        "entry_date": smr_entry_date,
        "entry_price": round(smr_entry_price, 4) if smr_entry_price is not None else None,
        "trade_chiusi": smr_trade_chiusi,
        "pct_vincenti": smr_pct_vincenti,
        "delta_medio_pct": smr_delta_medio_pct,
        "rendimento_cumulato": smr_rendimento_cumulato,
    }

    # Super Best Buy storico (stesso calcolo di azionario: SAR rialzista, flip <= 2 barre, AO in miglioramento)
    sar_flip_v = sar_flip.values
    bars_since_flip_arr = np.zeros(len(sar_flip_v), dtype=int)
    last_flip_idx = None
    for fidx, is_flip in enumerate(sar_flip_v):
        if is_flip:
            last_flip_idx = fidx
        bars_since_flip_arr[fidx] = (fidx - last_flip_idx) if last_flip_idx is not None else 9999
    sbb = (sar_bullish_series.values & (bars_since_flip_arr <= 2) & ao_improving_series.fillna(False).values)
    inv = inversione_flags(close.values, kama_fast.values, atr_series_full, obv.values, rsi14.values, ao.values)
    return {
        "dates": df.index, "close": close.values,
        "buy": seg_vals, "mr": mr_seg, "smr_entries": smr_entries,
        "sbb": [bool(x) for x in sbb], "inv": inv,
    }
