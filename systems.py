"""Sistemi di acquisto di big. Ogni sistema e' un'entry di SYSTEMS: per aggiungerne uno basta una funzione
ev_xxx(df, ctx) -> lista di (indice_barra, prezzo, etichetta, nota) e una riga in SYSTEMS.
Si disegnano solo gli ingressi long: le uscite servono solo a sapere quando un sistema puo' rientrare."""
import numpy as np
import pandas as pd
import az_engine as A
import core_engine as G

COOLDOWN_GIORNI = 10   # come il ledger di azionario: stesso ticker + stesso sistema, niente doppioni entro 10 giorni
MINTICK = 0.01
MOM_LEN = 12
SOGLIE = {"rsi": (45, 55), "stoch": (45, 55), "adx": 25, "aggregato_compra": 20}   # come investing (regole.json)


# ---------------- Momentum (strategia TradingView "Momentum strategy", length 12) ----------------
def ev_momentum(df, ctx):
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    n = len(c); mom0 = np.full(n, np.nan); mom0[MOM_LEN:] = c[MOM_LEN:] - c[:-MOM_LEN]
    mom1 = np.full(n, np.nan); mom1[1:] = mom0[1:] - mom0[:-1]
    out, pos, pend = [], 0, None
    for i in range(n):
        if pend is not None:                       # ordine stop messo alla chiusura precedente
            side, stop = pend
            if side == "L" and h[i] >= stop:
                if pos != 1: out.append((i, float(max(o[i], stop)), "MOM", "Stop buy alto+mintick"))
                pos = 1
            elif side == "S" and l[i] <= stop:
                pos = -1                           # il lato short non si disegna, ma chiude il long come in TradingView
        if mom0[i] > 0 and mom1[i] > 0: pend = ("L", h[i] + MINTICK)
        elif mom0[i] < 0 and mom1[i] < 0: pend = ("S", l[i] - MINTICK)
        else: pend = None
    return out


# ---------------- Compra dal (G): saldo aggregato indicatori+medie di investing >= Compra ----------------
def _wil(s, n): return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
def _vote(v, lo, hi): return np.where(np.isnan(v), np.nan, np.where(v >= hi, 1.0, np.where(v <= lo, -1.0, 0.0)))
def _sgn(v): return np.where(np.isnan(v), np.nan, np.sign(v))

def saldo_series(d):
    """Saldo (-100..+100) di ogni giorno: stessi indicatori e stessi voti di investing/compute_tech.py calc()."""
    h, l, c = d["High"], d["Low"], d["Close"]; pc = c.shift(1); up = c.diff()
    rsi = 100 - 100 / (1 + _wil(up.clip(lower=0), 14) / _wil((-up).clip(lower=0), 14))
    k = (100 * (c - l.rolling(9).min()) / (h.rolling(9).max() - l.rolling(9).min())).rolling(6).mean()
    srsi = 100 * (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())
    mac = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    diff = mac - mac.ewm(span=9, adjust=False).mean()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1); atr = _wil(tr, 14)
    pdm = h.diff().where((h.diff() > -l.diff()) & (h.diff() > 0), 0.0); mdm = (-l.diff()).where((-l.diff() > h.diff()) & (-l.diff() > 0), 0.0)
    pdi = 100 * _wil(pdm, 14) / atr; mdi = 100 * _wil(mdm, 14) / atr; adx = _wil(100 * (pdi - mdi).abs() / (pdi + mdi), 14)
    tp = (h + l + c) / 3; sma = tp.rolling(14).mean()
    mad = tp.rolling(14).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma) / (0.015 * mad)
    roc = 100 * (c / c.shift(12) - 1)
    lo_ = pd.concat([l, pc], axis=1).min(axis=1); bp = c - lo_; t2 = pd.concat([h, pc], axis=1).max(axis=1) - lo_
    av = lambda n: bp.rolling(n).sum() / t2.rolling(n).sum(); uo = 100 * (4 * av(7) + 2 * av(14) + av(28)) / 7
    e13 = c.ewm(span=13, adjust=False).mean(); bull = h - e13; bear = l - e13
    el = pd.Series(0.0, index=c.index)
    el[(e13 > e13.shift(1)) & (bear < 0) & (bear > bear.shift(1))] = 1.0
    el[(e13 < e13.shift(1)) & (bull > 0) & (bull < bull.shift(1)) & ~((e13 > e13.shift(1)) & (bear < 0) & (bear > bear.shift(1)))] = -1.0
    adx_v = np.where(np.isnan(adx.values), np.nan, np.where(adx.values <= SOGLIE["adx"], 0.0, np.where(pdi.values > mdi.values, 1.0, -1.0)))
    V = [_vote(rsi.values, *SOGLIE["rsi"]), _vote(k.values, *SOGLIE["stoch"]), _vote(srsi.values, *SOGLIE["stoch"]),
         _sgn(diff.values), adx_v, _sgn(cci.values), _sgn(roc.values), _sgn((uo - 50).values), el.values]
    for n in (5, 10, 20, 50, 100, 200):
        sm = c.rolling(n).mean(); em = c.ewm(span=n, adjust=False, min_periods=n).mean()
        V.append(np.where(np.isnan(sm.values), np.nan, np.where(c.values > sm.values, 1.0, -1.0)))
        V.append(np.where(np.isnan(em.values), np.nan, np.where(c.values > em.values, 1.0, -1.0)))
    M = np.vstack(V); cnt = (~np.isnan(M)).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        saldo = (np.nansum(M == 1, axis=0) - np.nansum(M == -1, axis=0)) / np.where(cnt == 0, np.nan, cnt) * 100
    saldo = saldo.astype(float); saldo[:29] = np.nan   # investing richiede almeno 30 barre
    return saldo

def ev_compra_dal(df, ctx):
    s = saldo_series(df); c = df["Close"].values; out = []
    ok = np.nan_to_num(s, nan=-999) >= SOGLIE["aggregato_compra"]
    for i in range(1, len(s)):
        if ok[i] and not ok[i - 1]: out.append((i, float(c[i]), "CD", "Saldo %+.0f%%" % s[i]))
    return out


# ---------------- Core: BUY1/BUY2/BUY3 (inizio di una fase di acquisto) ----------------
def core_segnali(df):
    hi, lo, cl = (df[k].round(4).tolist() for k in ("High", "Low", "Close"))
    ka = G.calc_kama(cl); _, sb = G.calc_sar_array(hi, lo); ao = G.calc_ao_array(hi, lo)
    return G.calc_segnale_array(cl, ka, G.calc_er_array(cl), G.calc_baffetti_array(hi, lo), G.calc_ao_improving_array(ao), sb,
                                G.calc_cross_days_array(cl, ka), G.calc_mm_align_array(cl), G.calc_rsi_array(cl))

def ev_core(df, ctx):
    seg = ctx["core_seg"]; c = df["Close"].values; out = []
    for i in range(1, len(seg)):
        if seg[i] and seg[i].startswith("BUY") and not (seg[i - 1] or "").startswith("BUY"):
            out.append((i, float(c[i]), "CORE " + seg[i], seg[i]))
    return out


# ---------------- Motori di azionario ----------------
def ev_az_buy(df, ctx):
    r = ctx["az"]; return [(i, float(r["close"][i]), "AZ " + s, s) for i, s in enumerate(r["buy"]) if s in ("BUY2", "BUY3")]

def _fronti(flags, close, etic, nota):
    return [(i, float(close[i]), etic, nota) for i in range(len(flags)) if flags[i] and (i == 0 or not flags[i - 1])]

def ev_inversione(df, ctx): r = ctx["az"]; return _fronti(r["inv"], r["close"], "INV", "Inversione (2+ trigger)")
def ev_sbb(df, ctx): r = ctx["az"]; return _fronti(r["sbb"], r["close"], "SBB", "Flip SAR + AO in miglioramento")
def ev_mr(df, ctx): r = ctx["az"]; return [(i, float(r["close"][i]), "MR", "Banda Bollinger + RSI<30") for i, s in enumerate(r["mr"]) if s == "MR_BUY"]
def ev_smr(df, ctx): r = ctx["az"]; return [(i, float(r["close"][i]), "SMR", "Calo >=25% + conferma rimbalzo") for i in r["smr_entries"]]


# ---------------- Registro ----------------
SYSTEMS = [
 dict(id="mom", nome="Momentum", colore="#1d4ed8", fn=ev_momentum,
      regole=["Momentum = chiusura meno chiusura di 12 barre prima (mom0); mom1 = variazione di mom0 rispetto alla barra precedente.",
              "Se mom0>0 e mom1>0 si piazza un ordine stop buy a massimo della barra + mintick (0,01), valido per la barra successiva; altrimenti si cancella.",
              "La freccia e' sulla barra in cui lo stop scatta, al prezzo massimo tra apertura e stop (gap). Non c'e' freccia se si e' gia' long.",
              "Il lato short non si disegna ma, come in TradingView, chiude il long quando scatta: dopo uno short si puo' avere un nuovo ingresso."]),
 dict(id="cd", nome="Compra dal (G)", colore="#15803d", fn=ev_compra_dal,
      regole=["Saldo giornaliero = (indicatori Compra - Vendi) / indicatori con voto, su 9 indicatori e 12 medie (SMA/EMA 5-200) come nella pagina investing.",
              "Soglie voti: RSI e stocastico 45/55, ADX 25, saldo aggregato >= +20% = Compra.",
              "La freccia e' sul primo giorno in cui il saldo sale a Compra dopo essere stato sotto."]),
 dict(id="core", nome="Core BUY1/2/3", colore="#7e22ce", fn=ev_core,
      regole=["Motore del repo core. BUY1: SAR rialzista, incrocio KAMA entro 3 barre, AO in miglioramento.",
              "BUY2: prezzo sopra KAMA e almeno 2 barre sopra. BUY3: sopra KAMA, ER>=0,50, almeno 3 barre, medie allineate.",
              "La freccia e' sul primo giorno di una fase BUY (dopo WATCH, EXIT o MEAN REV). 'Acq. nuovo' = fase iniziata da 3 barre o meno."]),
 dict(id="azb", nome="Az BUY2/BUY3", colore="#ea580c", fn=ev_az_buy,
      regole=["Motore di azionario (regole diverse dal BUY di core). BUY2: prezzo sopra KAMA veloce, SAR rialzista, AO in miglioramento.",
              "BUY3: come BUY2 piu' zona LONG confermata, 3 barre sopra KAMA, ER>=0,35, gap KAMA>=0,3%, Hurst 60>=0,55.",
              "Ingresso solo da flat: dopo un ingresso il motore resta in posizione fino a Chandelier (massimo - 3 ATR) o stop -8%."]),
 dict(id="inv", nome="Inversione", colore="#db2777", fn=ev_inversione,
      regole=["Almeno 2 trigger su 4: incrocio sopra KAMA negli ultimi 5 giorni; ATR e prezzo in salita; divergenza OBV con KAMA piatta; RSI14<35 con AO in miglioramento.",
              "La freccia e' sul primo giorno in cui il segnale si accende."]),
 dict(id="sbb", nome="Super Best Buy", colore="#0891b2", fn=ev_sbb,
      regole=["SAR rialzista con flip da 2 barre o meno e AO in miglioramento (2 rialzi consecutivi). Freccia sul primo giorno."]),
 dict(id="mr", nome="Mean Reversion", colore="#854d0e", fn=ev_mr,
      regole=["Prezzo sotto banda di Bollinger inferiore (20, 2), RSI14<30 e Hurst 60<0,45 (titolo che oscilla). Uscita alla banda superiore."]),
 dict(id="smr", nome="Super Mean Reversion", colore="#475569", fn=ev_smr,
      regole=["Fase 1: calo >=25% dall'ultimo massimo ZigZag e RSI14<35. Fase 2 (entro 15 barre): RSI5 in salita e chiusura in rialzo = ingresso.",
              "Stop -15% dall'ingresso, uscita alla banda di Bollinger superiore."]),
]
RUMORE = ("Il segnale core scatta spesso: tra un'entrata e l'altra passa poco tempo. Per questo vale il cooldown di %d giorni per ticker e sistema." % COOLDOWN_GIORNI)

def calcola(df):
    """df: Open High Low Close Volume, indice date. Ritorna (eventi, meta).
    eventi = [(data, sys_id, etichetta, prezzo, nota)]; meta = stato attuale del segnale core."""
    ctx = {"core_seg": core_segnali(df), "az": A.az_serie(df)}
    if ctx["az"] is None: ctx["az"] = {"close": df["Close"].values, "buy": [], "mr": [], "smr_entries": [], "sbb": [], "inv": []}
    seg = ctx["core_seg"]; n = len(seg); i = n - 1
    core_ora = seg[-1]; nuovo = False; inizio = None
    if core_ora and core_ora.startswith("BUY"):
        while i > 0 and (seg[i - 1] or "").startswith("BUY"): i -= 1
        nuovo = (n - 1 - i) <= 3; inizio = str(df.index[i].date())
    out = []; idx = df.index
    for s in SYSTEMS:
        last = None
        for i, price, etic, nota in sorted(s["fn"](df, ctx)):
            d = idx[i]
            if last is not None and (d - last).days < COOLDOWN_GIORNI: continue
            last = d
            if s["id"] == "core" and nuovo and str(d.date()) == inizio: etic += " nuovo"; nota += " - Acq. nuovo"
            out.append((str(d.date()), s["id"], etic, round(price, 4), nota))
    return sorted(out), {"core": core_ora, "nuovo": nuovo}
