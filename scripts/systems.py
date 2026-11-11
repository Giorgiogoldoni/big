"""Sistemi di acquisto di big. Ogni sistema e' un'entry di SYSTEMS: per aggiungerne uno basta una funzione
ev_xxx(df, ctx) -> lista di (indice_barra, prezzo, etichetta, nota) e una riga in SYSTEMS.
Gli ingressi long sono le frecce in su. Le uscite si registrano con _esci(ctx, id, ...) dentro ev_xxx (uscite del sistema) oppure con una
regola che parte dall'ingresso (USCITE, uscita()); da ingresso e uscita nascono le posizioni (trades) con il guadagno in %."""
import numpy as np
import pandas as pd
import az_engine as A
import core_engine as G

CHAND_ATR = 3.0          # stop comune: massimo dall'ingresso - 3 x ATR14 (stesso moltiplicatore del motore di azionario)
TEMPO_MAX_INV = 20       # Inversione: uscita a tempo se la KAMA veloce non la chiude prima
USCITA_RECENTE_GG = 5    # stato ESCI: posizione Trend chiusa negli ultimi 5 giorni di calendario
FAM_TREND = ("core", "azb", "sbb", "psar", "adx", "ema")
COOLDOWN_GIORNI = 10   # come il ledger di azionario: stesso ticker + stesso sistema, niente doppioni entro 10 giorni
MINTICK = 0.01
MOM_LEN = 12
ADX_MIN = 20             # ADX: soglia di forza del trend (investing usa 25 per il voto, qui 20 per entrare prima)
EMA_FAST, EMA_SLOW = 20, 50
KAMAL_LEN = 5            # Kamal 5: barre del canale (massimo/minimo delle 5 barre precedenti, la corrente esclusa)
def _esci(ctx, sid, i, prezzo, motivo):
    ctx.setdefault("uscite", {}).setdefault(sid, []).append((i, float(prezzo), motivo))


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
                if pos == 1: _esci(ctx, "mom", i, min(o[i], stop), "Stop short (mom<0)")
                pos = -1                           # il lato short non si disegna, ma chiude il long come in TradingView
        if mom0[i] > 0 and mom1[i] > 0: pend = ("L", h[i] + MINTICK)
        elif mom0[i] < 0 and mom1[i] < 0: pend = ("S", l[i] - MINTICK)
        else: pend = None
    return out


# ---------------- Parabolic SAR strategy (TradingView, 0,02 / 0,02 / 0,2) ----------------
def ev_psar(df, ctx, start=0.02, inc=0.02, mx=0.2):
    """Port letterale dello script Pine 'Parabolic SAR strategy'. Ingresso long = scatta lo stop di acquisto sul SAR della barra dopo
    (ParLE). Il lato short (ParSE) non si disegna ma chiude il long, come in TradingView."""
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    n = len(c); out = []; pos = 0; pend = None
    up = False; EP = AF = nxt = None; AF = start
    for i in range(n):
        if pend is not None:                                   # ordine stop messo alla chiusura precedente
            side, stop = pend
            if side == "L" and h[i] >= stop:
                if pos != 1: out.append((i, float(max(o[i], stop)), "PSAR", "Stop buy sul SAR (ParLE)"))
                pos = 1
            elif side == "S" and l[i] <= stop:
                if pos == 1: _esci(ctx, "psar", i, min(o[i], stop), "Stop short (ParSE)")
                pos = -1
        pend = None
        if i == 0: continue
        first = False; sar = nxt
        if i == 1:
            if c[1] > c[0]: up = True; EP = h[1]; psar = l[0]; pep = h[1]
            else: up = False; EP = l[1]; psar = h[0]; pep = l[1]
            first = True; sar = psar + start * (pep - psar)
        if up:
            if sar > l[i]: first = True; up = False; sar = max(EP, h[i]); EP = l[i]; AF = start
        else:
            if sar < h[i]: first = True; up = True; sar = min(EP, l[i]); EP = h[i]; AF = start
        if not first:
            if up:
                if h[i] > EP: EP = h[i]; AF = min(AF + inc, mx)
            else:
                if l[i] < EP: EP = l[i]; AF = min(AF + inc, mx)
        if up:
            sar = min(sar, l[i - 1])
            if i > 1: sar = min(sar, l[i - 2])
        else:
            sar = max(sar, h[i - 1])
            if i > 1: sar = max(sar, h[i - 2])
        nxt = sar + AF * (EP - sar)
        pend = ("S", nxt) if up else ("L", nxt)
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
        if ok[i - 1] and not ok[i]: _esci(ctx, "cd", i, c[i], "Saldo sotto Compra")
    return out


# ---------------- Core: BUY1/BUY2/BUY3 (inizio di una fase di acquisto) ----------------
def core_segnali(df):
    hi, lo, cl = (df[k].round(4).tolist() for k in ("High", "Low", "Close"))
    ka = G.calc_kama(cl); _, sb = G.calc_sar_array(hi, lo); ao = G.calc_ao_array(hi, lo)
    return G.calc_segnale_array(cl, ka, G.calc_er_array(cl), G.calc_baffetti_array(hi, lo), G.calc_ao_improving_array(ao), sb,
                                G.calc_cross_days_array(cl, ka), G.calc_mm_align_array(cl), G.calc_rsi_array(cl))

def ev_core(df, ctx):
    seg = ctx["core_seg"]; c = df["Close"].values; out = []
    for i in range(len(seg)):
        if seg[i] in ("EXIT1", "EXIT2"): _esci(ctx, "core", i, c[i], seg[i])
    for i in range(1, len(seg)):
        if seg[i] and seg[i].startswith("BUY") and not (seg[i - 1] or "").startswith("BUY"):
            out.append((i, float(c[i]), "CORE " + seg[i], seg[i]))
    return out


# ---------------- Motori di azionario ----------------
def ev_az_buy(df, ctx):
    r = ctx["az"]
    for i, s in enumerate(r["buy"]):
        if s in ("SELL", "STOP"): _esci(ctx, "azb", i, r["close"][i], "Chandelier 3 ATR" if s == "SELL" else "Stop -8%")
    return [(i, float(r["close"][i]), "AZ " + s, s) for i, s in enumerate(r["buy"]) if s in ("BUY2", "BUY3")]

def _fronti(flags, close, etic, nota):
    return [(i, float(close[i]), etic, nota) for i in range(len(flags)) if flags[i] and (i == 0 or not flags[i - 1])]

def ev_inversione(df, ctx): r = ctx["az"]; return _fronti(r["inv"], r["close"], "INV", "Inversione (2+ trigger)")
def ev_sbb(df, ctx):
    r = ctx["az"]
    for i, b in enumerate(r.get("sarb", [])):
        if not b: _esci(ctx, "sbb", i, r["close"][i], "SAR ribassista")
    return _fronti(r["sbb"], r["close"], "SBB", "Flip SAR + AO in miglioramento")
def ev_mr(df, ctx):
    r = ctx["az"]
    for i, s in enumerate(r["mr"]):
        if s == "MR_SELL": _esci(ctx, "mr", i, r["close"][i], "Banda Bollinger sup.")
    return [(i, float(r["close"][i]), "MR", "Banda Bollinger + RSI<30") for i, s in enumerate(r["mr"]) if s == "MR_BUY"]
def ev_smr(df, ctx):
    r = ctx["az"]
    for i, motivo in r.get("smr_exits", []): _esci(ctx, "smr", i, r["close"][i], motivo)
    return [(i, float(r["close"][i]), "SMR", "Calo >=25% + conferma rimbalzo") for i in r["smr_entries"]]


# ---------------- ADX / DMI (Wilder, 14): +DI sopra -DI con trend abbastanza forte ----------------
def _dmi(df):
    h, l, c = df["High"], df["Low"], df["Close"]; pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1); atr = _wil(tr, 14)
    pdm = h.diff().where((h.diff() > -l.diff()) & (h.diff() > 0), 0.0); mdm = (-l.diff()).where((-l.diff() > h.diff()) & (-l.diff() > 0), 0.0)
    pdi = 100 * _wil(pdm, 14) / atr; mdi = 100 * _wil(mdm, 14) / atr
    return pdi.values, mdi.values, _wil(100 * (pdi - mdi).abs() / (pdi + mdi), 14).values

def ev_adx(df, ctx):
    pdi, mdi, adx = _dmi(df); c = df["Close"].values; out = []
    for i in range(1, len(c)):
        if np.isnan(pdi[i]) or np.isnan(mdi[i]) or np.isnan(pdi[i - 1]) or np.isnan(mdi[i - 1]): continue
        if pdi[i] > mdi[i] and pdi[i - 1] <= mdi[i - 1]:
            if not np.isnan(adx[i]) and adx[i] >= ADX_MIN: out.append((i, float(c[i]), "ADX", "+DI sopra -DI, ADX %.0f" % adx[i]))
        elif pdi[i] < mdi[i] and pdi[i - 1] >= mdi[i - 1]:
            _esci(ctx, "adx", i, c[i], "-DI sopra +DI")
    return out


# ---------------- Incrocio medie mobili: EMA 20 sopra EMA 50 ----------------
def ev_ema(df, ctx):
    c = df["Close"]
    f = c.ewm(span=EMA_FAST, adjust=False, min_periods=EMA_SLOW).mean().values
    s = c.ewm(span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW).mean().values; cv = c.values; out = []
    for i in range(1, len(cv)):
        if np.isnan(f[i]) or np.isnan(s[i]) or np.isnan(f[i - 1]) or np.isnan(s[i - 1]): continue
        if f[i] > s[i] and f[i - 1] <= s[i - 1]: out.append((i, float(cv[i]), "EMA", "EMA %d sopra EMA %d" % (EMA_FAST, EMA_SLOW)))
        elif f[i] < s[i] and f[i - 1] >= s[i - 1]: _esci(ctx, "ema", i, cv[i], "EMA %d sotto EMA %d" % (EMA_FAST, EMA_SLOW))
    return out


# ---------------- Kamal 5 (TradingView "Kamal 5 Tick Trading Setup"): rottura del canale a 5 barre ----------------
def ev_kamal5(df, ctx):
    """Port letterale dello script Pine. Chiusura sopra il massimo delle 5 barre precedenti = BUY (se non si e' gia' long); chiusura sotto il
    minimo delle 5 barre precedenti = SELL (se non si e' gia' short). Il SELL non si disegna ma chiude il long, come in TradingView."""
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    out, pos = [], 0
    for i in range(KAMAL_LEN, len(c)):
        hh = h[i - KAMAL_LEN:i].max(); ll = l[i - KAMAL_LEN:i].min()
        buy = c[i] > hh and pos != 1
        sell = c[i] < ll and pos != -1
        if buy:
            pos = 1; out.append((i, float(c[i]), "K5", "Chiusura sopra il massimo di %d barre (%.6g)" % (KAMAL_LEN, hh)))
        if sell:
            pos = -1; _esci(ctx, "k5", i, c[i], "Chiusura sotto il minimo di %d barre" % KAMAL_LEN)
    return out


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
 dict(id="psar", nome="Parabolic SAR", colore="#ca8a04", fn=ev_psar,
      regole=["Strategia TradingView 'Parabolic SAR strategy' (partenza 0,02, incremento 0,02, massimo 0,2), copiata dallo script Pine.",
              "Con SAR ribassista si piazza uno stop di acquisto sul SAR della barra dopo. La freccia e' sulla barra in cui scatta, al maggiore tra apertura e SAR.",
              "Il lato short non si disegna ma chiude il long come in TradingView. Nella confluenza e' nella famiglia Trend."]),
 dict(id="adx", nome="ADX", colore="#65a30d", fn=ev_adx,
      regole=["DMI di Wilder a 14 periodi (+DI, -DI, ADX). La freccia e' sul giorno in cui +DI incrocia sopra -DI con ADX >= 20 (trend abbastanza forte).",
              "Se l'incrocio avviene con ADX sotto 20 non c'e' freccia. Nella confluenza e' nella famiglia Trend."]),
 dict(id="ema", nome="Incrocio EMA 20/50", colore="#6366f1", fn=ev_ema,
      regole=["Media esponenziale a 20 giorni che incrocia al rialzo quella a 50 giorni, alla chiusura. Si parte dopo 50 barre di dati.",
              "Nella confluenza e' nella famiglia Trend. E' un segnale in ritardo e dà molti falsi segnali nei periodi laterali."]),
 dict(id="k5", nome="Kamal 5", colore="#0f766e", fn=ev_kamal5,
      regole=["Strategia TradingView 'Kamal 5 Tick Trading Setup', copiata dallo script Pine. Nonostante il nome lavora sulle barre giornaliere, non sui tick.",
              "Canale a 5 barre: massimo e minimo delle 5 barre precedenti (la barra corrente e' esclusa). BUY = chiusura sopra il massimo, solo se non si e' gia' long.",
              "SELL = chiusura sotto il minimo, solo se non si e' gia' short. Il SELL non si disegna ma chiude il long: dopo un SELL si puo' avere un nuovo BUY.",
              "Nella confluenza e' una famiglia a se' (Breakout). Segnale molto sensibile: nei periodi laterali da' molti falsi segnali, spesso per pochi centesimi di scarto."]),
 dict(id="mr", nome="Mean Reversion", colore="#854d0e", fn=ev_mr,
      regole=["Prezzo sotto banda di Bollinger inferiore (20, 2), RSI14<30 e Hurst 60<0,45 (titolo che oscilla). Uscita alla banda superiore."]),
 dict(id="smr", nome="Super Mean Reversion", colore="#475569", fn=ev_smr,
      regole=["Fase 1: calo >=25% dall'ultimo massimo ZigZag e RSI14<35. Fase 2 (entro 15 barre): RSI5 in salita e chiusura in rialzo = ingresso.",
              "Stop -15% dall'ingresso, uscita alla banda di Bollinger superiore."]),
]
# Uscite: testo per le pagine/PDF e stop comune. chand=True: oltre all'uscita del sistema vale lo stop Chandelier (massimo dall'ingresso - 3 ATR14, chiusura sotto).
# I sistemi che hanno gia' uno stop proprio (azb, smr) o che sono copie fedeli di TradingView (mom, psar) non hanno lo stop comune.
USCITE = {
 "mom":  dict(chand=False, testo="Uscita: scatta lo stop short dello script (mom0<0 e mom1<0); chiude il long al minore tra apertura e stop. Nessuno stop aggiuntivo (copia fedele di TradingView)."),
 "cd":   dict(chand=True,  testo="Uscita: il saldo scende sotto +20% (non e' piu' Compra), alla chiusura. In piu' stop Chandelier."),
 "core": dict(chand=True,  testo="Uscita: primo EXIT1 o EXIT2 del motore core (SAR ribassista senza condizioni BUY), alla chiusura. WATCH non e' un'uscita. In piu' stop Chandelier."),
 "azb":  dict(chand=False, testo="Uscita: Chandelier (massimo dall'ingresso - 3 ATR14) o stop -8% dall'ingresso, come nel motore di azionario."),
 "inv":  dict(chand=True,  testo="Uscita: prima chiusura sotto la KAMA veloce dopo essere stata sopra, oppure dopo %d barre. In piu' stop Chandelier." % TEMPO_MAX_INV),
 "sbb":  dict(chand=True,  testo="Uscita: il SAR torna ribassista (chiusura sotto il SAR). In piu' stop Chandelier."),
 "adx":  dict(chand=True,  testo="Uscita: -DI torna sopra +DI, alla chiusura (anche con ADX basso). In piu' stop Chandelier."),
 "ema":  dict(chand=True,  testo="Uscita: la EMA 20 scende sotto la EMA 50, alla chiusura. In piu' stop Chandelier."),
 "k5":   dict(chand=False, testo="Uscita: la chiusura scende sotto il minimo delle %d barre precedenti (SELL dello script), alla chiusura. Nessuno stop aggiuntivo (copia fedele di TradingView)." % KAMAL_LEN),
 "psar": dict(chand=False, testo="Uscita: scatta lo stop short dello script (ParSE); chiude il long al minore tra apertura e stop. Nessuno stop aggiuntivo (copia fedele di TradingView)."),
 "mr":   dict(chand=True,  testo="Uscita: chiusura sopra la banda di Bollinger superiore. In piu' stop Chandelier (il motore originale non ha stop)."),
 "smr":  dict(chand=False, testo="Uscita: chiusura sopra la banda di Bollinger superiore oppure stop -15% dall'ingresso, come nel motore di azionario."),
}
# Indicatori solo grafici (nessuna freccia, nessuna posizione): compaiono nel PDF delle regole.
REGOLE_GRAFICO = [
    "KAMA Cloud (nuvola): replica dello script TradingView 'KAMA Cloud' di mertenes3. Solo grafico, spenta di default (casella sopra il grafico); non genera segnali ne' posizioni.",
    "Si calcolano 19 KAMA su ohlc4 con lunghezze 5, 10, 15 ... 90 e 100 (costanti veloce 0,666 e lenta 0,0645). Distanza media = media delle 18 distanze relative tra KAMA adiacenti: (KAMA corta - KAMA lunga) / KAMA lunga.",
    "Linea 1 = KAMA 100 sulla chiusura. Linea 2 = linea 1 x (1 + 10 x distanza media). Linea 3 = KAMA 10 calcolata sulla linea 2.",
    "Riempimento tra linea 1 e linea 2: verde se la distanza media e' positiva, rosso se negativa. Riempimento tra linea 2 e linea 3: verde se la linea 2 e' sopra la linea 3, rosso altrimenti.",
    "Differenza dallo script: ogni KAMA parte dal prezzo della prima barra utile invece che da zero, perche' il grafico ha solo 2 anni di storia. La nuvola compare dopo 100 barre e coincide con TradingView solo dopo circa un anno di storia.",
]
REGOLE_USCITE = [
    "Posizione = un ingresso e la sua uscita. Per ogni sistema una posizione alla volta: una freccia che compare mentre la posizione e' ancora aperta non apre un'altra posizione.",
    "Guadagno % = prezzo di uscita / prezzo di ingresso - 1; se la posizione e' aperta, ultima chiusura / prezzo di ingresso - 1. Lordo, senza commissioni, senza simulare nulla: non e' un backtest.",
    "Stop Chandelier comune (dove indicato): massimo dall'ingresso - %g x ATR14; esce alla prima chiusura sotto lo stop." % CHAND_ATR,
    "Stato dell'ETF sulla famiglia Trend (Core, Az BUY, Super Best Buy, Parabolic SAR, ADX, Incrocio EMA): TIENI = almeno una posizione Trend aperta (guadagno della piu' vecchia); ESCI = nessuna aperta e una chiusa negli ultimi %d giorni (guadagno dell'ultima chiusa); FUORI = altrimenti." % USCITA_RECENTE_GG,
]
RUMORE = ("Il segnale core scatta spesso: tra un'entrata e l'altra passa poco tempo. Per questo vale il cooldown di %d giorni per ticker e sistema." % COOLDOWN_GIORNI)

def _chand(ctx, i0):
    h, c, atr = ctx["hi"], ctx["cl"], ctx["atr"]; hh = h[i0]
    for j in range(i0 + 1, len(c)):
        hh = max(hh, h[j]); a = atr[j]
        if a and c[j] < hh - CHAND_ATR * a: return (j, float(c[j]), "Chandelier %g ATR" % CHAND_ATR)
    return None


def _uscita_inv(ctx, i0):
    c, kf = ctx["cl"], ctx["az"].get("kf") or []
    ok = lambda j: j < len(kf) and kf[j] is not None
    sopra = ok(i0) and c[i0] > kf[i0]
    for j in range(i0 + 1, len(c)):
        if ok(j):
            if c[j] > kf[j]: sopra = True
            elif sopra: return (j, float(c[j]), "Chiusura sotto KAMA veloce")
        if j - i0 >= TEMPO_MAX_INV: return (j, float(c[j]), "Tempo massimo %d barre" % TEMPO_MAX_INV)
    return None


def uscita(sid, ctx, i0):
    """Prima uscita dopo la barra di ingresso i0: (indice, prezzo, motivo) oppure None se la posizione e' ancora aperta."""
    cand = [e for e in [next((e for e in ctx.get("uscite", {}).get(sid, []) if e[0] > i0), None),
                        _uscita_inv(ctx, i0) if sid == "inv" else None,
                        _chand(ctx, i0) if USCITE[sid]["chand"] else None] if e]
    return min(cand, key=lambda e: e[0]) if cand else None


def posizioni(df, ctx, ingressi):
    """ingressi = eventi gia' filtrati dal cooldown. Ritorna [[sistema, data_in, prezzo_in, data_out|None, prezzo_out|None, motivo|None, guadagno%]]."""
    idx = df.index; pos = {str(d.date()): k for k, d in enumerate(idx)}; ult = float(df["Close"].values[-1]); out = []
    for s in SYSTEMS:
        fine = None; aperto = False
        for e in ingressi:
            if e[1] != s["id"]: continue
            i0 = pos[e[0]]
            if aperto and (fine is None or fine[0] > i0): continue      # ancora in posizione: freccia ripetuta, non e' un nuovo trade
            fine = uscita(s["id"], ctx, i0); aperto = True
            if fine: out.append([s["id"], e[0], round(e[3], 4), str(idx[fine[0]].date()), round(fine[1], 4), fine[2], round((fine[1] / e[3] - 1) * 100, 2)])
            else: out.append([s["id"], e[0], round(e[3], 4), None, None, None, round((ult / e[3] - 1) * 100, 2)])
    return out


def stato_etf(tr, ult):
    """Stato sulla famiglia Trend: ["TIENI"|"ESCI"|"FUORI", guadagno%|None, data_ingresso|None, data_uscita|None, n_aperte]."""
    t = [x for x in tr if x[0] in FAM_TREND]
    ap = sorted([x for x in t if x[3] is None], key=lambda x: x[1])
    if ap: return ["TIENI", ap[0][6], ap[0][1], None, len(ap)]
    gg = lambda a, b: (pd.Timestamp(b) - pd.Timestamp(a)).days
    ch = sorted([x for x in t if x[3] and gg(x[3], ult) <= USCITA_RECENTE_GG], key=lambda x: x[3])
    if ch: return ["ESCI", ch[-1][6], ch[-1][1], ch[-1][3], 0]
    return ["FUORI", None, None, None, 0]


def calcola(df):
    """df: Open High Low Close Volume, indice date. Ritorna (eventi, meta).
    eventi = [(data, sys_id, etichetta, prezzo, nota)]; meta = stato attuale del segnale core."""
    ctx = {"core_seg": core_segnali(df), "az": A.az_serie(df)}
    if ctx["az"] is None: ctx["az"] = {"close": df["Close"].values, "buy": [], "mr": [], "smr_entries": [], "sbb": [], "inv": [], "sarb": [], "smr_exits": [], "kf": []}
    ctx["uscite"] = {}; ctx["hi"] = df["High"].values; ctx["cl"] = df["Close"].values
    ctx["atr"] = A.calc_atr_series(df["High"].tolist(), df["Low"].tolist(), df["Close"].tolist())
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
    out = sorted(out)
    return out, {"core": core_ora, "nuovo": nuovo, "tr": posizioni(df, ctx, out)}
