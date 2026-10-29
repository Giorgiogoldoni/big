"""Classifica dei sistemi per ETF: cosa ha fatto il prezzo dopo ogni freccia (nessuna simulazione di operazioni, costi ignorati).
Per ogni ETF, sistema e orizzonte (5, 10, 20, 60 barre) misura: segnali, % positivi, rendimento mediano, eccesso sul rendimento mediano
dello stesso ETF in tutti i giorni, e l'eccesso nelle due meta' della storia (per vedere se regge)."""
import json
import numpy as np
from pathlib import Path

ORIZZONTI = (5, 10, 20, 60)
MIN_SEGNALI = 10     # sotto questa soglia la coppia ETF-sistema non si giudica
MIN_META = 3         # segnali minimi in ciascuna meta' per misurarla
T_MIN = 2.0          # soglia di significativita' (t di Student su segnali non sovrapposti): taglia i falsi positivi del caso
REGOLE = [
    "Si misura il rendimento della chiusura 5, 10, 20 e 60 barre dopo ogni freccia, partendo dal prezzo del segnale. Nessuna simulazione di operazioni, costi ignorati (rendimenti lordi).",
    "Eccesso = rendimento mediano dopo i segnali meno il rendimento mediano dello stesso ETF su tutti i giorni dello stesso periodo: toglie l'effetto di un ETF che sale comunque.",
    "Una coppia ETF-sistema si giudica solo con almeno %d segnali misurabili a quell'orizzonte; sotto la soglia resta 'dati insufficienti'." % MIN_SEGNALI,
    "Affidabile = eccesso positivo sull'intera storia, positivo anche in entrambe le meta' (almeno %d segnali ciascuna) e statisticamente distinguibile dal caso: t >= %.1f calcolato solo su segnali non sovrapposti (distanti almeno quanto l'orizzonte). Non verificato = eccesso positivo ma una di queste prove manca." % (MIN_META, T_MIN),
    "Il miglior sistema e' quello con l'eccesso piu' alto tra gli affidabili; se nessuno lo e', tra i non verificati. Se nessuno ha eccesso positivo: 'nessuno'.",
    "Calibrazione su 150 serie casuali (nessun vantaggio reale): il test segnala 'affidabile' nel 9% degli ETF a +5 barre, 4% a +10, 2% a +20, 1% a +60. Su serie con regimi di trend simulati ne riconosce solo il 7% a +10 e il 3% a +20: il test e' prudente e riconosce poco, quindi la maggior parte degli ETF restera' 'non verificato'.",
    "Attenzione: con 9 sistemi per ETF il migliore del passato puo' essere frutto del caso, e i segnali vicini hanno finestre sovrapposte (soprattutto a 60 barre). E' una misura del passato, non una previsione.",
]


def _p(x):
    return None if x is None else round(float(x) * 100, 1)


def stat_coppia(C, base, N, segnali, m):
    """segnali = [(indice_barra, prezzo)]. Ritorna [n, %pos, mediana%, eccesso%, ecc_1a_meta%, ecc_2a_meta%, n1, n2, t]."""
    n = len(C)
    r = [(i, C[i + N] / p - 1) for i, p in segnali if i + N < n and p > 0]
    if not r: return None
    b0, b1, b2, bm = base[N]
    med = float(np.median([x for _, x in r]))
    r1 = [x for i, x in r if i < m]; r2 = [x for i, x in r if i >= m]
    e1 = float(np.median(r1)) - b1 if len(r1) >= MIN_META and b1 is not None else None
    e2 = float(np.median(r2)) - b2 if len(r2) >= MIN_META and b2 is not None else None
    sel, last = [], -10**9                       # segnali non sovrapposti: ognuno ad almeno N barre dal precedente
    for i, x in sorted(r):
        if i - last >= N: sel.append(x); last = i
    t = None
    if len(sel) >= 5 and np.std(sel, ddof=1) > 0:
        t = round(float((np.mean(sel) - bm) / (np.std(sel, ddof=1) / np.sqrt(len(sel)))), 1)
    return [len(r), int(round(100 * np.mean([x > 0 for _, x in r]))), _p(med), _p(med - b0), _p(e1), _p(e2), len(r1), len(r2), t]


def migliore(st):
    """st = {sistema: statistiche}. Ritorna [id|None, stato] con stato ok / nv / no / nd."""
    ok = [(v[3], k) for k, v in st.items() if v[0] >= MIN_SEGNALI and v[3] > 0 and v[4] is not None and v[5] is not None and v[4] > 0 and v[5] > 0 and v[8] is not None and v[8] >= T_MIN]
    if ok: return [max(ok)[1], "ok"]
    pos = [(v[3], k) for k, v in st.items() if v[0] >= MIN_SEGNALI and v[3] > 0]
    if pos: return [max(pos)[1], "nv"]
    return [None, "no" if any(v[0] >= MIN_SEGNALI for v in st.values()) else "nd"]


def riga(ticker, df, eventi):
    """eventi = [(data, sistema, etichetta, prezzo, nota)] calcolati sull'intera storia di df."""
    C = df["Close"].values.astype(float); n = len(C); m = n // 2
    pos = {str(d.date()): i for i, d in enumerate(df.index)}
    per = {}
    for d, s, _, p, _ in eventi:
        if d in pos: per.setdefault(s, []).append((pos[d], float(p)))
    base = {}
    for N in ORIZZONTI:
        if n <= N + 5: continue
        B = C[N:] / C[:-N] - 1
        base[N] = (float(np.median(B)), float(np.median(B[:m])) if m >= 1 else None, float(np.median(B[m:])) if len(B) > m else None, float(np.mean(B)))
    st, mig = {}, {}
    for N in base:
        cur = {s: x for s in per for x in [stat_coppia(C, base, N, per[s], m)] if x}
        st[str(N)] = cur; mig[str(N)] = migliore(cur)
    return dict(t=ticker, b=n, da=str(df.index[0].date()), a=str(df.index[-1].date()), ns={s: len(v) for s, v in per.items()}, st=st, mig=mig)


def scrivi(cartella, righe, gen):
    cartella = Path(cartella)
    base = dict(gen=gen, orizzonti=list(ORIZZONTI), minimo=MIN_SEGNALI, tmin=T_MIN)
    (cartella / "classifica.json").write_text(json.dumps(dict(base, righe=righe), separators=(",", ":")), encoding="utf-8")
    sint = {}
    for r in righe:
        sint[r["t"]] = {N: (r["mig"][N] + ([r["st"][N][r["mig"][N][0]][3], r["st"][N][r["mig"][N][0]][0], r["st"][N][r["mig"][N][0]][1]] if r["mig"][N][0] else [None, None, None]))
                        for N in r["mig"]}
    (cartella / "classifica_sintesi.json").write_text(json.dumps(dict(base, mig=sint), separators=(",", ":")), encoding="utf-8")
    print("classifica:", len(righe), "ETF | classifica.json %.1f MB | sintesi %.0f KB" % ((cartella / "classifica.json").stat().st_size / 1e6, (cartella / "classifica_sintesi.json").stat().st_size / 1e3))
