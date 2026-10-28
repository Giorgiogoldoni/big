"""big: scarica le barre giornaliere (Yahoo, 2 anni), calcola i segnali di acquisto di ogni sistema e scrive i dati per le pagine.
Universo: tutti i .MI di core + gli iShares .DE che non hanno gia' un .MI (core tiene un solo ticker per ISIN, priorita' al .MI)."""
import json, re, sys, time, datetime as dt, urllib.request
from pathlib import Path
import pandas as pd
import yfinance as yf
import systems as S
from issuers import issuer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; CH = DATA / "charts"
UNIV_URL = "https://raw.githubusercontent.com/Giorgiogoldoni/core/main/data/tickers_universe.json"
ISHARES = re.compile(r"ISHARES|^ISH|^IS[. ]", re.I)
# inverso riconosciuto dal nome: "-2x", "Inverse", "Bear", "Daily/Dly ... Short", "Dly ... Inv", "3x Short", "ShortDAX".
# Non scatta su "Short Duration", "Ultra Short", "Invesco/INV", "Short JPY Long EUR" (non sono inversi di indici).
INVERSO = re.compile(r"-\s?\d\s?X|\bINVERSE\b|\bBEAR\b|(DAILY|DLY).*\bSHORT\b|\bSHORT\b.*(DAILY|DLY)|(DAILY|DLY).*\bINV\b|\d\s?X\s*(SHORT|INV)\b|\bSHORT[A-Z]*DAX", re.I)
BATCH, PAUSA, MIN_BARRE = 40, 3, 60


def carica_universo():
    for k in range(3):
        try:
            return json.load(urllib.request.urlopen(UNIV_URL, timeout=60))["instruments"]
        except Exception as e:
            print("universo: tentativo", k + 1, repr(e)[:80]); time.sleep(5)
    sys.exit("ERRORE: impossibile leggere tickers_universe.json da core")


def seleziona(strumenti):
    mi = [x for x in strumenti if x["ticker_yf"].endswith(".MI")]
    isin_mi = {x["isin"] for x in mi}
    de = [x for x in strumenti if x["ticker_yf"].endswith(".DE") and ISHARES.search(x["name"]) and x["isin"] not in isin_mi]
    return [dict(t=x["ticker_yf"], n=x["name"], i=x["isin"], b="Borsa Italiana" if x["ticker_yf"].endswith(".MI") else "Xetra",
                 e=issuer(x["name"]), lv=bool(x.get("is_leveraged")), inv=bool(INVERSO.search(x["name"])),
                 tp=x.get("product_type"), ac=x.get("asset_class"), acd=x.get("asset_class_confidence") == "default_fallback") for x in mi + de]


def pulisci(d):
    if d is None or len(d) == 0: return None
    d = d[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if getattr(d.index, "tz", None) is not None: d.index = d.index.tz_localize(None)
    d.index = pd.to_datetime(d.index).normalize()
    d = d[~((d["Volume"].fillna(0) == 0) & (d["High"] == d["Low"]))]      # barre piatte senza scambi (Yahoo): scartate
    d = d[~d.index.duplicated(keep="last")].sort_index()
    return d if len(d) >= MIN_BARRE else None


def scarica(tickers):
    out = {}
    for k in range(0, len(tickers), BATCH):
        b = tickers[k:k + BATCH]
        try:
            raw = yf.download(b, period="2y", interval="1d", auto_adjust=False, group_by="ticker", threads=True, progress=False)
        except Exception as e:
            print("batch", k, "errore", repr(e)[:80]); raw = None
        for t in b:
            try:
                if raw is None or len(raw) == 0: continue
                d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                d = pulisci(d)
                if d is not None: out[t] = d
            except Exception:
                pass
        print("scaricati", min(k + BATCH, len(tickers)), "/", len(tickers), flush=True); time.sleep(PAUSA)
    for t in [t for t in tickers if t not in out]:           # secondo tentativo, uno per volta
        try:
            d = pulisci(yf.Ticker(t).history(period="2y", interval="1d", auto_adjust=False))
            if d is not None: out[t] = d
        except Exception:
            pass
        time.sleep(1)
    return out


def scrivi(universo, dati):
    CH.mkdir(parents=True, exist_ok=True)
    gen = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    righe, errori = [], []
    for u in universo:
        d = dati.get(u["t"])
        if d is None: errori.append(u["t"]); continue
        try:
            ev, meta = S.calcola(d)
        except Exception as e:
            print("ERRORE calcolo", u["t"], repr(e)[:100]); errori.append(u["t"]); continue
        last, prev = d.iloc[-1], d.iloc[-2]; ult = str(d.index[-1].date())
        r = lambda x: round(float(x), 4)
        ch = dict(t=u["t"], n=u["n"], i=u["i"], b=u["b"], gen=gen, u=ult,
                  d=[[str(i.date()), r(a), r(b), r(c), r(e_)] for i, (a, b, c, e_) in zip(d.index, d[["Open", "High", "Low", "Close"]].values)],
                  ev=[list(e) for e in ev])
        (CH / (u["t"].replace(".", "_") + ".json")).write_text(json.dumps(ch, separators=(",", ":")), encoding="utf-8")
        ultimi = {}
        for e in ev: ultimi[e[1]] = e[0]
        righe.append(dict(t=u["t"], n=u["n"], i=u["i"], b=u["b"], e=u["e"], lv=u["lv"], inv=u["inv"], tp=u["tp"], ac=u["ac"], acd=u["acd"], p=r(last["Close"]), v=round(float(last["Close"] / prev["Close"] - 1) * 100, 2),
                          u=ult, s=ultimi, nuovo=bool(meta["nuovo"]), core=meta["core"], ne=len(ev)))
    sistemi = [dict(id=s["id"], nome=s["nome"], colore=s["colore"]) for s in S.SYSTEMS]
    (DATA / "index.json").write_text(json.dumps(dict(gen=gen, sistemi=sistemi, righe=righe), separators=(",", ":")), encoding="utf-8")
    (DATA / "errori.json").write_text(json.dumps(errori), encoding="utf-8")
    print("titoli con dati:", len(righe), "| senza dati:", len(errori))


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    uni = seleziona(carica_universo())
    (DATA / "universo.json").write_text(json.dumps(uni, ensure_ascii=False), encoding="utf-8")
    print("universo:", len(uni), "(.MI %d, iShares .DE %d)" % (sum(x["b"] == "Borsa Italiana" for x in uni), sum(x["b"] == "Xetra" for x in uni)))
    scrivi(uni, scarica([x["t"] for x in uni]))
