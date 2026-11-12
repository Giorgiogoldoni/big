"""Pagina Mercato di big: lettura descrittiva dello stato dei segnali sugli ETF iShares (azionari e, in una sezione separata, obbligazionari).
Non e' una previsione: conta quanti ETF hanno posizioni Trend aperte e quanti segnali nuovi ci sono ogni giorno.
build.py chiama riga() per ogni ETF mentre calcola i segnali e scrivi() una volta sola; il risultato e' data/mercato.json."""
import json
import re
from pathlib import Path
import systems as S

EMITTENTE = "iShares"
GIORNI = 21               # ultimi 21 giorni di Borsa (circa un mese)
MIN_TREND = 3             # ampiezza: ETF con almeno 3 sistemi Trend con una posizione aperta
SOGLIA_CALENDARIO = 0.5   # un giorno entra nel calendario se almeno meta' degli ETF ha una barra quel giorno

# Correzioni a mano dopo la revisione dell'elenco "da controllare" (pagina Mercato): ticker -> "azionari" | "obbligazionari" | "escluso".
# Esempio: CORREZIONI = {"IGLT.MI": "obbligazionari", "SXRV.DE": "azionari"}
CORREZIONI = {}

# Nomi che indicano un obbligazionario. Servono perche' per il 46% degli iShares la classe di core e' solo un valore di ripiego ("equity"):
# la regola e' stata provata sugli iShares con classe sicura (274 su 274 concordi) e poi letta a mano sugli altri.
OBBL = re.compile("|".join([
    r"\bBONDS?\b", r"\bBD\b", r"\bGOVT?\b", r"\bGOV\.", r"GOVERNMENT", r"\bGVT\b", r"TREASURY", r"\bTREAS?\b", r"\bTIPS\b", r"\bCORP\b", r"CORPORATE", r"CORPOR\b",
    r"\bAGG\b", r"\bAGGR\b", r"AGGREG", r"HIGH YLD", r"HIGH YIELD", r"\bHY\b", r"INFL", r"LINKED", r"LNKD", r"SOVEREIGN", r"MORTGAGE", r"\bMBS\b", r"MUNI",
    r"\bCREDIT\b", r"\bCASH\b", r"MONEY MARKET", r"ULTRASH", r"\bULTRA\b", r"DURATION", r"\bGILTS?\b", r"\bCLO\b", r"IBNDS?\b", r"IBONDS", r"\bIBD\b", r"IBD\d{2}",
    r"IBO\d{2}", r"EB\.REXX", r"\bFRN\b", r"FLOATING", r"FLOAT RATE", r"FLTG", r"EM LOCAL", r"LOCAL GOV", r"PFANDBR", r"\bTB\b", r"T\.BD", r"TB\.\d",
    r"\bCOR B\b", r"\bC\.B\.", r"\bG\.B\.", r"GO\.B\.", r"GHYCB", r"HYCB", r"SDHYCB", r"DCB\d", r"EOCB", r"COESG", r"FALL(EN)?[ .]?A", r"\bEMB\b", r"J\.A\.",
    r"J\.P\. MORGAN \$ EM", r"SHT DUR", r"YLD COR", r"\.BD\b", r"\bBD\d{2}", r"E\.G\.B", r"\bDUR B\b", r"\bL\.C\.B\b", r"\bBBB"]), re.I)
# Sigle che non so interpretare con certezza (restano azionari ma finiscono nell'elenco da controllare).
AMBIGUI = re.compile(r"BRDL|DBB|ISH\.C HGD|OV\.RA|E\.[FD]\.F\.E", re.I)

REGOLE = [
    "Solo gli ETF iShares dell'universo di big (Borsa Italiana e Xetra), divisi in azionari e, in una sezione separata, obbligazionari. Materie prime e cripto sono escluse. Non e' una previsione: descrive lo stato dei segnali di big.",
    "Classe: si usa quella di core quando e' sicura. Per gli iShares con classe solo di ripiego (circa meta') decide il nome dell'ETF: parole come Bond, Gov, Corp, Treasury, TIPS, High Yield, iBonds e le sigle abbreviate di Xetra lo fanno risultare obbligazionario. I casi decisi dal nome o dubbi sono nell'elenco 'da controllare' in fondo alla pagina; si correggono in scripts/mercato.py (CORREZIONI).",
    "Giorni: gli ultimi %d giorni di Borsa. Un giorno conta se almeno meta' degli ETF ha una barra quel giorno (si escludono i giorni in cui e' aperta una sola Borsa)." % GIORNI,
    "Ampiezza = quota di ETF del gruppo con almeno %d sistemi della famiglia Trend (%s) con una posizione aperta alla chiusura di quel giorno. Una posizione e' aperta dal giorno di ingresso fino al giorno prima dell'uscita. Si contano gli ETF che in quel giorno hanno dati." % (MIN_TREND, ", ".join(S.FAM_TREND)),
    "Nuove frecce = numero di ETF con almeno una freccia di acquisto di qualsiasi sistema in quel giorno. Uscite Trend = numero di ETF con almeno una posizione Trend chiusa in quel giorno.",
    "Le soglie (%d sistemi, %d giorni) sono scelte di big, non sono state ottimizzate. L'ampiezza puo' restare alta a lungo prima di un ribasso e bassa a lungo prima di un rialzo: e' una fotografia, non un segnale." % (MIN_TREND, GIORNI),
]


def classe(u):
    """(classe, motivo, dubbio) di un ETF iShares: classe = azionari | obbligazionari | escluso."""
    if u["t"] in CORREZIONI: return CORREZIONI[u["t"]], "correzione manuale", False
    ac, ripiego = u.get("ac"), bool(u.get("acd"))
    if not ripiego and ac in ("commodity", "crypto"): return "escluso", "classe core: " + str(ac), False
    if not ripiego: return ("obbligazionari" if ac == "bond" else "azionari"), "classe core sicura", False
    if OBBL.search(u["n"]): return "obbligazionari", "classe core di ripiego, deciso dal nome", True
    return "azionari", "classe core di ripiego, deciso dal nome", bool(AMBIGUI.search(u["n"]))


def riga(u, d, ev, tr):
    """Dati di un ETF per la pagina Mercato (None se non e' iShares). d = barre, ev = frecce di acquisto, tr = posizioni di calcola()."""
    if u["e"] != EMITTENTE: return None
    cl, motivo, dubbio = classe(u)
    trend = [(t[1], t[3]) for t in tr if t[0] in S.FAM_TREND]       # (ingresso, uscita|None)
    return dict(t=u["t"], n=u["n"], cl=cl, mo=motivo, du=dubbio, da=str(d.index[0].date()), a=str(d.index[-1].date()),
                date=[str(x.date()) for x in d.index], trend=trend, acq={e[0] for e in ev}, usc={t[3] for t in tr if t[0] in S.FAM_TREND and t[3]})


def _calendario(righe):
    cont = {}
    for r in righe:
        for x in r["date"]: cont[x] = cont.get(x, 0) + 1
    return sorted(x for x, n in cont.items() if n >= SOGLIA_CALENDARIO * len(righe))[-GIORNI:]


def _serie(righe, giorni):
    """[[data, ETF con dati, ETF con >=MIN_TREND posizioni Trend aperte, ampiezza %, ETF con nuove frecce, ETF con uscite Trend]]"""
    out = []
    for g in giorni:
        att = [r for r in righe if r["da"] <= g <= r["a"]]
        n3 = sum(1 for r in att if sum(1 for i, u in r["trend"] if i <= g and (u is None or u > g)) >= MIN_TREND)
        out.append([g, len(att), n3, round(100 * n3 / len(att), 1) if att else None,
                    sum(1 for r in att if g in r["acq"]), sum(1 for r in att if g in r["usc"])])
    return out


def scrivi(cartella, righe, gen):
    righe = [r for r in righe if r]
    giorni = _calendario(righe) if righe else []
    gruppi = {g: dict(n=sum(1 for r in righe if r["cl"] == g), serie=_serie([r for r in righe if r["cl"] == g], giorni)) for g in ("azionari", "obbligazionari")}
    classi = [[r["t"], r["n"], r["cl"], r["mo"], r["du"]] for r in sorted(righe, key=lambda r: (r["cl"], not r["du"], r["t"]))]
    dati = dict(gen=gen, ultima=giorni[-1] if giorni else None, giorni=GIORNI, min_trend=MIN_TREND, gruppi=gruppi, classi=classi)
    (Path(cartella) / "mercato.json").write_text(json.dumps(dati, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print("mercato: iShares", len(righe), "| azionari", gruppi["azionari"]["n"], "| obbligazionari", gruppi["obbligazionari"]["n"], "| giorni", len(giorni))
