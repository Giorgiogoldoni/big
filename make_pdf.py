"""Regole PDF di big: generato dai sistemi registrati in systems.py, quindi sempre allineato al codice."""
import datetime as dt
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import systems as S

OUT = Path(__file__).resolve().parent.parent / "data" / "regole_big.pdf"
st = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=st["Title"], fontSize=18, alignment=0)
H2 = ParagraphStyle("h2", parent=st["Heading2"], fontSize=12, spaceBefore=10)
P = ParagraphStyle("p", parent=st["BodyText"], fontSize=9.5, leading=13)

def main():
    OUT.parent.mkdir(exist_ok=True)
    adesso = dt.datetime.now(dt.timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    f = [Paragraph("big - Regole dei segnali di acquisto", H1), Paragraph("Aggiornato il " + adesso, P), Spacer(1, 6)]
    gen = ["Universo: tutti gli ETF/ETP .MI di core e gli iShares .DE che non hanno un .MI (un solo ticker per ISIN, priorita' al .MI).",
           "Dati: barre giornaliere Yahoo Finance, 2 anni, non adjusted. Le barre piatte senza scambi sono scartate prima di ogni calcolo.",
           "Il grafico mostra solo ingressi di acquisto. Nessun backtest, nessuna commissione, nessun target o stop.",
           "Cooldown: stesso ticker e stesso sistema non producono un secondo segnale entro %d giorni. Sistemi diversi non si bloccano a vicenda." % S.COOLDOWN_GIORNI,
           "Le regole dei sistemi sono quelle dei repo di origine (core, azionario, investing, TradingView) e vengono copiate: se cambiano la', big non cambia da solo.",
           "Lettura tecnica a regole proprie, non e' una raccomandazione e non garantisce risultati."]
    f.append(Paragraph("Regole generali", H2)); f += [Paragraph("- " + x, P) for x in gen]
    for s in S.SYSTEMS:
        f.append(Paragraph('<font color="%s">&#9632;</font> %s' % (s["colore"], s["nome"]), H2)); f += [Paragraph("- " + x.replace("<", "&lt;"), P) for x in s["regole"]]
    SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm, title="big - regole").build(f)
    print("PDF:", OUT)

if __name__ == "__main__":
    main()
