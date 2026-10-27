# big

Grafico **scannerv10**: candele giornaliere con una freccia di acquisto per ogni sistema, più l’elenco datato dei segnali.
Universo: tutti i .MI di core + gli iShares .DE che non hanno un .MI (`data/universo.json`, rigenerato a ogni run).
Nessun backtest, nessuna commissione. Dettagli delle regole: bottone “Regole PDF” (generato dal codice a ogni run).

## Messa online (una volta sola)
1. Carica tutti i file nel repo `big` (cartelle `.github/workflows`, `scripts`, i due `.html`).
2. Settings → Pages → Source: **GitHub Actions**.
3. Actions → “Aggiorna big” → Run workflow. Poi gira da solo lun-ven alle 17:30 UTC.
I dati non vengono salvati nel repo: li pubblica il workflow insieme alle pagine.

## Aggiungere un sistema
In `scripts/systems.py`: scrivi `ev_nome(df, ctx)` che restituisce `(indice_barra, prezzo, etichetta, nota)` e aggiungi una riga a `SYSTEMS` (id, nome, colore, regole). Pagine e PDF si aggiornano da soli.

## Origine del codice
`core_engine.py` da core/scripts/generate_charts.py, `az_engine.py` da azionario/fetch_azionario.py (estratti, stesse regole), `compra dal` da investing/scripts/compute_tech.py (ricalcolato su ogni giorno), momentum da TradingView.
Se cambi le regole in quei repo, big non cambia da solo.
