# Plani i punës: Luna Tomato Logistics

## Objektivi

Zgjidhja sa më e mirë e tri instancave të Luna Tomato Logistics duke krahasuar strategjitë, testuar vlefshmërinë dhe dorëzuar JSON në ESA. **Pa premtuar një renditje të caktuar.**

## Faza A: Matching I

1. Shkarko skedarin zyrtar nga `About > Instance Files > Matching I`.
2. Ruaje si `data/matching-i.txt` pa ndryshuar rendin e rreshtave.
3. Kontrollo që ka 25,000 rreshta transferesh.
4. Ekzekuto testet dhe fillimisht `--method heuristic`.
5. Provo `--method hybrid --seconds 300`, pastaj rrite kohën.
6. Ruaj rezultatin pozitiv lokal, statusin e MILP, gap, seed dhe kohën.
7. Dërgo `results/submission_matching-i.json` te faqja `Matching I` e ESA-s.

## Faza B: Matching II

Si më lart, me `data/matching-ii.txt`, 92,103 rreshta, dhe `--seconds 600` ose më shumë sipas resurseve.

## Faza C: Trajectory Matching

1. Shkarko tri dataset-et orbitale, kodin zyrtar të evaluatorit dhe rastin shembull.
2. Krijo ambient Linux/conda të testuar me versionin e duhur të `pykep` dhe `heyoka`.
3. Përsërit score-in e zgjidhjes shembull me evaluatorin zyrtar.
4. Modeli fizik: BCP (Toka, Hëna, Dielli). Kontrollo njësitë dhe kufijtë e kohës.
5. Gjenero trajektore fillestare për çifte E-L (p.sh. trajektore me 2 impulse dhe differential evolution).
6. Përmirëso me multi-start, targeting `(a,e,i)`, përfshirë tolerancat dhe zonat e ndaluara.
7. Valido çdo kandidat me kodin ESA dhe regjistro proveniencën.
8. Për çdo trajektore valide llogarit masën e transportuar për mundësitë l-d.
9. Zgjidh probleme përzgjedhjeje MILP për tre grupe E-L-D pa përsëritje.
10. Plotëso 8,400 vlera; ri-evaluso dhe dorëzo JSON final.

## Gjërat që nuk duhet t'i ngatërrojmë

- Kaggle është për llogaritje; ESA Optimise është gjyqtari i rezultateve.
- GitHub është për versionet e kodit; `git push` nuk bën submission në ESA.
- Vektor me 8,400 numra nuk do të thotë trajektore e vlefshme.
- Një rezultat negativ më i vogël (më negativ) është më i mirë, vetëm kur përfaqëson më shumë masë pozitive dhe të vlefshme.
- Nuk ndryshojmë rendin e rreshtave të inputit.

## Për profesorin

Shpjego: modeli (hypergraph 3D matching), kufizimet, pse Greedy është heuristik, pse 3D matching është NP-hard, si funksionon Local Search, pse MILP jep kufi të sipërm dhe kur mund të dëshmojë optimalitetin; për Advanced shpjego ekuacionin e raketës, kohën, orbitat dhe BCP, duke dalluar qartë pjesën e realizuar nga ajo e parealizuar.
