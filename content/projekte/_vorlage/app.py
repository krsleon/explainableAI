"""Skelett für eine interaktive Projektseite (Weg 2).

Diese Datei liegt in `_vorlage/` und wird deshalb *nicht* als Seite
angezeigt, denn Ordner mit führendem Unterstrich überspringt die App. Kopiert sie
zusammen mit der `projekt.md` in euren eigenen Projektordner.

Ein ausgebautes Beispiel mit echten Daten findet ihr in
`content/projekte/beispielprojekt/`.

Regeln:
- Kein st.set_page_config() aufrufen, das erledigt die Haupt-App.
- Nur Pakete aus requirements.txt benutzen.
- Widget-keys mit eurem Projektnamen präfixen, damit sie sich nicht mit
  anderen Seiten überschneiden.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Falls ihr Dateien aus eurem Ordner ladet:
# from pathlib import Path
# daten = pd.read_csv(Path(__file__).parent / "daten.csv")

st.markdown("# 🚀 Euer Projekttitel")
st.caption("Team: Vorname N., Vorname N.")

st.markdown(
    """
Ein, zwei Sätze zu eurer Fragestellung. Danach kommt der interaktive Teil.
"""
)

# ------------------------------------------------------------ Eingabe
stichprobe = st.slider("Stichprobengröße", 20, 500, 100, key="vorlage_n")
streuung = st.slider("Streuung", 0.1, 3.0, 1.0, step=0.1, key="vorlage_sigma")

# ------------------------------------------------------------ Berechnung
rng = np.random.default_rng(42)
x = rng.uniform(0, 10, stichprobe)
y = 2.0 + 0.5 * x + rng.normal(0, streuung, stichprobe)
steigung, achsenabschnitt = np.polyfit(x, y, 1)

# ------------------------------------------------------------ Darstellung
fig = go.Figure()
fig.add_scatter(x=x, y=y, mode="markers", name="Beobachtungen")
raster = np.linspace(0, 10, 50)
fig.add_scatter(
    x=raster,
    y=achsenabschnitt + steigung * raster,
    mode="lines",
    name="Regressionsgerade",
)
fig.update_layout(xaxis_title="x", yaxis_title="y", height=420)
st.plotly_chart(fig, use_container_width=True)

st.metric("Geschätzte Steigung", f"{steigung:.2f}", delta=f"{steigung - 0.5:+.2f}")
st.caption("Wahrer Wert: 0.50. Mit wachsender Stichprobe rückt die Schätzung näher.")
