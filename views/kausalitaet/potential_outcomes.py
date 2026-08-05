"""Kapitel Kausalität: Potential Outcomes & Randomized Controlled Trials.

Gott-Tabelle, Selection Bias, Randomisierung und der Umgang mit
Non-Compliance (ITT, Per-Protocol, IV) als Grundlage für das Gruppenthema
RCTs in der Medizin.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.theming import FARBEN, gruppen_aufgabe, kapitel_kopf, merkkasten, vertiefung

kapitel_kopf(
    "⚖️",
    "Potential Outcomes & RCTs",
    "Was wäre gewesen? Das fundamentale Problem der Kausalinferenz und die Rolle der Randomisierung",
)

# ---------------------------------------------------------------- Intro
st.markdown(
    r"""
Was bedeutet die Aussage „das Medikament wirkt“? Die präziseste Antwort gibt
das **Potential-Outcomes-Framework** (Neyman/Rubin). Wir betrachten ein
binäres Treatment

$$
D_i = \begin{cases} 1, & \text{wenn Person } i \text{ behandelt wird,} \\
0, & \text{sonst,} \end{cases}
$$

und definieren für jede Person $i$ zwei **potenzielle Outcomes**:

- $Y_i^1$: das Ergebnis von Person $i$, *wenn* sie behandelt wird,
- $Y_i^0$: das Ergebnis derselben Person, *wenn* sie nicht behandelt wird.

Beide beziehen sich auf dieselbe Person zum selben Zeitpunkt, beobachtet
wird jedoch stets nur eines von beiden. Welches, bestimmt der
Treatment-Status über die *Switching Equation*:

$$
Y_i = D_i\, Y_i^1 + (1 - D_i)\, Y_i^0 .
$$

Der **Individual Treatment Effect** $\mathrm{ITE}_i = Y_i^1 - Y_i^0$ ist
deshalb prinzipiell nicht identifizierbar. Dies ist das **fundamentale
Problem der Kausalinferenz**: Kausalinferenz ist im Kern ein Problem
fehlender Daten. Auf Populationsebene definieren wir daher den
**Average Treatment Effect**

$$
\mathrm{ATE} = E\big[Y^1 - Y^0\big] = E\big[Y^1\big] - E\big[Y^0\big]
$$

sowie den Effekt auf die Behandelten, den **Average Treatment Effect on the
Treated** $\mathrm{ATT} = E\big[Y^1 - Y^0 \mid D = 1\big]$, und analog den
$\mathrm{ATU} = E\big[Y^1 - Y^0 \mid D = 0\big]$ für die Unbehandelten.
"""
)

merkkasten(
    "Begriffe: Estimand, Estimator, Estimate",
    "Drei Begriffe, die in der gesamten Kausalinferenz sauber getrennt "
    "werden: Der <b>Estimand</b> ist die Zielgröße, die wir wissen wollen "
    "(etwa der ATE), definiert über potenzielle Outcomes und damit nicht "
    "direkt beobachtbar. Ein <b>Estimator</b> ist die Rechenvorschrift, die "
    "aus Daten eine Antwort produziert (etwa die Differenz der "
    "Gruppenmittelwerte). Das <b>Estimate</b> ist der konkrete Zahlenwert, "
    "den der Estimator in einer Stichprobe liefert. <b>Identification</b> "
    "bezeichnet den Schritt, den Estimand unter Annahmen durch beobachtbare "
    "Größen auszudrücken, erst danach beginnt die <b>Estimation</b>. Ein "
    "präziser Estimator nützt nichts, wenn die Identification scheitert.",
    typ="definition",
)

# ------------------------------------------------ Demo 1: Gott-Tabelle
st.markdown("## Demo: Die Gott-Tabelle")
st.markdown(
    """
Stell dir vor, wir wären allwissend und sähen **beide** potenziellen
Outcomes (Gesundheits-Score 0 bis 100) von acht Patientinnen und Patienten.
Das Medikament hilft in diesem Beispiel jeder einzelnen Person, mit Effekten
zwischen +5 und +15 Punkten. Schalte anschließend den Gott-Modus aus und
betrachte, was in der Realität beobachtbar bleibt, wenn, wie so häufig, vor
allem die Kränkeren zum Medikament greifen.
"""
)

GOTT = pd.DataFrame(
    {
        "Person": ["A", "B", "C", "D", "E", "F", "G", "H"],
        "Y⁰ (ohne)": [45, 50, 55, 60, 75, 80, 85, 90],
        "Y¹ (mit)": [60, 62, 65, 72, 82, 88, 92, 98],
    }
)
GOTT["Effekt Y¹−Y⁰"] = GOTT["Y¹ (mit)"] - GOTT["Y⁰ (ohne)"]
GOTT["nimmt Medikament?"] = GOTT["Y⁰ (ohne)"] < 70  # die Kränkeren greifen zu

wahrer_ate = GOTT["Effekt Y¹−Y⁰"].mean()

gott_modus = st.toggle("Gott-Modus: beide potenzielle Outcomes anzeigen", value=True)

if gott_modus:
    st.dataframe(
        GOTT[["Person", "Y⁰ (ohne)", "Y¹ (mit)", "Effekt Y¹−Y⁰"]],
        hide_index=True,
        use_container_width=True,
    )
    st.metric("Wahrer durchschnittlicher Effekt (ATE)", f"+{wahrer_ate:.1f} Punkte")
else:
    beobachtet = GOTT.copy()
    beobachtet["Gruppe"] = np.where(
        beobachtet["nimmt Medikament?"], "Medikament", "kein Medikament"
    )
    beobachtet["beobachtetes Y"] = np.where(
        beobachtet["nimmt Medikament?"],
        beobachtet["Y¹ (mit)"],
        beobachtet["Y⁰ (ohne)"],
    )
    st.dataframe(
        beobachtet[["Person", "Gruppe", "beobachtetes Y"]],
        hide_index=True,
        use_container_width=True,
    )
    mit_med = beobachtet.loc[beobachtet["nimmt Medikament?"], "beobachtetes Y"].mean()
    ohne_med = beobachtet.loc[~beobachtet["nimmt Medikament?"], "beobachtetes Y"].mean()
    naive = mit_med - ohne_med
    metrik_naiv, metrik_wahr = st.columns(2)
    metrik_naiv.metric(
        "Naiver Vergleich (Mittel mit minus Mittel ohne)", f"{naive:+.1f} Punkte"
    )
    metrik_wahr.metric("Wahrer ATE (aus dem Gott-Modus)", f"+{wahrer_ate:.1f} Punkte")
    st.error(
        "**Das Medikament erscheint schädlich, obwohl es allen hilft.** "
        "Die Medikamentengruppe besteht aus den Kränkeren. Verglichen wird "
        "nicht mit gegen ohne Medikament, sondern krank gegen gesund. Das "
        "ist **Selection Bias**: Wer behandelt wird, unterscheidet sich "
        "systematisch von den Unbehandelten."
    )

st.markdown(
    r"""
Diese Verzerrung lässt sich allgemein zerlegen. Für den einfachen
Mittelwertvergleich (*Simple Difference in Outcomes*, SDO) gilt:

$$
\underbrace{E[Y \mid D{=}1] - E[Y \mid D{=}0]}_{\text{SDO}}
= \mathrm{ATE}
+ \underbrace{E\big[Y^0 \mid D{=}1\big] - E\big[Y^0 \mid D{=}0\big]}_{\text{Selection Bias}}
+ \underbrace{(1 - \pi)\,(\mathrm{ATT} - \mathrm{ATU})}_{\text{Heterogeneous Treatment Effect Bias}},
$$

wobei $\pi$ den Anteil der Behandelten bezeichnet. Der Selection Bias
vergleicht die *kontrafaktischen* unbehandelten Outcomes beider Gruppen. Er
ist genau dann von null verschieden, wenn sich Behandelte und Unbehandelte
auch ohne Behandlung unterschieden hätten, wie in der Tabelle oben, in der
gerade die Kränkeren zum Medikament greifen.
"""
)

# --------------------------------------- Demo 2: Randomisierung
st.markdown("## Demo: Randomisierung und die Independence-Annahme")
st.markdown(
    r"""
Der **Randomized Controlled Trial (RCT)** löst das Selektionsproblem per
Design: Der Zufall entscheidet über die Behandlung. Formal stellt die
Randomisierung die **Independence Assumption** (auch *Ignorability*) sicher:

$$
(Y^1, Y^0) \;\perp\; D .
$$

Die Zuteilung enthält dann keine Information über die potenziellen Outcomes,
alle Mechanismen der Selbstselektion sind ausgeschaltet. Daraus folgt
$E\big[Y^0 \mid D{=}1\big] = E\big[Y^0 \mid D{=}0\big]$: Selection Bias und
Heterogenitätsbias verschwinden, und der einfache Mittelwertvergleich
identifiziert den ATE:

$$
E[Y \mid D{=}1] - E[Y \mid D{=}0] = E\big[Y^1\big] - E\big[Y^0\big] = \mathrm{ATE}.
$$

Vergleiche beide Zuteilungsmechanismen in der Simulation:
"""
)

zuteilung = st.radio(
    "Wie kommt die Behandlung zu den Menschen?",
    ["Selbstselektion (Kränkere greifen eher zu)", "Randomisierung (Münzwurf)"],
    horizontal=True,
)
n_studie = st.select_slider(
    "Studiengröße", options=[50, 100, 200, 500, 1000, 5000], value=200
)

WAHRER_EFFEKT = 8.0


@st.cache_data
def studie_simulieren(randomisiert: bool, n: int, seed: int = 4):
    rng = np.random.default_rng(seed)
    schwere = rng.uniform(0, 1, n)  # Krankheitsschwere (auch unbeobachtet denkbar)
    if randomisiert:
        d = rng.integers(0, 2, n)
    else:
        d = (rng.uniform(0, 1, n) < 0.15 + 0.7 * schwere).astype(int)
    y0 = 85 - 30 * schwere + rng.normal(0, 5, n)
    y = y0 + WAHRER_EFFEKT * d
    return schwere, d, y


schwere, d, y_beob = studie_simulieren(zuteilung.startswith("Randomisierung"), n_studie)

diff = y_beob[d == 1].mean() - y_beob[d == 0].mean()
se = np.sqrt(
    y_beob[d == 1].var(ddof=1) / (d == 1).sum() + y_beob[d == 0].var(ddof=1) / (d == 0).sum()
)

spalte_balance, spalte_schaetzung = st.columns(2)
with spalte_balance:
    fig_balance = go.Figure()
    fig_balance.add_bar(
        x=["kein Medikament", "Medikament"],
        y=[schwere[d == 0].mean(), schwere[d == 1].mean()],
        marker_color=[FARBEN["gletscher"], FARBEN["sonne"]],
    )
    fig_balance.update_layout(
        title="Balance-Check: mittlere Krankheitsschwere",
        yaxis_title="Ø Schwere (0–1)", height=360, yaxis=dict(range=[0, 1]),
    )
    st.plotly_chart(fig_balance, use_container_width=True)
with spalte_schaetzung:
    fig_schaetz = go.Figure()
    fig_schaetz.add_bar(
        x=["Schätzung"], y=[diff],
        error_y=dict(type="data", array=[1.96 * se]),
        marker_color=FARBEN["nacht"], width=[0.4],
    )
    fig_schaetz.add_hline(
        y=WAHRER_EFFEKT, line_dash="dash", line_color=FARBEN["wiese"],
        annotation_text="wahrer Effekt (+8)",
    )
    fig_schaetz.update_layout(
        title="Geschätzter Effekt (± 95 %-Intervall)",
        yaxis_title="Punkte", height=360,
    )
    st.plotly_chart(fig_schaetz, use_container_width=True)

if zuteilung.startswith("Randomisierung"):
    st.success(
        "**Balance:** Beide Gruppen sind im Mittel gleich krank, der "
        "Gruppenvergleich isoliert also den Medikamenteneffekt. Mit "
        "wachsender Studiengröße zieht sich das Intervall um den wahren "
        "Wert zusammen."
    )
else:
    st.error(
        "**Unbalanciert:** Die Medikamentengruppe ist deutlich kränker. Der "
        "naive Vergleich vermengt Medikamentenwirkung und Krankheitsschwere, "
        "und mehr Daten helfen hier nicht: Der Bias bleibt bestehen, nur das "
        "Intervall wird enger. Man schätzt präzise das Falsche."
    )

# --------------------------------------- Demo 3: Non-Compliance
with vertiefung("Wenn nicht alle mitmachen: ITT, Per-Protocol und IV"):
    st.markdown(
        r"""
    In klinischen Studien lässt sich die *Zuweisung* $Z$ randomisieren, nicht
    aber die tatsächliche *Einnahme* $D$: Ein Teil der Zugewiesenen nimmt das
    Medikament nicht (**Non-Compliance**). Angenommen, vor allem die weniger
    schwer Erkrankten halten die Therapie durch. Für die Auswertung stehen drei
    Ansätze zur Wahl:

    - **Intention-to-Treat (ITT):** Vergleich nach *Zuweisung* $Z$. Die
      Randomisierung bleibt intakt, geschätzt wird jedoch der Effekt des
      Angebots, bei Non-Compliance eine verwässerte Größe.
    - **Per-Protocol:** Vergleich nach tatsächlicher *Einnahme* $D$. Da die
      Einnahme selbstselektiert ist, kehrt der Selection Bias zurück.
    - **Instrumental Variables (IV):** Die randomisierte Zuweisung $Z$ dient als
      **Instrument** für die Einnahme $D$. Der **Wald Estimator**

    $$
    \widehat{\mathrm{LATE}}
    = \frac{E[Y \mid Z{=}1] - E[Y \mid Z{=}0]}{E[D \mid Z{=}1] - E[D \mid Z{=}0]}
    = \frac{\mathrm{ITT}}{\text{Complier-Anteil}}
    $$

    identifiziert den **Local Average Treatment Effect (LATE)** für die
    **Complier**. Er ist gültig unter der *Exclusion Restriction*, nach der die
    Zuweisung ausschließlich über die Einnahme wirkt, und unter *Monotonicity*,
    nach der niemand systematisch das Gegenteil der Zuweisung tut.
    """
    )

    anteil_complier = st.slider("Anteil Complier (halten sich an die Zuweisung)", 0.2, 1.0, 0.6, step=0.05)


    @st.cache_data
    def compliance_studie(anteil: float, n: int = 20000, seed: int = 8):
        rng = np.random.default_rng(seed)
        schwere = rng.uniform(0, 1, n)
        z = rng.integers(0, 2, n)
        complier = schwere < anteil          # die weniger Kranken halten durch
        d = z * complier.astype(int)         # einseitige Non-Compliance
        y = 85 - 30 * schwere + WAHRER_EFFEKT * d + rng.normal(0, 5, n)
        itt = y[z == 1].mean() - y[z == 0].mean()
        per_protocol = y[d == 1].mean() - y[d == 0].mean()
        iv = itt / complier.mean()
        return itt, per_protocol, iv


    itt, per_protocol, iv = compliance_studie(anteil_complier)

    fig_nc = go.Figure()
    fig_nc.add_bar(
        x=["ITT (nach Zuweisung)", "Per-Protocol (nach Einnahme)", "IV / LATE"],
        y=[itt, per_protocol, iv],
        marker_color=[FARBEN["gletscher"], FARBEN["beere"], FARBEN["wiese"]],
    )
    fig_nc.add_hline(
        y=WAHRER_EFFEKT, line_dash="dash", line_color=FARBEN["schiefer"],
        annotation_text="wahrer Effekt der Einnahme (+8)",
    )
    fig_nc.update_layout(yaxis_title="geschätzter Effekt (Punkte)", height=380)
    st.plotly_chart(fig_nc, use_container_width=True)

    st.markdown(
        f"""
    Bei **{anteil_complier:.0%} Compliern** ergibt sich: ITT ≈ {itt:.1f}
    (verwässert, denn geschätzt wird der Effekt des Angebots, was allerdings
    häufig genau die politikrelevante Größe ist), Per-Protocol ≈
    {per_protocol:.1f} (verzerrt, weil die Einnehmenden systematisch gesünder
    sind) und IV ≈ {iv:.1f} (trifft den wahren Einnahme-Effekt, gültig für die
    Complier).
    """
    )

merkkasten(
    "Merke",
    "Randomisierung macht Behandlungs- und Kontrollgruppe <b>in allem</b> "
    "vergleichbar, auch im Unbeobachtbaren. Non-Compliance zerstört diese "
    "Eigenschaft für die <i>Einnahme</i>, nicht für die <i>Zuweisung</i>. "
    "Deshalb ist ITT der ehrliche Standard klinischer Studien, während "
    "IV-Methoden den Einnahme-Effekt für die Complier zurückgewinnen.",
    typ="merke",
)
gruppen_aufgabe(
    "Was eure Gruppe hier herausfindet",
    [
        (
            "Wie plant man ein Experiment, <i>bevor</i> Daten existieren? "
            "Fallzahlberechnung, Präregistrierung, Block- oder "
            "Stratifizierungs-Randomisierung. Hier entscheidet sich, ob eine "
            "Studie überhaupt etwas zeigen kann."
        ),
        (
            "Was tun, wenn nicht alle mitmachen? Die Vertiefung oben stellt "
            "ITT, Per-Protocol und IV nebeneinander. Welche der drei Größen "
            "beantwortet welche Frage, und welche würdet ihr einer "
            "Zulassungsbehörde vorlegen?"
        ),
        (
            "Wie geht man mit mehreren Endpunkten um? Wer zwanzig Outcomes "
            "testet, findet fast garantiert eines mit p &lt; 0,05. Bonferroni, "
            "Holm, False Discovery Rate: Was davon ist in klinischen Studien "
            "Standard?"
        ),
    ],
    hinweis=(
        "Startpunkt: das CONSORT-Statement als Checkliste für "
        "RCT-Berichte, <code>statsmodels.stats.power</code> für "
        "Fallzahlen, und ein echtes Studienprotokoll von "
        "ClinicalTrials.gov zum Auseinandernehmen."
    ),
)

# -------------------------------------------------------------- Ausblick
st.markdown("## Weiterführende Literatur")
st.markdown(
    """
- S. Cunningham (2021), *Causal Inference: The Mixtape*, Yale University Press, Kap. 4 (frei online)
- M. Huber (2023), *Causal Analysis: Impact Evaluation and Causal Machine Learning with Applications in R*, MIT Press, Kap. 1 bis 3
- G. W. Imbens & D. B. Rubin (2015), *Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction*, Cambridge University Press
"""
)

st.markdown("## Wie geht es weiter?")
weiter_quasi, weiter_bayes = st.columns(2)
with weiter_quasi:
    st.page_link(
        "views/kausalitaet/quasi_experimente.py",
        label="Kein Experiment möglich? Quasi-Experimente: DiD & RDD",
        icon="📐",
    )
with weiter_bayes:
    st.page_link(
        "views/kausalitaet/bayes.py",
        label="Oder: Bayesian Methods",
        icon="🎲",
    )
