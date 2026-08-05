"""Kapitel ML: Explainable ML.

Permutation Importance, Partial Dependence Plots und exakt berechnete
Shapley Values (Mini-SHAP) an einem fiktiven Kreditausfall-Modell.
"""

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils.theming import FARBEN, gruppen_aufgabe, kapitel_kopf, merkkasten, vertiefung

kapitel_kopf(
    "🔍",
    "Explainable ML",
    "Warum sagt das Modell das voraus? Black Boxes öffnen mit Importance, PDP und SHAP",
)

# ---------------------------------------------------------------- Intro
st.markdown(
    """
Ein Random Forest mit 300 Trees oder ein Neural Network mit Millionen von
Gewichten trifft gute Vorhersagen, doch niemand kann direkt ablesen,
**warum**. Sobald Modelle über Kredite, Diagnosen oder Bewerbungen
mitentscheiden, wird das zum Problem: Betroffene haben ein berechtigtes
Interesse an einer Erklärung, und Fachleute müssen Fehler finden können,
bevor sie Schaden anrichten.

**Explainable ML (XAI)** stellt Werkzeuge bereit, die einem trainierten
Modell nachträglich Erklärungen entlocken. Wir betrachten die drei
wichtigsten. Alle sind **modellagnostisch**, funktionieren also für jede
Black Box, und alle bauen wir von Hand nach, damit keine Magie übrig
bleibt.

Unser Untersuchungsobjekt ist ein Random Forest, der **Kreditausfälle** aus
drei Features vorhersagt: monatliches Einkommen, Schuldenquote und Alter.
Die Daten sind simuliert, der wahre Zusammenhang ist uns daher bekannt.
"""
)

FEATURES = ["Einkommen (Tsd. €)", "Schuldenquote", "Alter"]


@st.cache_data
def kredit_daten(n: int = 900, seed: int = 3):
    rng = np.random.default_rng(seed)
    einkommen = rng.uniform(1.5, 8.0, n)
    schulden = rng.uniform(0.0, 0.8, n)
    alter = rng.uniform(18, 75, n)
    # Wahrer Zusammenhang: mehr Schulden ↑, mehr Einkommen ↓,
    # Alter u-förmig (jung und alt riskanter).
    logit = -0.5 - 0.9 * einkommen + 6.0 * schulden + 0.0025 * (alter - 42) ** 2
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    X = pd.DataFrame(
        {FEATURES[0]: einkommen, FEATURES[1]: schulden, FEATURES[2]: alter}
    )
    return X, y


@st.cache_data
def modelle_trainieren():
    X, y = kredit_daten()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    wald = RandomForestClassifier(n_estimators=300, random_state=0)
    wald.fit(X_train, y_train)
    linear = LogisticRegression(max_iter=1000)
    linear.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test, wald, linear


X_train, X_test, y_train, y_test, wald, linear = modelle_trainieren()

# --------------------------------------- Tradeoff Performanz/Transparenz
st.markdown("## Der Tradeoff: Performanz vs. Transparenz")
spalte_linear, spalte_wald = st.columns(2)
with spalte_linear, st.container(border=True):
    st.markdown("**Logistic Regression** (transparent)")
    st.metric("Testgenauigkeit", f"{linear.score(X_test, y_test):.1%}")
    st.caption("Jeder Koeffizient ist direkt lesbar, das Modell erfasst jedoch nur lineare Effekte.")
with spalte_wald, st.container(border=True):
    st.markdown("**Random Forest** (Black Box)")
    st.metric("Testgenauigkeit", f"{wald.score(X_test, y_test):.1%}")
    st.caption("Erfasst Nichtlinearitäten wie das u-förmige Altersrisiko, doch 300 Trees liest niemand.")

st.markdown(
    """
Der Random Forest gewinnt hier gerade deshalb, weil der wahre Zusammenhang
nichtlinear ist: Das u-förmige Altersrisiko kann die Logistic Regression
nicht abbilden. Am transparenten Ende des Spektrums stehen die linearen
Modelle aus den Kapiteln **Lineare Regression** und **Lasso & Ridge**,
deren Koeffizienten man direkt lesen kann (ein sparsames Lasso-Modell mit
einer Handvoll Variablen ist oft die interpretierbarste Wahl überhaupt);
am performanten Ende stehen Forests und Neural Networks. Genau dieser
Tradeoff motiviert das gesamte XAI-Feld. Lässt sich die Performanz der
Black Box behalten und trotzdem verstehen, was sie tut?
"""
)

# --------------------------------------- Demo 1: Permutation Importance
st.markdown("## Werkzeug 1: Permutation Importance")
st.markdown(
    """
Die Idee ist bestechend einfach. Man zerstört die Information eines Features,
indem man seine Spalte zufällig durchmischt, und misst anschließend, wie
stark die Genauigkeit einbricht. Ein großer Einbruch bedeutet, dass das
Modell dieses Feature tatsächlich benötigt.
"""
)


@st.cache_data
def perm_importance():
    ergebnis = permutation_importance(
        wald, X_test, y_test, n_repeats=15, random_state=0
    )
    return ergebnis.importances_mean, ergebnis.importances_std


imp_mittel, imp_std = perm_importance()
reihenfolge = np.argsort(imp_mittel)
fig_imp = go.Figure(
    go.Bar(
        x=imp_mittel[reihenfolge],
        y=[FEATURES[i] for i in reihenfolge],
        orientation="h",
        error_x=dict(type="data", array=imp_std[reihenfolge]),
        marker_color=FARBEN["gletscher"],
    )
)
fig_imp.update_layout(
    xaxis_title="Genauigkeitsverlust beim Durchmischen", height=300, margin=dict(t=20)
)
st.plotly_chart(fig_imp, use_container_width=True)

# ------------------------------------------------- Demo 2: PDP und ICE
st.markdown("## Werkzeug 2: Partial Dependence Plot (PDP)")
st.markdown(
    r"""
Die Importance zeigt, *dass* ein Feature relevant ist. Der PDP zeigt,
***wie*** es wirkt. Für Feature $j$ wird das Feature künstlich durch seinen
Wertebereich geschoben und die Vorhersage über die Verteilung der übrigen
Features $X_{-j}$ gemittelt:

$$
\mathrm{PDP}_j(x) = E_{X_{-j}}\big[\hat{f}(x,\, X_{-j})\big].
$$

Die dünnen Linien (*Individual Conditional Expectation*, ICE) zeigen
einzelne Beobachtungen. Laufen sie auseinander, wirkt das Feature nicht für
alle Personen gleich.
"""
)

pdp_feature = st.selectbox("Feature wählen", FEATURES, index=2)


@st.cache_data
def pdp_berechnen(feature: str, gitterpunkte: int = 40, n_ice: int = 30):
    werte = np.linspace(
        X_test[feature].quantile(0.02), X_test[feature].quantile(0.98), gitterpunkte
    )
    stichprobe = X_test.sample(150, random_state=0).reset_index(drop=True)
    kurven = np.zeros((len(stichprobe), gitterpunkte))
    for j, wert in enumerate(werte):
        variante = stichprobe.copy()
        variante[feature] = wert
        kurven[:, j] = wald.predict_proba(variante)[:, 1]
    return werte, kurven[:n_ice], kurven.mean(axis=0)


werte, ice_kurven, pdp_kurve = pdp_berechnen(pdp_feature)

fig_pdp = go.Figure()
for kurve in ice_kurven:
    fig_pdp.add_scatter(
        x=werte, y=kurve, mode="lines", showlegend=False,
        line=dict(color=FARBEN["himmel"], width=1), opacity=0.35,
    )
fig_pdp.add_scatter(
    x=werte, y=pdp_kurve, mode="lines", name="PDP (Mittel)",
    line=dict(color=FARBEN["nacht"], width=4),
)
fig_pdp.update_layout(
    xaxis_title=pdp_feature,
    yaxis_title="Vorhergesagte Ausfallwahrscheinlichkeit",
    height=400,
)
st.plotly_chart(fig_pdp, use_container_width=True)

if pdp_feature == "Alter":
    st.success(
        "Die U-Form entspricht exakt dem nichtlinearen Effekt, den wir in die "
        "Simulation eingebaut haben. Der Random Forest hat ihn gefunden, die "
        "Logistic Regression konnte das nicht."
    )

# ----------------------------------------------- Demo 3: Mini-SHAP
with vertiefung("SHAP als faire Aufteilung einer Vorhersage"):
    st.markdown(
        r"""
    PDP und Importance erklären das Modell **im Durchschnitt**. Häufig ist aber
    eine Erklärung **für eine einzelne Entscheidung** gefragt: Warum erhält
    genau diese Person eine hohe Risikoprognose?

    **SHAP** beantwortet diese Frage mit einem Konzept der kooperativen
    Spieltheorie. Die Features sind Mitspieler, die gemeinsam die Vorhersage
    erzeugen, und der **Shapley Value** verteilt die Abweichung vom Durchschnitt
    fair auf sie. Für Feature $j$ ist er der gewichtete mittlere
    Marginalbeitrag über alle Koalitionen $S$ der Feature-Menge $F$:

    $$
    \phi_j = \sum_{S \subseteq F \setminus \{j\}}
    \frac{|S|!\,\big(|F| - |S| - 1\big)!}{|F|!}
    \Big(v\big(S \cup \{j\}\big) - v(S)\Big),
    $$

    wobei $v(S)$ die erwartete Modellvorhersage bezeichnet, wenn nur die
    Features in $S$ auf die Werte der erklärten Instanz gesetzt werden. Zentral
    ist die **Effizienz-Eigenschaft**
    $\sum_j \phi_j = \hat{f}(x) - E\big[\hat{f}(X)\big]$: Die Beiträge summieren
    sich exakt zur Abweichung vom Durchschnitt. Bei drei Features lässt sich die
    Summe über alle $2^3$ Koalitionen exakt auswerten, und genau das tut diese
    Demo. Bei vielen Features greift die SHAP-Bibliothek auf effiziente
    Näherungen zurück.

    Stelle eine Person ein:
    """
    )

    regler_e, regler_s, regler_a = st.columns(3)
    p_einkommen = regler_e.slider("Einkommen (Tsd. €)", 1.5, 8.0, 2.5, step=0.1)
    p_schulden = regler_s.slider("Schuldenquote", 0.0, 0.8, 0.6, step=0.05)
    p_alter = regler_a.slider("Alter", 18, 75, 27)


    @st.cache_data
    def shapley_exakt(einkommen: float, schulden: float, alter: float):
        """Exakte Shapley Values über alle 2^3 Koalitionen.

        Wert einer Koalition S: mittlere Vorhersage, wenn die Features in S auf
        die Instanzwerte gesetzt und die übrigen über Hintergrunddaten
        marginalisiert werden.
        """
        instanz = {FEATURES[0]: einkommen, FEATURES[1]: schulden, FEATURES[2]: alter}
        hintergrund = X_train.sample(250, random_state=0).reset_index(drop=True)

        def koalitionswert(koalition: tuple) -> float:
            variante = hintergrund.copy()
            for f in koalition:
                variante[f] = instanz[f]
            return float(wald.predict_proba(variante)[:, 1].mean())

        werte = {
            koalition: koalitionswert(koalition)
            for groesse in range(4)
            for koalition in combinations(FEATURES, groesse)
        }
        k = len(FEATURES)

        def kanonisch(koalition) -> tuple:
            # Schlüssel in FEATURES-Reihenfolge, passend zu combinations() oben
            return tuple(sorted(koalition, key=FEATURES.index))

        beitraege = {}
        for f in FEATURES:
            summe = 0.0
            for groesse in range(k):
                for koalition in combinations([g for g in FEATURES if g != f], groesse):
                    gewicht = factorial(len(koalition)) * factorial(
                        k - len(koalition) - 1
                    ) / factorial(k)
                    summe += gewicht * (
                        werte[kanonisch((*koalition, f))] - werte[koalition]
                    )
            beitraege[f] = summe
        return werte[()], beitraege


    basis, beitraege = shapley_exakt(p_einkommen, p_schulden, p_alter)
    gesamt = basis + sum(beitraege.values())

    fig_shap = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=["Durchschnitt", *FEATURES, "Vorhersage für diese Person"],
            y=[basis, *[beitraege[f] for f in FEATURES], None],
            text=[f"{basis:.2f}", *[f"{beitraege[f]:+.2f}" for f in FEATURES], f"{gesamt:.2f}"],
            textposition="outside",
            increasing=dict(marker=dict(color=FARBEN["beere"])),
            decreasing=dict(marker=dict(color=FARBEN["wiese"])),
            totals=dict(marker=dict(color=FARBEN["nacht"])),
            connector=dict(line=dict(color=FARBEN["schiefer"], width=1)),
        )
    )
    fig_shap.update_layout(
        yaxis_title="Ausfallwahrscheinlichkeit", height=420, showlegend=False,
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    staerkstes = max(beitraege, key=lambda f: abs(beitraege[f]))
    st.markdown(
        f"Vom Durchschnittsrisiko **{basis:.0%}** zur persönlichen Prognose "
        f"**{gesamt:.0%}**: Der größte Beitrag stammt für diese Person von "
        f"**{staerkstes}** ({beitraege[staerkstes]:+.2f}). Rote Beiträge erhöhen "
        "das prognostizierte Risiko, grüne senken es."
    )

merkkasten(
    "Achtung: Erklärung des Modells ist nicht Erklärung der Welt",
    "SHAP, LIME und verwandte Verfahren erklären, <b>wie das Modell "
    "rechnet</b>, nicht, was in der Welt was verursacht. Nutzt das Modell "
    "einen Confounder oder ein diskriminierendes Korrelat, erhält eben dieser "
    "einen großen Shapley Value. XAI ersetzt daher keine Kausalanalyse. Beide "
    "Perspektiven zu verbinden gehört zu den spannendsten offenen Fragen des "
    "Feldes und ist Gegenstand eures Gruppenthemas.",
    typ="achtung",
)

st.markdown(
    """
Ergänzend gehört **LIME** in den Werkzeugkasten: Es approximiert die Black
Box in der Umgebung einer einzelnen Vorhersage durch ein kleines lineares
Modell und beantwortet damit dieselbe Frage wie SHAP auf anderem Weg. In der
Praxis verwendet ihr die Bibliotheken `shap` und `lime`; nach diesem Kapitel
wisst ihr, was sie unter der Haube tun.
"""
)
gruppen_aufgabe(
    "Was eure Gruppe hier herausfindet",
    [
        (
            "SHAP-Werte sind <b>keine</b> kausalen Effekte. Wo genau bricht "
            "die Interpretation, wenn Merkmale untereinander korreliert sind, "
            "und warum verteilt SHAP dann Wichtigkeit auf Variablen, die "
            "nichts bewirken?"
        ),
        (
            "Wie stabil sind Erklärungen? Trainiert zwei Modelle mit ähnlicher "
            "Genauigkeit auf denselben Daten und vergleicht ihre Erklärungen. "
            "Wenn sie auseinanderlaufen: welcher sollte man glauben?"
        ),
        (
            "SHAP, LIME und kontrafaktische Erklärungen beantworten "
            "unterschiedliche Fragen. Welche Methode passt zu welcher "
            "Situation? Und was will eine Person eigentlich wissen, deren "
            "Kreditantrag abgelehnt wurde?"
        ),
    ],
    hinweis=(
        "Startpunkt: <code>shap</code> und <code>dice-ml</code> für "
        "kontrafaktische Erklärungen. Als Nachschlagewerk C. Molnar, "
        "<i>Interpretable Machine Learning</i> (frei online)."
    ),
)

# -------------------------------------------------------------- Ausblick
st.markdown("## Weiterführende Literatur")
st.markdown(
    """
- C. Molnar, *Interpretable Machine Learning* (frei online)
- S. Lundberg & S.-I. Lee (2017), *A Unified Approach to Interpreting Model Predictions*, NeurIPS (SHAP)
- M. T. Ribeiro, S. Singh & C. Guestrin (2016), *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*, KDD (LIME)
"""
)

st.markdown("## Wie geht es weiter?")
weiter_kausal, weiter_llm = st.columns(2)
with weiter_kausal:
    st.page_link(
        "views/kausalitaet/korrelation.py",
        label="Passt dazu: Korrelation ≠ Kausalität",
        icon="🔀",
    )
with weiter_llm:
    st.page_link(
        "views/ml/llms_kausalitaet.py", label="Weiter: LLMs & kausales Denken", icon="💬"
    )
