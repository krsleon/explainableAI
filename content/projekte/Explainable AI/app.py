""" Explainable AI App
Frage: Wie koennen wir die Art von Pinguinen erkennen 
und die Entscheidung des KI-Modells nachvollziehen?"""

import sys
from pathlib import Path
import os

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

# Der Projektordner liegt nicht automatisch im Suchpfad, weil Streamlit die
# Seite aus dem Hauptverzeichnis heraus startet. Diese drei Zeilen braucht
# ihr, sobald ihr eigenen Code aus Nachbardateien importieren wollt.
ORDNER = Path(__file__).parent
if str(ORDNER) not in sys.path:
    sys.path.insert(0, str(ORDNER))

import analyse  # noqa: E402  (erst nach dem sys.path-Eintrag importierbar)
from utils.theming import FARBEN, merkkasten

data = analyse.load_data()
model, X_test, y_test, y_pred = analyse.train_blackbox(data, report=False)

st.markdown("# Explainable AI: Wie können wir Black-Box Modelle verstehen?")
st.caption(
    "Projekt von Katharina Gudat und Leon Kraus · Daten: Palmer Penguins, 2014, 344 Penguins"
)

merkkasten(
    "Dieses Projekt veranschaulicht zwei Modelle, die Black-Box Entscheidungen nachvollziehbar machen.",
    "Vor allem geht es um <b>Lime</b> und <b>SHAP</b>. "
    "Tab 1 ist eine Einleitung in Explainable AI, Tab 2 zeigt das Pinguin-Datenset, Tab 3 beschreibt das verwendete Blackbox-Modell, Tab 4 behandelt die LIME-Methode, Tab 5 die SHAP-Methode, und Tab 6 geht kurz auf andere Möglichkeiten der Explainable AI ein.",
    typ="merke",
)

tab_intro, tab_daten, tab_blackbox, tab_lime, tab_shap, tab_ausblick = st.tabs(
    [
        "1 · Einleitung",
        "2 · Daten: Palmer Penguins",
        "3 · Das Blackbox-Modell",
        "4 · LIME",
        "5 · SHAP",
        "6 · Ausblick",
    ]
)
#============================================================= Intro to xAI
with tab_intro:
    st.markdown("## 1 · Einleitung: Explainable AI")
    st.markdown(
        "Explainable AI (xAI) ist ein Forschungsgebiet, das sich mit der Nachvollziehbarkeit von Entscheidungen von KI-Modellen beschäftigt. "
        "Es geht darum, die Funktionsweise von Modellen zu verstehen und ihre Entscheidungen zu erklären."
        "Dazu gibt es verschiedene Herangehensweisen, die schematisch in der folgenden Abbildung dargestellt werden."
    )
    st.image(os.path.join(ORDNER, "images", "xAI_overview.png"), caption="Abbildung 1: Überblick über Explainable AI Methoden")
    st.markdown(
        " Die erste Unterscheidung in der Grafik meint den Unterschied zwischen intrinsisch interpretierbaren Modellen (z. B. lineare Regression, Entscheidungsbäume) und post-hoc Methoden, die im Nachhinein auf kompliziertere Modelle (z. B. Random Forest, Neural Networks) angewendet werden können."
    )
    st.markdown(
        "Klassische, lineare Machine Learning Modelle bieten eine gewisse Interpretierbarkeit in ihren Entscheidungen, indem Koeffizienten und Verzweigungen direkt abzulesen sind, sie sind also erklärbar by design."
        "Bei hochdimensionalen Modellen hingegen leidet die Interpretierbarkeit unter der besseren Performance – die Millionen nicht-linear verknüpften Parameter entziehen sich der menschlichen Intuition."
    )
    st.markdown(
        "Man spricht dabei von *Blackbox-Modellen*. "
    )
    st.markdown(
        """
        ## Darum suchen wir Ansätze, um Ergebnisse interpretierbar zu machen...

        Ein reines Vorhersageergebnis reicht in der Praxis selten aus:
        - **Vertrauen & Validierung:** Trifft das Modell Entscheidungen anhand biologisch plausibler Merkmale oder verlässt es sich auf zufällige Artefakte im Datensatz?
        - **Fehlersuche & Debugging:** Warum wurde ein bestimmter Pinguin falsch klassifiziert? Welches Merkmal hat den Ausschlag gegeben?
        - **Verantwortung & Nachvollziehbarkeit:** In kritischen Anwendungen müssen Entscheidungen gegenüber Anwendern und Regulatoren begründet werden können.

        Um den Konflikt zwischen hoher Vorhersageleistung und mangelnder Transparenz aufzulösen, setzt **Explainable AI (XAI)** unter anderem auf *modellagnostische Post-Hoc-Erklärungen*, also Verfahren, die auf beliebige Modelle angewandt werden können. 
        Dabei wird das Verhalten der Black Box von außen beobachtet und analysiert – wie in den folgenden Tabs an zwei führenden Verfahren demonstriert wird:
        """
    )

    merkkasten(
        "Ansätze für mehr Erklärbarkeit",
        "• <b>LIME (Tab 4):</b> Erklärt einzelne Vorhersagen lokal durch ein einfaches lineares Ersatzmodell in der direkten Nachbarschaft eines Datenpunkts.<br>"
        "• <b>SHAP (Tab 5):</b> Nutzt Konzepte der kooperativen Spieltheorie, um den exakten, fairen Beitrag jedes einzelnen Merkmals zur Entscheidung zu quantifizieren.",
        typ="definition",
    )
    
    st.markdown("LIME und SHAP sind lokale Methoden, die Auskunft über die Entscheidung eines Modells in der Nähe eines bestimmten Datenpunkts geben. Die modell-agnostischen Methoden können auch global sein und Auskunft über das Gesamtverhalten des Modells geben. ")
    st.markdown("Es existieren auch modell-spezifische Ansätze, die auf bestimmte Modelltypen zugeschnitten sind, z. B. neuronale Netze oder Entscheidungsbäume. Diese werden in Tab 6 kurz vorgestellt.")


#============================================================= Daten
with tab_daten:
    st.markdown("## 2 · Daten: Palmer Penguins")
    st.caption(
        "Die Daten stammen aus dem Datensatz 'Palmer Penguins' von Allison Horst, 2014. "
        "Er enthält Informationen zu 344 Pinguinen von drei Arten (Adelie, Chinstrap, Gentoo) "
        "und vier Inseln (Biscoe, Dream, Torgersen, und Palmer)."
    )
    st.markdown(
        "Die Daten enthalten die folgenden Spalten: `species`, `island`, `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, `sex`."
    )
    st.dataframe(data.sample(7, random_state=42))
    st.markdown(
        "Wir werden die Spalten `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, und `body_mass_g` als Features verwenden, um die Art der Pinguine vorherzusagen. "
        "Im Plot unten können wir die Verteilung der Pinguinarten in Abhängigkeit von zwei ausgewählten Features visualisieren."
        "Es fällt auf, dass die Arten je nach ausgewählten Features unterschiedlich gut trennbar sind. "
        "Unser Blackbox-Modell soll die Art der Pinguine anhand der Features vorhersagen, und wir wollen danach nachvollziehen, warum eine bestimmte Klassifikation stattgefunden hat."
    )
    numeric_features = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
    ]

    col1, col2 = st.columns(2)

    with col1:
        x_feature = st.selectbox(
            "x-axis",
            numeric_features,
            index=0
        )

    with col2:
        y_feature = st.selectbox(
            "y-axis",
            numeric_features,
            index=3
        )

    fig = px.scatter(
        data,
        x=x_feature,
        y=y_feature,
        color="species",
        hover_data=["sex", "island"],
    )

    st.plotly_chart(fig, use_container_width=True)
    
    
#============================================================= Blackbox-Modell
with tab_blackbox:
    st.markdown("## 3 · Das Blackbox-Modell")
    st.markdown("Wir trainieren ein Random Forest Modell, um die Art der Pinguine anhand der vier ausgewählten Features vorherzusagen. "
                "Dafur verwenden wir die sklearn Bibliothek. Das Modell wird auf 70% der Daten trainiert und auf den restlichen 30% getestet. "
                "Der Random Forest enthält 70 Bäume und ist damit ein komplexes Modell, das nicht mehr direkt interpretierbar ist. ")
    accuracy = accuracy_score(y_test, y_pred)

    st.metric(
        "Accuracy",
        f"{accuracy:.1%}"
    )
    cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
    )

    fig_cm = px.imshow(
        cm,
        x=model.classes_,
        y=model.classes_,
        text_auto=True,
        labels={
            "x": "Vorhergesagte Spezies",
            "y": "Tatsächliche Spezies",
            "color": "Anzahl"
        },
        title="Confusion Matrix"
    )

    fig_cm.update_layout(
        height=400
    )

    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )
    
    st.markdown("Die Confusion Matrix zeigt, dass das Modell die Arten der Pinguine sehr gut vorhersagen kann. Die meisten Fehler treten bei der Unterscheidung zwischen Adelie und Chinstrap auf, allerdings irrt sich das Modell nur in 3 Fällen. ")
    
    importance_df = pd.DataFrame({
    "Feature": analyse.standard_features,
    "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=True
    )
    fig_importance = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    text="Importance",
    title="Feature Importance des Random Forests",
    labels={
        "Importance": "Importance",
        "Feature": "Feature"
    }
    )

    fig_importance.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    fig_importance.update_layout(
        xaxis_range=[0, importance_df["Importance"].max() * 1.15],
        height=400
    )

    st.plotly_chart(
        fig_importance,
        use_container_width=True
    )
    
    st.markdown("Die Feature-Importance des Random Forests gibt an, wie oft das Modell ueber ein bestimmtes Feature splittet, um seine Klassifizierung abzuleiten."
                " Die Feature-Importance ist eine globale Metrik, die das Verhalten des Modells über alle Pinguine hinweg beschreibt. Sie sagt jedoch nichts darüber aus, wie das Modell eine bestimmte Vorhersage für einen einzelnen Pinguin getroffen hat. "
                "Die Schnabellänge ist das wichtigste Feature, gefolgt von der Flossenlänge. Die Körpermasse und die Schnabeltiefe sind weniger wichtig für die Vorhersage der Pinguinart. "
                "Das kann man auch aus den Scatter-Plots des Datensatzes ableiten.")
    


#============================================================= LIME
with tab_lime:
    st.markdown("## 4 · LIME: Local Interpretable Model-agnostic Explanations")
    st.caption(
        "LIME ist eine Methode, die lokale Erklärungen für die Vorhersagen von Black-Box-Modellen liefert. "
    )
    st.markdown(
        "Die Grundidee von LIME ist, dass wir ein komplexes Modell durch ein einfaches, interpretiertes Modell approximieren können, "
        "das in der Nähe der Vorhersage des komplexen Modells gut funktioniert. "
        "Dazu werden zufällige Perturbationen der Eingabedaten erzeugt und die Vorhersagen des Black-Box-Modells für diese Perturbationen gesammelt. "
        "Anschließend wird ein einfaches Modell (z.B. lineares Modell, Decision Tree) auf diesen Daten trainiert, um die Vorhersage des Black-Box-Modells zu erklären."
        "Es handelt sich um eine lokale Methode, da eine Erklärung nur für eine einzelne Vorhersage/Instanz - in unserem Beispiel für einen Pinguin - erzeugt wird, und nicht für das gesamte Modell."
    )
    
    st.markdown("### Mathematische Grundlagen von LIME")

    st.markdown(
        r"""
        Für eine einzelne Instanz $x$ versucht LIME, das Verhalten des komplexen
        Black-Box-Modells $f$ in einer lokalen Umgebung durch ein einfaches,
        interpretierbares Modell $g$ zu approximieren. Für unsere Anwendung ist
        $f$ der Random Forest und $x$ ein einzelner Pinguin.

        Als einfaches Erklärungsmodell kann beispielsweise ein lineares Modell
        verwendet werden:
        """
    )

    st.latex(
        r"""
        g(x') = \beta_0 + \sum_{j=1}^{p} \beta_j x'_j .
        """
    )

    st.markdown(
        r"""
        Dabei beschreibt $x'$ die Merkmale einer (möglicherweise perturbierten)
        Instanz und $\beta_j$ gibt an, welchen lokalen Einfluss das Merkmal
        $j$ auf die Vorhersage des Surrogatmodells hat. Ein großer Betrag
        $|\beta_j|$ bedeutet somit einen starken lokalen Einfluss. Das Vorzeichen
        gibt die Richtung des Einflusses an: Ein positiver Koeffizient unterstützt
        die betrachtete Klasse, ein negativer Koeffizient spricht gegen sie.

        Wichtig ist dabei, dass diese Koeffizienten <b>keine globalen
        Feature-Importances des Random Forests</b> sind. Sie beschreiben nur,
        wie das vereinfachte Modell das Verhalten des Random Forests in der
        Umgebung der ausgewählten Instanz approximiert.
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        r"""
        Um die lokale Umgebung zu erzeugen, verändert LIME die Merkmale der
        ursprünglichen Instanz und lässt das Black-Box-Modell die erzeugten
        Datenpunkte klassifizieren. Datenpunkte, die näher an der ursprünglichen
        Instanz liegen, sollen dabei stärker zur lokalen Erklärung beitragen.
        """
    )
    st.markdown("#### Optimierungsproblem von LIME")
    st.markdown(
    r"""
    LIME sucht dabei ein möglichst einfaches Modell $g$, das das Verhalten
    des Black-Box-Modells $f$ in der lokalen Umgebung von $x$ gut beschreibt:

    """
    )

    st.latex(
        r"""
        \xi(x)
        =
        \underset{g\in G}{\operatorname{arg\,min}}
        \left[
            L(f,g,\pi_x) + \Omega(g)
        \right].
        """
    )

    st.markdown(
        r"""
        Dabei misst $L(f,g,\pi_x)$, wie gut das Surrogatmodell $g$ das
        Black-Box-Modell in der lokalen Umgebung reproduziert, während
        $\Omega(g)$ die Komplexität des Erklärungsmodells bestraft.
        LIME sucht also einen Kompromiss zwischen **Genauigkeit der lokalen
        Approximation** und **Einfachheit der Erklärung**.
        """
    )

    st.markdown("#### Kernel und Kernel-Breite")

    st.markdown(
        r"""
        Die Nähe eines perturbierten Datenpunkts $x'$ zur ursprünglichen Instanz
        $x$ wird durch einen Kernel gewichtet. Vereinfacht kann man sich die
        Gewichtsfunktion als
        """
    )

    st.latex(
        r"""
        \pi_x(x') =
        \exp\left(
        -\frac{D(x,x')^2}{\sigma^2}
        \right)
        """
    )

    st.markdown(
        r"""
        vorstellen. Hier bezeichnet $D(x,x')$ einen Distanzmaß zwischen den
        beiden Instanzen und $\sigma$ die <b>Kernel-Breite</b>
        (<i>kernel width</i>).

        Die Kernel-Breite bestimmt damit, wie groß die lokale Umgebung ist:
        """
    )

    st.markdown(
        r"""
        - **Kleine Kernel-Breite:** Nur Datenpunkte, die sehr nahe an der
          ursprünglichen Instanz liegen, erhalten ein hohes Gewicht.
          Die Erklärung ist dadurch stärker lokalisiert.
        - **Große Kernel-Breite:** Auch weiter entfernte Datenpunkte erhalten
          noch ein relevantes Gewicht. Die Erklärung berücksichtigt dadurch
          einen größeren Bereich des Datenraums.

        Die Wahl der Kernel-Breite beeinflusst daher direkt die resultierende
        Erklärung. Es gibt nicht notwendigerweise eine einzige, eindeutig
        "richtige" lokale Erklärung.
        """,
        unsafe_allow_html=True
    )

    if "lime_instance_idx" not in st.session_state:
        st.session_state["lime_instance_idx"] = 7

    # Bounds check
    if st.session_state["lime_instance_idx"] < 0:
        st.session_state["lime_instance_idx"] = 0
    elif st.session_state["lime_instance_idx"] >= len(X_test):
        st.session_state["lime_instance_idx"] = len(X_test) - 1

    def _prev_penguin():
        if st.session_state["lime_instance_idx"] > 0:
            st.session_state["lime_instance_idx"] -= 1

    def _next_penguin():
        if st.session_state["lime_instance_idx"] < len(X_test) - 1:
            st.session_state["lime_instance_idx"] += 1

    st.markdown("### Ihr seid dran: LIME-Vorhersage für einen Pinguin")
    col_prev, col_num, col_next = st.columns([1, 2, 1], vertical_alignment="center")

    with col_prev:
        st.button(
            "◀ Vorheriger",
            key="lime_btn_prev",
            on_click=_prev_penguin,
            disabled=(st.session_state["lime_instance_idx"] <= 0),
            use_container_width=True,
        )

    with col_num:
        st.markdown(
            f"<div style='text-align: center;'>"
            f"<span style='font-size: 1.8rem; font-weight: 700;'>Pinguin #{st.session_state['lime_instance_idx']}</span>"
            f"<br><span style='font-size: 0.85rem; color: #888;'>Index {st.session_state['lime_instance_idx']} von {len(X_test) - 1}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_next:
        st.button(
            "Nächster ▶",
            key="lime_btn_next",
            on_click=_next_penguin,
            disabled=(st.session_state["lime_instance_idx"] >= len(X_test) - 1),
            use_container_width=True,
        )

    instance_idx = st.session_state["lime_instance_idx"]
    instance_to_explain = X_test.iloc[instance_idx].to_numpy()

    actual_class = y_test.iloc[instance_idx]

    predicted_class = model.predict(
        instance_to_explain.reshape(1, -1)
    )[0]

    st.write(f"**Tatsächliche Art:** {actual_class}")
    st.write(f"**Vorhergesagte Art:** {predicted_class}")
    st.markdown("Eigenschaften des gewählten Pinguins:")
    st.dataframe(
    X_test.iloc[[instance_idx]],
    hide_index=True,
    use_container_width=True
    )
    st.markdown("### Kernel-Abhängigkeit von LIME Erklärungen")
    kernel_width = st.slider(
        "Kernel width - Wie lokal ist unsere Erklärung?",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
    )
    explainer = analyse.create_lime_explainer(model, data, kernel_width)
    explanation = analyse.explain_instance(explainer, model, instance_to_explain)
    lime_values = explanation.as_list(explanation.top_labels[0])
    
    # Convert LIME output into a DataFrame
    lime_df = pd.DataFrame(
        lime_values,
        columns=["Feature", "Beitrag zur Vorhersage"]
    )

    # Sort so the strongest contributions appear at the top
    lime_df = lime_df.sort_values(
        "Beitrag zur Vorhersage",
        ascending=True
    )

    fig = px.bar(
        lime_df,
        x="Beitrag zur Vorhersage",
        y="Feature",
        orientation="h",
        title=f"LIME Vorhersage — {predicted_class}",
        labels={
            "Contribution": "Beitrag zur Vorhersage",
            "Feature": "Feature"
        }
    )

    # Add vertical line at zero
    fig.add_vline(
        x=0,
        line_width=2
    )

    fig.update_layout(
        title=(
            f"LIME explanation for {predicted_class}"
            f"<br><sup>Kernel width = {kernel_width}</sup>"
        ),
        height=400,
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    
    fig.update_layout(
    title=(
        f"LIME Erklärung für {predicted_class}"
        f"<br><sup>Kernelbreite = {kernel_width}</sup>"
    ),
    height=400,
    showlegend=False
    )
    
    st.markdown("Im Beispiel könnt ihr sehen, dass die Kernelbreite die Bedeutung der einzelnen Features für die Vorhersage verändert."
                "Bei zu geringer Kernelbreite kann überhaupt keine signifikante Kontribution festgestellt werden (Werte von $10^{-21}$)."
                "Teilweise ändern sich die aufgeführten wichtigsten Features bei steigender Kernelbreite.")
    
    st.markdown("### Stärken und Schwächen von LIME")

    col1, col2 = st.columns(2)

    with col1:
        st.success(
            """
            **✓ Stärken**

            - **Modellunabhängig:** LIME benötigt keinen direkten Zugriff auf
              die interne Struktur des Black-Box-Modells.

            - **Lokale Erklärungen:** Es erklärt konkrete einzelne
              Vorhersagen und kann dadurch sehr detaillierte Einzelfälle
              untersuchen.

            - **Intuitiv:** Die Beiträge einzelner Merkmale können als
              positive oder negative Gewichte dargestellt werden.

            - **Vielseitig:** Das Grundprinzip kann auf unterschiedliche
              Datentypen und Modelle angewendet werden.

            - **Einfach visualisierbar:** Die lokalen Feature-Beiträge lassen
              sich beispielsweise als Balkendiagramm darstellen.
            """
        )

    with col2:
        st.warning(
            """
            **⚠ Schwächen**

            - **Lokale Methode:** Eine Erklärung für einen Pinguin sagt
              nicht automatisch etwas über andere Pinguine aus.

            - **Abhängigkeit von Hyperparametern:** Die Erklärung kann sich
              beispielsweise mit der Kernel-Breite oder der Anzahl der
              erzeugten Perturbationen verändern.

            - **Stochastisch:** Durch die zufällige Erzeugung von
              Perturbationen können sich Erklärungen zwischen verschiedenen
              Läufen unterscheiden.

            - **Surrogatmodell ist nur eine Approximation:** Das einfache
              Modell muss das Black-Box-Modell nicht außerhalb der lokalen
              Umgebung korrekt beschreiben.

            - **Keine Kausalität:** Ein hoher Feature-Beitrag bedeutet nicht,
              dass dieses Merkmal die Vorhersage kausal verursacht.
            """
        )
    st.warning(
     """
     ⚠️ **Wichtig: Erklärung ≠ Kausalität**

     Die von LIME oder SHAP berechneten Feature-Beiträge beschreiben,
     welche Merkmale für die Vorhersage des Modells relevant sind.
     Sie zeigen **keine kausalen Zusammenhänge** zwischen den Merkmalen
     und der Zielvariable.

     Insbesondere sind sie **keine Counterfactuals**: Aus einem positiven
     Beitrag von z.B. „Körpermasse“ folgt nicht, dass eine Änderung der
     Körpermasse tatsächlich die Spezies eines Pinguins verändern würde.
     """    
    )
    st.markdown("### Good to know")
    st.info(
    """
    💡 **LIME ist nicht auf Tabellendaten beschränkt**

    LIME wurde als allgemeines, modellunabhängiges Erklärungsverfahren
    konzipiert. Das Grundprinzip kann auf verschiedene Datentypen
    angewendet werden.

    **Tabellendaten:** Einzelne Merkmale einer Instanz werden perturbiert.
    In unserem Beispiel sind dies z.B. Schnabellänge, Schnabeltiefe,
    Flossenlänge und Körpermasse.

    **Text:** Wörter bzw. Wortbestandteile können entfernt oder verändert
    werden. LIME untersucht dann, welche Wörter besonders stark zur
    Klassifikation eines Textes beitragen.

    **Bilder:** Bildbereiche bzw. Superpixel können verändert oder
    ausgeblendet werden. So lässt sich untersuchen, welche Bildregionen
    für eine Klassifikation besonders relevant sind.
    """
    )
    
    
    st.markdown("## Weiterführende Literatur")
    st.markdown(
    "- <b>Why Should I Trust You?: Explaining the Predictions of Any Classifier</b>, Ribeiro et al., 2016."
    "- <b>Interpretable Machine Learning</b>, Christoph Molnar, 2020."
    )


#============================================================= SHAP
with tab_shap:
    st.markdown("## 5 · SHAP: Shapley Additive Explanations")
    st.caption(
        "Kooperative Spieltheorie zur Quantifizierung der Beiträge zur Modellentscheidung."
    )

    # ------------------------------------------------ 1. Erklärung & Formel
    st.markdown("### Grundlagen: Spieltheorie & Mathematische Theorie")
    st.markdown(
        r"""
Die Methode **SHAP (Shapley Additive Explanations)** überträgt ein klassisches Konzept der kooperativen Spieltheorie auf das maschinelle Lernen:

- **Die Spieler:** Die einzelnen Features $x_j$ (Schnabellänge, Schnabeltiefe, Flossenlänge, Körpergewicht) eines Pinguins.
- **Das Spiel:** Das trainierte Vorhersagemodell $f(x)$ (unser Random Forest).
- **Die Auszahlung (Payout):** Die Differenz zwischen der konkreten Vorhersage $f(x)$ und der durchschnittlichen Baseline-Vorhersage $\mathbb{E}[f(X)]$.

Der **Shapley Value** $\phi_j$ weist jedem Merkmal $j$ einen fairen Anteil an diesem Gewinn zu. Er wird berechnet als der gewichtete mittlere **Marginalbeitrag** über alle möglichen Teilmengen (Koalitionen) $S$ der Merkmalsmenge $F$:

$$
\phi_j(x) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|! \, (|F| - |S| - 1)!}{|F|!} \cdot \Big( v(S \cup \{j\}) - v(S) \Big)
$$

**Bedeutung der Komponenten in der Formel:**
- $F$: Die Menge aller Merkmale im Datensatz ($|F| = 4$)
- $v(S)$: der Modell-Erwartungswert ist, wenn nur die Merkmale in $S$ bekannt sind
- $\frac{|S|! \, (|F| - |S| - 1)!}{|F|!}$: Das kombinatorische Gewicht – die Wahrscheinlichkeit, dass Merkmal $j$ bei einer zufälligen Reihenfolgebildung genau nach der Koalition $S$ hinzugefügt wird
- $v(S \cup \{j\}) - v(S)$: Der **marginale Mehrwert**, den Merkmal $j$ zur bestehenden Koalition $S$ beisteuert 
"""
    )

    merkkasten(
        "Die vier mathematischen Axiome von SHAP",
        "• <b>1. Effizienz (Additivität):</b> ∑ φ<sub><i>j</i></sub>(<i>x</i>) = <i>f</i>(<i>x</i>) − 𝔼[<i>f</i>(<i>X</i>)] — die Summe aller Feature-Beiträge ergibt exakt die Abweichung vom globalen Durchschnitt.<br>"
        "• <b>2. Symmetrie:</b> Zwei Merkmale mit identischem Einfluss auf alle Koalitionen erhalten immer denselben Shapley Value.<br>"
        "• <b>3. Dummy (Null-Effekt):</b> Ein Merkmal, das den Vorhersagewert in keiner Koalition ändert, erhält exakt φ<sub><i>j</i></sub> = 0.<br>"
        "• <b>4. TreeSHAP:</b> Für baumbasierte Ensembles (wie Random Forests) wertet TreeSHAP diese Summe nicht exponentiell (2<sup>|<i>F</i>|</sup>), sondern in polynomieller Laufzeit <i>O</i>(<i>T · L · D</i>²) exakt entlang der Baumstrukturen aus.",
        typ="definition",
    )

    # ------------------------------------------------ Modell & Caching
    @st.cache_resource
    def load_shap_model_and_data():
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        import shap

        df = analyse.load_data()
        feature_names = [
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
        ]
        X = df[feature_names]
        Y = df["species"]

        X_train, X_test, Y_train, Y_test = train_test_split(
            X,
            Y,
            test_size=0.35,
            random_state=42,
            stratify=Y,
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, Y_train)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)
        classes = list(model.classes_)

        return model, explainer, shap_values, X_test, Y_test, classes, feature_names

    rf_model, tree_explainer, shap_vals, X_test, Y_test, class_names, feat_names = load_shap_model_and_data()

    # ------------------------------------------------ 2. Globale Ergebnisse
    st.markdown("---")
    st.markdown("### Globale Ergebnisse: Merkmalseinfluss pro Spezies")
    st.markdown(
        "Wähle eine Pinguinart aus. Anschließend werden die globalen Feature-Contributions für diese Art berechnet und visualisiert:"
    )

    selected_species = st.radio(
        "Ziel-Spezies auswählen:",
        options=class_names,
        index=2,  # Default: Gentoo
        horizontal=True,
    )
    species_idx = class_names.index(selected_species)

    import matplotlib.pyplot as plt
    import shap

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown(f"**SHAP Beeswarm Plot ({selected_species})**")
        st.caption("Punkte = Pinguine im Testset; Farbe = Feature-Wert; Position = Einfluss auf die Klassifikation")
        fig_bee, ax_bee = plt.subplots(figsize=(6.5, 4.2))
        shap.plots.beeswarm(shap_vals[:, :, species_idx], show=False)
        plt.title(f"Einfluss der Features auf '{selected_species}'", fontsize=12, pad=10)
        plt.tight_layout()
        st.pyplot(fig_bee, use_container_width=True)
        plt.close(fig_bee)

    with col_g2:
        st.markdown(f"**Globale Feature Importance |SHAP| ({selected_species})**")
        st.caption("Mittlere absolute SHAP-Werte $\\frac{1}{n}\\sum |\\phi_j|$ ")
        fig_bar, ax_bar = plt.subplots(figsize=(6.5, 4.2))
        shap.plots.bar(shap_vals[:, :, species_idx], show=False)
        plt.title(f"Mittlere Wichtigkeit für '{selected_species}'", fontsize=12, pad=10)
        plt.tight_layout()
        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)

    # Spezifische Interpretation je nach ausgewählter Spezies
    if selected_species == "Gentoo":
        st.info(
            "🐧 **Erkenntnis für Gentoo:** Eine hohe Flossenlänge (`flipper_length_mm`) und eine hohe Körpermasse (`body_mass_g`) treiben die Modellvorhersage stark in Richtung Gentoo (rote Punkte weit rechts). Eine geringere Schnabeltiefe (`bill_depth_mm`) und hohe Schnabellänge (`bill_length_mm`) begünstigen die Entscheidung für Gentoo zusätzlich."
        )
    elif selected_species == "Chinstrap":
        st.info(
            "🐧 **Erkenntnis für Chinstrap:** Eine lange Schnabellänge (`bill_length_mm`) treibt die Modellvorhersage stark in Richtung Chinstrap (rote Punkte weit rechts). Eine geringere Körpermasse (`body_mass_g`), geringe Flossenlänge (`flipper_length_mm`) und hohe Schnabeltiefe (`bill_depth_mm`) begünstigen die Entscheidung für Chinstrap zusätzlich."
        )
    else:  # Adelie
        st.info(
            "🐧 **Erkenntnis für Adelie:** Eine kurze Schnabellänge (`bill_length_mm`) und kurze Flossenlänge (`flipper_length_mm`) treiben die Modellvorhersage stark in Richtung Adelie (blaue Punkte weit rechts). Eine hohe Schnabeltiefe (`bill_depth_mm`) und eine geringere Körpermasse (`body_mass_g`) begünstigen die Entscheidung für Adelie zusätzlich."
        )

    st.markdown("---")
    st.markdown("### Lokale Ergebnisse: Einzelfall-Erklärung")
    st.markdown(
        "Hier untersuchen wir konkrete Pinguine aus dem Testdatensatz. Wähle einen Test-Pinguin aus, um den exakten Zerlegungspfad der Entscheidung nachzuvollziehen:"
    )

    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        preset_choice = st.selectbox(
            "Pinguin-Instanz auswählen:",
            options=[
                "Pinguin #4 (Gentoo)",
                "Pinguin #0 (Gentoo)",
                "Pinguin #2 (Chinstrap)",
                "Pinguin #59 (Chinstrap)",
                "Pinguin #10 (Adelie)",
                "Pinguin #20 (Adelie — Fehlklassifikation)",
                "Eigene Index-Eingabe...",
            ],
            index=0,
        )
    with col_sel2:
        if preset_choice == "Eigene Index-Eingabe...":
            instance_idx = int(
                st.number_input(
                    "Test-Index (0 bis 116):",
                    min_value=0,
                    max_value=len(X_test) - 1,
                    value=4,
                )
            )
        else:
            preset_map = {
                "Pinguin #4 (Gentoo)": 4,
                "Pinguin #0 (Gentoo)": 0,
                "Pinguin #2 (Chinstrap)": 2,
                "Pinguin #59 (Chinstrap)": 59,
                "Pinguin #10 (Adelie)": 10,
                "Pinguin #20 (Adelie — Fehlklassifikation)": 20,
            }
            instance_idx = preset_map[preset_choice]

    # Daten des ausgewählten Pinguins
    p_instance = X_test.iloc[[instance_idx]]
    p_true = Y_test.iloc[instance_idx]
    p_pred = rf_model.predict(p_instance)[0]
    p_pred_idx = class_names.index(p_pred)
    p_proba = rf_model.predict_proba(p_instance)[0]

    # Datenübersicht anzeigen
    st.markdown("**Merkmalswerte und Modellprognose für diesen Pinguin:**")
    metrics_cols = st.columns(6)
    metrics_cols[0].metric("Schnabellänge", f"{p_instance['bill_length_mm'].values[0]:.1f} mm")
    metrics_cols[1].metric("Schnabeltiefe", f"{p_instance['bill_depth_mm'].values[0]:.1f} mm")
    metrics_cols[2].metric("Flossenlänge", f"{p_instance['flipper_length_mm'].values[0]:.0f} mm")
    metrics_cols[3].metric("Körpermasse", f"{p_instance['body_mass_g'].values[0]:.0f} g")
    metrics_cols[4].metric("Tatsächlich", p_true)
    metrics_cols[5].metric("Vorhersage", f"{p_pred} ({p_proba[p_pred_idx]:.1%})")

    # Waterfall Plot für die vorhergesagte Klasse
    fig_water, ax_water = plt.subplots(figsize=(8, 4.5))
    shap.plots.waterfall(shap_vals[instance_idx, :, p_pred_idx], show=False)
    plt.title(f"SHAP Waterfall Plot für Pinguin #{instance_idx} (Erklärung für Vorhersage: '{p_pred}')", fontsize=12, pad=12)
    plt.tight_layout()
    st.pyplot(fig_water, use_container_width=True)
    plt.close(fig_water)

    st.markdown(
        f"""

#### 💡 Der Unterschied zwischen SHAP und LIME einfach erklärt:
- **LIME:**
  LIME nimmt den Pinguin, verändert seine Maße ein bisschen zufällig (Perturbationen) und schaut, wie das Modell reagiert. Daraus wird eine **Näherung (Schätzung)** berechnet. Weil Zufall im Spiel ist, können die Ergebnisse bei jedem Durchlauf leicht schwanken.
- **SHAP:**
  SHAP betrachtet alle Merkmale wie Spieler in einer Mannschaft. Es berechnet mathematisch exakt, welchen **fairen Anteil** jedes Merkmal zum Gesamtergebnis beigetragen hat (basierend auf der Nobelpreis-gekrönten Shapley-Formel).
- **Der große Vorteil von SHAP:** Alle Merkmalsbeiträge addieren sich **zu 100 % exakt** zur finalen Wahrscheinlichkeit auf (kein Rundungsverlust, kein Zufall).
"""
    )
    st.markdown("---")
    st.markdown("### Limitationen von SHAP")
    merkkasten(
        "Kritische Grenzen und Fallstricke von SHAP",
        "1. <b>Erklärung des Modells ≠ Erklärung der Wirklichkeit (Keine Kausalität):</b> SHAP erklärt ausschließlich die interne mathematische Funktionsweise des trainierten Modells. Wenn das Modell einen falschen Zusammenhang lernt, erhält dieser einen hohen SHAP-Wert, obwohl in der realen Biologie kein Kausalzusammenhang vorliegt.<br><br>"
        "2. <b>Annahme unabhängiger Merkmale:</b> Features werden unabhängig permutiert. Bei korrelierten Merkmalen (z. B. Flossenlänge und Körpergewicht) werden Koalitionen evaluiert, die in der Realität physikalisch unmöglich sind (z. B. 6 kg Pinguin mit 130 mm Flossen).<br><br>"
        "3. <b>Rechenaufwand bei modellagnostischer Nutzung:</b> Für beliebige Blackbox-Modelle müssen bei <i>M</i> Features potenziell alle 2<sup><i>M</i></sup> Koalitionen berechnet werden. TreeSHAP umgeht dies für Entscheidungsbäume, ist jedoch an diese Modellklasse gebunden.<br><br>"
        "4. <b>Multi-Class-Komplexität:</b> Da SHAP für jede Klasse separate Attributionswerte berechnet, kann die gleichzeitige Interpretation von 3 oder mehr Klassen schnell anspruchsvoll sein.",
        typ="achtung",
    )

    st.markdown("---")
    st.markdown("### 5 · Weiterführende Literatur")
    st.markdown(
        """
- **Lundberg, S. M., & Lee, S.-I. (2017):** *A Unified Approach to Interpreting Model Predictions.* Advances in Neural Information Processing Systems (NeurIPS 2017), 4765–4774. *(Das Grundlagen-Paper zu SHAP)*
- **Lundberg, S. M. et al. (2020):** *From local explanations to global understanding with explainable AI for trees.* Nature Machine Intelligence, 2(1), 56–67. *(TreeSHAP-Algorithmus)*
- **Shapley, L. S. (1953):** *A Value for n-Person Games.* Contributions to the Theory of Games, 2(28), 307–317. *(Originale spieltheoretische Formulierung)*
- **Molnar, C. (2022):** *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable.* Kapitel 9.5 (Shapley Values) und Kapitel 9.6 (SHAP).
- **Slack, D. et al. (2020):** *Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods.* AIES 2020.
"""
    )


#============================================================= Ausblick
with tab_ausblick:
    st.markdown("## 6 · Ausblick: Weitere Ansätze der Explainable AI")
    st.caption(
        "Ein Überblick über moderne Forschungsrichtungen: Von inhärent interpretierbaren Modellen (By Design) bis zu fortgeschrittenen Post-Hoc Methoden."
        "Fazit und weitere globale und lokale Methoden zur Interpretation von KI-Modellen."
    )
    st.markdown("### Fazit: Was lernen wir aus LIME und SHAP?")

    st.markdown(
        """
        LIME und SHAP bieten zwei Möglichkeiten, die Vorhersagen eines
        Black-Box-Modells verständlicher zu machen. Sie können zeigen, welche
        Merkmale für eine Vorhersage besonders relevant sind und damit einen
        Einblick in das Verhalten des Modells geben.

        Gleichzeitig sind solche Erklärungen nicht unbedingt eindeutig.
        Unterschiedliche Methoden können einem Merkmal unterschiedliche
        Beiträge zuweisen. Auch bei LIME kann sich die Erklärung beispielsweise
        mit der Wahl der Kernel-Breite oder der zufälligen Perturbationen
        verändern. Eine einzelne Erklärung sollte daher nicht als die
        „wahre“ Begründung einer Modellentscheidung interpretiert werden.

        Damit ist Explainable AI weniger eine Methode, mit der man eine
        einzelne „richtige“ Erklärung erhält, sondern vielmehr ein Werkzeug,
        um die Funktionsweise und mögliche Schwächen eines Modells kritisch
        zu untersuchen.
        """
    )

    st.info(
        """
        💡 **Eine XAI-Methode erklärt das Modell – nicht die Realität.**

        Feature-Beiträge von LIME oder SHAP beschreiben, wie das Modell seine
        Vorhersage aus den Eingabedaten konstruiert. Sie zeigen weder
        automatisch kausale Zusammenhänge noch, was bei einer tatsächlichen
        Änderung eines Merkmals passieren würde.

        Für eine zuverlässige Interpretation ist es deshalb sinnvoll,
        verschiedene Instanzen und – wenn möglich – verschiedene
        Erklärungsverfahren miteinander zu vergleichen.
        """
    )

    st.markdown(
        """
        **LIME** und **SHAP** sind zwei prominente, *modellagnostische Post-Hoc-Verfahren*.
        Das Forschungsfeld **Explainable AI (XAI)** ist jedoch deutlich breiter gefächert und teilt sich grundlegend in zwei Konzepte auf:
        """
    )

    merkkasten(
        "Die zwei Grundkonzepte der Explainable AI",
        "• <b>1. Interpretierbarkeit 'By Design' (Intrinsisch):</b> Das Modell ist von vornherein so konstruiert, dass Menschen jeden Rechenschritt und die interne Logik direkt verstehen können — ganz ohne nachgelagerte Hilfsmodelle.<br>"
        "• <b>2. Post-Hoc Erklärungen (Nachträglich):</b> Ein beliebig komplexes (Black Box-)Modell wird trainiert, und erst im Nachhinein analysieren externe Verfahren, wie das Modell zu seinen Entscheidungen gekommen ist.",
        typ="definition",
    )

    st.markdown("---")
    st.markdown("### 1 · Forschungsansätze 'By Design' (Intrinsische Interpretierbarkeit)")
    st.markdown(
        "Der Leitgedanke von *Interpretable Machine Learning by Design* lautet: **Warum eine Black Box mühsam von außen approximieren, wenn man direkt ein hochpräzises, verständliches Modell bauen kann?** *(Rudin, 2019)*."
    )

    col_bd1, col_bd2 = st.columns(2)

    with col_bd1:
        st.markdown("#### 🌲 Explainable Boosting Machines (EBM / GAMs)")
        st.markdown(
            """
            **Generalized Additive Models (GAMs)** zerlegen Vorhersagen in die Summe einzelner Merkmalsfunktionen:
            
            $$g(E[y]) = \\beta_0 + f_1(x_1) + f_2(x_2) + \\dots + f_{ij}(x_i, x_j)$$
            
            - **Funktionsweise:** Für jedes Merkmal (und wichtige Interaktionen) wird eine separate, 1-dimensionale Kurve trainiert.
            - **Vorteil:** Erreicht auf Tabellendaten oft die gleiche Performance wie XGBoost oder Random Forests, bleibt aber als exakter Funktionsgraph lesbar.
            """
        )

        st.markdown("#### 🧩 Konzept-Flaschenhals-Modelle (CBMs)")
        st.markdown(
            """
            - **Prinzip:** Tiefe neuronale Netze lernen zuerst menschlich verständliche Zwischenkonzepte (z. B. *„Schnabel gebogen?“*, *„Gelbe Federn?“*).
            - **Entscheidung:** Erst aus diesen Konzepten trifft ein einfaches lineares Modell die Endklassifikation.
            - **Intervention:** Fachexperten können bei Fehldiagnosen direkt an den Zwischenkonzepten eingreifen und Fehler korrigieren.
            """
        )

    with col_bd2:
        st.markdown("#### 🔍 Prototypen-Netzwerke (ProtoPNet)")
        st.markdown(
            """
            - **Leitmotiv:** *„This looks like that“* – Entscheidungen werden durch Analogien zu repräsentativen Beispielen (Prototypen) begründet.
            - **Bildverarbeitung:** Das Modell zeigt: *„Ich klassifiziere diesen Pinguin als Gentoo, weil dieser Ausschnitt dem Flügelprototyp #14 entspricht.“*
            - **Transparenz:** Die Entscheidungsfindung ist direkt im Bildraum visuell nachvollziehbar.
            """
        )

        st.markdown("#### 📋 Optimale Regellisten & Symbolische Regression")
        st.markdown(
            """
            - **Optimale Entscheidungsbäume:** Moderne Solver (z. B. OSDT) finden mathematisch garantiert die kürzeste, lesbarste Regelmenge mit maximaler Treffsicherheit.
            - **Symbolische Regression:** Findet exakte, physikalisch motivierte mathematische Gleichungen direkt aus den Daten via genetischer Programmierung.
            """
        )

    st.markdown("---")
    st.markdown("### 2 · Fortgeschrittene Post-Hoc Forschungsansätze")
    st.markdown(
        "Wenn bereits ein hochdimensionales Black-Box-Modell (z. B. Deep Neural Network, Vision Transformer oder LLM) existiert, kommen spezialisierte Post-Hoc Verfahren zum Einsatz:"
    )

    col_ph1, col_ph2 = st.columns(2)

    with col_ph1:
        st.markdown("#### 🔄 Kontrafaktische Erklärungen (Counterfactuals)")
        st.markdown(
            """
            - **Fragestellung:** *„Was ist die kleinste minimale Änderung an den Merkmalen, damit das Modell seine Entscheidung ändert?“*
            - **Beispiel Pinguin:** *„Wenn dieser Adelie-Pinguin nur 3 mm längere Schnabelmaße hätte, wäre er als Chinstrap klassifiziert worden.“*
            - **Praxisnutzen:** Extrem wertvoll für Anwender (z. B. Kreditvergabe oder Medizin: *„Was muss sich konkret ändern, damit der Kredit bewilligt wird?“*).
            """
        )

        st.markdown("#### 🎯 Anchor Explanations (Regel-Anker)")
        st.markdown(
            """
            - **Prinzip:** Findet hinreichende Wenn-Dann-Bedingungen mit garantierter hoher Präzision (*Coverage & Precision*).
            - **Beispiel:** *„WENN Flossenlänge > 215 mm UND Körpergewicht > 4800 g, DANN gilt zu 99% Gentoo — völlig egal, wie die restlichen Maße aussehen.“*
            """
        )

    with col_ph2:
        st.markdown("#### 🎨 Saliency Maps & Grad-CAM (Computer Vision)")
        st.markdown(
            """
            - **Grad-CAM:** Berechnet Gradienten der Zielklasse bezüglich der letzten Faltungsschichten.
            - **Visualisierung:** Eine farbige Heatmap zeigt pixelgenau, auf welche Bildbereiche das neuronale Netz für die Entscheidung „geschaut“ hat.
            - **LRP (Layer-wise Relevance Propagation):** Propagiert Relevanzwerte deterministisch rückwärts durch alle Netzwerkschichten.
            """
        )

        st.markdown("#### 🧠 Mechanistic Interpretability & Concept Vectors (TCAV)")
        st.markdown(
            """
            - **TCAV:** Prüft, ob ein Modell intern abstrakte Konzepte (z. B. *„Streifenmuster“* oder *„Geschlecht“*) gelernt hat, ohne dass dafür Labels vorlagen.
            - **Mechanistic Interpretability:** Untersucht neuronale Schaltkreise (Attention Heads, Induction Heads) in Sprachmodellen (LLMs) wie elektronische Schaltpläne.
            """
        )

    st.markdown("---")
    st.markdown("### 3 · Gegenüberstellung: By Design vs. Post-Hoc")

    vergleich_data = {
        "Kriterium": [
            "Treue (Fidelity)",
            "Modellflexibilität",
            "Performance / Kapazität",
            "Erklärungsaufwand",
            "Typische Einsatzgebiete",
        ],
        "By Design (Intrinsisch)": [
            "100 % exakt (Modell IST die Erklärung)",
            "Eingeschränkt auf verständliche Architekturen",
            "Exzellent bei Tabellendaten, schwieriger bei rohen Bildern/Audio",
            "Kein zusätzlicher Rechenaufwand nach dem Training",
            "Medizinische Diagnostik, Justiz, Kreditentscheidungen",
        ],
        "Post-Hoc (Nachträglich)": [
            "Approximation / Näherung (Gefahr von Scheinerklärungen)",
            "Universell modellagnostisch (jedes Modell nutzbar)",
            "Höchste Kapazität (Deep Learning, LLMs, Ensembles)",
            "Teilweise sehr hoher zusätzlicher Rechenaufwand (Sampling)",
            "Computer Vision, NLP / LLMs, komplexe Ensembles",
        ],
    }
    st.dataframe(pd.DataFrame(vergleich_data), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### 4 · Globale Erklärungen")

    st.markdown(
        """
        Während eine lokale Methode wie LIME fragt

        > *„Warum hat das Modell diesen Pinguin als Gentoo klassifiziert?“*

        untersuchen globale Methoden die Frage

        > *„Wie verhält sich das Modell insgesamt?“*

        Einige wichtige modellunabhängige Verfahren sind:
        """
    )

    st.markdown(
        """
        **Permutation Feature Importance**  
        Misst, wie stark sich die Modellleistung verschlechtert, wenn die
        Werte eines Merkmals zufällig vertauscht werden. Dadurch lässt sich
        abschätzen, welche Merkmale für die Vorhersage insgesamt wichtig sind.

        **Partial Dependence Plots (PDP)**  
        Zeigen, wie sich die durchschnittliche Modellvorhersage verändert,
        wenn ein Merkmal systematisch variiert wird, während die anderen
        Merkmale berücksichtigt werden.

        **Accumulated Local Effects (ALE)**  
        Haben ein ähnliches Ziel wie PDPs, berücksichtigen jedoch lokale
        Änderungen der Vorhersage und sind insbesondere bei korrelierten
        Merkmalen häufig besser geeignet.
        """
    )

    st.info(
        """
        💡 **Global oder lokal?**

        **Globale Methoden** beschreiben das Verhalten des Modells über viele
        oder alle Datenpunkte hinweg. **Lokale Methoden** erklären dagegen
        eine einzelne Vorhersage.

        Eine Übersicht und ausführlichere Erklärungen zu diesen Methoden
        findest du hier:
        """
    )
    st.page_link("views/ml/explainable_ml.py", label="Globale xAI", icon="🗂️")

    st.markdown("---")
    st.markdown("### 5 · Explainable AI für neuronale Netze")

    st.markdown(
        """
        Für neuronale Netze können zusätzlich Methoden verwendet werden, die
        deren interne Struktur oder Gradienten ausnutzen. Dadurch können
        beispielsweise einzelne Eingabemerkmale oder Pixel eines Bildes
        bestimmten Vorhersagen zugeordnet werden.
        """
    )

    st.markdown(
        """
        **Saliency Maps**  
        Untersuchen den Gradienten der Modellvorhersage bezüglich der
        Eingabedaten. Bei Bildklassifikationen kann so sichtbar gemacht
        werden, welche Pixel besonders stark mit der Vorhersage verbunden sind.

        **Integrated Gradients**  
        Integrieren die Gradienten der Modellvorhersage entlang eines Pfades
        von einer Referenz-Eingabe zur tatsächlichen Eingabe. Dadurch wird
        der Einfluss einzelner Eingabemerkmale auf die Vorhersage
        quantifiziert.

        **DeepLIFT**  
        Vergleicht die Aktivierungen eines neuronalen Netzes für die
        tatsächliche Eingabe mit denen für eine Referenz-Eingabe und
        propagiert diese Unterschiede rückwärts durch das Netzwerk.
        """
    )

    st.markdown(
        """
        Diese Verfahren sind besonders für neuronale Netze interessant, da
        sie Informationen über deren Differenzierbarkeit und interne
        Aktivierungen verwenden können. Im Gegensatz zu LIME sind sie daher
        nicht vollständig modellunabhängig.
        """
    )

    st.markdown("---")
    st.markdown("### 6 · Zentrale Herausforderungen der aktuellen XAI-Forschung")

    merkkasten(
        "Offene Fragen und zukünftige Forschungsfelder",
        "• <b>1. Faithfulness vs. Plausibility:</b> Eine Erklärung kann für Menschen plausibel und schön aussehen, aber das tatsächliche (möglicherweise fehlerhafte) Modellverhalten verschleiern.<br>"
        "• <b>2. Adversarial Robustness:</b> LIME- und SHAP-Erklärungen können durch gezielte Störungen manipuliert werden, um diskriminierende Modelle nach außen 'fair' wirken zu lassen.<br>"
        "• <b>3. Kausalität statt Korrelation:</b> Reine Merkmalsattributionen zeigen Korrelationen, erfassen aber keine biologischen oder physikalischen Kausalzusammenhänge.<br>"
        "• <b>4. Interpretierbarkeit von Large Language Models (LLMs):</b> Wie verstehen und steuern wir Milliarden von Parametern in generativen Modellen (Activation Steering, Chain-of-Thought)?",
        typ="achtung",
    )

    st.markdown("---")
    st.markdown("### 7 · Weiterführende Literatur zu modernen XAI-Ansätzen")
    st.markdown(
        """
- **Rudin, C. (2019):** *Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead.* Nature Machine Intelligence, 1(5), 206–215. *(Plädoyer für By-Design Interpretierbarkeit)*
- **Nori, H. et al. (2019):** *InterpretML: A Unified Framework for Machine Learning Interpretability.* arXiv:1909.09223. *(Explainable Boosting Machines / EBMs)*
- **Selvaraju, R. R. et al. (2017):** *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* ICCV 2017, 618–626.
- **Kim, B. et al. (2018):** *Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV).* ICML 2018, 2668–2677.
- **Wachter, S., Mittelstadt, B., & Russell, C. (2017):** *Counterfactual Explanations Without Opening the Black Box: Automated Decisions and the GDPR.* Harvard Journal of Law & Technology.
- **Molnar, C. (2022):** *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable.* (Frei online verfügbar).
"""
    )
