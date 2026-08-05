"""Beispielprojekt: Bringen kleinere Schulklassen bessere Leistungen?

Vorlage für die Gruppenprojekte. Die sechs Tabs entsprechen genau den sechs
Abschnitten, die wir von einer Projektarbeit erwarten:

    Frage → Daten → Naive Analyse → Identifikation → Ergebnis → Limitationen

Wichtigste technische Regel: KEIN st.set_page_config() aufrufen, das erledigt
die Haupt-App.
"""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

# Der Projektordner liegt nicht automatisch im Suchpfad, weil Streamlit die
# Seite aus dem Hauptverzeichnis heraus startet. Diese drei Zeilen braucht
# ihr, sobald ihr eigenen Code aus Nachbardateien importieren wollt.
ORDNER = Path(__file__).parent
if str(ORDNER) not in sys.path:
    sys.path.insert(0, str(ORDNER))

import analyse  # noqa: E402  (erst nach dem sys.path-Eintrag importierbar)

from utils.theming import FARBEN, merkkasten

st.markdown("# 🏫 Bringen kleinere Klassen bessere Leistungen?")
st.caption(
    "Beispielprojekt des Dozenten-Teams · Daten: Project STAR, Tennessee 1985–1989"
)

st.markdown(
    """
Dieses Projekt ist als **Vorlage** gedacht. Es zeigt an echten Daten, wie eine
Projektarbeit aufgebaut sein kann: von der Frage über die Identifikation bis
zu den Grenzen der Aussage. Die sechs Reiter sind die sechs Abschnitte, die
wir auch von euch sehen wollen.
"""
)

daten = analyse.lade_daten()

tab_frage, tab_daten, tab_naiv, tab_ident, tab_ergebnis, tab_grenzen = st.tabs(
    [
        "1 · Frage",
        "2 · Daten & EDA",
        "3 · Naive Analyse",
        "4 · Identifikation",
        "5 · Ergebnis",
        "6 · Limitationen",
    ]
)

# =================================================================== 1 Frage
with tab_frage:
    st.markdown("## Die Frage")
    st.markdown(
        """
Kleinere Klassen kosten Geld: weniger Kinder pro Lehrkraft bedeutet mehr
Lehrkräfte und mehr Räume. Bildungspolitisch ist die Klassengröße deshalb
eine der teuersten Stellschrauben überhaupt. Die Frage lautet schlicht:

> **Lernen Kinder in kleineren Klassen mehr?**

Der naheliegende Weg wäre, bestehende Klassen zu vergleichen. Der taugt aber
nichts: Kleine Klassen entstehen selten zufällig. Sie stehen in wohlhabenden
Gemeinden, an Privatschulen oder dort, wo besonders viele Kinder zusätzliche
Förderung brauchen. Wer kleine mit großen Klassen vergleicht, vergleicht in
Wahrheit unterschiedliche Familien, Schulen und Regionen.
"""
    )

    merkkasten(
        "Warum dieses Projekt eine gute Vorlage ist",
        "Der Datensatz enthält <b>zwei</b> Fragen: Die Klassengröße wurde "
        "verlost, die Erfahrung der Lehrkraft nicht. Beide Vergleiche liefern "
        "eine Zahl, aber nur eine davon darf man kausal lesen. Genau diese "
        "Unterscheidung ist der Kern eurer Projektarbeit.",
        typ="definition",
    )

    st.markdown("## Das Experiment")
    st.markdown(
        """
Der Bundesstaat Tennessee hat die Frage zwischen 1985 und 1989 tatsächlich
experimentell beantwortet: Im **Project STAR** (Student/Teacher Achievement
Ratio) wurden rund 11.600 Kinder beim Schuleintritt **per Los** einem von drei
Versuchsarmen zugeteilt, und die Lehrkräfte gleich mit:

- **Normale Klasse** mit 22–25 Kindern (Referenzgruppe)
- **Normale Klasse plus Hilfskraft**, die billigere Alternative
- **Kleine Klasse** mit 13–17 Kindern

Am Ende des Kindergartenjahres schrieben alle Kinder dieselben standardisierten
Tests in Lesen und Mathematik. Wir betrachten hier die Kindergarten-Kohorte,
also das erste Jahr des Experiments.
"""
    )

    st.markdown(
        """
**Warum die Randomisierung alles ändert:** Weil das Los entscheidet, gibt es
keinen Grund, warum die Kinder in kleinen Klassen im Schnitt aus reicheren
Familien kämen oder von Haus aus begabter wären. Alle Störfaktoren, auch die,
die niemand gemessen hat, verteilen sich gleichmäßig auf die drei Arme. Der
schlichte Mittelwertvergleich wird damit zur kausalen Aussage. Das ist der
Grund, warum sich der ganze Aufwand eines Experiments lohnt.
"""
    )

    st.page_link(
        "views/kausalitaet/potential_outcomes.py",
        label="Theorie dazu: Potential Outcomes & RCTs",
        icon="⚖️",
    )

# ============================================================ 2 Daten & EDA
with tab_daten:
    st.markdown("## Was steckt in den Daten?")
    st.markdown(
        f"""
Der Datensatz umfasst **{len(daten):,} Kinder** der Kindergarten-Kohorte mit
{daten.shape[1] - 2} erhobenen Merkmalen. Das Ergebnis (`gesamt`) ist der
Mittelwert aus Lese- und Mathematik-Score.
""".replace(",", ".")
    )

    st.dataframe(
        analyse.ueberblick(daten).style.format(
            {"Mittlerer Score": "{:.1f}", "Standardabweichung": "{:.1f}"}
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### Verteilung der Testergebnisse")
    st.markdown(
        """
Erster Blick vor jeder Modellierung: Wie sehen die Rohdaten aus? Die drei
Verteilungen liegen fast übereinander. Was wir suchen, ist eine leichte
Verschiebung, kein dramatischer Sprung.
"""
    )

    fig_dichte = go.Figure()
    farben_arm = [FARBEN["gletscher"], FARBEN["schiefer"], FARBEN["sonne"]]
    for (label, werte), farbe in zip(analyse.verteilung_je_arm(daten).items(), farben_arm):
        fig_dichte.add_histogram(
            x=werte,
            name=label,
            opacity=0.55,
            nbinsx=45,
            marker_color=farbe,
        )
    fig_dichte.update_layout(
        barmode="overlay",
        xaxis_title="Gesamtscore (Mittel aus Lesen und Mathematik)",
        yaxis_title="Anzahl Kinder",
        height=400,
    )
    st.plotly_chart(fig_dichte, use_container_width=True)

    st.markdown("### Fehlende Werte")
    st.markdown(
        """
Nicht jedes Kind hat am Test teilgenommen: Umzug, Krankheit, Wechsel auf eine
andere Schule. Solche Lücken gehören in jede ehrliche Datenbeschreibung, nicht
in eine Fußnote.
"""
    )
    st.dataframe(
        analyse.ausfall_uebersicht(daten).style.format({"Anteil fehlend": "{:.1%}"}),
        hide_index=True,
        use_container_width=True,
    )
    st.info(
        "Die Ausfallquote liegt in allen drei Armen bei rund 7 %. Wäre sie in "
        "einem Arm deutlich höher, hätten wir ein Problem. Dazu mehr im Tab "
        "„Limitationen“."
    )

# ========================================================= 3 Naive Analyse
with tab_naiv:
    st.markdown("## Zwei Vergleiche, zwei Zahlen")
    st.markdown(
        """
Bevor wir irgendetwas modellieren, rechnen wir das Einfachste: Mittelwerte
vergleichen. Und zwar für **zwei** Merkmale gleichzeitig: die Klassengröße
und die Berufserfahrung der Lehrkraft.
"""
    )

    spalte_links, spalte_rechts = st.columns(2)

    with spalte_links:
        st.markdown("### A · Klassengröße")
        roh_klasse = analyse.rohdifferenz(daten, "stark", "regular")
        st.dataframe(
            roh_klasse.style.format(
                {"Mittelwert": "{:.1f}", "Differenz zur Referenz": "{:+.1f}"}
            ),
            hide_index=True,
            use_container_width=True,
        )
        klein = roh_klasse.loc[roh_klasse["Gruppe"] == "small", "Differenz zur Referenz"]
        st.metric("Kleine vs. normale Klasse", f"{klein.iloc[0]:+.1f} Punkte")

    with spalte_rechts:
        st.markdown("### B · Erfahrung der Lehrkraft")
        mit_erfahrung = daten.copy()
        mit_erfahrung["erfahrung"] = analyse.erfahrungsgruppe(mit_erfahrung)
        roh_erfahrung = analyse.rohdifferenz(mit_erfahrung, "erfahrung", "0–5 Jahre")
        st.dataframe(
            roh_erfahrung.style.format(
                {"Mittelwert": "{:.1f}", "Differenz zur Referenz": "{:+.1f}"}
            ),
            hide_index=True,
            use_container_width=True,
        )
        erfahren = roh_erfahrung.loc[
            roh_erfahrung["Gruppe"] == "über 15 Jahre", "Differenz zur Referenz"
        ]
        st.metric("Über 15 Jahre vs. 0–5 Jahre", f"{erfahren.iloc[0]:+.1f} Punkte")

    st.markdown("## Die entscheidende Frage")
    merkkasten(
        "Welche der beiden Zahlen darf man kausal lesen?",
        "Der Erfahrungseffekt ist der <b>größere</b> von beiden. Wer nur auf "
        "die Zahlen schaut, würde folgern: Erfahrene Lehrkräfte bringen mehr "
        "als kleine Klassen. Überlege selbst, bevor du weiterliest.",
        typ="achtung",
    )

    with st.expander("Auflösung"):
        st.markdown(
            """
Nur **A**, die Klassengröße, wurde per Los zugeteilt. Die Berufserfahrung
nicht, und sie verteilt sich alles andere als zufällig: Erfahrene Lehrkräfte
können sich ihre Schule eher aussuchen, landen häufiger an begehrten
Standorten und unterrichten dort Kinder mit günstigerem Hintergrund. Der
Unterschied von rund 11 Punkten mischt den Beitrag der Lehrkraft mit dem der
Schule und des Elternhauses. Wie viel davon auf welchen Anteil entfällt, sagen
uns diese Daten nicht.

Drei Irrtümer, die hier nahe liegen:

- *„Es sind dieselben Daten aus demselben Experiment, also gilt beides.“*
  Ein Experiment adelt nicht den ganzen Datensatz, sondern nur die Variable,
  die tatsächlich verlost wurde.
- *„Der Erfahrungseffekt ist größer, also deutlicher.“* Die Größe eines
  Unterschieds sagt nichts über seine kausale Gültigkeit. Ein großer
  verzerrter Effekt bleibt verzerrt.
- *„Mit einer Regression ließe sich das reparieren.“* Eine Regression ist ein
  Rechenwerkzeug, kein Identifikationsargument. Ohne Randomisierung oder eine
  andere glaubwürdige Quelle exogener Variation macht auch die beste
  Regression aus Korrelation keine Kausalität.
"""
        )

    st.page_link(
        "views/kausalitaet/korrelation.py",
        label="Mehr dazu: Korrelation ≠ Kausalität",
        icon="🔀",
    )

# ========================================================= 4 Identifikation
with tab_ident:
    st.markdown("## Warum wir dem Vergleich glauben")
    st.markdown(
        r"""
Der Effekt, den wir schätzen wollen, ist der durchschnittliche
Behandlungseffekt

$$
\mathrm{ATE} = E\big[Y^{\text{klein}} - Y^{\text{normal}}\big],
$$

also die Differenz zwischen zwei Welten, von denen wir pro Kind immer nur eine
beobachten. Die Randomisierung schließt diese Lücke: Sie macht die Zuteilung
unabhängig von den potenziellen Ergebnissen, und damit wird die beobachtete
Mittelwertdifferenz zum ATE.

Das ist eine Behauptung über das Design, keine über die Daten. Prüfen lässt
sie sich trotzdem, wenn auch nur indirekt.
"""
    )

    st.markdown("### Prüfung 1: Balance-Check")
    st.markdown(
        """
Wenn das Los funktioniert hat, müssen sich die drei Arme in den Merkmalen
ähneln, die **vor** der Zuteilung feststanden. Diese Merkmale kann die
Behandlung nicht beeinflusst haben, ein systematischer Unterschied wäre also
ein Warnsignal.
"""
    )
    balance = analyse.balance_tabelle(daten)
    st.dataframe(
        balance.style.format(
            {
                "Normale Klasse (22–25 Kinder)": "{:.1%}",
                "Normale Klasse + Hilfskraft": "{:.1%}",
                "Kleine Klasse (13–17 Kinder)": "{:.1%}",
                "Max. Differenz (%-Punkte)": "{:.1f}",
                "p-Wert (Merkmal)": "{:.3f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.success(
        "**Merkmale der Kinder:** Geschlecht, ethnische Zugehörigkeit und der "
        "Anspruch auf kostenloses Mittagessen verteilen sich sehr gleichmäßig "
        "(alle p > 0,05). Das ist genau das, was ein funktionierendes Los "
        "produzieren soll, und was der Vergleich erfahrener und unerfahrener "
        "Lehrkräfte aus Tab 3 nicht vorweisen kann."
    )
    st.warning(
        "**Der Schultyp fällt aus der Reihe** (p ≈ 0,001). Bevor man daraus "
        "auf eine kaputte Randomisierung schließt, lohnt der Blick auf das "
        "Design. Die Auflösung steht direkt darunter."
    )

    with st.expander("Warum der Schultyp nicht balanciert sein muss"):
        st.markdown(
            """
Der Schultyp ist ein Merkmal der **Schule**, nicht des Kindes. Zugelost wurde
aber innerhalb jeder Schule. Über die Zusammensetzung der gepoolten Arme sagt
das Los deshalb nichts aus, und zwei Dinge sorgen dafür, dass sie sich
tatsächlich leicht unterscheidet:

1. Kleine Klassen fassen weniger Kinder (13–17 statt 22–25). Eine Schule
   steuert zu einem Arm also automatisch unterschiedlich viele Kinder bei.
2. Nicht jede Schule richtete gleich viele Klassen pro Arm ein.

Das Ergebnis: rund 49 % der Kinder in normalen Klassen kommen von ländlichen
Schulen, aber nur 45 % in kleinen Klassen. Der Unterschied ist mit maximal
5 Prozentpunkten klein, bei 6.325 Kindern reicht das trotzdem für ein
signifikantes Testergebnis. Ein schönes Beispiel dafür, dass ein p-Wert die
*Größe* eines Unterschieds nicht misst.

Gefährlich wäre der Befund, wenn er auf Kindebene aufträte: Dann hätte das
Los versagt. Hier folgt er aus der Bauart des Experiments. Und er ist genau
der Grund, warum der nächste Abschnitt Schul-Dummies in die Regression
aufnimmt: Sie nehmen den Schulvergleich vollständig heraus.
"""
        )

    st.markdown("### Prüfung 2: Zugelost wurde innerhalb der Schule")
    st.markdown(
        """
Ein Detail des Designs mit Folgen für die Schätzung: Das Los entschied nicht
landesweit, sondern **innerhalb jeder Schule**. Kinder wurden also nicht auf
andere Schulen verteilt, sondern nur auf Klassen ihrer eigenen Schule.

Der Vergleich zwischen Schulen gehört damit nicht in den Effekt. Wir nehmen
deshalb Schul-Dummies (*School Fixed Effects*) in die Regression auf. Das ist
hier ausdrücklich **keine** Korrektur für Confounding, denn das erledigt die
Randomisierung bereits, sondern schlicht die Abbildung des Designs. Der
Punktschätzer ändert sich kaum, das Konfidenzintervall wird enger.

Weil die Zuteilung klassen- und schulweise erfolgte, sind die Ergebnisse von
Kindern derselben Schule nicht unabhängig. Die Standardfehler clustern wir
darum auf Schulebene.
"""
    )

    st.markdown("### Was wir zusätzlich annehmen")
    merkkasten(
        "Annahmen jenseits der Randomisierung",
        "<b>SUTVA:</b> Das Ergebnis eines Kindes hängt nur von seiner eigenen "
        "Klassenzuteilung ab, nicht von der anderer Kinder. Bei Kindern "
        "derselben Schule, deren Lehrkräfte sich im Lehrerzimmer austauschen, "
        "ist das eine echte Annahme und keine Selbstverständlichkeit. "
        "<b>Vollständige Umsetzung:</b> Kinder blieben im zugelosten Arm. "
        "<b>Ausfälle sind zufällig:</b> Fehlende Testergebnisse hängen nicht "
        "mit dem Arm zusammen. Belege dafür stehen im Tab „Limitationen“.",
        typ="definition",
    )

# =============================================================== 5 Ergebnis
with tab_ergebnis:
    st.markdown("## Der geschätzte Effekt")

    mit_fe = st.toggle(
        "School Fixed Effects einbeziehen (entspricht dem Design)",
        value=True,
        key="beispiel_star_fe",
    )
    ergebnis = analyse.ate_mit_ci(daten, schul_fe=mit_fe)

    spalten = st.columns(2)
    for spalte, (_, zeile) in zip(spalten, ergebnis.iterrows()):
        with spalte:
            st.metric(
                zeile["Vergleich"].split(" vs. ")[0],
                f"{zeile['Effekt (Punkte)']:+.1f} Punkte",
                delta=f"{zeile['Effekt (SD)']:+.2f} SD",
                delta_color="off",
            )
            st.caption(
                f"95 %-KI: [{zeile['KI unten']:.1f}, {zeile['KI oben']:.1f}] · "
                f"p = {zeile['p-Wert']:.4f}"
            )

    fig_effekt = go.Figure()
    for i, (_, zeile) in enumerate(ergebnis.iterrows()):
        fig_effekt.add_scatter(
            x=[zeile["Effekt (Punkte)"]],
            y=[zeile["Vergleich"].split(" vs. ")[0]],
            mode="markers",
            marker=dict(size=14, color=FARBEN["sonne"] if i else FARBEN["schiefer"]),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[zeile["KI oben"] - zeile["Effekt (Punkte)"]],
                arrayminus=[zeile["Effekt (Punkte)"] - zeile["KI unten"]],
                thickness=2,
            ),
            showlegend=False,
        )
    fig_effekt.add_vline(x=0, line_dash="dot", line_color=FARBEN["schiefer"])
    fig_effekt.update_layout(
        xaxis_title="Effekt auf den Gesamtscore (Punkte, gegenüber normaler Klasse)",
        height=280,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_effekt, use_container_width=True)

    st.markdown(
        """
### Wie liest man das?

Kleine Klassen heben den Gesamtscore um gut **7 Punkte**, das entspricht rund
**0,2 Standardabweichungen**. In der Bildungsforschung gilt das als spürbarer,
aber nicht spektakulärer Effekt, grob die Größenordnung, die ein zusätzliches
Drittel Schuljahr bringt. Das Konfidenzintervall schließt die Null deutlich
aus.

Der zweite Befund ist der interessantere: Die **Hilfskraft in der normalen
Klasse bringt praktisch nichts**. Das Konfidenzintervall umfasst die Null,
der Punktschätzer liegt nahe bei ihr. Für die Bildungspolitik ist das die
teure Nachricht: Es reicht nicht, eine weitere erwachsene Person in den Raum
zu stellen, es kommt offenbar auf die tatsächliche Gruppengröße an.
"""
    )

    merkkasten(
        "Ein Nullbefund ist ein Befund",
        "Dass die Hilfskraft nichts bewirkt, ist kein Misserfolg der Analyse, "
        "sondern ein Ergebnis. Wichtig ist die Unterscheidung: „kein Effekt "
        "nachweisbar“ heißt nicht „Effekt bewiesenermaßen null“. Hier ist das "
        "Konfidenzintervall eng genug, um große Effekte auszuschließen, und "
        "genau das macht die Aussage belastbar.",
        typ="merke",
    )

    st.markdown("### Wirkt der Effekt bei allen gleich?")
    st.markdown(
        """
Der ATE ist ein Durchschnitt und kann sehr unterschiedliche Einzeleffekte
verdecken. Wir teilen die Stichprobe deshalb nach dem Anspruch auf kostenloses
Mittagessen, dem Indikator für den sozioökonomischen Hintergrund, und
schätzen den Effekt getrennt. Beide Teilgruppen sind **vorab** festgelegt.
"""
    )
    hetero = analyse.heterogenitaet(daten)
    hetero_anzeige = hetero.copy()
    hetero_anzeige["Teilgruppe"] = hetero_anzeige["Teilgruppe"].map(
        {"free": "Anspruch auf kostenloses Essen", "non-free": "Kein Anspruch"}
    )
    st.dataframe(
        hetero_anzeige.style.format(
            {
                "Effekt (Punkte)": "{:+.1f}",
                "KI unten": "{:.1f}",
                "KI oben": "{:.1f}",
                "p-Wert": "{:.4f}",
                "Effekt (SD)": "{:+.2f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.markdown(
        """
Kinder aus einkommensschwächeren Haushalten profitieren stärker (rund 0,26 SD
gegenüber 0,17 SD). Die Konfidenzintervalle überlappen allerdings deutlich,
als Hinweis interessant, als Beweis zu schwach. Wer solche Unterschiede
belastbar untersuchen will, braucht Methoden, die für dieses Suchen nach
Teilgruppen ausgelegt sind (Causal Forests, CATE-Schätzung).
"""
    )
    st.page_link(
        "views/kausalitaet/kausales_ml.py",
        label="Genau darum geht es hier: Kausales Machine Learning",
        icon="🎯",
    )

# =========================================================== 6 Limitationen
with tab_grenzen:
    st.markdown("## Was diese Analyse nicht zeigt")
    st.markdown(
        """
Der ehrlichste Teil jeder Arbeit. Wer die Grenzen der eigenen Ergebnisse
benennt, macht sie nicht schwächer, sondern glaubwürdig.
"""
    )

    with st.expander("1 · Ausfälle: Wer fehlt am Ende?", expanded=True):
        st.markdown(
            """
Rund 7 % der Kinder haben kein Testergebnis. Solange dieser Anteil in allen
Armen gleich hoch ist, bleibt der Vergleich intakt, und das ist hier der
Fall (7,3 % / 6,9 % / 7,2 %).

Bedenklich wäre der umgekehrte Fall: Wenn zum Beispiel schwächere Kinder aus
großen Klassen häufiger die Schule wechselten, würde die Kontrollgruppe
künstlich besser aussehen und wir würden den Effekt unterschätzen. Die
Randomisierung schützt vor Verzerrung bei der Zuteilung, nicht vor Verzerrung
beim Ausscheiden.
"""
        )

    with st.expander("2 · Keine Verblindung"):
        st.markdown(
            """
Lehrkräfte, Eltern und Schulleitungen wussten, in welchem Arm sie waren. Ein
Teil des Effekts könnte daher aus veränderter Motivation stammen statt aus der
Gruppengröße selbst: Lehrkräfte in kleinen Klassen könnten sich beobachtet
gefühlt und mehr Einsatz gezeigt haben. Was wir messen, ist der Effekt der
Zuteilung *inklusive* aller Reaktionen darauf. Für die politische Frage
(„Was passiert, wenn wir Klassen verkleinern?“) ist das sogar die
relevantere Größe.
"""
        )

    with st.expander("3 · Externe Validität"):
        st.markdown(
            """
Tennessee, 1985, Kindergarten. Die Schulen waren überwiegend ländlich, die
Lehrpläne von damals sind nicht die von heute, und „klein“ hieß 13–17 Kinder.
In Ländern mit Klassen von 35 Kindern ist das eine ganz andere Intervention.
Der Effekt gilt zunächst für diese Population. Die Übertragung auf andere
Kontexte ist eine inhaltliche Behauptung, die Daten allein stützen sie nicht.
"""
        )

    with st.expander("4 · Nur das erste Jahr"):
        st.markdown(
            """
Wir betrachten hier die Kindergarten-Kohorte. Der vollständige Datensatz
verfolgt die Kinder über vier Jahrgänge, und die spannenden Fragen fangen dort
erst an: Hält der Effekt an? Wächst er? Und wie geht man damit um, dass
Kinder in späteren Jahren die Schule wechselten? Ein lohnender nächster
Schritt für eine Gruppe, die auf diesem Projekt aufsetzen will.
"""
        )

    with st.expander("5 · SUTVA zwischen Klassen derselben Schule"):
        st.markdown(
            """
Wenn eine Schule Kinder in kleine Klassen umverteilt, werden die verbleibenden
Klassen tendenziell voller oder die besten Lehrkräfte anders eingesetzt. Dann
hängt das Ergebnis eines Kindes auch von der Zuteilung anderer ab, und die
SUTVA-Annahme wackelt. STAR hat dieses Problem durch zusätzliche Lehrkräfte
klein gehalten, ausschließen lässt es sich nicht.
"""
        )

    st.markdown("## Fazit")
    merkkasten(
        "Was bleibt",
        "Kleine Klassen wirken, mit rund 0,2 SD moderat, und stärker bei "
        "Kindern aus einkommensschwächeren Haushalten. Eine zusätzliche "
        "Hilfskraft wirkt nicht. Wir glauben diesen Zahlen nicht wegen der "
        "Regression, sondern wegen des Losverfahrens. Die Regression macht "
        "die Schätzung nur präziser. Und der größere Rohunterschied bei der "
        "Lehrererfahrung aus Tab 3 bleibt genau das: ein Rohunterschied.",
        typ="merke",
    )

    st.markdown("## Quellen")
    st.markdown(
        """
- C. M. Achilles et al. (2008), *Tennessee's Student Teacher Achievement Ratio (STAR) Project*, Harvard Dataverse
- A. B. Krueger (1999), *Experimental Estimates of Education Production Functions*, Quarterly Journal of Economics 114(2), 497–532
- Datensatz bezogen über das R-Paket `AER` (Kleiber & Zeileis 2008). Details in `daten/QUELLE.md`
"""
    )

st.divider()
st.markdown(
    """
### Für eure eigene Projektseite

Der Code liegt in eurem Projektordner und besteht aus zwei Dateien:
`analyse.py` enthält ausschließlich die Berechnungen, `app.py` nur die
Darstellung. Diese Trennung lohnt sich: Die Analysefunktionen lassen sich
testen und in einem Notebook wiederverwenden, ohne dass Streamlit läuft.

Wie ihr euren Ordner anlegt, steht in der Projektübersicht.
"""
)
st.page_link("views/projekte/uebersicht.py", label="Zur Projektübersicht", icon="🗂️")
