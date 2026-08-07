"""Werkzeuge: Was ist Streamlit?

Erklärt das Ausführungsmodell (Skript läuft bei jeder Interaktion neu), die
wichtigsten Bausteine (Ausgabe, Widgets, Layout, Diagramme), Session State
und Caching, jeweils mit lauffähigen Mini-Demos auf der Seite selbst.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.theming import FARBEN, kapitel_kopf, merkkasten, vertiefung

kapitel_kopf(
    "🎈",
    "Was ist Streamlit?",
    "Von einem Python-Skript zur interaktiven Website, ganz ohne HTML",
)

# ---------------------------------------------------------------- Intro
st.markdown(
    """
Diese Website ist keine klassische Webseite, sondern ein paar Python-Dateien.
Genau das ist die Idee von **Streamlit**: Du schreibst normales Python, und
Streamlit macht daraus eine Seite im Browser. Kein HTML, kein CSS, kein
JavaScript, keine Serverkonfiguration.

Für eure Gruppenprojekte heißt das: Ihr analysiert eure Daten wie gewohnt mit
`pandas` und `scikit-learn`, hängt ein paar `st.`-Zeilen darum, und heraus
kommt eine Seite, die auch Leute bedienen können, die euren Code nie sehen.
Jede Abbildung in diesem Kapitelteil, jeder Schieberegler, jeder Knopf ist auf
diese Weise entstanden.
"""
)

st.markdown("## Das Grundprinzip: dein Skript ist die App")
st.markdown(
    """
Streamlit kennt nur eine einzige Regel, und wenn du die verstanden hast, ist
der Rest Vokabeln:

**Bei jeder Interaktion läuft das komplette Skript von oben nach unten neu.**

Du schiebst einen Regler, und Streamlit startet die Datei erneut. Diesmal
liefert der Regler den neuen Wert, alles darunter wird mit diesem Wert neu
berechnet, und die Seite wird ausgetauscht. Es gibt keine Callbacks, die du
verdrahten musst, und keinen Zustand, den du selbst aktualisierst. Ein Widget
ist einfach eine Funktion, die einen Wert zurückgibt.
"""
)

st.code(
    """import streamlit as st

st.title("Meine erste App")

# Das Widget gibt seinen aktuellen Wert direkt zurück.
name = st.text_input("Wie heißt du?", "Welt")

st.write(f"Hallo, {name}!")""",
    language="python",
)

merkkasten(
    "Merke",
    "Ein Streamlit-Skript beschreibt <b>nicht</b>, was sich ändern soll, "
    "sondern wie die Seite bei den aktuellen Werten aussieht. Bei jeder "
    "Interaktion läuft es komplett neu, und Streamlit zeichnet das Ergebnis "
    "neu. Deshalb steht in einer Streamlit-Datei ganz normaler Python-Code "
    "von oben nach unten, ohne Ereignisbehandlung.",
    typ="merke",
)

# --------------------------------------------------- Bausteine in Tabs
st.markdown("## Die wichtigsten Bausteine")
st.markdown(
    """
Streamlit hat viele Funktionen, aber mit diesen vier Gruppen kommst du durch
ein ganzes Projekt. Die Reiter unten sind selbst ein Streamlit-Element
(`st.tabs`).
"""
)

tab_ausgabe, tab_widgets, tab_layout, tab_daten = st.tabs(
    ["Text ausgeben", "Widgets", "Layout", "Daten & Diagramme"]
)

with tab_ausgabe:
    st.code(
        """st.title("Größte Überschrift")
st.header("Abschnitt")
st.subheader("Unterabschnitt")

st.markdown("Normaler Text mit **fett**, *kursiv* und $E = mc^2$.")
st.write(irgendein_objekt)   # rät das passende Format, auch für DataFrames
st.code("print('hallo')", language="python")

st.metric("Geschätzter Effekt", "0.42", delta="+0.05")""",
        language="python",
    )
    st.markdown(
        "`st.markdown` versteht LaTeX zwischen Dollarzeichen, deshalb sehen "
        "die Formeln in den Kapiteln so aus, wie sie aussehen. `st.write` ist "
        "der bequeme Allrounder, wenn du nur schnell etwas sehen willst."
    )

with tab_widgets:
    st.code(
        """zahl   = st.slider("Stichprobengröße", 10, 500, 100)
wahl   = st.selectbox("Modell", ["OLS", "Lasso", "Ridge"])
an_aus = st.checkbox("Regressionsgerade zeigen", value=True)
text   = st.text_input("Titel der Abbildung")
datei  = st.file_uploader("CSV hochladen", type="csv")

if st.button("Neue Stichprobe ziehen"):
    ...   # Knöpfe geben True zurück, aber nur im Lauf direkt nach dem Klick""",
        language="python",
    )
    st.markdown(
        "Jedes Widget gibt seinen aktuellen Wert zurück. Wenn zwei Widgets "
        "dieselbe Beschriftung haben, brauchen sie ein eigenes `key=`, sonst "
        "hält Streamlit sie für dasselbe Element."
    )

with tab_layout:
    st.code(
        """links, rechts = st.columns(2)
with links:
    st.plotly_chart(figur_a)
with rechts:
    st.plotly_chart(figur_b)

with st.expander("Details zur Herleitung"):
    st.markdown("...")

with st.sidebar:
    st.slider("Globale Einstellung", 0, 10)

with st.container(border=True):
    st.markdown("Ein Kasten mit Rahmen")""",
        language="python",
    )
    st.markdown(
        "In diesem Projekt sind die wiederkehrenden Bausteine in "
        "`utils/theming.py` gebündelt: `kapitel_kopf`, `merkkasten`, "
        "`vertiefung` und `gruppen_aufgabe`. So sehen alle Seiten gleich aus, "
        "auch wenn verschiedene Gruppen sie schreiben."
    )

with tab_daten:
    st.code(
        """st.dataframe(df)              # scrollbare, sortierbare Tabelle
st.table(df.head())           # statische Tabelle
st.data_editor(df)            # Tabelle, die man bearbeiten kann

st.line_chart(df)             # schnell und ohne Konfiguration
st.plotly_chart(figur, use_container_width=True)   # volle Kontrolle
st.pyplot(matplotlib_figur)""",
        language="python",
    )
    st.markdown(
        "Für die Kapitel auf dieser Website nutzen wir durchgehend Plotly, "
        "weil man die Diagramme zoomen und einzelne Linien ausblenden kann. "
        "Für einen schnellen Blick in die Daten reicht `st.line_chart` völlig."
    )

# ------------------------------------------------ Demo 1: Widget + Plot
st.markdown("## Demo: Regler und Diagramm")
st.markdown(
    """
Ein vollständiges Beispiel mit allem, was eine kleine Analyse braucht: zwei
Widgets, eine Rechnung, eine Abbildung. Links steht der Code, rechts das
Ergebnis, und das Ergebnis ist echt. Probier die Regler aus.
"""
)

spalte_code, spalte_demo = st.columns([1, 1])

with spalte_code:
    st.code(
        """import numpy as np
import plotly.graph_objects as go
import streamlit as st

n = st.slider("Beobachtungen", 20, 500, 120)
rauschen = st.slider("Rauschen", 0.0, 3.0, 1.0)

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, n)
y = 0.8 * x + rng.normal(0, rauschen, n)

figur = go.Figure()
figur.add_scatter(x=x, y=y, mode="markers")
st.plotly_chart(figur)

st.write(f"Korrelation: {np.corrcoef(x, y)[0, 1]:.2f}")""",
        language="python",
    )

with spalte_demo:
    demo_n = st.slider("Beobachtungen", 20, 500, 120, key="sl_demo_n")
    demo_rauschen = st.slider("Rauschen", 0.0, 3.0, 1.0, key="sl_demo_noise")

    demo_rng = np.random.default_rng(0)
    demo_x = demo_rng.uniform(0, 10, demo_n)
    demo_y = 0.8 * demo_x + demo_rng.normal(0, demo_rauschen, demo_n)

    fig_demo = go.Figure()
    fig_demo.add_scatter(
        x=demo_x,
        y=demo_y,
        mode="markers",
        marker=dict(color=FARBEN["gletscher"], size=7, opacity=0.75),
        name="Beobachtungen",
    )
    fig_demo.update_layout(
        xaxis_title="x", yaxis_title="y", height=320, showlegend=False
    )
    st.plotly_chart(fig_demo, use_container_width=True)
    st.write(f"Korrelation: {np.corrcoef(demo_x, demo_y)[0, 1]:.2f}")

st.markdown(
    """
Beachte, was hier **nicht** steht: kein Code, der auf ein Ereignis wartet, und
keine Anweisung, die das Diagramm aktualisiert. Du beschreibst nur, wie die
Seite bei den aktuellen Reglerwerten aussieht. Um den Rest kümmert sich
Streamlit.
"""
)

# ------------------------------------------- Demo 2: Rerun sichtbar machen
st.markdown("## Demo: Jede Interaktion startet das Skript neu")

st.session_state["sl_laeufe"] = st.session_state.get("sl_laeufe", 0) + 1
st.session_state.setdefault("sl_klicks", 0)

if st.button("Knopf drücken", key="sl_knopf"):
    st.session_state["sl_klicks"] += 1

spalte_laeufe, spalte_klicks = st.columns(2)
spalte_laeufe.metric("Läufe dieses Skripts", st.session_state["sl_laeufe"])
spalte_klicks.metric("Gezählte Klicks", st.session_state["sl_klicks"])

st.markdown(
    """
Der linke Zähler steigt bei **jeder** Interaktion auf dieser Seite, auch wenn
du oben nur an einem Regler ziehst. Das ist der Beweis für die Regel von
vorhin: Streamlit führt die Datei jedes Mal komplett neu aus.

Damit stellt sich sofort eine Frage. Wenn alles neu läuft, wie überlebt dann
der rechte Zähler? Normale Variablen tun das nicht, sie werden bei jedem Lauf
neu angelegt. Wer etwas über Läufe hinweg behalten will, braucht
**`st.session_state`**, ein Wörterbuch, das pro Browser-Sitzung bestehen
bleibt.
"""
)

st.code(
    """# Startwert nur setzen, falls noch nichts da ist.
st.session_state.setdefault("klicks", 0)

if st.button("Knopf drücken"):
    st.session_state["klicks"] += 1

st.metric("Gezählte Klicks", st.session_state["klicks"])""",
    language="python",
)

st.markdown("## Teure Rechnungen nur einmal machen")
st.markdown(
    """
Die zweite Folge des Neustarts: Eine Simulation, die drei Sekunden braucht,
würde bei jedem Reglerzug drei Sekunden brauchen. Dagegen hilft der Decorator
**`@st.cache_data`**. Streamlit merkt sich das Ergebnis pro Argumentkombination
und ruft die Funktion nur dann wirklich auf, wenn die Argumente neu sind.
"""
)

st.code(
    """@st.cache_data
def simuliere(n: int, wiederholungen: int = 500):
    # Läuft nur beim ersten Mal pro (n, wiederholungen).
    return teure_rechnung(n, wiederholungen)

ergebnis = simuliere(n)   # ab dem zweiten Aufruf sofort da""",
    language="python",
)

with vertiefung("Stolperfallen bei Caching und Session State"):
    st.markdown(
        """
    **Alles, wovon das Ergebnis abhängt, muss ein Argument sein.** Greift die
    Funktion auf eine globale Variable oder eine Datei zu, die sich ändert,
    merkt Streamlit davon nichts und gibt weiter das alte Ergebnis zurück. In
    den Kapiteln dieser Website siehst du deshalb Muster wie
    `def daten(seed: int)`: Der Knopf „Neue Stichprobe“ erhöht den Seed, und
    erst dadurch entsteht ein neuer Cache-Eintrag.

    **Gecachte Objekte nicht verändern.** `@st.cache_data` gibt bei jedem
    Aufruf eine Kopie zurück, das ist sicher, kostet aber Zeit bei großen
    DataFrames. `@st.cache_resource` gibt dasselbe Objekt zurück und ist für
    Dinge gedacht, die man nur einmal aufbauen will, etwa eine
    Datenbankverbindung oder ein geladenes Modell.

    **Widgets schreiben selbst in den Session State.** Ein Widget mit
    `key="alpha"` legt seinen Wert unter `st.session_state["alpha"]` ab. Du
    kannst diesen Eintrag lesen, solltest ihn aber nicht im selben Lauf
    überschreiben, in dem das Widget gezeichnet wird. Sonst streiten sich zwei
    Quellen um denselben Wert, und Streamlit meldet einen Fehler.

    **Reihenfolge zählt.** Weil das Skript von oben nach unten läuft, kannst du
    einen Wert erst benutzen, nachdem das Widget dazu erzeugt wurde. Wer ein
    Ergebnis oben und die Steuerung unten haben will, nimmt Platzhalter
    (`platz = st.empty()` und später `platz.write(...)`).
    """
    )

# ---------------------------------------------------------- Projektseiten
st.markdown("## Wie eine Projektseite aussieht")
st.markdown(
    """
Eure Gruppenseite ist genau so eine Datei. Ihr legt einen Ordner unter
`content/projekte/` an, schreibt entweder ein Markdown-Dokument (ohne jedes
Streamlit-Wissen) oder eine eigene `app.py`, und die Seite erscheint
automatisch in der Navigation links. Als Gerüst gibt es
`content/projekte/_vorlage/` und das ausgearbeitete Beispielprojekt.
"""
)

merkkasten(
    "Drei Regeln für eure app.py",
    "Erstens kein <code>st.set_page_config()</code>, das macht nur die "
    "Hauptdatei. Zweitens nur Pakete, die in <code>requirements.txt</code> "
    "stehen, sonst startet die Seite online nicht. Drittens Dateien immer "
    "relativ zum eigenen Ordner laden, also "
    "<code>Path(__file__).parent / \"daten.csv\"</code> statt eines Pfades, "
    "der nur auf deinem Rechner existiert.",
    typ="achtung",
)

st.markdown("## Weiterführende Links")
st.markdown(
    """
- [Streamlit-Dokumentation](https://docs.streamlit.io) mit vollständiger API-Referenz
- [30 Days of Streamlit](https://30days.streamlit.app) als kurzer Einstiegskurs
- [Streamlit Gallery](https://streamlit.io/gallery) mit Beispiel-Apps und deren Quellcode
"""
)

st.markdown("## Wie geht es weiter?")
st.markdown(
    """
Bevor du selbst etwas startest, braucht dein Rechner Python und eine
Arbeitsumgebung. Die nächste Seite richtet beides mit `uv` und Visual Studio
Code ein.
"""
)

weiter_setup, weiter_projekte = st.columns(2)
with weiter_setup:
    st.page_link("views/python_setup.py", label="Weiter: Python Setup", icon="🐍")
with weiter_projekte:
    st.page_link(
        "views/projekte/uebersicht.py", label="Zu den Gruppenprojekten", icon="🗂️"
    )
