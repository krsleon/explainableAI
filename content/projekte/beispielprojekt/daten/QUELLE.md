# Datenquelle: Project STAR

## Was ist das?

**Project STAR** (Student/Teacher Achievement Ratio) war ein großangelegtes
randomisiertes Bildungsexperiment im US-Bundesstaat Tennessee. Zwischen 1985
und 1989 wurden rund 11.600 Kinder an 79 Schulen beim Schuleintritt per Los
einem von drei Versuchsarmen zugeteilt:

- `regular`: normale Klasse (22–25 Kinder)
- `regular+aide`: normale Klasse mit zusätzlicher Hilfskraft
- `small`: kleine Klasse (13–17 Kinder)

Auch die Lehrkräfte wurden den Klassen zugelost. Die Kinder wurden über vier
Jahrgänge (Kindergarten bis 3. Klasse) begleitet und schrieben jeweils
standardisierte Tests in Lesen und Mathematik.

## Woher diese Datei kommt

Bezogen über [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/), das
den Datensatz `STAR` aus dem R-Paket **AER** (Kleiber & Zeileis 2008) als CSV
spiegelt:

```
https://vincentarelbundock.github.io/Rdatasets/csv/AER/STAR.csv
```

## Aufbereitung

Aus dem Originaldatensatz (11.598 Zeilen × 48 Spalten, ~2,1 MB) wurde die
**Kindergarten-Kohorte** herausgeschnitten, also die Kinder, für die eine
Zuteilung im ersten Jahr vorliegt (`stark` nicht fehlend), und auf die
Spalten reduziert, die dieses Projekt benutzt. Sonst wurde nichts verändert:
keine Zeilen gefiltert, keine fehlenden Werte ersetzt, keine Werte
umkodiert.

```python
import pandas as pd

d = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/AER/STAR.csv")
cols = [
    "gender", "ethnicity", "birth", "stark", "readk", "mathk", "lunchk",
    "schoolk", "degreek", "ladderk", "experiencek", "tethnicityk", "schoolidk",
]
d.loc[d["stark"].notna(), cols].to_csv("star_kindergarten.csv", index=False)
```

Ergebnis: 6.325 Zeilen × 13 Spalten, rund 520 KB.

## Spalten

Das Suffix `k` steht durchgehend für *kindergarten*, also das erste
Erhebungsjahr.

| Spalte | Bedeutung |
|--------|-----------|
| `stark` | Versuchsarm: `regular`, `regular+aide`, `small`. **Die zugeloste Variable** |
| `readk` | Score im Lesetest (Stanford Achievement Test) |
| `mathk` | Score im Mathematiktest |
| `gender` | Geschlecht des Kindes |
| `ethnicity` | Ethnische Zugehörigkeit des Kindes |
| `birth` | Geburtsquartal |
| `lunchk` | Anspruch auf kostenloses Mittagessen (`free` / `non-free`), üblicher Indikator für den sozioökonomischen Hintergrund |
| `schoolk` | Schultyp: `rural`, `suburban`, `urban`, `inner-city` |
| `schoolidk` | Schul-ID, Grundlage für Fixed Effects und geclusterte Standardfehler |
| `degreek` | Höchster Abschluss der Lehrkraft |
| `ladderk` | Karrierestufe der Lehrkraft im Tennessee-System |
| `experiencek` | Berufsjahre der Lehrkraft. **Nicht zugelost**, dient im Projekt als Gegenbeispiel |
| `tethnicityk` | Ethnische Zugehörigkeit der Lehrkraft |

Fehlende Werte sind vorhanden und wurden bewusst beibehalten: 536 Kinder ohne
Lesescore, 454 ohne Mathescore, 596 ohne Angabe zur Karrierestufe. Sie sind
Thema im Tab „Limitationen“ der Projektseite.

## Lizenz und Zitation

Der Datensatz ist Teil des R-Pakets `AER`, das unter GPL-2/GPL-3 steht. Die
Erhebung selbst wurde vom Bundesstaat Tennessee finanziert und ist über das
Harvard Dataverse öffentlich zugänglich.

Bei Verwendung zu zitieren:

> Achilles, C. M., Bain, H. P., Bellott, F., Boyd-Zaharias, J., Finn, J.,
> Folger, J., Johnston, J., & Word, E. (2008). *Tennessee's Student Teacher
> Achievement Ratio (STAR) project.* Harvard Dataverse.

Die kanonische ökonometrische Auswertung:

> Krueger, A. B. (1999). Experimental Estimates of Education Production
> Functions. *Quarterly Journal of Economics*, 114(2), 497–532.

> Kleiber, C., & Zeileis, A. (2008). *Applied Econometrics with R.* Springer.
