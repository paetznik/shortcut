# ShortCut

Erstellt aus langen deutschsprachigen Videos automatisch Shorts/Reels im Hochformat 9:16 – mit Highlight-Auswahl und animierten Wort-für-Wort-Untertiteln.

## Ablauf

1. Video holen: Upload im Browser (in 8-MB-Stücken, daher auch große Dateien), Datei aus einem Nextcloud-Ordner oder automatisch: neue Videos im Nextcloud-Unterordner `Eingang` werden erkannt, nach `Eingang/Verarbeitet` verschoben und mit den Standard-Einstellungen verarbeitet.
2. Transkription auf Deutsch mit [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (lokal, CPU). Eine Liste mit Fachbegriffen und Namen in den Einstellungen verbessert die Schreibweise.
3. Tonanalyse (CPU, wenige Minuten pro Stunde Video): Lautstärke, Tonhöhen-Schwankung (lebhafte Stimme) und Sprechtempo relativ zum normalen Sprechen, dazu laute Stellen ohne Sprache (Lachen, Applaus). Das fließt in die Auswahl ein; ein Lacher direkt nach einem Clip wird mitgenommen und beim Pausen-Schnitt nicht entfernt. Die Projektansicht zeigt den Energieverlauf mit den Clips.
4. Highlight-Auswahl:
   - mit Anthropic-API-Schlüssel (in den Einstellungen hinterlegen): Claude wählt die Ausschnitte und schreibt Titel und Beschreibungen. Standardmäßig in zwei Durchgängen: erst 12–20 Kandidaten, streng bewertet nach Hook, Verständlichkeit ohne Vorwissen, Mehrwert/Emotion und Pointe; dann ein Feinschnitt der Besten auf Satzebene (schwache Einstiege weg, Ende nach der Pointe), bei dem schwache Kandidaten verworfen werden,
   - ohne Schlüssel: automatische Auswahl nach Sprechtempo, Fragen und Schlüsselwörtern.
5. Optional Prüfmodus: Titel, Hook-Text (mit KI-Varianten), Titelbild-Text und Untertitel jedes Clips vor dem Rendern korrigieren; Wörter im Transkript anklicken schneidet sie heraus.
6. Schnitt mit ffmpeg:
   - 9:16-Zuschnitt, der dem Sprecher folgt, oder geteilter Bildschirm bei zwei Personen (links oben, rechts unten) (Gesichtserkennung mit OpenCV, Mundbewegung entscheidet bei mehreren Personen), alternativ Mitte oder ganzes Bild mit unscharfem Hintergrund; optional zusätzlich 4:5 und 1:1,
   - Pausen, Füllwörter und angeklickte Wörter werden herausgeschnitten, dynamische Zooms an Schnitten und Satzanfängen,
   - Hook-Text in den ersten Sekunden, Untertitel in vier Stilen (Pop, Box, Minimal, ganze Sätze) mit farbigen Schlüsselwörtern, innerhalb der Sicherheitszonen der Plattformen,
   - Tonaufbereitung (Rumpelfilter, Rauschunterdrückung, De-Esser, Kompressor, Lautheit −14 LUFS) und optionale Hintergrundmusik, die unter Sprache leiser wird,
   - Titelbild mit großem Text für das Profilraster.
7. Bildausschnitt von Hand: im Editor auf einem Standbild den 9:16-Rahmen ziehen, auch mit Wechseln ab bestimmten Sekunden.
8. Best-of-Trailer (30–60 s) aus den stärksten Sätzen mehrerer Clips; übersetzte Versionen jedes Clips (Untertitel, Hook, Titel, Beschreibung) per Claude in Englisch, Spanisch, Französisch, Italienisch, Türkisch oder Polnisch.
9. Qualitäts-Check nach dem Rendern: Lautstärke, Schnitt mitten im Wort, angeschnittene Personen, fehlende Gesichter; mit Claude-Schlüssel zusätzlich Rechtschreibprüfung der Untertitel mit Ein-Klick-Korrektur.
10. Texte zum Video per Claude: YouTube-Kapitel, LinkedIn-Beitrag, Blog-/Newsletter-Text und Karussell-Folien.
11. Export für Schnittprogramme: ZIP mit EDL je Clip (Premiere, DaVinci), FCPXML (Final Cut, DaVinci), SRTs und Anleitung.
12. Betrieb: Warteschlange mit Restzeit (lernt die Geschwindigkeit des Servers), Umsortieren und Abbrechen; Transkript und Tonanalyse werden übernommen, wenn dasselbe Video erneut hochgeladen wird; Speicheranzeige mit Löschen der Originalvideos; Claude-Kosten pro Projekt und pro Monat.
13. Download (MP4 je Format, Titelbild, SRT), automatischer Export nach Nextcloud und Versand an Postiz (planen oder sofort veröffentlichen). Clips lassen sich nachträglich ändern und neu rendern oder als Variante B mit anderem Hook duplizieren.

## Deployment

`docker-compose.yaml` baut das Image direkt aus diesem Repository und hängt es an Traefik (Let's-Encrypt-Zertifikat).
Beim ersten Aufruf legst du in der Weboberfläche ein Passwort fest.

Daten (Uploads, Clips, Einstellungen) liegen im Volume `data` und werden nach 14 Tagen gelöscht (`RETENTION_DAYS`).
Die Whisper-Modelle werden beim ersten Gebrauch ins Volume `models` geladen.

## Lokal starten

```bash
docker build -t shortcut .
docker run -p 8000:8000 -v shortcut-data:/data shortcut
```
