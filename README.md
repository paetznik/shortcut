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

## Qualität

- Bild: Bildrate des Originals (24–60 fps), Lanczos-Skalierung mit leichtem Nachschärfen, CRF 19; Warnung, wenn das Material für Hochformat zu wenig Auflösung hat.
- Bildausschnitt: YuNet-Gesichtserkennung (OpenCV) mit Mundbewegung als Sprechersignal, weiche Kamerafahrt mit Totzone, harte Schnitte nur bei Sprecherwechsel; Szenenwechsel werden erkannt, Clip-Grenzen auf Kameraschnitte gelegt.
- Ton: Schnitte bildgenau, 12-ms-Blenden an jeder Schnittstelle, RNNoise-Rauschunterdrückung.
- Text: Spracherkennung wahlweise mit large-v3-turbo, wortgenaue Zeitstempel per Forced Alignment (torchaudio MMS) für die Clip-Bereiche, Untertitel-Umbruch nach Sinngrenzen.
- Kontrolle: optionale KI-Endkontrolle anhand von Standbildern des fertigen Clips.
- Schnitt-Feinschliff: 40-ms-Kreuzblenden an jedem Tonschnitt (Länge bleibt exakt), Sprecherwechsel im Bild 0,2 s nach dem ersten Wort (J-Cut), Bildausschnitt wechselt an jedem Jump-Cut (Punch-in), Schnittpunkte rücken um wenige Bilder weg von geschlossenen Augen und Bewegungsunschärfe, Pausen nach Sprechtempo mit Spannungspausen nach Fragen und vor kurzen Pointen, Atmer in Pausen −9 dB, Raumton-Teppich aus der leisesten Stelle der Aufnahme, optionale Stabilisierung (vidstab), Blickraum in Blickrichtung und Zooms auf Augenhöhe, Untertitel 70 ms vor dem Wort, Loop-Ende, Bild steht nie länger als ca. 4,5 s still.

## Bedienung per Klick

- Gesamtes Transkript: erstes und letztes Wort anklicken → als eigener Clip oder als Serie („Teil 1/2/3“) anlegen.
- Clip-Editor: „In Teile zerlegen“ teilt einen Clip an Satzenden in eine Serie; jeder Teil außer dem letzten endet mit „Teil x folgt →“.
- Wörter überpiepen: im Editor auf „🔇 Überpiepen“ schalten und Wörter anklicken – Ton wird stummgeschaltet und mit 1-kHz-Piep überlagert, im Untertitel steht „a***“.
- Titelbild: automatisch das beste Standbild (scharf, gut belichtet, großes Gesicht mit offenen Augen, mittig).
- Texte zum Video: YouTube-Titel und fertige Beschreibung mit Kapiteln (Kapitel beginnen am Satzanfang, mind. 10 s, mind. 3). Mit dem Link zum veröffentlichten Video bekommt jeder Clip-Text „▶ Ganzes Video ab 12:34 – Kapitel …“ mit Sprungmarke.
- Schnitt-Ansicht im Clip-Editor: Tonwelle des Originals (Clip ±15 s), Anfang/Ende per Griff ziehen (rastet an Wortgrenzen ein, Alt = frei), Klick setzt die Abspielposition; Tasten: Leertaste abspielen, I/O Anfang/Ende, ←/→ 1 s (Umschalt 0,1 s), P Auswahl abspielen.
- Papierkorb: gelöschte Clips und durch Neu-Rendern ersetzte Fassungen bleiben 14 Tage erhalten und lassen sich wiederherstellen (eine ältere Fassung tauscht mit der aktuellen).
- Mehrere Videos auf einmal hochladen: alle bekommen dieselben Einstellungen und werden nacheinander verarbeitet.
- Morph-Cut (Standard): bei Jump-Cuts in derselben Einstellung werden Zwischenbilder mit Bewegungskompensation berechnet (minterpolate), Kopf und Körper gleiten in die neue Position; bei Kamera-, Ausschnitt- oder Sprecherwechsel und bei zu großer Bildänderung automatisch Überblendung. Bildanzahl exakt, Ton bleibt synchron.
- Feinregler: Länge von Morph-Cut/Überblendung (0,1–0,4 s) pro Projekt, Übergang pro Schnitt (Automatisch, Morph, Überblendung, hart) – an der Stelle im Original verankert.
- Vorher/Nachher: alte Fassung (Papierkorb) oder Vorschau neben der aktuellen, synchron abgespielt, Ton wahlweise links/rechts.
- Zwischenspeicher je Projekt für Gesichts-, Szenen-, Stimm-, Morph-, Raumton-, Unkenntlich- und Kamera-Analysen: Neu-Rendern rechnet nur, was sich geändert hat.
- Editor fürs Handy: Bereiche Schnitt/Texte/Bild/Extras einklappbar, große Griffe, Knöpfe statt Tastenkürzel, Aktionen unten fixiert.
- Freigabe-Link: öffentliche, nicht erratbare Seite /s/… mit allen fertigen Clips; Freigeben oder Änderung wünschen mit Namen und Kommentar; Rückmeldungen am Clip, Link jederzeit zurückziehbar.
- Schnelle Vorschau im Clip-Editor: ungespeicherte Änderungen in 540×960 in wenigen Sekunden ansehen, ohne die langsamen Extras; erst „Speichern und neu rendern“ erzeugt die volle Qualität.
- Übergang an Schnitten wählbar: Morph-Cut, weiche Überblendung ( 0,2 s, Länge und Lippensynchronität bleiben exakt), Zoom-Sprung oder harter Schnitt; Pausen werden nur noch deutlich gekürzt, kein Stück zwischen zwei Schnitten unter 1 s.
- Weitere Highlights nachträglich finden: im fertigen Projekt Anzahl und Länge wählen, ohne neues Hochladen; vorhandene Clips werden nicht wiederholt.
- Schnitt-Assistent im Clip-Editor: Wunsch in eigenen Worten („Nimm den Teil über Preise raus“), Claude setzt Start/Ende, Schnitte, Piepser und Texte um.
- Musik im Takt: Tempo und Beat der Hintergrundmusik werden erkannt, die Musik startet so, dass möglichst viele Schnitte auf einen Beat fallen, Zooms innerhalb einer Einstellung liegen auf dem Beat.
- Optional: 15/30/60-s-Fassungen je Highlight (an Satzenden gekürzt), Sprecherfarben in den Untertiteln (zwei Stimmen, Erkennung über Stimmlage und Klang), Gesichter im Hintergrund und Kennzeichen (experimentell) unkenntlich, eigene Bereiche per Rechteck weichzeichnen.
- Mehrere Kameras: weitere Aufnahmen hochladen, Synchronisation über den Ton, beim Rendern automatisch Schnitt auf die Kamera mit der sprechenden Person in Großaufnahme.
- App installieren (PWA): Android/Chrome „Zum Startbildschirm hinzufügen“, danach Videos per „Teilen → ShortCut“ direkt hochladen. iPhone: nur „Zum Home-Bildschirm“, ohne Teilen-Ziel.

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
