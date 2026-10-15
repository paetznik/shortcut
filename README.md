# ShortCut

Erstellt aus langen deutschsprachigen Videos automatisch Shorts/Reels im Hochformat 9:16 – mit Highlight-Auswahl und animierten Wort-für-Wort-Untertiteln.

## Ablauf

1. Video holen: Upload im Browser (in 8-MB-Stücken, daher auch große Dateien) oder Datei aus einem Nextcloud-Ordner.
2. Transkription auf Deutsch mit [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (lokal, CPU). Eine Liste mit Fachbegriffen und Namen in den Einstellungen verbessert die Schreibweise.
3. Highlight-Auswahl:
   - mit Anthropic-API-Schlüssel (in den Einstellungen hinterlegen): Claude wählt die Ausschnitte und schreibt Titel und Beschreibungen,
   - ohne Schlüssel: automatische Auswahl nach Sprechtempo, Fragen und Schlüsselwörtern.
4. Optional Prüfmodus: Titel, Hook-Text und Untertitel jedes Clips vor dem Rendern korrigieren.
5. Schnitt mit ffmpeg:
   - 9:16-Zuschnitt, der dem Sprecher folgt (Gesichtserkennung mit OpenCV, Mundbewegung entscheidet bei mehreren Personen), alternativ Mitte oder ganzes Bild mit unscharfem Hintergrund,
   - Pausen und Füllwörter werden herausgeschnitten,
   - Hook-Text in den ersten Sekunden, animierte Wort-für-Wort-Untertitel, Lautstärke-Normalisierung.
6. Download als MP4 plus SRT, automatischer Export nach Nextcloud und Versand an Postiz (planen oder sofort veröffentlichen). Start, Ende, Titel und Text lassen sich pro Clip nachträglich ändern und neu rendern.

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
