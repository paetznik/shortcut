# ShortCut

Erstellt aus langen deutschsprachigen Videos automatisch Shorts/Reels im Hochformat 9:16 – mit Highlight-Auswahl und animierten Wort-für-Wort-Untertiteln.

## Ablauf

1. Video in der Weboberfläche hochladen (in 8-MB-Stücken, daher auch große Dateien).
2. Transkription auf Deutsch mit [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (lokal, CPU).
3. Highlight-Auswahl:
   - mit Anthropic-API-Schlüssel (in den Einstellungen hinterlegen): Claude wählt die Ausschnitte und schreibt Titel und Beschreibungen,
   - ohne Schlüssel: automatische Auswahl nach Sprechtempo, Fragen und Schlüsselwörtern.
4. Schnitt mit ffmpeg: Zuschnitt auf 9:16 (Mitte oder ganzes Bild mit unscharfem Hintergrund), eingebrannte Untertitel, Lautstärke-Normalisierung.
5. Download als MP4 plus SRT; Start, Ende und Titel lassen sich pro Clip anpassen und neu rendern.

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
