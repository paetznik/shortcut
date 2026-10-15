"""Video pipeline: audio extraction, transcription, highlight detection, clip rendering."""
import gc
import json
import math
import os
import re
import subprocess

import httpx

FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE_BIN", "ffprobe")
THREADS = int(os.environ.get("CPU_THREADS", "2"))
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")

WHISPER_PRESETS = {
    "schnell": {"model": "base", "beam_size": 1},
    "ausgewogen": {"model": "small", "beam_size": 3},
    "genau": {"model": "medium", "beam_size": 5},
}


def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError("ffmpeg-Fehler: " + p.stderr[-1500:])
    return p.stdout


def probe(path):
    out = run([FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
    info = json.loads(out)
    v = next((s for s in info["streams"] if s.get("codec_type") == "video"), None)
    if not v:
        raise RuntimeError("Die Datei enthält keine Videospur.")
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = abs(int(sd["rotation"]))
    w, h = int(v["width"]), int(v["height"])
    if rot in (90, 270):
        w, h = h, w
    return {"duration": float(info["format"].get("duration", 0)), "width": w, "height": h}


def extract_audio(video, wav):
    run([FFMPEG, "-y", "-i", video, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav])


# ---------------------------------------------------------------- transcription

def transcribe(wav, duration, preset, progress):
    from faster_whisper import WhisperModel

    cfg = WHISPER_PRESETS.get(preset, WHISPER_PRESETS["ausgewogen"])
    progress(None, "Lade Spracherkennungsmodell (" + cfg["model"] + ") …")
    model = WhisperModel(cfg["model"], device="cpu", compute_type="int8", cpu_threads=THREADS)
    try:
        import wave

        import numpy as np

        with wave.open(wav, "rb") as f:
            audio = np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float32) / 32768.0
        segments, _ = model.transcribe(
            audio,
            language="de",
            beam_size=cfg["beam_size"],
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        result = []
        for seg in segments:
            words = [
                {"w": w.word.strip(), "s": round(w.start, 2), "e": round(w.end, 2)}
                for w in (seg.words or [])
                if w.word.strip()
            ]
            if not words:
                continue
            result.append({"s": round(seg.start, 2), "e": round(seg.end, 2), "text": seg.text.strip(), "words": words})
            if duration:
                progress(min(99, seg.end / duration * 100), "Transkribiere … " + fmt_time(seg.end) + " / " + fmt_time(duration))
        return result
    finally:
        del model
        gc.collect()


def fmt_time(t):
    t = int(t)
    return "%d:%02d" % (t // 60, t % 60) if t < 3600 else "%d:%02d:%02d" % (t // 3600, t % 3600 // 60, t % 60)


# ---------------------------------------------------------------- highlights

def all_words(segments):
    return [w for s in segments for w in s["words"]]


def snap(clip, words, min_len, max_len):
    """Snap clip boundaries to word boundaries and enforce length limits."""
    if not words:
        return None
    s, e = float(clip["start"]), float(clip["end"])
    start_word = min(words, key=lambda w: abs(w["s"] - s))
    end_word = min(words, key=lambda w: abs(w["e"] - e))
    s, e = start_word["s"], end_word["e"]
    if e - s > max_len:
        cands = [w for w in words if s < w["e"] <= s + max_len]
        e = cands[-1]["e"] if cands else s + max_len
    if e - s < min_len * 0.6:
        return None
    clip = dict(clip)
    clip["start"] = round(max(0, s - 0.15), 2)
    clip["end"] = round(e + 0.35, 2)
    return clip


def remove_overlaps(clips):
    out = []
    for c in sorted(clips, key=lambda c: -c.get("score", 0)):
        if all(c["end"] <= o["start"] or c["start"] >= o["end"] for o in out):
            out.append(c)
    return out


def highlights_claude(segments, api_key, n, min_len, max_len, context):
    lines = ["[%.1f-%.1f] %s" % (s["s"], s["e"], s["text"]) for s in segments]
    transcript = "\n".join(lines)
    extra = ("\nZusätzlicher Kontext vom Ersteller: " + context.strip()) if context and context.strip() else ""
    prompt = (
        "Du bist ein erfahrener Social-Media-Editor für TikTok, Instagram Reels und YouTube Shorts.\n"
        "Unten steht das Transkript eines deutschsprachigen Videos mit Zeitstempeln in Sekunden.\n"
        "Wähle die %d stärksten Ausschnitte, die als eigenständige Kurzvideos funktionieren.\n\n"
        "Regeln:\n"
        "- Jeder Ausschnitt ist zwischen %d und %d Sekunden lang.\n"
        "- Er beginnt mit einem starken Hook (Frage, steile These, Überraschung, Emotion) und braucht keinen Vorkontext.\n"
        "- Er endet an einem natürlichen Satzende mit einer Pointe oder einem abgeschlossenen Gedanken.\n"
        "- Ausschnitte überlappen sich nicht.\n"
        "- Start und Ende müssen exakt auf Zeitstempeln aus dem Transkript liegen.%s\n\n"
        "Antworte NUR mit einem JSON-Array, ohne weiteren Text, in diesem Format:\n"
        '[{"start": 12.3, "end": 55.0, "title": "kurzer, packender Titel", '
        '"hook": "warum das funktioniert (1 Satz)", "score": 1-100, '
        '"caption": "Social-Media-Beschreibung mit 3-5 Hashtags"}]\n\n'
        "TRANSKRIPT:\n%s"
    ) % (n, min_len, max_len, extra, transcript)

    r = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={"model": CLAUDE_MODEL, "max_tokens": 4000, "messages": [{"role": "user", "content": prompt}]},
        timeout=300,
    )
    if r.status_code != 200:
        raise RuntimeError("Claude-API-Fehler %d: %s" % (r.status_code, r.text[:300]))
    text = "".join(b.get("text", "") for b in r.json().get("content", []))
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise RuntimeError("Claude hat kein gültiges JSON geliefert.")
    return json.loads(m.group(0))


HOOK_WORDS = [
    "wichtig", "geheimnis", "fehler", "tipp", "trick", "niemand", "nie", "immer", "krass", "unglaublich",
    "wirklich", "achtung", "warum", "wie", "so geht", "problem", "lösung", "ehrlich", "wahrheit", "beste",
    "schlimmste", "größte", "einfach", "sofort", "genau", "stell dir vor", "das heißt", "zum beispiel",
    "kein", "musst", "solltest", "verrückt", "endlich", "überraschend", "plötzlich", "geld", "zeit",
]


def highlights_heuristic(segments, n, min_len, max_len):
    """Fallback without AI: score windows by speech density, emphasis and hook words."""
    target = (min_len + max_len) / 2
    cands = []
    for i, seg in enumerate(segments):
        start = seg["s"]
        text, j = "", i
        while j < len(segments) and segments[j]["e"] - start <= max_len:
            text += " " + segments[j]["text"]
            dur = segments[j]["e"] - start
            if dur >= min_len:
                low = text.lower()
                words = len(text.split())
                score = words / dur * 10
                score += sum(low.count(k) for k in HOOK_WORDS) * 2
                score += text.count("!") * 2 + text.count("?") * 3
                first = seg["text"].strip()
                if first.endswith("?") or any(first.lower().startswith(k) for k in ("warum", "wie", "was", "stell")):
                    score += 8
                if segments[j]["text"].strip()[-1:] in ".!?":
                    score += 5
                score -= abs(dur - target) / target * 5
                cands.append({"start": start, "end": segments[j]["e"], "score": score, "text": text.strip()})
            j += 1
    clips = remove_overlaps(cands)[:n]
    for c in clips:
        first = re.split(r"(?<=[.!?])\s", c["text"])[0]
        c["title"] = (first[:70] + "…") if len(first) > 70 else first
        c["hook"] = "Automatisch gewählt (Sprechdichte, Fragen, Schlüsselwörter)"
        c["caption"] = ""
        del c["text"]
    if clips:
        top = max(c["score"] for c in clips) or 1
        for c in clips:
            c["score"] = max(1, round(c["score"] / top * 100))
    return clips


def find_highlights(segments, api_key, n, min_len, max_len, context, log):
    words = all_words(segments)
    clips, source = None, "heuristik"
    if api_key:
        try:
            clips = highlights_claude(segments, api_key, n, min_len, max_len, context)
            source = "claude"
        except Exception as e:  # fall back to heuristic
            log("KI-Auswahl fehlgeschlagen, nutze automatische Auswahl: " + str(e)[:200])
    if not clips:
        clips = highlights_heuristic(segments, n, min_len, max_len)
    snapped = [c for c in (snap(c, words, min_len, max_len) for c in clips) if c]
    snapped = remove_overlaps(snapped)[:n]
    snapped.sort(key=lambda c: c["start"])
    for c in snapped:
        c["score"] = int(c.get("score") or 50)
        c.setdefault("title", "")
        c.setdefault("hook", "")
        c.setdefault("caption", "")
    return snapped, source


# ---------------------------------------------------------------- subtitles

def ass_time(t):
    t = max(0, t)
    cs = int(round(t * 100))
    return "%d:%02d:%02d.%02d" % (cs // 360000, cs // 6000 % 60, cs // 100 % 60, cs % 100)


def srt_time(t):
    ms = int(round(max(0, t) * 1000))
    return "%02d:%02d:%02d,%03d" % (ms // 3600000, ms // 60000 % 60, ms // 1000 % 60, ms % 1000)


def hex_to_ass(color):
    c = color.lstrip("#")
    if len(c) != 6:
        c = "FFE600"
    return "&H00%s%s%s" % (c[4:6], c[2:4], c[0:2])


def chunk_words(words, max_words, max_chars=20):
    chunks, cur = [], []
    for i, w in enumerate(words):
        if cur:
            gap = w["s"] - cur[-1]["e"]
            length = len(" ".join(x["w"] for x in cur + [w]))
            if len(cur) >= max_words or length > max_chars or gap > 0.6 or cur[-1]["w"][-1:] in ".!?,;:":
                chunks.append(cur)
                cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def clip_words(segments, start, end):
    out = []
    for w in all_words(segments):
        if w["e"] > start and w["s"] < end:
            out.append({"w": w["w"], "s": max(0, w["s"] - start), "e": min(end, w["e"]) - start})
    return out


def write_subtitles(words, ass_path, srt_path, style):
    upper = style.get("uppercase", True)
    hl = hex_to_ass(style.get("highlight", "#FFE600"))
    size = int(style.get("font_size", 78))
    pos = style.get("position", "unten")
    margin_v = {"unten": 420, "mitte": 860, "oben": 1350}.get(pos, 420)
    max_words = int(style.get("words_per_line", 3))

    def txt(w):
        return (w.upper() if upper else w).replace("{", "(").replace("}", ")")

    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\nWrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Montserrat,%d,&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,1,0,1,7,3,2,70,70,%d,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    ) % (size, margin_v)
    events, srt = [], []
    for n, chunk in enumerate(chunk_words(words, max_words), 1):
        c_end = chunk[-1]["e"]
        srt.append("%d\n%s --> %s\n%s\n" % (n, srt_time(chunk[0]["s"]), srt_time(c_end), " ".join(w["w"] for w in chunk)))
        for i, w in enumerate(chunk):
            s = w["s"] if i else chunk[0]["s"]
            e = chunk[i + 1]["s"] if i + 1 < len(chunk) else c_end
            parts = []
            for k, x in enumerate(chunk):
                if k == i:
                    parts.append("{\\c%s&\\fscx112\\fscy112}%s{\\r}" % (hl, txt(x["w"])))
                else:
                    parts.append(txt(x["w"]))
            pop = "" if i else "{\\fscx80\\fscy80\\t(0,90,\\fscx100\\fscy100)}"
            events.append("Dialogue: 0,%s,%s,Default,,0,0,0,,%s%s" % (ass_time(s), ass_time(e), pop, " ".join(parts)))
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events) + "\n")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt))


# ---------------------------------------------------------------- rendering

def render_clip(video, info, start, end, workdir, name, mode, ass_name, burn):
    w, h = info["width"], info["height"]
    sub = (",ass=" + ass_name) if burn else ""
    if mode == "blur" and w / h > 9 / 16 + 0.01:
        vf = (
            "[0:v]split=2[a][b];"
            "[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=24:3,eq=brightness=-0.08[bg];"
            "[b]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1" + sub + "[v]"
        )
    else:
        vf = (
            "[0:v]crop='min(iw,ih*9/16)':'min(ih,iw*16/9)',scale=1080:1920,setsar=1" + sub + "[v]"
        )
    out = name + ".mp4"
    run(
        [
            FFMPEG, "-y", "-ss", "%.2f" % start, "-i", video, "-t", "%.2f" % (end - start),
            "-filter_complex", vf, "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-r", "30",
            "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
            "-movflags", "+faststart", "-threads", str(THREADS), out,
        ],
        cwd=workdir,
    )
    run([FFMPEG, "-y", "-ss", "%.2f" % min(1.0, (end - start) / 2), "-i", out, "-frames:v", "1", "-vf", "scale=360:-2", name + ".jpg"], cwd=workdir)
    return out
