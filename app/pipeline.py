"""Video pipeline: audio extraction, transcription, highlight detection, clip rendering."""
import gc
import json
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

def transcribe(wav, duration, preset, progress, vocabulary=""):
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
            # known names and terms steer the spelling of the whole transcript
            initial_prompt=("Wichtige Begriffe: " + vocabulary.strip()) if vocabulary and vocabulary.strip() else None,
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


# ---------------------------------------------------------------- transcript editing

def realign(old, new_text):
    """Map corrected text onto the timings of the original words.

    Unchanged words keep their timing; replaced or inserted words share the time span of what they replace."""
    import difflib

    tokens = new_text.split()
    if not old:
        return []
    sm = difflib.SequenceMatcher(a=[w["w"].lower() for w in old], b=[t.lower() for t in tokens], autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out += [{**old[i1 + k], "w": tokens[j1 + k]} for k in range(i2 - i1)]
        elif tag in ("replace", "insert"):
            if i2 > i1:
                s, e = old[i1]["s"], old[i2 - 1]["e"]
            else:
                s = old[i1 - 1]["e"] if i1 > 0 else old[0]["s"]
                e = old[i1]["s"] if i1 < len(old) else old[-1]["e"]
            n = j2 - j1
            if e - s < 0.15 * n:
                e = s + 0.15 * n
            step = (e - s) / n
            out += [{"w": tokens[j1 + k], "s": round(s + k * step, 2), "e": round(s + (k + 1) * step, 2)} for k in range(n)]
    for a, b in zip(out, out[1:]):  # keep timings monotonic
        if b["s"] < a["s"]:
            b["s"] = a["s"]
        b["e"] = max(b["e"], b["s"] + 0.05)
    return out


def replace_words(segments, start, end, new_words):
    """Replace all words between start and end (absolute seconds) in the transcript."""
    target = None
    for idx, seg in enumerate(segments):
        keep = [w for w in seg["words"] if not (w["e"] > start and w["s"] < end)]
        if len(keep) != len(seg["words"]) and target is None:
            target = idx
        seg["words"] = keep
    if target is None:
        target = next((i for i, s in enumerate(segments) if s["s"] >= start), len(segments) - 1)
    seg = segments[target]
    seg["words"] = sorted(seg["words"] + new_words, key=lambda w: w["s"])
    for seg in segments:
        seg["text"] = " ".join(w["w"] for w in seg["words"])
    return [s for s in segments if s["words"]]


# ---------------------------------------------------------------- pauses and fillers

FILLERS = {"äh", "ähm", "ähh", "ähmm", "öh", "öhm", "hm", "hmm", "mhm", "ehm", "em", "uh", "um", "uhm"}


def is_filler(word):
    return re.sub(r"[^\wäöüß]", "", word.lower()) in FILLERS


def keep_intervals(words, duration, remove_pauses, max_pause=0.35, pad=0.1):
    """Parts of the clip (relative seconds) that stay after cutting pauses and filler words."""
    if not remove_pauses:
        return [(0.0, duration)]
    spoken = [w for w in words if not is_filler(w["w"])]
    if not spoken:
        return [(0.0, duration)]
    out = []
    for w in spoken:
        a, b = max(0.0, w["s"] - pad), min(duration, w["e"] + pad)
        if out and a - out[-1][1] < max_pause:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    out[-1][1] = min(duration, out[-1][1] + 0.25)
    return [(round(a, 3), round(b, 3)) for a, b in out if b - a > 0.05]


def remap(t, intervals):
    """Time in the original clip -> time in the cut clip."""
    acc = 0.0
    for a, b in intervals:
        if t <= a:
            return acc
        if t <= b:
            return acc + t - a
        acc += b - a
    return acc


def remap_words(words, intervals):
    out = []
    for w in words:
        mid = (w["s"] + w["e"]) / 2
        if is_filler(w["w"]) or not any(a <= mid <= b for a, b in intervals):
            continue
        out.append({"w": w["w"], "s": remap(w["s"], intervals), "e": remap(w["e"], intervals)})
    return out


# ---------------------------------------------------------------- speaker tracking

def track_speaker(video, info, start, end):
    """Find who is talking and return crop positions [(t0, t1, x)] in clip time, or None for a centre crop.

    Faces are detected on a few frames per second; the face whose mouth area moves most within each second
    is treated as the speaker, and short switches are smoothed away so the framing does not jump around."""
    import cv2
    import numpy as np

    W, H = info["width"], info["height"]
    if W / H <= 9 / 16 + 0.01:
        return None
    fps, aw = 5, 480
    ah = int(round(H * aw / W / 2)) * 2
    p = subprocess.run(
        [FFMPEG, "-v", "error", "-ss", "%.2f" % start, "-t", "%.2f" % (end - start), "-i", video,
         "-vf", "fps=%d,scale=%d:%d" % (fps, aw, ah), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True,
    )
    if p.returncode != 0 or not p.stdout:
        return None
    frames = np.frombuffer(p.stdout, np.uint8)
    frames = frames[: len(frames) // (aw * ah) * aw * ah].reshape(-1, ah, aw)
    frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    min_size = (max(20, ah // 9),) * 2

    tracks = []  # {"cx": [..], "last": x, "seen": n, "motion": {frame: v}, "patch": (frame, arr)}
    for i, img in enumerate(frames):
        faces = list(frontal.detectMultiScale(img, 1.1, 5, minSize=min_size))
        if not faces:
            faces = list(profile.detectMultiScale(img, 1.1, 5, minSize=min_size))
        used = set()
        for x, y, w, h in faces:
            cx = x + w / 2
            best = min((t for t in tracks if id(t) not in used), key=lambda t: abs(t["last"] - cx), default=None)
            if best is None or abs(best["last"] - cx) > 0.12 * aw:
                best = {"cx": [], "last": cx, "seen": 0, "motion": {}, "patch": None}
                tracks.append(best)
            used.add(id(best))
            best["cx"].append(cx)
            best["last"] = cx
            best["seen"] += 1
            mouth = img[y + int(h * 0.62): y + h, x + int(w * 0.2): x + int(w * 0.8)]
            if mouth.size:
                patch = cv2.resize(mouth, (24, 12)).astype(np.float32)
                if best["patch"] is not None and best["patch"][0] == i - 1:
                    best["motion"][i] = float(np.mean(np.abs(patch - best["patch"][1])))
                best["patch"] = (i, patch)

    tracks = [t for t in tracks if t["seen"] >= max(3, len(frames) * 0.15)]
    if not tracks:
        return None
    cw = int(H * 9 / 16) // 2 * 2

    def crop_x(track):
        cx = float(np.median(track["cx"])) * W / aw
        return int(min(max(0, cx - cw / 2), W - cw))

    duration = end - start
    if len(tracks) == 1:
        return [(0.0, duration, crop_x(tracks[0]))]

    windows = max(1, int(np.ceil(len(frames) / fps)))
    choice, prev = [], max(range(len(tracks)), key=lambda k: tracks[k]["seen"])
    for wi in range(windows):
        rng = range(wi * fps, (wi + 1) * fps)
        scores = [sum(t["motion"].get(f, 0) for f in rng) for t in tracks]
        if max(scores) > 0:
            prev = int(np.argmax(scores))
        choice.append(prev)
    # merge runs shorter than two seconds into the previous shot
    runs = []
    for k in choice:
        if runs and runs[-1][0] == k:
            runs[-1][1] += 1
        else:
            runs.append([k, 1])
    merged = []
    for k, n in runs:
        if merged and (n < 2 or merged[-1][0] == k):
            merged[-1][1] += n
        else:
            merged.append([k, n])
    segs, t = [], 0.0
    for k, n in merged:
        segs.append((t, min(duration, t + n), crop_x(tracks[k])))
        t += n
    return segs


def crop_filter(segs, W, H):
    cw = int(H * 9 / 16) // 2 * 2
    if not segs:
        return "crop='min(iw,ih*9/16)':'min(ih,iw*16/9)'"
    expr = str(segs[-1][2])
    for t0, t1, x in reversed(segs[:-1]):
        expr = "if(lt(t,%.2f),%d,%s)" % (t1, x, expr)
    return "crop=%d:%d:'%s':0" % (cw, H, expr)


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
    for w in words:
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


def ass_escape(text):
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


def write_subtitles(words, ass_path, srt_path, style, hook_text="", total=0.0):
    upper = style.get("uppercase", True)
    hl = hex_to_ass(style.get("highlight", "#FFE600"))
    size = int(style.get("font_size", 78))
    pos = style.get("position", "unten")
    margin_v = {"unten": 420, "mitte": 860, "oben": 1350}.get(pos, 420)
    max_words = int(style.get("words_per_line", 3))

    def txt(w):
        return ass_escape(w.upper() if upper else w)

    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\nWrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Montserrat,%d,&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,1,0,1,7,3,2,70,70,%d,1\n"
        "Style: Hook,Montserrat,72,&H00111111,&H00111111,%s,&H00000000,-1,0,0,0,100,100,0,0,3,16,0,8,110,110,250,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    ) % (size, margin_v, hl)
    events, srt = [], []
    hook_secs = float(style.get("hook_seconds", 3))
    if style.get("hook", True) and hook_text.strip():
        end = min(hook_secs, total) if total else hook_secs
        events.append("Dialogue: 1,%s,%s,Hook,,0,0,0,,{\\fad(120,250)\\fscx90\\fscy90\\t(0,150,\\fscx100\\fscy100)}%s"
                      % (ass_time(0), ass_time(end), ass_escape(hook_text.strip())))
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
    return bool(events)


# ---------------------------------------------------------------- rendering

def select_expr(intervals):
    return "+".join("between(t,%.3f,%.3f)" % (a, b) for a, b in intervals)


def render_clip(video, info, start, end, workdir, name, mode, ass_name, burn, intervals=None, crop_segs=None):
    w, h = info["width"], info["height"]
    duration = end - start
    cut = intervals and not (len(intervals) == 1 and intervals[0][0] <= 0.01 and intervals[0][1] >= duration - 0.01)
    sel = (",select='%s',setpts=N/30/TB" % select_expr(intervals)) if cut else ""
    sub = (",ass=" + ass_name) if burn else ""
    portrait = w / h <= 9 / 16 + 0.01
    if mode == "blur" and not portrait:
        vf = (
            "[0:v]fps=30" + sel + ",split=2[a][b];"
            "[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=24:3,eq=brightness=-0.08[bg];"
            "[b]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1" + sub + "[v]"
        )
    else:
        crop = crop_filter(crop_segs if mode == "auto" else None, w, h)
        vf = "[0:v]fps=30," + crop + sel + ",scale=1080:1920,setsar=1" + sub + "[v]"
    af = ("aselect='%s',asetpts=N/SR/TB," % select_expr(intervals)) if cut else ""
    af += "loudnorm=I=-14:TP=-1.5:LRA=11"
    out = name + ".mp4"
    run(
        [
            FFMPEG, "-y", "-ss", "%.2f" % start, "-t", "%.2f" % duration, "-i", video,
            "-filter_complex", vf, "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-r", "30",
            "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-af", af,
            "-movflags", "+faststart", "-threads", str(THREADS), out,
        ],
        cwd=workdir,
    )
    out_len = sum(b - a for a, b in intervals) if cut else duration
    run([FFMPEG, "-y", "-ss", "%.2f" % min(1.0, out_len / 2), "-i", out, "-frames:v", "1", "-vf", "scale=360:-2", name + ".jpg"], cwd=workdir)
    return out_len
