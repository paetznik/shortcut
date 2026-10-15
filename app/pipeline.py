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
    has_audio = any(s.get("codec_type") == "audio" for s in info["streams"])
    return {"duration": float(info["format"].get("duration", 0)), "width": w, "height": h, "has_audio": has_audio}


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
        '"caption": "Social-Media-Beschreibung mit 3-5 Hashtags", '
        '"hooks": ["3 unterschiedliche Hook-Texte für die ersten Sekunden, je max. 8 Wörter: z. B. Frage, steile These, Neugier"], '
        '"keywords": ["2-6 Wörter, die im Ausschnitt wörtlich vorkommen und hervorgehoben werden sollen, z. B. Zahlen, Kernbegriffe"]}]\n\n'
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
        sentences = [s for s in re.split(r"(?<=[.!?])\s", c["text"]) if s.strip()]
        short = lambda s: (s[:70] + "…") if len(s) > 70 else s
        c["title"] = short(sentences[0])
        c["hook"] = "Automatisch gewählt (Sprechdichte, Fragen, Schlüsselwörter)"
        c["caption"] = ""
        # alternatives: the first question in the clip and the second sentence
        alts = [s for s in sentences[1:] if s.endswith("?")][:1] + sentences[1:2]
        c["hooks"] = list(dict.fromkeys([c["title"]] + [short(s) for s in alts]))[:3]
        c["keywords"] = guess_keywords(c["text"])
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
        hooks = [str(h).strip() for h in (c.get("hooks") or []) if str(h).strip()]
        c["hooks"] = list(dict.fromkeys(([c["title"]] if c["title"] else []) + hooks))[:4]
        kw = c.get("keywords") or []
        c["keywords"] = [str(k).strip() for k in kw if str(k).strip()][:8] if isinstance(kw, list) else []
    return snapped, source


def guess_keywords(text):
    """Words worth colouring without AI: numbers, hook words and long nouns, a handful per clip."""
    out = []
    for tok in text.split():
        w = re.sub(r"[^\wäöüÄÖÜß%€$]", "", tok)
        if not w:
            continue
        if re.search(r"\d", w) or w.lower() in HOOK_WORDS or (len(w) >= 9 and w[0].isupper()):
            if w.lower() not in (x.lower() for x in out):
                out.append(w)
    return out[:6]


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


def keep_intervals(words, duration, remove_pauses, removed=(), max_pause=0.35, pad=0.1):
    """Parts of the clip (relative seconds) that stay after cutting pauses, filler words and words the user struck out.

    `removed` holds the start times of struck-out words; a struck-out word always forces a cut."""
    cut = {round(float(s), 2) for s in removed}
    gone = [round(w["s"], 2) in cut for w in words]
    if not remove_pauses:
        out = [[0.0, duration]]
        for w, g in zip(words, gone):
            if not g:
                continue
            nxt = []
            for a, b in out:  # subtract the word from every kept interval
                if w["e"] <= a or w["s"] >= b:
                    nxt.append([a, b])
                    continue
                if w["s"] > a:
                    nxt.append([a, w["s"]])
                if w["e"] < b:
                    nxt.append([w["e"], b])
            out = nxt
        return [(round(a, 3), round(b, 3)) for a, b in out if b - a > 0.05]
    out, blocked, prev_cut_end = [], False, 0.0
    for i, w in enumerate(words):
        if gone[i]:
            blocked, prev_cut_end = True, w["e"]
            continue
        if is_filler(w["w"]):
            continue
        nxt_cut = next((x["s"] for x, g in zip(words[i + 1:], gone[i + 1:]) if g), duration)
        a = max(0.0, prev_cut_end if blocked else 0.0, w["s"] - pad)
        b = min(duration, nxt_cut, w["e"] + pad)
        if out and not blocked and a - out[-1][1] < max_pause:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
        blocked = False
    if not out:
        return [(0.0, duration)] if not any(gone) else [(0.0, 0.5)]
    tail_cut = next((x["s"] for x, g in zip(words, gone) if g and x["s"] >= out[-1][1]), duration)
    out[-1][1] = min(duration, tail_cut, out[-1][1] + 0.25)
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
    """Find who is talking and return [(t0, t1, centre_x)] in clip time (source pixels), or None for a centre crop.

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

    def centre(track):
        return int(float(np.median(track["cx"])) * W / aw)

    duration = end - start
    if len(tracks) == 1:
        return [(0.0, duration, centre(tracks[0]))]

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
        segs.append((t, min(duration, t + n), centre(tracks[k])))
        t += n
    return segs


def crop_box(W, H, ar):
    """Largest crop of the source with aspect ratio `ar` (width/height)."""
    if W / H > ar:
        return int(H * ar) // 2 * 2, H
    return W, int(W / ar) // 2 * 2


def crop_filter(segs, W, H, ar):
    cw, ch = crop_box(W, H, ar)
    y = (H - ch) // 2
    if not segs:
        return "crop=%d:%d:%d:%d" % (cw, ch, (W - cw) // 2, y)
    xs = [int(min(max(0, cx - cw / 2), W - cw)) for _, _, cx in segs]
    expr = str(xs[-1])
    for (t0, t1, _), x in reversed(list(zip(segs[:-1], xs[:-1]))):
        expr = "if(lt(t,%.2f),%d,%s)" % (t1, x, expr)
    return "crop=%d:%d:'%s':%d" % (cw, ch, expr, y)


# ---------------------------------------------------------------- subtitles

SUB_STYLES = ("pop", "box", "minimal", "satz")


def ass_time(t):
    t = max(0, t)
    cs = int(round(t * 100))
    return "%d:%02d:%02d.%02d" % (cs // 360000, cs // 6000 % 60, cs // 100 % 60, cs % 100)


def srt_time(t):
    ms = int(round(max(0, t) * 1000))
    return "%02d:%02d:%02d,%03d" % (ms // 3600000, ms // 60000 % 60, ms // 1000 % 60, ms % 1000)


def hex_to_ass(color, fallback="FFE600"):
    c = color.lstrip("#")
    if len(c) != 6:
        c = fallback
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


def norm_word(w):
    return re.sub(r"[^\wäöüß%€$]", "", w.lower())


def write_subtitles(words, ass_path, srt_path, style, hook_text="", total=0.0, keywords=(), size=(1080, 1920)):
    """Subtitles in one of several looks; keywords get their own colour, the active word the highlight colour."""
    pw, ph = size
    k = ph / 1920
    look = style.get("sub_style", "pop")
    upper = style.get("uppercase", True)
    hl = hex_to_ass(style.get("highlight", "#FFE600"))
    kw_col = hex_to_ass(style.get("keyword_color", "#4ADE80"), "4ADE80")
    fs = int(style.get("font_size", 78)) * max(0.72, k)
    margin_v = int({"unten": 0.22, "mitte": 0.45, "oben": 0.70}.get(style.get("position", "unten"), 0.22) * ph)
    max_words = int(style.get("words_per_line", 3))
    keys = {norm_word(x) for kw in keywords for x in str(kw).split() if norm_word(x)}

    def txt(w):
        return ass_escape(w.upper() if upper else w)

    # side margins keep text clear of the platform buttons on the right (safe zone)
    side = int(0.12 * pw)
    if look == "box":
        main = "Default,Montserrat,%d,&H00FFFFFF,&H00FFFFFF,&H64000000,&H64000000,-1,0,0,0,100,100,0,0,3,10,0,2,%d,%d,%d,1" % (fs * 0.92, side, side, margin_v)
    elif look == "minimal":
        fs *= 0.78
        main = "Default,Montserrat,%d,&H00FFFFFF,&H00FFFFFF,&H50000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,2,2,%d,%d,%d,1" % (fs, side, side, margin_v)
    elif look == "satz":
        fs *= 0.72
        main = "Default,Montserrat,%d,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,3,8,0,2,%d,%d,%d,1" % (fs, side, side, margin_v)
    else:
        main = "Default,Montserrat,%d,&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,1,0,1,%d,3,2,%d,%d,%d,1" % (fs, max(4, int(7 * k)), side, side, margin_v)
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: %d\nPlayResY: %d\nWrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        "Style: %s\n"
        "Style: Hook,Montserrat,%d,&H00111111,&H00111111,%s,&H00000000,-1,0,0,0,100,100,0,0,3,16,0,8,%d,%d,%d,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    ) % (pw, ph, main, 72 * max(0.75, k), hl, side, side, int(0.13 * ph))
    events, srt = [], []
    hook_secs = float(style.get("hook_seconds", 3))
    if style.get("hook", True) and hook_text.strip():
        end = min(hook_secs, total) if total else hook_secs
        events.append("Dialogue: 1,%s,%s,Hook,,0,0,0,,{\\fad(120,250)\\fscx90\\fscy90\\t(0,150,\\fscx100\\fscy100)}%s"
                      % (ass_time(0), ass_time(end), ass_escape(hook_text.strip())))

    def coloured(x):
        return ("{\\c%s&}%s{\\r}" % (kw_col, txt(x["w"]))) if norm_word(x["w"]) in keys else txt(x["w"])

    if look == "satz":
        chunks = chunk_words(words, 12, 42)
    else:
        chunks = chunk_words(words, max_words)
    for n, chunk in enumerate(chunks, 1):
        c_end = chunk[-1]["e"]
        srt.append("%d\n%s --> %s\n%s\n" % (n, srt_time(chunk[0]["s"]), srt_time(c_end), " ".join(w["w"] for w in chunk)))
        if look == "satz":  # whole sentence at once, keywords coloured
            events.append("Dialogue: 0,%s,%s,Default,,0,0,0,,%s" % (ass_time(chunk[0]["s"]), ass_time(c_end),
                                                                   " ".join(coloured(x) for x in chunk)))
            continue
        for i, w in enumerate(chunk):
            s = w["s"] if i else chunk[0]["s"]
            e = chunk[i + 1]["s"] if i + 1 < len(chunk) else c_end
            parts = []
            for j, x in enumerate(chunk):
                if j == i:
                    grow = "\\fscx112\\fscy112" if look == "pop" else ""
                    parts.append("{\\c%s&%s}%s{\\r}" % (hl, grow, txt(x["w"])))
                else:
                    parts.append(coloured(x))
            pop = "{\\fscx80\\fscy80\\t(0,90,\\fscx100\\fscy100)}" if (look == "pop" and not i) else ""
            events.append("Dialogue: 0,%s,%s,Default,,0,0,0,,%s%s" % (ass_time(s), ass_time(e), pop, " ".join(parts)))
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events) + "\n")
    if srt_path:
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt))
    return bool(events)


# ---------------------------------------------------------------- zooms

def zoom_spans(words, intervals, total, every=4.0):
    """Punch-in zoom spans (output time): the zoom toggles at cuts and at sentence starts, at least `every` s apart,
    so jump cuts look intentional and long takes get movement."""
    points = []
    acc = 0.0
    for a, b in intervals[:-1]:
        acc += b - a
        points.append(acc)
    for prev, w in zip(words, words[1:]):
        if prev["w"][-1:] in ".!?":
            points.append(w["s"])
    points = sorted(p for p in points if 1.0 < p < total - 1.0)
    picked, last = [], 0.0
    for p in points:
        if p - last >= 2.5:
            picked.append(p)
            last = p
    t = picked[-1] if picked else 0.0
    while t + every < total - 1.0:  # long stretches without sentence ends still get a change
        t += every
        picked.append(t)
    picked = sorted(set(round(p, 2) for p in picked))
    spans, on = [], False
    edges = [0.0] + picked + [total]
    for a, b in zip(edges, edges[1:]):
        if on:
            spans.append((a, b))
        on = not on
    return spans


def zoom_filter(spans, ow, oh, amount=0.1):
    if not spans:
        return ""
    z = "(1+%.3f*(%s))" % (amount, "+".join("between(t,%.2f,%.2f)" % s for s in spans))
    return (",scale=w='trunc(%d*%s/2)*2':h='trunc(%d*%s/2)*2':eval=frame,crop=%d:%d" % (ow, z, oh, z, ow, oh))


# ---------------------------------------------------------------- rendering

def select_expr(intervals):
    return "+".join("between(t,%.3f,%.3f)" % (a, b) for a, b in intervals)


def video_chain(info, mode, crop_segs, sel, ow, oh):
    w, h = info["width"], info["height"]
    if mode == "blur" and w / h > ow / oh + 0.01:
        return (
            "[0:v]fps=30" + sel + ",split=2[a][b];"
            "[a]scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d,boxblur=24:3,eq=brightness=-0.08[bg];"
            "[b]scale=%d:%d:force_original_aspect_ratio=decrease[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1" % (ow, oh, ow, oh, ow, oh)
        )
    crop = crop_filter(crop_segs if mode == "auto" else None, w, h, ow / oh)
    return "[0:v]fps=30," + crop + sel + ",scale=%d:%d,setsar=1" % (ow, oh)


def audio_chain(intervals, cut, enhance, music, music_volume):
    """Voice clean-up (rumble, noise, sibilance, levelling) and optional background music that ducks under speech."""
    a = "[0:a]aformat=sample_rates=48000:channel_layouts=stereo"
    if cut:
        a += ",aselect='%s',asetpts=N/SR/TB" % select_expr(intervals)
    if enhance:
        a += ",highpass=f=80,afftdn=nf=-25,deesser=i=0.4,acompressor=threshold=-20dB:ratio=3:attack=5:release=150"
    if music:
        a += (",asplit=2[vs][vm];"
              "[1:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=%.2f[m];"
              "[m][vs]sidechaincompress=threshold=0.03:ratio=10:attack=15:release=400[md];"
              "[vm][md]amix=inputs=2:duration=first:normalize=0" % (music_volume / 100))
    return a + ",loudnorm=I=-14:TP=-1.5:LRA=11[a]"


def render_clip(video, info, start, end, workdir, name, style, ass_name, burn, intervals, crop_segs,
                size=(1080, 1920), zoom_spans_out=None, music=None):
    ow, oh = size
    duration = end - start
    cut = bool(intervals) and not (len(intervals) == 1 and intervals[0][0] <= 0.01 and intervals[0][1] >= duration - 0.01)
    sel = (",select='%s',setpts=N/30/TB" % select_expr(intervals)) if cut else ""
    graph = video_chain(info, style.get("mode", "auto"), crop_segs, sel, ow, oh)
    if style.get("zoom", True) and zoom_spans_out:
        graph += zoom_filter(zoom_spans_out, ow, oh)
    graph += ((",ass=" + ass_name) if burn else "") + "[v]"
    cmd = [FFMPEG, "-y", "-ss", "%.2f" % start, "-t", "%.2f" % duration, "-i", video]
    has_audio = info.get("has_audio", True)
    if has_audio:
        if music:
            cmd += ["-stream_loop", "-1", "-i", music]
        graph += ";" + audio_chain(intervals, cut, style.get("enhance_audio", True), music,
                                   int(style.get("music_volume", 15)))
    out = name + ".mp4"
    cmd += ["-filter_complex", graph, "-map", "[v]"] + (["-map", "[a]", "-c:a", "aac", "-b:a", "160k", "-ar", "48000"] if has_audio else [])
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-r", "30",
            "-shortest", "-movflags", "+faststart", "-threads", str(THREADS), out]
    run(cmd, cwd=workdir)
    return sum(b - a for a, b in intervals) if cut else duration


def render_thumb(workdir, video_name, name, at):
    run([FFMPEG, "-y", "-ss", "%.2f" % at, "-i", video_name, "-frames:v", "1", "-vf", "scale=360:-2", name + ".jpg"], cwd=workdir)


def render_cover(video, info, start, rel, workdir, name, style, crop_segs, text):
    """Cover image for the profile grid: a frame of the clip, darkened, with the title in large type."""
    ass = name + ".ass"
    hl = hex_to_ass(style.get("highlight", "#FFE600"))
    with open(os.path.join(workdir, ass), "w", encoding="utf-8") as f:
        f.write(
            "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
            "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
            "MarginL, MarginR, MarginV, Encoding\n"
            "Style: Cover,Montserrat,112,&H00111111,&H00111111,%s,&H00000000,-1,0,0,0,100,100,0,0,3,24,0,5,110,110,0,1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            "Dialogue: 0,0:00:00.00,0:00:10.00,Cover,,0,0,0,,%s\n" % (hl, ass_escape(text.upper()))
        )
    at = start + rel
    segs = [s for s in (crop_segs or []) if s[0] <= rel < s[1]] or (crop_segs or [])[:1]
    graph = video_chain(info, style.get("mode", "auto"), segs and [(0, 999, segs[0][2])], "", 1080, 1920)
    graph += ",eq=brightness=-0.12:saturation=1.1,ass=" + ass + "[v]"
    run([FFMPEG, "-y", "-ss", "%.2f" % at, "-i", video, "-filter_complex", graph, "-map", "[v]", "-frames:v", "1",
         "-q:v", "3", name + ".jpg"], cwd=workdir)
    os.remove(os.path.join(workdir, ass))
