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
    "genau": {"model": "large-v3-turbo", "beam_size": 3},  # newest large model, distilled decoder; 3-5x video length on 2 cores
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
    try:
        num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1").split("/")
        fps = round(float(num) / float(den), 3) if float(den) else 30.0
    except (ValueError, ZeroDivisionError):
        fps = 30.0
    return {"duration": float(info["format"].get("duration", 0)), "width": w, "height": h, "has_audio": has_audio,
            "fps": fps if 10 <= fps <= 120 else 30.0}


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


def overlap(a, b):
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def remove_overlaps(clips, tolerance=0.0):
    """Best first; a clip is dropped only if it overlaps an accepted one by more than `tolerance` seconds."""
    out = []
    for c in sorted(clips, key=lambda c: (c.get("rejected", False), -c.get("score", 0))):
        if all(overlap(c, o) <= tolerance for o in out):
            out.append(c)
    return out


def split_touching(clips):
    """Clips that touch because of the cut padding share the boundary instead of overlapping."""
    clips.sort(key=lambda c: c["start"])
    for a, b in zip(clips, clips[1:]):
        if b["start"] < a["end"]:
            mid = round((a["end"] + b["start"]) / 2, 2)
            a["end"], b["start"] = mid, mid
    return clips


def annotated_transcript(segments, audio):
    """Transcript lines with hints from the audio analysis, e.g. ⟨laut, lebhafte Stimme⟩ or ⟨Lachen/Reaktion⟩."""
    lines = []
    for seg in segments:
        line = "[%.1f-%.1f] %s" % (seg["s"], seg["e"], seg["text"])
        w = audio_window(audio, seg["s"], seg["e"]) if audio else None
        if w:
            tags = [TAG_LABELS[t] for t in w["tags"] if t != "reaktion"]
            if w["mean"] > 1.2 and not tags:
                tags.append("energiegeladen")
            if tags:
                line += " ⟨" + ", ".join(tags) + "⟩"
        lines.append((seg["s"], line))
    for ev in (audio or {}).get("events", []):
        if ev["type"] == "reaktion":
            lines.append((ev["s"], "[%.1f-%.1f] ⟨Lachen/Reaktion, keine Sprache⟩" % (ev["s"], ev["e"])))
    return "\n".join(l for _, l in sorted(lines, key=lambda x: x[0]))


def highlights_claude(segments, api_key, n, min_len, max_len, context, audio=None):
    transcript = annotated_transcript(segments, audio)
    extra = ("\nZusätzlicher Kontext vom Ersteller: " + context.strip()) if context and context.strip() else ""
    prompt = (
        "Du bist ein erfahrener Social-Media-Editor für TikTok, Instagram Reels und YouTube Shorts.\n"
        "Unten steht das Transkript eines deutschsprachigen Videos mit Zeitstempeln in Sekunden.\n"
        "Wähle die %d stärksten Ausschnitte, die als eigenständige Kurzvideos funktionieren, sortiert nach "
        "Stärke (bester zuerst). Die letzten Vorschläge sind Reserve, falls einzelne nicht passen.\n\n"
        "Regeln:\n"
        "- Jeder Ausschnitt ist zwischen %d und %d Sekunden lang.\n"
        "- Er beginnt mit einem starken Hook (Frage, steile These, Überraschung, Emotion) und braucht keinen Vorkontext.\n"
        "- Er endet an einem natürlichen Satzende mit einer Pointe oder einem abgeschlossenen Gedanken.\n"
        "- Ausschnitte überlappen sich nicht.\n"
        "- Start und Ende müssen exakt auf Zeitstempeln aus dem Transkript liegen.\n"
        "- Hinweise in ⟨⟩ stammen aus einer automatischen Tonanalyse (Lautstärke, lebhafte Stimme, Sprechtempo, "
        "Lachen/Applaus). Emotionale Stellen und Momente kurz vor einer Reaktion sind oft starke Highlights – "
        "der Inhalt geht aber vor.%s\n\n"
        "Antworte NUR mit einem JSON-Array, ohne weiteren Text, in diesem Format:\n"
        '[{"start": 12.3, "end": 55.0, "title": "kurzer, packender Titel", '
        '"hook": "warum das funktioniert (1 Satz)", "score": 1-100, '
        '"caption": "Social-Media-Beschreibung mit 3-5 Hashtags", '
        '"hooks": ["3 unterschiedliche Hook-Texte für die ersten Sekunden, je max. 8 Wörter: z. B. Frage, steile These, Neugier"], '
        '"keywords": ["2-6 Wörter, die im Ausschnitt wörtlich vorkommen und hervorgehoben werden sollen, z. B. Zahlen, Kernbegriffe"]}]\n\n'
        "TRANSKRIPT:\n%s"
    ) % (n + 3, min_len, max_len, extra, transcript)

    return claude_json(api_key, prompt, r"\[.*\]", feature="KI-Auswahl")


# US-$ per million tokens (input, output), Anthropic API list prices
PRICES = {"claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0), "claude-haiku-5-5": (0.10, 0.50)}
USAGE_HOOK = None  # set by the app: fn(feature, model, input_tokens, output_tokens, usd)


def cost_usd(model, tokens_in, tokens_out):
    pin, pout = PRICES.get(model, PRICES["claude-sonnet-5-5"])
    return (tokens_in * pin + tokens_out * pout) / 1e6


def claude_json(api_key, prompt, pattern=r"\{.*\}", max_tokens=4000, feature="Sonstiges", images=()):
    """Ask Claude (optionally with JPEG images) and parse the first JSON array/object in the answer;
    usage is reported to USAGE_HOOK."""
    import base64

    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                            "data": base64.b64encode(img).decode()}} for img in images]
    content.append({"type": "text", "text": prompt})
    r = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={"model": CLAUDE_MODEL, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]},
        timeout=300,
    )
    if r.status_code != 200:
        raise RuntimeError("Claude-API-Fehler %d: %s" % (r.status_code, r.text[:300]))
    usage = r.json().get("usage") or {}
    if USAGE_HOOK:
        tin, tout = int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
        try:
            USAGE_HOOK(feature, CLAUDE_MODEL, tin, tout, cost_usd(CLAUDE_MODEL, tin, tout))
        except Exception:
            pass
    text = "".join(b.get("text", "") for b in r.json().get("content", []))
    m = re.search(pattern, text, re.S)
    if not m:
        raise RuntimeError("Claude hat kein gültiges JSON geliefert.")
    return json.loads(m.group(0))


HOOK_WORDS = [
    "wichtig", "geheimnis", "fehler", "tipp", "trick", "niemand", "nie", "immer", "krass", "unglaublich",
    "wirklich", "achtung", "warum", "wie", "so geht", "problem", "lösung", "ehrlich", "wahrheit", "beste",
    "schlimmste", "größte", "einfach", "sofort", "genau", "stell dir vor", "das heißt", "zum beispiel",
    "kein", "musst", "solltest", "verrückt", "endlich", "überraschend", "plötzlich", "geld", "zeit",
]


def score_windows(segments, min_len, max_len, audio=None):
    """All candidate windows (whole segments, min_len..max_len) with a score from speech density, emphasis,
    hook words and the audio analysis."""
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
                if audio:
                    end = segments[j]["e"]
                    aw = audio_window(audio, start, end)
                    hook = audio_window(audio, start, min(end, start + 5))
                    after = audio_window(audio, end, end + 3)
                    if aw:
                        score += 4 * aw["mean"] + 2 * len(aw["tags"])
                    if hook:
                        score += 3 * max(0.0, hook["peak"])
                    if after and "reaktion" in after["tags"]:
                        score += 6
                cands.append({"start": start, "end": segments[j]["e"], "score": score, "text": text.strip()})
            j += 1
    return cands


def decorate(c):
    """Title, hook ideas and keywords for a window chosen without AI."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s", c["text"]) if s.strip()] or [c["text"]]
    short = lambda s: (s[:70] + "…") if len(s) > 70 else s
    c["title"] = short(sentences[0])
    c["hook"] = "Automatisch gewählt (Sprechdichte, Fragen, Schlüsselwörter)"
    c["caption"] = ""
    alts = [s for s in sentences[1:] if s.endswith("?")][:1] + sentences[1:2]
    c["hooks"] = list(dict.fromkeys([c["title"]] + [short(s) for s in alts]))[:3]
    c["keywords"] = guess_keywords(c["text"])
    del c["text"]
    return c


def highlights_heuristic(segments, n, min_len, max_len, audio=None, cands=None):
    """Fallback without AI: the best non-overlapping scored windows."""
    cands = cands if cands is not None else score_windows(segments, min_len, max_len, audio)
    clips = [decorate(dict(c)) for c in remove_overlaps(cands, 1.0)[:n]]
    if clips:
        top = max(c["score"] for c in clips) or 1
        for c in clips:
            c["score"] = max(1, round(c["score"] / top * 100))
    return clips


def find_highlights(segments, api_key, n, min_len, max_len, context, log, audio=None, two_stage=True):
    words = all_words(segments)
    clips, source = None, "heuristik"
    if api_key:
        try:
            if two_stage:
                clips = highlights_claude_two_stage(segments, api_key, n, min_len, max_len, context, audio, log)
            else:
                clips = highlights_claude(segments, api_key, n, min_len, max_len, context, audio)
            source = "claude"
        except Exception as e:  # fall back to heuristic
            log("KI-Auswahl fehlgeschlagen, nutze automatische Auswahl: " + str(e)[:200])
    cands = score_windows(segments, min_len, max_len, audio)
    if not clips:
        clips = highlights_heuristic(segments, n, min_len, max_len, audio, cands)
    for rank, c in enumerate(clips):  # Claude lists the best first; keep that order if it gives no score
        c.setdefault("score", 100 - rank)

    def select(primary, pool, max_l):
        chosen = [c for c in (snap(c, words, min_len, max_l) for c in primary) if c]
        chosen = remove_overlaps(chosen, 1.5)[:n]
        top = max([x["score"] for x in pool] or [1]) or 1
        for c in sorted(pool, key=lambda c: -c["score"]):  # fill free stretches with the best remaining windows
            if len(chosen) >= n:
                break
            sc = snap(dict(c), words, min_len, max_l)
            if sc and all(overlap(sc, o) <= 1.5 for o in chosen):
                sc = decorate(sc)
                sc["score"] = max(1, round(c["score"] / top * 100))
                sc["filled"] = True
                if source == "claude":
                    sc["hook"] = "Ergänzt durch die Automatik (KI hatte nicht genug passende Vorschläge)"
                chosen.append(sc)
        return chosen

    snapped = select(clips, cands, max_len)
    # not enough room for n clips of this length: try shorter ones, as long as that yields more clips
    for frac in (0.6, 0.3, 0.0):
        if len(snapped) >= n:
            break
        mx = min_len + (max_len - min_len) * frac + 5
        short = [c for c in cands if c["end"] - c["start"] <= mx]
        alt = select([dict(c) for c in clips if c["end"] - c["start"] <= mx + 1] if source == "claude"
                      else highlights_heuristic(segments, n, min_len, mx, audio, short), short, mx)
        if len(alt) > len(snapped):
            snapped = alt
    for c in snapped:
        if source != "claude":
            c.pop("filled", None)
    if len(snapped) < n:
        spoken = sum(s["e"] - s["s"] for s in segments)
        log("Nur %d von %d Clips möglich – %s" % (len(snapped), n,
            "das Video hat zu wenig Sprache für so viele Clips dieser Länge" if spoken < n * min_len * 1.3
            else "versuche eine kürzere Mindestlänge"))
    split_touching(snapped)
    for k, c in enumerate(snapped):
        if audio:  # keep the laugh or applause that follows the punchline
            nxt = snapped[k + 1]["start"] if k + 1 < len(snapped) else float("inf")
            ev = next((e for e in audio["events"] if e["type"] == "reaktion" and c["end"] - 0.5 <= e["s"] <= c["end"] + 1.5), None)
            if ev:
                c["end"] = round(min(ev["e"] + 0.3, c["end"] + 3.5, nxt), 2)
                c["reaction"] = True
            c["audio"] = audio_window(audio, c["start"], c["end"])
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

YUNET = os.environ.get("YUNET_MODEL", "/opt/models/face_detection_yunet_2023mar.onnx")


def detect_faces(video, info, start, end, faces=True):
    """Scene cuts and faces on 5 frames per second.

    Faces come from YuNet (a small modern CNN detector that also finds profiles and small faces; falls back to the
    older Haar cascades if the model file is missing) and are grouped into one track per person by position.
    Scene cuts are found from the same frames (large change of the whole picture between two frames).
    Returns {"tracks": [{"cx", "w", "seen", "motion", "pos"}], "cuts": [s], "frames", "fps"}, positions in source
    pixels, or None for portrait sources and when nothing could be decoded."""
    import cv2
    import numpy as np

    W, H = info["width"], info["height"]
    fps, aw = 5, 480
    ah = int(round(H * aw / W / 2)) * 2
    p = subprocess.run(
        [FFMPEG, "-v", "error", "-ss", "%.2f" % start, "-t", "%.2f" % (end - start), "-i", video,
         "-vf", "fps=%d,scale=%d:%d" % (fps, aw, ah), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        capture_output=True,
    )
    if p.returncode != 0 or not p.stdout:
        return None
    frames = np.frombuffer(p.stdout, np.uint8)
    frames = frames[: len(frames) // (aw * ah * 3) * aw * ah * 3].reshape(-1, ah, aw, 3)
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    # scene cuts: the whole picture changes much more than between neighbouring frames usually
    small = [cv2.resize(g, (48, 27)).astype(np.float32) for g in grays]
    diffs = [0.0] + [float(np.mean(np.abs(a - b))) for a, b in zip(small[1:], small[:-1])]
    base = float(np.median(diffs[1:])) if len(diffs) > 2 else 0.0
    cuts = [round(i / fps - 0.1, 2) for i, d in enumerate(diffs) if i and d > max(22.0, 4 * base)]

    result = {"tracks": [], "cuts": cuts, "frames": len(frames), "fps": fps}
    if not faces or W / H <= 9 / 16 + 0.01:
        return result

    yunet = None
    if os.path.exists(YUNET):
        try:
            yunet = cv2.FaceDetectorYN.create(YUNET, "", (aw, ah), 0.6, 0.3, 50)
        except cv2.error:
            yunet = None
    if yunet is None:
        frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    min_size = (max(20, ah // 9),) * 2
    cut_frames = {int(round((c + 0.1) * fps)) for c in cuts}

    tracks = []
    for i, (img, gray) in enumerate(zip(frames, grays)):
        if i in cut_frames:  # positions after a camera cut are not continuations of the old ones
            for t in tracks:
                t["last"] = -9999
        if yunet is not None:
            found = yunet.detect(img)[1]
            faces_ = [] if found is None else [(f[0], f[1], f[2], f[3], f[10:14]) for f in found]
        else:
            boxes = list(frontal.detectMultiScale(gray, 1.1, 5, minSize=min_size)) or \
                list(profile.detectMultiScale(gray, 1.1, 5, minSize=min_size))
            faces_ = [(x, y, w, h, None) for x, y, w, h in boxes]
        used = set()
        for x, y, w, h, mouth_pts in faces_:
            cx = x + w / 2
            best = min((t for t in tracks if id(t) not in used), key=lambda t: abs(t["last"] - cx), default=None)
            if best is None or abs(best["last"] - cx) > 0.12 * aw:
                best = {"cx": [], "w": [], "last": cx, "seen": 0, "motion": {}, "patch": None, "pos": {}}
                tracks.append(best)
            used.add(id(best))
            best["cx"].append(cx)
            best["w"].append(w)
            best["last"] = cx
            best["seen"] += 1
            best["pos"][i] = cx
            if mouth_pts is not None:  # mouth corners from the landmarks
                mx0, my0, mx1, my1 = [float(v) for v in mouth_pts]
                mw = max(8, abs(mx1 - mx0))
                x0, x1 = int(min(mx0, mx1) - mw * 0.2), int(max(mx0, mx1) + mw * 0.2)
                y0, y1 = int((my0 + my1) / 2 - mw * 0.45), int((my0 + my1) / 2 + mw * 0.45)
            else:
                x0, x1, y0, y1 = int(x + w * 0.2), int(x + w * 0.8), int(y + h * 0.62), int(y + h)
            mouth = gray[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
            if mouth.size:
                patch = cv2.resize(mouth, (24, 12)).astype(np.float32)
                if best["patch"] is not None and best["patch"][0] == i - 1:
                    best["motion"][i] = float(np.mean(np.abs(patch - best["patch"][1])))
                best["patch"] = (i, patch)

    scale = W / aw
    result["tracks"] = [{"cx": int(float(np.median(t["cx"])) * scale), "w": int(float(np.median(t["w"])) * scale * 1.6),
                         "seen": t["seen"] / max(1, len(frames)), "motion": t["motion"],
                         "pos": {k: v * scale for k, v in t["pos"].items()}}
                        for t in tracks if t["seen"] >= max(3, len(frames) * 0.08)]
    return result


def speaker_segments(det, duration):
    """Framing over time: [(t0, t1, centre_x, keyframes)] in clip time.

    Per second the person whose mouth moves most is the speaker; a change of speaker or a camera cut starts a new
    shot (hard cut, like a real edit), shots shorter than two seconds are merged away. Within a shot the frame
    follows the person smoothly: one keyframe per second, small moves ignored (dead zone), the rest eased."""
    import numpy as np

    if not det or not det["tracks"]:
        return None
    tracks, fps = det["tracks"], det["fps"]
    windows = max(1, int(np.ceil(det["frames"] / fps)))
    cut_at = {int(c): c for c in det.get("cuts", [])}  # window -> exact cut time
    cut_wins = set(cut_at)
    visible = lambda k, rng: any(f in tracks[k]["pos"] for f in rng)
    choice, prev = [], max(range(len(tracks)), key=lambda k: tracks[k]["seen"])
    for wi in range(windows):
        rng = range(wi * fps, (wi + 1) * fps)
        scores = [sum(t["motion"].get(f, 0) for f in rng) if visible(k, rng) else -1 for k, t in enumerate(tracks)]
        if max(scores) > 0:
            prev = int(np.argmax(scores))
        elif not visible(prev, rng):  # nobody talks visibly: keep someone who is in the picture
            vis = [k for k in range(len(tracks)) if visible(k, rng)]
            prev = max(vis, key=lambda k: tracks[k]["seen"]) if vis else prev
        choice.append(prev)
    runs = []
    for wi, k in enumerate(choice):
        if runs and runs[-1][0] == k and wi not in cut_wins:
            runs[-1][2] = wi + 1
        else:
            runs.append([k, wi, wi + 1])
    merged = []
    for k, a, b in runs:
        if merged and (b - a < 2 and a not in cut_wins or merged[-1][0] == k and a not in cut_wins):
            merged[-1][2] = b
        else:
            merged.append([k, a, b])

    W_guess = max(t["cx"] for t in tracks) * 2 or 1
    segs = []
    for k, a, b in merged:
        t = tracks[k]
        pts, last = [], None
        for wi in range(a, b):
            xs = [t["pos"][f] for f in range(wi * fps, (wi + 1) * fps) if f in t["pos"]]
            if not xs:
                continue
            x = float(np.median(xs))
            if last is not None and abs(x - last) < 0.035 * W_guess:  # dead zone against jitter
                x = last
            elif last is not None:
                x = last + 0.6 * (x - last)  # ease towards the new position
            pts.append((wi + 0.5, x))
            last = x
        t0 = float(cut_at.get(a, a)) if segs else 0.0
        t1 = float(min(duration, b))
        if segs:
            segs[-1] = (segs[-1][0], t0) + segs[-1][2:]
        if not pts:
            pts = [(t0, float(t["cx"]))]
        segs.append((t0, t1, int(pts[0][1]), [(max(t0, min(t1, pt)), int(x)) for pt, x in pts]))
    return segs


def split_pair(det):
    """The two most visible people, left one first – for the split-screen layout; None if fewer than two."""
    if not det or len(det["tracks"]) < 2:
        return None
    a, b = sorted(sorted(det["tracks"], key=lambda t: -t["seen"])[:2], key=lambda t: t["cx"])
    if b["seen"] < 0.25 or a["seen"] < 0.25:  # the second person is hardly ever in the picture
        return None
    return a["cx"], b["cx"]


def track_speaker(video, info, start, end):
    return speaker_segments(detect_faces(video, info, start, end), end - start)


def crop_box(W, H, ar):
    """Largest crop of the source with aspect ratio `ar` (width/height)."""
    if W / H > ar:
        return int(H * ar) // 2 * 2, H
    return W, int(W / ar) // 2 * 2


def crop_filter(segs, W, H, ar):
    """Crop that follows the framing: hard switches between shots, linear glides between keyframes within one."""
    cw, ch = crop_box(W, H, ar)
    y = (H - ch) // 2
    if not segs:
        return "crop=%d:%d:%d:%d" % (cw, ch, (W - cw) // 2, y)
    clampx = lambda cx: int(min(max(0, cx - cw / 2), W - cw))

    def seg_expr(seg):
        pts = seg[3] if len(seg) > 3 and seg[3] else [(seg[0], seg[2])]
        pts = [(t, clampx(cx)) for t, cx in pts]
        expr = str(pts[-1][1])
        for (ta, xa), (tb, xb) in reversed(list(zip(pts[:-1], pts[1:]))):
            body = str(xa) if xa == xb else "%d+%d*(t-%.2f)/%.2f" % (xa, xb - xa, ta, max(0.01, tb - ta))
            expr = "if(lt(t,%.2f),%s,%s)" % (tb, body, expr)
        return "if(lt(t,%.2f),%d,%s)" % (pts[0][0], pts[0][1], expr) if pts[0][0] > seg[0] + 0.01 else expr

    expr = seg_expr(segs[-1])
    for seg in reversed(segs[:-1]):
        expr = "if(lt(t,%.2f),%s,%s)" % (seg[1], seg_expr(seg), expr)
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


# words that belong to what follows: a subtitle line should not end on them
GLUE = {
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "einer", "eines", "und", "oder",
    "aber", "sondern", "mit", "in", "im", "ins", "zu", "zum", "zur", "von", "vom", "für", "auf", "an", "am", "bei",
    "beim", "nach", "aus", "über", "unter", "vor", "durch", "gegen", "ohne", "um", "als", "wie", "dass", "weil",
    "wenn", "ob", "nicht", "kein", "keine", "keinen", "sehr", "so", "noch", "schon", "mein", "meine", "meinen",
    "dein", "deine", "sein", "seine", "ihr", "ihre", "unser", "unsere", "dieser", "diese", "dieses", "diesen",
    "jeder", "jede", "jedes", "jeden", "the", "a", "an", "of", "to", "and",
}


def _len(ws):
    return len(" ".join(x["w"] for x in ws))


def chunk_words(words, max_words, max_chars=20):
    """Subtitle lines that break where the meaning allows it.

    Hard breaks at sentence marks and pauses; otherwise a line is full at `max_words`/`max_chars`, but it never
    ends on an article, preposition or conjunction, and names (capitalised words in a row) stay together.
    Lines that would only flash up for a moment are merged with the next one."""
    chunks, cur = [], []
    for w in words:
        if cur:
            prev = cur[-1]
            hard = w["s"] - prev["e"] > 0.6 or prev["w"][-1:] in ".!?,;:"
            full = len(cur) >= max_words or _len(cur + [w]) > max_chars
            name = (len(cur) >= 2 or not chunks) and prev["w"][:1].isupper() and w["w"][:1].isupper() \
                and norm_word(prev["w"]) not in GLUE and _len(cur + [w]) <= max_chars + 10
            if hard or (full and not name):
                carry = []
                if not hard and len(cur) > 1 and norm_word(cur[-1]["w"]) in GLUE:
                    carry = [cur.pop()]
                chunks.append(cur)
                cur = carry
        cur.append(w)
    if cur:
        chunks.append(cur)
    merged = []
    for ch in chunks:  # a line shown for less than ~0.45 s cannot be read: join it with the previous one
        if merged and ch[-1]["e"] - ch[0]["s"] < 0.45 and merged[-1][-1]["w"][-1:] not in ".!?" \
                and ch[0]["s"] - merged[-1][-1]["e"] < 0.3 and _len(merged[-1] + ch) <= max_chars + 8:
            merged[-1] = merged[-1] + ch
        else:
            merged.append(ch)
    return merged


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
    if style.get("split_layout"):  # split screen: subtitles sit on the seam between the two people
        margin_v = int(0.455 * ph)
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

def out_fps(info):
    """Keep the source frame rate (24-60 fps) instead of forcing 30."""
    return int(min(60, max(24, round(info.get("fps") or 30))))


def quantize(intervals, fps):
    """Snap cut points to whole frames so picture and sound are cut at exactly the same place (no drift)."""
    out = []
    for a, b in intervals:
        a, b = round(a * fps) / fps, round(b * fps) / fps
        if b - a >= 1 / fps:
            out.append((a, b))
    return out


def select_expr(intervals, fps=30):
    h = 0.5 / fps  # frame k (at k/fps) belongs to [a, b) -> exactly (b-a)*fps frames per part
    return "+".join("between(t,%.4f,%.4f)" % (a - h, b - h) for a, b in intervals)


SCALE = "flags=lanczos"
SHARPEN = "unsharp=5:5:0.55:5:5:0.0"  # a touch of luma sharpening after enlarging the crop


def video_chain(info, mode, crop_segs, sel, ow, oh, split=None):
    w, h = info["width"], info["height"]
    fps = out_fps(info)
    if split:  # two people stacked: left person on top, right person below, thin line in between
        hh = oh // 2
        cw, ch = crop_box(w, h, ow / hh)
        y = (h - ch) // 2
        xs = [int(min(max(0, cx - cw / 2), w - cw)) for cx in split]
        return (
            "[0:v]fps=%d,split=2[pa][pb];"
            "[pa]crop=%d:%d:%d:%d,scale=%d:%d:%s,setsar=1[top];"
            "[pb]crop=%d:%d:%d:%d,scale=%d:%d:%s,setsar=1[bot];"
            "[top][bot]vstack=inputs=2,%s,drawbox=x=0:y=%d:w=%d:h=6:color=black@0.7:t=fill%s"
            % (fps, cw, ch, xs[0], y, ow, hh, SCALE, cw, ch, xs[1], y, ow, oh - hh, SCALE, SHARPEN, hh - 3, ow, sel)
        )
    if mode == "blur" and w / h > ow / oh + 0.01:
        return (
            "[0:v]fps=%d" % fps + sel + ",split=2[a][b];"
            "[a]scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d,boxblur=24:3,eq=brightness=-0.08[bg];"
            "[b]scale=%d:%d:force_original_aspect_ratio=decrease:%s[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1" % (ow, oh, ow, oh, ow, oh, SCALE)
        )
    crop = crop_filter(crop_segs if mode == "auto" else None, w, h, ow / oh)
    return "[0:v]fps=%d," % fps + crop + sel + ",scale=%d:%d:%s,%s,setsar=1" % (ow, oh, SCALE, SHARPEN)


RNNOISE = os.environ.get("RNNOISE_MODEL", "/opt/models/sh.rnnn")


def filter_path(path):
    return "'" + path.replace("\\", "/").replace(":", "\\:") + "'"


def audio_chain(intervals, cut, enhance, music, music_volume):
    """Voice clean-up (rumble, AI noise reduction, sibilance, levelling), soft 12 ms fades at every cut so the
    pause cutter never clicks, and optional background music that ducks under speech."""
    head = "[0:a]aformat=sample_rates=48000:channel_layouts=stereo"
    if cut and len(intervals) > 1:
        n = len(intervals)
        parts = [head + ",asplit=%d%s" % (n, "".join("[s%d]" % i for i in range(n)))]
        for i, (a, b) in enumerate(intervals):
            ln = b - a
            fade = ",afade=t=in:st=0:d=0.012,afade=t=out:st=%.4f:d=0.012" % (ln - 0.012) if ln > 0.05 else ""
            parts.append("[s%d]atrim=start=%.4f:end=%.4f,asetpts=PTS-STARTPTS%s[p%d]" % (i, a, b, fade, i))
        parts.append("%sconcat=n=%d:v=0:a=1" % ("".join("[p%d]" % i for i in range(n)), n))
        a = ";".join(parts)
    elif cut:
        a = head + ",atrim=start=%.4f:end=%.4f,asetpts=PTS-STARTPTS" % intervals[0]
    else:
        a = head
    if enhance:
        denoise = ("arnndn=m=%s:mix=0.85" % filter_path(RNNOISE)) if os.path.exists(RNNOISE) else "afftdn=nf=-25"
        a += ",highpass=f=80," + denoise + ",deesser=i=0.4,acompressor=threshold=-20dB:ratio=3:attack=5:release=150"
    if music:
        a += (",asplit=2[vs][vm];"
              "[1:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=%.2f[m];"
              "[m][vs]sidechaincompress=threshold=0.03:ratio=10:attack=15:release=400[md];"
              "[vm][md]amix=inputs=2:duration=first:normalize=0" % (music_volume / 100))
    return a + ",loudnorm=I=-14:TP=-1.5:LRA=11[a]"


def render_clip(video, info, start, end, workdir, name, style, ass_name, burn, intervals, crop_segs,
                size=(1080, 1920), zoom_spans_out=None, music=None, split=None):
    ow, oh = size
    fps = out_fps(info)
    duration = end - start
    cut = bool(intervals) and not (len(intervals) == 1 and intervals[0][0] <= 0.01 and intervals[0][1] >= duration - 0.01)
    sel = (",select='%s',setpts=N/%d/TB" % (select_expr(intervals, fps), fps)) if cut else ""
    graph = video_chain(info, style.get("mode", "auto"), crop_segs, sel, ow, oh, split)
    if style.get("zoom", True) and zoom_spans_out and not split:
        graph += zoom_filter(zoom_spans_out, ow, oh)
    graph += ((",ass=" + ass_name) if burn else "") + "[v]"
    cmd = [FFMPEG, "-y", "-ss", "%.3f" % start, "-t", "%.3f" % duration, "-i", video]
    has_audio = info.get("has_audio", True)
    if has_audio:
        if music:
            cmd += ["-stream_loop", "-1", "-i", music]
        graph += ";" + audio_chain(intervals, cut, style.get("enhance_audio", True), music,
                                   int(style.get("music_volume", 15)))
    out = name + ".mp4"
    cmd += ["-filter_complex", graph, "-map", "[v]"] + (["-map", "[a]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000"] if has_audio else [])
    cmd += ["-c:v", "libx264", "-preset", "faster", "-crf", "19", "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-r", str(fps), "-shortest", "-movflags", "+faststart", "-threads", str(THREADS), out]
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


# ---------------------------------------------------------------- translation

LANGUAGES = {"en": "Englisch", "es": "Spanisch", "fr": "Französisch", "it": "Italienisch", "tr": "Türkisch", "pl": "Polnisch"}


def sentences(words):
    out, cur = [], []
    for w in words:
        cur.append(w)
        if w["w"][-1:] in ".!?" or len(cur) >= 25:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def translate_clip(out_words, api_key, lang, meta):
    """Translate subtitles sentence by sentence; each translated sentence keeps the time span of the original,
    its words spread evenly across it. Returns (words, translated meta)."""
    sents = sentences(out_words)
    payload = {
        "sentences": [" ".join(w["w"] for w in s) for s in sents],
        "hook": meta.get("hook", ""), "title": meta.get("title", ""), "caption": meta.get("caption", ""),
        "cover": meta.get("cover", ""), "keywords": meta.get("keywords", []),
    }
    prompt = (
        "Übersetze die folgenden Texte eines deutschsprachigen Kurzvideos ins %s. Die Sätze sind Untertitel und "
        "müssen kurz, natürlich und gesprochen klingen; behalte Anzahl und Reihenfolge der Sätze exakt bei. "
        "Übersetze Hashtags sinnvoll. \"keywords\" sind Wörter, die farbig hervorgehoben werden: gib die "
        "entsprechenden Wörter aus deiner Übersetzung an.\n"
        "Antworte NUR mit JSON im selben Format:\n%s"
    ) % (LANGUAGES.get(lang, lang), json.dumps(payload, ensure_ascii=False))
    res = claude_json(api_key, prompt, max_tokens=6000, feature="Übersetzung")
    trans = res.get("sentences") or []
    if len(trans) != len(sents):
        raise RuntimeError("Übersetzung unvollständig (%d von %d Sätzen)." % (len(trans), len(sents)))
    words = []
    for src, text in zip(sents, trans):
        tokens = str(text).split()
        if not tokens:
            continue
        s, e = src[0]["s"], src[-1]["e"]
        step = (e - s) / len(tokens)
        words += [{"w": t, "s": s + k * step, "e": s + (k + 1) * step} for k, t in enumerate(tokens)]
    return words, {k: res.get(k, "") for k in ("hook", "title", "caption", "cover", "keywords")}


# ---------------------------------------------------------------- best-of trailer

def bestof_segments(segments, clips, target=45.0):
    """Strongest clip first: take its opening sentence, then the next clip's, and so on; if one round is not long
    enough, further sentences of the same clips follow. Each part stays in story order within its clip."""
    words = all_words(segments)
    ranked = [c for c in sorted(clips, key=lambda c: -c.get("score", 0)) if c.get("variant_of") is None]
    sents = {}
    for c in ranked:
        cw = [w for w in words if w["s"] >= c["start"] - 0.01 and w["e"] <= c["end"] + 0.01]
        merged = []
        for s in sentences(cw):  # sentences under 2.5 s are joined with the next one
            if merged and merged[-1][-1]["e"] - merged[-1][0]["s"] < 2.5:
                merged[-1] = merged[-1] + s
            else:
                merged.append(s)
        sents[id(c)] = merged
    picks, total = [], 0.0
    for rnd in range(4):
        for c in ranked:
            if total >= target or rnd >= len(sents[id(c)]):
                continue
            sent = sents[id(c)][rnd]
            a, b = sent[0]["s"], sent[-1]["e"]
            if b - a > 14:
                b = next((w["e"] for w in sent if w["e"] - a >= 10), a + 14)
            if total + (b - a) > target + 5 and picks:
                continue
            picks.append((round(max(0, a - 0.1), 2), round(b + 0.3, 2), c))
            total += b - a
        if total >= target * 0.85:
            break
    # within the trailer, keep the clips in score order but each clip's sentences in their natural order
    order = {id(c): k for k, c in enumerate(ranked)}
    picks.sort(key=lambda p: (order[id(p[2])], p[0]))
    joined = []
    for a, b, c in picks:  # neighbouring sentences of the same clip become one part (no needless cut)
        if joined and joined[-1][2] is c and a - joined[-1][1] < 0.8:
            joined[-1] = (joined[-1][0], b, c)
        else:
            joined.append((a, b, c))
    return joined


def concat_parts(workdir, parts, out, music=None, music_volume=15):
    """Join rendered parts (same encoding settings) losslessly; optional music is mixed over the whole trailer."""
    lst = out + ".txt"
    with open(os.path.join(workdir, lst), "w", encoding="utf-8") as f:
        f.write("".join("file '%s'\n" % p for p in parts))
    joined = out if not music else "joined_" + out
    run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", joined], cwd=workdir)
    if music:
        graph = ("[0:a]asplit=2[vs][vm];[1:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=%.2f[m];"
                 "[m][vs]sidechaincompress=threshold=0.03:ratio=10:attack=15:release=400[md];"
                 "[vm][md]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.9[a]" % (music_volume / 100))
        run([FFMPEG, "-y", "-i", joined, "-stream_loop", "-1", "-i", music, "-filter_complex", graph, "-map", "0:v",
             "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", out],
            cwd=workdir)
        os.remove(os.path.join(workdir, joined))
    os.remove(os.path.join(workdir, lst))


def frame_jpeg(video, t, width=640):
    p = subprocess.run([FFMPEG, "-v", "error", "-ss", "%.2f" % max(0, t), "-i", video, "-frames:v", "1",
                        "-vf", "scale=%d:-2" % width, "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "4", "-"],
                       capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError("Standbild konnte nicht erstellt werden.")
    return p.stdout


# ---------------------------------------------------------------- audio analysis

def _robust_z(x, mask=None):
    import numpy as np

    ref = x[mask] if mask is not None and mask.any() else x
    med = float(np.median(ref)) if len(ref) else 0.0
    mad = float(np.median(np.abs(ref - med))) * 1.4826 if len(ref) else 1.0
    return (x - med) / (mad if mad > 1e-6 else 1.0)


def _merge_events(flags, step, kind, min_len):
    out, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if (i - start) * step >= min_len:
                out.append({"s": round(start * step, 2), "e": round(i * step, 2), "type": kind})
            start = None
    return out


def analyze_audio(wav, words, step=0.5):
    """Cheap prosody analysis of the whole recording (CPU, numpy only).

    Per `step` seconds: loudness, pitch variation (how lively the voice is) and speech rate, each relative to the
    speaker's own typical level, combined into one energy value. Loud stretches without words are reported as
    reactions (laughter, applause, shouting)."""
    import wave

    import numpy as np

    with wave.open(wav, "rb") as f:
        sr = f.getframerate()
        a = np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float32) / 32768.0
    hop = int(sr * step)
    n = len(a) // hop
    if n < 4:
        return None

    # loudness per window
    win = a[: n * hop].reshape(n, hop)
    db = 20 * np.log10(np.sqrt((win ** 2).mean(1)) + 1e-6)

    # pitch per 20 ms frame via autocorrelation (70-400 Hz), only where there is signal
    fl, fh = int(sr * 0.04), int(sr * 0.02)
    nf = (len(a) - fl) // fh
    f0 = np.zeros(max(nf, 0), np.float32)
    lo, hi = sr // 400, sr // 70
    gate = 10 ** ((np.median(db) - 10) / 20)
    for c0 in range(0, nf, 4000):
        idx = np.arange(c0, min(nf, c0 + 4000))
        frames = a[idx[:, None] * fh + np.arange(fl)[None, :]]
        frames = frames - frames.mean(1, keepdims=True)
        loud = np.sqrt((frames ** 2).mean(1)) > gate
        if not loud.any():
            continue
        spec = np.fft.rfft(frames[loud] * np.hanning(fl), 1024)
        ac = np.fft.irfft(np.abs(spec) ** 2, 1024)[:, : hi + 1]
        lag = ac[:, lo:].argmax(1) + lo
        strength = ac[np.arange(len(lag)), lag] / (ac[:, 0] + 1e-9)
        vals = np.where(strength > 0.35, sr / lag, 0.0)
        f0[idx[loud]] = vals
    per = max(1, int(step / 0.02))
    pitch_var = np.zeros(n, np.float32)
    for i in range(n):
        seg = f0[i * per:(i + 1) * per]
        seg = seg[seg > 0]
        if len(seg) >= 5:
            pitch_var[i] = float(np.std(12 * np.log2(seg / np.median(seg))))

    # speech rate and where words are
    rate = np.zeros(n, np.float32)
    has_words = np.zeros(n, bool)
    for w in words:
        i = int(w["s"] / step)
        if 0 <= i < n:
            rate[i] += 1
        for j in range(int(w["s"] / step), min(n, int(w["e"] / step) + 1)):
            has_words[j] = True
    k = max(1, int(2 / step))  # 2 s moving window for the rate
    rate = np.convolve(rate, np.ones(k) / (k * step), "same")

    speech = has_words & (db > np.median(db) - 20)
    z_loud = _robust_z(db, speech)
    z_pitch = _robust_z(pitch_var, speech & (pitch_var > 0))
    z_rate = _robust_z(rate, speech)
    energy = 0.5 * np.clip(z_loud, -3, 4) + 0.3 * np.clip(z_pitch, -3, 4) + 0.2 * np.clip(z_rate, -3, 4)
    energy = np.where(speech, energy, np.where(db > np.median(db[speech]) - 6 if speech.any() else False, 1.0, -1.0))
    energy = np.convolve(energy, np.ones(3) / 3, "same")

    events = (_merge_events(speech & (z_loud > 1.8), step, "laut", 1.0)
              + _merge_events(speech & (z_pitch > 1.8), step, "lebhaft", 1.0)
              + _merge_events(speech & (z_rate > 1.8), step, "schnell", 2.0))
    if speech.any():  # loud without words: laughter, applause, shouting
        ref = float(np.median(db[speech]))
        events += _merge_events(~has_words & (db > ref - 4), step, "reaktion", 1.0)
    events.sort(key=lambda e: e["s"])
    return {"step": step, "energy": [round(float(x), 2) for x in energy], "events": events}


def audio_window(audio, s, e):
    """Mean and peak energy plus the event types inside [s, e]."""
    if not audio:
        return None
    st, en = audio["step"], audio["energy"]
    vals = en[int(s / st): max(int(s / st) + 1, int(e / st))]
    if not vals:
        return None
    kinds = sorted({ev["type"] for ev in audio["events"] if ev["e"] > s and ev["s"] < e})
    return {"mean": round(sum(vals) / len(vals), 2), "peak": round(max(vals), 2), "tags": kinds}


def audio_curve(audio, points=600):
    """Downsampled energy curve for the timeline in the web interface."""
    if not audio:
        return None
    en = audio["energy"]
    k = max(1, len(en) // points)
    return {"step": audio["step"] * k, "values": [round(max(en[i:i + k]), 2) for i in range(0, len(en), k)]}


TAG_LABELS = {"laut": "laut", "lebhaft": "lebhafte Stimme", "schnell": "schnell gesprochen", "reaktion": "Lachen/Reaktion"}


def add_spans(intervals, spans, duration):
    """Union of kept intervals with spans that must stay (e.g. a laugh after the punchline)."""
    allv = sorted([list(i) for i in intervals] + [[max(0.0, a), min(duration, b)] for a, b in spans if b > a])
    out = []
    for a, b in allv:
        if out and a <= out[-1][1] + 0.05:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(round(a, 3), round(b, 3)) for a, b in out]


# ---------------------------------------------------------------- two-stage AI selection

CRITERIA = (("hook", "Hook", 0.35), ("standalone", "Verständlich", 0.25), ("value", "Mehrwert/Emotion", 0.25),
            ("payoff", "Pointe", 0.15))


def sentence_lines(words, start, end):
    """Sentences with timestamps for a stretch of the video, the material for the fine cut."""
    sel = [w for w in words if w["e"] > start and w["s"] < end]
    return "\n".join("[%.1f-%.1f] %s" % (s[0]["s"], s[-1]["e"], " ".join(w["w"] for w in s)) for s in sentences(sel))


def highlights_claude_two_stage(segments, api_key, n, min_len, max_len, context, audio, log):
    """Stage 1: many candidates, each rated on fixed criteria. Stage 2: the best ones get a sentence-level fine cut,
    titles and hooks, and may be rejected after a closer look."""
    words = all_words(segments)
    k = min(20, max(12, n * 3))
    extra = ("\nKontext vom Ersteller: " + context.strip()) if context and context.strip() else ""
    prompt1 = (
        "Du bist ein sehr anspruchsvoller Social-Media-Editor für TikTok, Instagram Reels und YouTube Shorts.\n"
        "Unten steht das Transkript eines deutschsprachigen Videos mit Zeitstempeln in Sekunden. Hinweise in ⟨⟩ "
        "stammen aus einer automatischen Tonanalyse (Lautstärke, lebhafte Stimme, Tempo, Lachen/Applaus).%s\n\n"
        "Sammle die %d vielversprechendsten Kandidaten für eigenständige Kurzvideos (je %d-%d Sekunden, Start und Ende "
        "auf Zeitstempeln des Transkripts, Überschneidungen erlaubt). Bewerte jeden streng von 0 bis 10:\n"
        "- hook: Fesseln die ersten 3 Sekunden sofort (Frage, steile These, Überraschung, Konflikt, Emotion)?\n"
        "- standalone: Versteht man den Ausschnitt ohne den Rest des Videos?\n"
        "- value: Bietet er einen echten Mehrwert (Tipp, Erkenntnis, Geschichte) oder starke Emotion?\n"
        "- payoff: Endet er mit Pointe, Auflösung oder abgeschlossenem Gedanken?\n"
        "10 ist außergewöhnlich und selten, 5 ist Durchschnitt. Nutze die ganze Skala.\n\n"
        "Antworte NUR mit einem JSON-Array: "
        '[{"start": 12.3, "end": 48.0, "scores": {"hook": 7, "standalone": 8, "value": 6, "payoff": 7}, '
        '"reason": "ein Satz, warum der Ausschnitt funktioniert"}]\n\nTRANSKRIPT:\n%s'
    ) % (extra, k, min_len, max_len, annotated_transcript(segments, audio))
    cands = claude_json(api_key, prompt1, r"\[.*\]", max_tokens=6000, feature="KI-Auswahl")
    rated = []
    for c in cands:
        try:
            sc = {key: max(0.0, min(10.0, float((c.get("scores") or {}).get(key, 5)))) for key, _, _ in CRITERIA}
            start, end = float(c["start"]), float(c["end"])
        except (TypeError, ValueError, KeyError):
            continue
        total = sum(sc[key] * w for key, _, w in CRITERIA)
        rated.append({"start": start, "end": end, "criteria": sc, "score": round(total * 10),
                      "reason": str(c.get("reason", ""))[:300]})
    if not rated:
        raise RuntimeError("Claude hat keine Kandidaten geliefert.")
    snapped = [x for x in (snap(c, words, min_len, max_len) for c in rated) if x]
    best = remove_overlaps(snapped, 1.5)[: n + 2]
    log("KI-Auswahl Durchgang 1: %d Kandidaten bewertet, %d in der engeren Wahl" % (len(rated), len(best)))

    # stage 2: fine cut with sentence-level timestamps and some room before and after
    blocks = []
    for i, c in enumerate(best):
        lo, hi = max(0.0, c["start"] - 20), c["end"] + 20
        blocks.append("### Clip %d (Rohschnitt %.1f-%.1f, Note %d)\n%s" % (i, c["start"], c["end"], c["score"],
                                                                         sentence_lines(words, lo, hi)))
    prompt2 = (
        "Du bekommst %d Rohschnitte für Kurzvideos aus einem deutschsprachigen Video, jeweils mit Sätzen davor und "
        "danach (Zeitstempel in Sekunden).%s\nPrüfe jeden Rohschnitt genau und lege den finalen Schnitt fest:\n"
        "- Start auf den Satz, der sofort fesselt; lass schwache Einleitungen, Begrüßungen und Füllsätze weg.\n"
        "- Ende nach dem abgeschlossenen Gedanken bzw. der Pointe, nicht mitten in der Auflösung.\n"
        "- Länge %d-%d Sekunden; Start und Ende exakt auf Satzgrenzen aus der Liste.\n"
        "- \"keep\": false, wenn der Ausschnitt bei genauem Hinsehen schwach oder ohne Vorwissen unverständlich ist.\n"
        "Schreibe außerdem Titel, eine Begründung, 3 Hook-Texte (max. 8 Wörter, unterschiedliche Ansätze), eine "
        "Beschreibung mit 3-5 Hashtags und 2-6 Schlüsselwörter, die wörtlich im Clip vorkommen.\n\n"
        "Antworte NUR mit einem JSON-Array: "
        '[{"id": 0, "keep": true, "start": 14.2, "end": 51.8, "title": "...", "reason": "...", '
        '"hooks": ["...", "...", "..."], "caption": "...", "keywords": ["..."]}]\n\n%s'
    ) % (len(best), extra, min_len, max_len, "\n\n".join(blocks))
    try:
        refined = {int(r.get("id", -1)): r for r in claude_json(api_key, prompt2, r"\[.*\]", max_tokens=6000,
                                                                  feature="KI-Feinschnitt")}
    except Exception as e:  # the stage-1 result is still usable
        log("KI-Feinschnitt fehlgeschlagen, nutze Durchgang 1: " + str(e)[:200])
        refined = {}
    out, dropped = [], 0
    for i, c in enumerate(best):
        r = refined.get(i)
        if r is not None and r.get("keep") is False:
            dropped += 1
            c["rejected"] = True
        if r:
            try:
                c["start"], c["end"] = float(r.get("start", c["start"])), float(r.get("end", c["end"]))
            except (TypeError, ValueError):
                pass
            for key in ("title", "caption", "hooks", "keywords"):
                if r.get(key):
                    c[key] = r[key]
            c["reason"] = str(r.get("reason") or c["reason"])[:300]
        c["hook"] = c["reason"]
        c.setdefault("title", c["reason"][:70])
        out.append(c)
    # rejected ones go to the back: they are used only if nothing better is left
    out.sort(key=lambda c: (c.get("rejected", False), -c["score"]))
    if refined:
        log("KI-Feinschnitt: %d Clips verfeinert%s" % (len(out) - dropped, (", %d verworfen" % dropped) if dropped else ""))
    return out


# ---------------------------------------------------------------- quality check

def volume_stats(workdir, name):
    p = subprocess.run([FFMPEG, "-hide_banner", "-i", name, "-vn", "-af", "volumedetect", "-f", "null", "-"], cwd=workdir,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    mean = re.search(r"mean_volume: (-?[\d.]+) dB", p.stderr)
    peak = re.search(r"max_volume: (-?[\d.]+) dB", p.stderr)
    return (float(mean.group(1)) if mean else None), (float(peak.group(1)) if peak else None)


def quality_checks(workdir, name, info, segments, start, end, det, crop_segs, mode, split_wanted, split, out_len):
    """Automatic checks of a rendered clip; returns a list of {"level": "warn"|"info", "msg": ...}."""
    out = []
    if not info.get("has_audio", True):
        out.append({"level": "warn", "msg": "Das Video hat keine Tonspur."})
    else:
        mean, peak = volume_stats(workdir, name)
        if mean is not None and mean < -28:
            out.append({"level": "warn", "msg": "Ton sehr leise (%.0f dB im Mittel)." % mean})
        if peak is not None and peak >= -0.1:
            out.append({"level": "warn", "msg": "Ton übersteuert – Spitzen erreichen 0 dB."})
    if out_len < 5:
        out.append({"level": "warn", "msg": "Clip ist nach dem Schnitt nur %.0f s lang." % out_len})
    for w in all_words(segments):  # cut points inside a word
        if w["s"] + 0.08 < start < w["e"] - 0.08:
            out.append({"level": "warn", "msg": "Beginnt mitten im Wort „%s“." % w["w"]})
        if w["s"] + 0.08 < end < w["e"] - 0.08:
            out.append({"level": "warn", "msg": "Endet mitten im Wort „%s“." % w["w"]})
    landscape = info["width"] / info["height"] > 9 / 16 + 0.01
    crop_w = crop_box(info["width"], info["height"], 9 / 16)[0]
    if landscape and mode in ("auto", "crop") and not split and crop_w < 700:
        out.append({"level": "warn", "msg": "Für Hochformat stehen nur %d Pixel Breite zur Verfügung (wird auf 1080 hochgerechnet). "
                                            "Schärfer wird es mit 4K-Aufnahmen oder direkt im Hochformat." % crop_w})
    if landscape and mode == "auto" and not split:
        if not det or not det["tracks"]:
            out.append({"level": "info", "msg": "Kein Gesicht erkannt – Bildmitte verwendet. Bei Bedarf Ausschnitt von Hand setzen."})
        elif crop_segs:
            cw, _ = crop_box(info["width"], info["height"], 9 / 16)
            for seg in crop_segs:
                t0, cx = seg[0], seg[2]
                x = min(max(0, cx - cw / 2), info["width"] - cw)
                for t in det["tracks"]:
                    if t["seen"] < 0.25:
                        continue
                    left, right = t["cx"] - t["w"] / 2, t["cx"] + t["w"] / 2
                    if left < x < right or left < x + cw < right:
                        out.append({"level": "warn", "msg": "Person am Bildrand angeschnitten (ab %s)." % fmt_time(t0)})
                        break
                else:
                    continue
                break
    if split_wanted and not split and landscape:
        out.append({"level": "info", "msg": "Geteilter Bildschirm nicht möglich – keine zwei Personen erkannt, normaler Zuschnitt verwendet."})
    return out


def spelling_check(api_key, texts, vocabulary=""):
    """Likely recognition or spelling errors in subtitle texts: {clip_index: [{"wrong", "suggestion"}]}."""
    if not texts:
        return {}
    vocab = ("\nRichtig geschriebene Fachbegriffe und Namen: " + vocabulary.strip()) if vocabulary.strip() else ""
    prompt = (
        "Die folgenden Untertitel stammen aus automatischer Spracherkennung eines deutschsprachigen Videos.%s\n"
        "Finde nur echte Fehler: falsch erkannte Wörter, Rechtschreibfehler, falsch geschriebene Namen oder Marken. "
        "Ignoriere Umgangssprache, fehlende Kommas und Stil. \"wrong\" muss wörtlich im Text vorkommen (1-3 Wörter).\n"
        "Antworte NUR mit JSON: {\"0\": [{\"wrong\": \"...\", \"suggestion\": \"...\"}], \"1\": []}\n\n%s"
    ) % (vocab, "\n\n".join("### %s\n%s" % (k, t) for k, t in texts.items()))
    res = claude_json(api_key, prompt, max_tokens=3000, feature="Rechtschreibung")
    out = {}
    for k, items in (res or {}).items():
        try:
            idx = int(k)
        except ValueError:
            continue
        good = [{"wrong": str(i["wrong"]), "suggestion": str(i["suggestion"])} for i in items or []
                if isinstance(i, dict) and i.get("wrong") and i.get("suggestion") and str(i["wrong"]) in texts.get(idx, "")
                and str(i["wrong"]) != str(i["suggestion"])]
        out[idx] = good[:10]
    return out


# ---------------------------------------------------------------- text content

def generate_texts(api_key, segments, context="", titles=()):
    """YouTube chapters, a LinkedIn post, a blog/newsletter text and carousel slides from the whole transcript."""
    lines = "\n".join("[%s] %s" % (fmt_time(s["s"]), s["text"]) for s in segments)
    extra = ("\nKontext vom Ersteller: " + context.strip()) if context and context.strip() else ""
    prompt = (
        "Hier ist das Transkript eines deutschsprachigen Videos mit Zeitstempeln (Minuten:Sekunden).%s\n"
        "Erstelle daraus Inhalte für die Verwertung auf anderen Kanälen, auf Deutsch, im Stil und mit den Aussagen des "
        "Sprechers (nichts erfinden):\n"
        "1. chapters: YouTube-Kapitel, das erste bei 0:00, mindestens 3, sinnvolle Abschnitte mit mind. 30 s Abstand, "
        "kurze Titel (max. 6 Wörter).\n"
        "2. linkedin: LinkedIn-Beitrag (120-200 Wörter): starke erste Zeile, kurze Absätze, eine Frage am Ende, 3 Hashtags.\n"
        "3. blog: Blog- bzw. Newsletter-Artikel mit title und text (300-500 Wörter, Zwischenüberschriften mit ##).\n"
        "4. carousel: 6-8 Folien für ein Instagram-Karussell, je title (max. 6 Wörter) und text (max. 25 Wörter); "
        "Folie 1 ist der Aufhänger, die letzte eine Aufforderung.\n"
        "Antworte NUR mit JSON: {\"chapters\": [{\"time\": \"0:00\", \"title\": \"...\"}], \"linkedin\": \"...\", "
        "\"blog\": {\"title\": \"...\", \"text\": \"...\"}, \"carousel\": [{\"title\": \"...\", \"text\": \"...\"}]}\n\n"
        "TRANSKRIPT:\n%s"
    ) % (extra, lines)
    res = claude_json(api_key, prompt, max_tokens=6000, feature="Texte")

    def secs(t):
        parts = [float(x) for x in str(t).split(":")]
        return sum(v * 60 ** k for k, v in enumerate(reversed(parts)))

    chapters, last = [], -999
    for c in sorted(res.get("chapters") or [], key=lambda c: secs(c.get("time", 0))):
        try:
            t = secs(c.get("time", 0))
        except ValueError:
            continue
        if not chapters:
            t = 0
        if t - last >= 10 and c.get("title"):
            chapters.append({"time": fmt_time(t), "title": str(c["title"])[:80]})
            last = t
    if len(chapters) < 3:
        chapters = []
    blog = res.get("blog") or {}
    return {
        "chapters": chapters,
        "linkedin": str(res.get("linkedin") or ""),
        "blog": {"title": str(blog.get("title") or ""), "text": str(blog.get("text") or "")},
        "carousel": [{"title": str(s.get("title", "")), "text": str(s.get("text", ""))}
                     for s in (res.get("carousel") or []) if isinstance(s, dict)][:10],
    }


# ---------------------------------------------------------------- re-use and export

def quick_hash(path):
    """Fingerprint of a video file (size + first and last 8 MB) to recognise the same video uploaded again."""
    import hashlib

    size = os.path.getsize(path)
    h = hashlib.sha1(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(8 << 20))
        if size > 16 << 20:
            f.seek(-(8 << 20), 2)
            h.update(f.read())
    return h.hexdigest()


def _tc(sec, fps):
    fr = int(round(sec * fps))
    f = fr % fps
    s_ = fr // fps
    return "%02d:%02d:%02d:%02d" % (s_ // 3600, s_ // 60 % 60, s_ % 60, f)


def edl(title, source_name, start, intervals, fps):
    """CMX3600 edit list: every kept part of the clip as an event, laid out back to back (Premiere, DaVinci, Avid)."""
    fps = int(round(fps))
    out = ["TITLE: %s" % title[:60], "FCM: NON-DROP FRAME", ""]
    rec = 0.0
    for n, (a, b) in enumerate(intervals, 1):
        out.append("%03d  AX       B     C        %s %s %s %s" % (n, _tc(start + a, fps), _tc(start + b, fps),
                                                                 _tc(rec, fps), _tc(rec + b - a, fps)))
        out.append("* FROM CLIP NAME: %s" % source_name)
        out.append("")
        rec += b - a
    return "\n".join(out)


def fcpxml(project_name, source_name, info, clips):
    """Final Cut Pro / DaVinci Resolve project: one vertical project per clip with all its cut parts.

    `clips` = [(name, start, intervals)]. The source has to be relinked to the original file after import."""
    from xml.sax.saxutils import quoteattr

    fps = int(round(info.get("fps", 30)))

    def t(sec):
        return "%d/%ds" % (int(round(sec * fps)), fps)

    parts = ['<?xml version="1.0" encoding="UTF-8"?>', "<!DOCTYPE fcpxml>", '<fcpxml version="1.9">', "  <resources>",
             '    <format id="r1" frameDuration="1/%ds" width="%d" height="%d"/>' % (fps, info["width"], info["height"]),
             '    <format id="r2" frameDuration="1/%ds" width="1080" height="1920"/>' % fps,
             '    <asset id="r3" name=%s start="0s" duration="%s" hasVideo="1" hasAudio="1" format="r1">'
             % (quoteattr(source_name), t(info.get("duration", 0))),
             '      <media-rep kind="original-media" src=%s/>' % quoteattr("file:///" + source_name),
             "    </asset>", "  </resources>", "  <library>", "    <event name=%s>" % quoteattr(project_name)]
    for name, start, intervals in clips:
        total = sum(b - a for a, b in intervals)
        parts.append('      <project name=%s><sequence format="r2" duration="%s"><spine>' % (quoteattr(name), t(total)))
        off = 0.0
        for a, b in intervals:
            parts.append('        <asset-clip ref="r3" offset="%s" start="%s" duration="%s" name=%s/>'
                         % (t(off), t(start + a), t(b - a), quoteattr(name)))
            off += b - a
        parts.append("      </spine></sequence></project>")
    parts += ["    </event>", "  </library>", "</fcpxml>"]
    return "\n".join(parts)


# ---------------------------------------------------------------- word-exact timing (forced alignment)

_ALIGNER = {}


def _aligner():
    """Meta MMS forced aligner from torchaudio (multilingual wav2vec2), loaded once and kept until released."""
    if not _ALIGNER:
        import torch
        import torchaudio

        torch.set_num_threads(THREADS)
        bundle = torchaudio.pipelines.MMS_FA
        _ALIGNER.update(model=bundle.get_model(with_star=False).eval(), dictionary=bundle.get_dict(star=None),
                        align=bundle.get_aligner(), tokenizer=bundle.get_tokenizer(), rate=bundle.sample_rate)
    return _ALIGNER


def release_aligner():
    _ALIGNER.clear()
    gc.collect()


def _roman(word):
    w = word.lower().replace("ä", "a").replace("ö", "o").replace("ü", "u").replace("ß", "ss")
    return re.sub(r"[^a-z']", "", w)


def align_words(video, words, start, end):
    """Re-time words (absolute seconds) between start and end against the audio itself.

    Whisper's word times are often off by one or two tenths of a second; the aligner finds where each word's sounds
    actually are. Words without letters (numbers) or implausible results keep their original times."""
    import numpy as np
    import torch

    al = _aligner()
    lo, hi = max(0.0, start - 0.4), end + 0.4
    p = subprocess.run([FFMPEG, "-v", "error", "-ss", "%.3f" % lo, "-t", "%.3f" % (hi - lo), "-i", video, "-vn", "-ac", "1",
                        "-ar", str(al["rate"]), "-f", "s16le", "-"], capture_output=True)
    if p.returncode != 0 or not p.stdout:
        return words
    wave_ = torch.from_numpy(np.frombuffer(p.stdout, np.int16).astype(np.float32) / 32768.0).unsqueeze(0)
    idx = [i for i, w in enumerate(words) if _roman(w["w"])]
    if not idx:
        return words
    with torch.inference_mode():
        emission, _ = al["model"](wave_)
        spans = al["align"](emission[0], al["tokenizer"]([_roman(words[i]["w"]) for i in idx]))
    sec = wave_.size(1) / emission.size(1) / al["rate"]
    out = [dict(w) for w in words]
    for i, sp in zip(idx, spans):
        if not sp:
            continue
        s_, e_ = lo + sp[0].start * sec, lo + sp[-1].end * sec
        if abs(s_ - words[i]["s"]) < 0.7 and abs(e_ - words[i]["e"]) < 0.7 and e_ > s_:
            out[i]["s"], out[i]["e"] = round(s_, 3), round(e_, 3)
    for a, b in zip(out, out[1:]):  # keep the order strict after re-timing
        if b["s"] < a["s"]:
            b["s"] = a["s"]
        b["e"] = max(b["e"], b["s"] + 0.03)
    return out


# ---------------------------------------------------------------- AI final check

def visual_check(api_key, video_path, out_len, hook, title, text):
    """Claude looks at three stills of the finished clip and reports visible problems."""
    frames = [frame_jpeg(video_path, out_len * f, 540) for f in (0.12, 0.5, 0.85)]
    prompt = (
        "Du prüfst ein fertig geschnittenes Kurzvideo im Hochformat 9:16 anhand von drei Standbildern (Anfang, Mitte, "
        "Ende). Oben steht in den ersten Sekunden der Hook-Text „%s“, unten bzw. mittig laufen Untertitel.\n"
        "Titel: %s\nGesprochener Text: %s\n\n"
        "Melde nur echte, sichtbare Probleme, zum Beispiel: Person oder Kopf angeschnitten oder außerhalb des Bildes, "
        "Untertitel oder Hook-Text abgeschnitten, verdeckt oder schlecht lesbar, Bild deutlich unscharf, zu dunkel oder "
        "überbelichtet, Hook-Text passt nicht zum Inhalt. Wenn alles in Ordnung ist, gib eine leere Liste zurück.\n"
        "Antworte NUR mit JSON: {\"issues\": [\"kurzer Hinweis auf Deutsch\"]}"
    ) % (hook[:200], title[:200], text[:1500])
    res = claude_json(api_key, prompt, max_tokens=800, feature="Endkontrolle", images=frames)
    return [str(i)[:200] for i in (res.get("issues") or []) if str(i).strip()][:5]
