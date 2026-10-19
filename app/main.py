"""ShortCut – erstellt aus langen Videos Shorts/Reels mit Highlights und Untertiteln."""
import hashlib
import hmac
import json
import os
import io
import re
import secrets
import shutil
import subprocess
import threading
import time
import traceback
import uuid
import zipfile
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

import integrations
import pipeline

DATA = os.environ.get("DATA_DIR", "/data")
JOBS = os.path.join(DATA, "jobs")
UPLOADS = os.path.join(DATA, "uploads")
MUSIC = os.path.join(DATA, "music")
CONFIG = os.path.join(DATA, "config.json")
RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "14"))
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(JOBS, exist_ok=True)
os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(MUSIC, exist_ok=True)

DEFAULTS = {
    "clips": 5, "min_len": 20, "max_len": 60, "mode": "auto", "quality": "ausgewogen",
    "highlight": "#FFE600", "uppercase": True, "font_size": 78, "position": "unten", "words_per_line": 3,
    "remove_pauses": True, "hook": True, "hook_seconds": 3, "review": False,
    "sub_style": "pop", "keyword_color": "#4ADE80", "zoom": True, "enhance_audio": True, "music": "", "music_volume": 15,
    "fmt_4x5": False, "fmt_1x1": False, "cover": True, "audio_analysis": True, "two_stage": True, "reuse_transcript": True, "precise_timing": True,
    "ai_check": True, "jumpcut_zoom": True, "blink_safe": True, "natural_pauses": True, "roomtone": True,
    "stabilize": False, "loop_end": True,
}
# settings stored in config.json; secrets are never sent back to the browser
CONFIG_TEXT = ("vocabulary", "nc_url", "nc_user", "nc_folder", "pz_url")
CONFIG_SECRET = ("anthropic_key", "nc_pass", "pz_key")
CONFIG_DEFAULTS = {"nc_url": "https://cloud.paetze.digital", "nc_folder": "ShortCut", "nc_export": False, "nc_watch": False,
                   "pz_url": "https://postiz.srv2011660.hstgr.cloud/api/public/v1", "vocabulary": ""}

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
lock = threading.Lock()


class WorkQueue:
    """FIFO of (kind, job id, argument) that the web interface can inspect, reorder and thin out."""

    def __init__(self):
        self.items, self.cond = [], threading.Condition()

    def put(self, item):
        with self.cond:
            self.items.append(item)
            self.cond.notify()

    def get(self):
        with self.cond:
            while not self.items:
                self.cond.wait()
            return self.items.pop(0)

    def snapshot(self):
        with self.cond:
            return list(self.items)

    def remove_job(self, jid):
        with self.cond:
            before = len(self.items)
            self.items = [i for i in self.items if i[1] != jid]
            return before - len(self.items)

    def move_job(self, jid, delta):
        """Move all entries of a job up (-1) or down (+1) past the neighbouring job."""
        with self.cond:
            order = list(dict.fromkeys(i[1] for i in self.items))
            if jid not in order:
                return False
            k = order.index(jid)
            j = max(0, min(len(order) - 1, k + delta))
            order.insert(j, order.pop(k))
            self.items.sort(key=lambda i: order.index(i[1]))
            return True


work = WorkQueue()
CURRENT = {"jid": None, "kind": None, "since": None}


# ---------------------------------------------------------------- config & auth

def load_config():
    try:
        with open(CONFIG, encoding="utf-8") as f:
            return {**CONFIG_DEFAULTS, **json.load(f)}
    except FileNotFoundError:
        return dict(CONFIG_DEFAULTS)


def save_config(cfg):
    tmp = CONFIG + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    os.replace(tmp, CONFIG)


def hash_pw(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 200_000).hex()


def make_token(cfg):
    exp = str(int(time.time()) + 30 * 86400)
    sig = hmac.new(cfg["secret"].encode(), exp.encode(), "sha256").hexdigest()
    return exp + "." + sig


def valid_token(token):
    cfg = load_config()
    if not token or "secret" not in cfg or "." not in token:
        return False
    exp, sig = token.split(".", 1)
    good = hmac.new(cfg["secret"].encode(), exp.encode(), "sha256").hexdigest()
    return hmac.compare_digest(sig, good) and exp.isdigit() and int(exp) > time.time()


def require_auth(request: Request):
    if not valid_token(request.cookies.get("sc_session")):
        raise HTTPException(401, "Nicht angemeldet")


def session_response(cfg, request):
    resp = JSONResponse({"ok": True})
    resp.set_cookie("sc_session", make_token(cfg), max_age=30 * 86400, httponly=True, samesite="lax",
                    secure=request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https")
    return resp


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
def health():
    """Which optional quality components are available in this build (no data, no auth needed)."""
    try:
        import torchaudio
        torch_ok = hasattr(torchaudio.pipelines, "MMS_FA")
    except Exception:
        torch_ok = False
    return {"ok": True, "face_model": os.path.exists(pipeline.YUNET), "denoise_model": os.path.exists(pipeline.RNNOISE),
            "aligner": torch_ok}


@app.get("/api/state")
def state(request: Request):
    cfg = load_config()
    return {"setup_needed": "pw_hash" not in cfg, "logged_in": valid_token(request.cookies.get("sc_session"))}


@app.post("/api/setup")
async def setup(request: Request):
    body = await request.json()
    with lock:
        cfg = load_config()
        if "pw_hash" in cfg:
            raise HTTPException(400, "Bereits eingerichtet")
        pw = body.get("password", "")
        if len(pw) < 8:
            raise HTTPException(400, "Das Passwort muss mindestens 8 Zeichen haben.")
        cfg["salt"] = secrets.token_hex(16)
        cfg["pw_hash"] = hash_pw(pw, cfg["salt"])
        cfg["secret"] = secrets.token_hex(32)
        save_config(cfg)
    return session_response(cfg, request)


@app.post("/api/login")
async def login(request: Request):
    body = await request.json()
    cfg = load_config()
    if "pw_hash" not in cfg or not hmac.compare_digest(hash_pw(body.get("password", ""), cfg["salt"]), cfg["pw_hash"]):
        time.sleep(1.5)
        raise HTTPException(401, "Falsches Passwort")
    return session_response(cfg, request)


@app.post("/api/logout")
def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("sc_session")
    return resp


@app.get("/api/settings")
def get_settings(request: Request):
    require_auth(request)
    cfg = load_config()
    key = cfg.get("anthropic_key", "")
    out = {"has_key": bool(key), "key_hint": ("…" + key[-4:]) if key else "",
           "defaults": {**DEFAULTS, **cfg.get("defaults", {})}, "nc_export": bool(cfg.get("nc_export")),
           "nc_watch": bool(cfg.get("nc_watch")), "watch": {k: v for k, v in WATCH.items() if k != "seen"},
           "languages": pipeline.LANGUAGES}
    for k in CONFIG_TEXT:
        out[k] = cfg.get(k, "")
    for k in CONFIG_SECRET:
        out["has_" + k] = bool(cfg.get(k))
    return out


@app.post("/api/settings")
async def set_settings(request: Request):
    require_auth(request)
    body = await request.json()
    with lock:
        cfg = load_config()
        for k in CONFIG_SECRET:
            if k in body:
                v = (body[k] or "").strip()
                if v:
                    cfg[k] = v
                else:
                    cfg.pop(k, None)
        for k in CONFIG_TEXT:
            if k in body:
                cfg[k] = str(body[k] or "").strip()[:2000]
        for k in ("nc_export", "nc_watch"):
            if k in body:
                cfg[k] = bool(body[k])
        if "defaults" in body:
            cfg["defaults"] = clean_settings(body["defaults"])
        if body.get("new_password"):
            if len(body["new_password"]) < 8:
                raise HTTPException(400, "Das Passwort muss mindestens 8 Zeichen haben.")
            cfg["salt"] = secrets.token_hex(16)
            cfg["pw_hash"] = hash_pw(body["new_password"], cfg["salt"])
        save_config(cfg)
    return {"ok": True}


def clean_settings(s):
    out = dict(DEFAULTS)
    for k, v in (s or {}).items():
        if k not in DEFAULTS:
            continue
        d = DEFAULTS[k]
        try:
            if isinstance(d, bool):
                out[k] = v in (True, "true", "1", 1, "on")
            elif isinstance(d, int):
                out[k] = int(float(v))
            else:
                out[k] = str(v)
        except (TypeError, ValueError):
            pass
    out["clips"] = max(1, min(15, out["clips"]))
    out["min_len"] = max(5, min(170, out["min_len"]))
    out["max_len"] = max(out["min_len"] + 5, min(180, out["max_len"]))
    out["font_size"] = max(40, min(140, out["font_size"]))
    out["words_per_line"] = max(1, min(6, out["words_per_line"]))
    out["hook_seconds"] = max(1, min(10, out["hook_seconds"]))
    if out["mode"] not in ("auto", "split", "crop", "blur"):
        out["mode"] = "auto"
    if out["quality"] not in pipeline.WHISPER_PRESETS:
        out["quality"] = "ausgewogen"
    for k in ("highlight", "keyword_color"):
        if not re.fullmatch(r"#[0-9A-Fa-f]{6}", out[k]):
            out[k] = DEFAULTS[k]
    if out["sub_style"] not in pipeline.SUB_STYLES:
        out["sub_style"] = "pop"
    out["music_volume"] = max(2, min(60, out["music_volume"]))
    if out["music"] and not os.path.isfile(os.path.join(MUSIC, os.path.basename(out["music"]))):
        out["music"] = ""
    out["music"] = os.path.basename(out["music"])
    return out


# ---------------------------------------------------------------- jobs

def job_dir(jid):
    if not re.fullmatch(r"[0-9a-f]{12}", jid):
        raise HTTPException(404, "Unbekannt")
    return os.path.join(JOBS, jid)


def load_job(jid):
    with open(os.path.join(job_dir(jid), "job.json"), encoding="utf-8") as f:
        return json.load(f)


def save_job(job):
    path = os.path.join(JOBS, job["id"], "job.json")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False)
    os.replace(path + ".tmp", path)


def load_transcript(jid):
    path = os.path.join(JOBS, jid, "transcript.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_transcript(jid, segments):
    path = os.path.join(JOBS, jid, "transcript.json")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False)
    os.replace(path + ".tmp", path)


def public_job(job, with_text=False):
    out = {k: v for k, v in job.items() if k not in ("log",)} | {"log": job.get("log", [])[-5:]}
    if job.get("status") in ("wartet", "läuft"):
        out["eta"] = eta_for(job)
    if with_text and job.get("clips"):
        segments = load_transcript(job["id"])
        if segments:
            words = pipeline.all_words(segments)
            out["clips"] = []
            for c in job["clips"]:
                cw = [w for w in words if w["e"] > c["start"] and w["s"] < c["end"]]
                out["clips"].append(dict(c, text=" ".join(w["w"] for w in cw), words=[[w["w"], w["s"]] for w in cw]))
    return out


def new_job(name, settings, context, source=None, remote=None, size=0, enqueue=True):
    jid = uuid.uuid4().hex[:12]
    os.makedirs(os.path.join(JOBS, jid))
    job = {
        "id": jid, "name": name, "created": time.time(), "status": "wartet", "progress": 0,
        "message": "In der Warteschlange", "settings": settings, "context": context, "source": source,
        "remote": remote, "size": size, "clips": [], "log": [], "selector": None,
    }
    save_job(job)
    if enqueue:
        work.put(("process", jid, None))
    return job


@app.get("/api/jobs")
def list_jobs(request: Request):
    require_auth(request)
    jobs = []
    for jid in os.listdir(JOBS):
        try:
            jobs.append(public_job(load_job(jid)))
        except Exception:
            continue
    jobs.sort(key=lambda j: -j["created"])
    return jobs


@app.get("/api/jobs/{jid}")
def get_job(jid: str, request: Request):
    require_auth(request)
    try:
        return public_job(load_job(jid), with_text=True)
    except FileNotFoundError:
        raise HTTPException(404, "Unbekannt")


def upload_path(uid):
    if not re.fullmatch(r"[0-9a-f]{24}", uid):
        raise HTTPException(404, "Unbekannt")
    return os.path.join(UPLOADS, uid + ".part")


@app.post("/api/uploads")
def start_upload(request: Request):
    """Uploads arrive in chunks so no single request hits the reverse proxy timeout."""
    require_auth(request)
    uid = uuid.uuid4().hex[:24]
    open(upload_path(uid), "wb").close()
    return {"id": uid}


@app.put("/api/uploads/{uid}")
async def upload_chunk(uid: str, request: Request, offset: int = 0):
    require_auth(request)
    path = upload_path(uid)
    if not os.path.exists(path):
        raise HTTPException(404, "Upload unbekannt")
    size = os.path.getsize(path)
    if offset != size:
        if offset > size:
            raise HTTPException(409, "Falscher Offset")
        with open(path, "r+b") as f:  # a retried chunk: drop the partial tail and rewrite
            f.truncate(offset)
    with open(path, "ab") as f:
        async for chunk in request.stream():
            f.write(chunk)
    return {"size": os.path.getsize(path)}


def source_name(filename):
    ext = os.path.splitext(filename)[1].lower()
    return "source" + (ext if re.fullmatch(r"\.[a-z0-9]{2,5}", ext) else ".mp4")


@app.post("/api/jobs")
async def create_job(request: Request):
    require_auth(request)
    body = await request.json()
    part = upload_path(str(body.get("upload_id", "")))
    if not os.path.exists(part) or os.path.getsize(part) < 1000:
        raise HTTPException(400, "Upload fehlt oder ist leer")
    filename = os.path.basename(str(body.get("filename") or "video.mp4"))[:120] or "video.mp4"
    src = source_name(filename)
    size = os.path.getsize(part)
    job = new_job(filename, clean_settings(body.get("settings")), str(body.get("context") or "")[:1000], src, None, size,
                  enqueue=False)
    os.replace(part, os.path.join(JOBS, job["id"], src))
    try:  # the length is known right away, so the queue can estimate the waiting time
        job = update(job["id"], info=pipeline.probe(os.path.join(JOBS, job["id"], src)))
    except Exception:
        pass
    work.put(("process", job["id"], None))
    return public_job(job)


@app.get("/api/nextcloud/files")
def nextcloud_files(request: Request):
    require_auth(request)
    try:
        return integrations.Nextcloud(load_config()).list_videos()
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, "Nextcloud nicht erreichbar: " + str(e)[:200])


@app.post("/api/import/nextcloud")
async def import_nextcloud(request: Request):
    require_auth(request)
    body = await request.json()
    name = os.path.basename(str(body.get("name") or ""))
    if not name.lower().endswith(integrations.VIDEO_EXT):
        raise HTTPException(400, "Keine Videodatei")
    job = new_job(name, clean_settings(body.get("settings")), str(body.get("context") or "")[:1000],
                  None, {"type": "nextcloud", "name": name})
    return public_job(job)


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str, request: Request):
    require_auth(request)
    d = job_dir(jid)
    job = load_job(jid)
    work.remove_job(jid)
    if job["status"] not in ("fertig", "fehler", "prüfen", "abgebrochen"):
        job["cancel"] = True
        save_job(job)
    shutil.rmtree(d, ignore_errors=True)
    return {"ok": True}


@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: str, request: Request):
    """Stop a waiting or running job; a running one stops at the next step (after the current clip or segment)."""
    require_auth(request)
    with lock:
        job = load_job(jid)
        if job["status"] not in ("wartet", "läuft"):
            raise HTTPException(400, "Projekt läuft gerade nicht")
        removed = work.remove_job(jid)
        if CURRENT["jid"] == jid:
            job["cancel"] = True
            job["message"] = "Wird abgebrochen …"
        else:
            job.update(status="abgebrochen", message="Abgebrochen")
            job["clips"] = [dict(c, status="fertig") if c.get("file") else c for c in job.get("clips", [])]
        save_job(job)
    return public_job(job)


@app.post("/api/jobs/{jid}/restart")
def restart_job(jid: str, request: Request):
    require_auth(request)
    with lock:
        job = load_job(jid)
        if job["status"] not in ("abgebrochen", "fehler"):
            raise HTTPException(400, "Nur abgebrochene oder fehlgeschlagene Projekte lassen sich neu starten")
        require_source(job)
        job.pop("cancel", None)
        job.update(status="wartet", message="In der Warteschlange", progress=0)
        save_job(job)
    work.put(("render_all", jid, None) if job.get("clips") and job.get("info") else ("process", jid, None))
    return public_job(job)


def require_source(job):
    if job.get("source_deleted"):
        raise HTTPException(400, "Das Originalvideo wurde gelöscht. Lade das Video neu hoch, um Clips neu zu rendern.")


@app.post("/api/jobs/{jid}/clips/{idx}")
async def edit_clip(jid: str, idx: int, request: Request):
    """Change timing, titles or transcript of a clip; renders it again unless the job is still in review."""
    require_auth(request)
    body = await request.json()
    with lock:
        job = load_job(jid)
        if job["status"] not in ("fertig", "fehler", "prüfen") or idx >= len(job["clips"]):
            raise HTTPException(400, "Clip kann gerade nicht bearbeitet werden")
        require_source(job)
        clip = job["clips"][idx]
        if job["status"] != "prüfen":
            keep_previous(clip)
        try:
            s, e = float(body.get("start", clip["start"])), float(body.get("end", clip["end"]))
        except (TypeError, ValueError):
            raise HTTPException(400, "Ungültige Zeit")
        if e - s < 3 or e - s > 180 or s < 0:
            raise HTTPException(400, "Clip muss zwischen 3 und 180 Sekunden lang sein")
        if "text" in body:
            segments = load_transcript(jid)
            old = [w for w in pipeline.all_words(segments) if w["e"] > clip["start"] and w["s"] < clip["end"]]
            new_text = re.sub(r"\s+", " ", str(body["text"])).strip()
            if new_text != " ".join(w["w"] for w in old):
                save_transcript(jid, pipeline.replace_words(segments, clip["start"], clip["end"],
                                                            pipeline.realign(old, new_text)))
        clip.update({"start": round(s, 2), "end": round(e, 2)})
        for k in ("title", "hook_text", "caption", "cover_text"):
            if k in body:
                clip[k] = str(body[k])[:2000]
        if "keywords" in body:
            kw = body["keywords"]
            kw = kw.split(",") if isinstance(kw, str) else kw
            clip["keywords"] = [str(k).strip() for k in kw if str(k).strip()][:12]
        if "crop_keys" in body:
            try:
                keys = [[round(max(0.0, float(t)), 2), round(min(1.0, max(0.0, float(c))), 4)] for t, c in body["crop_keys"] or []]
            except (TypeError, ValueError):
                raise HTTPException(400, "Ungültiger Bildausschnitt")
            clip["crop_keys"] = sorted(keys)[:50]
        if "beeped" in body:
            try:
                clip["beeped"] = sorted({round(float(x), 2) for x in body["beeped"]})[:500]
            except (TypeError, ValueError):
                raise HTTPException(400, "Ungültige Wortauswahl")
        if "removed" in body:
            try:
                clip["removed"] = sorted({round(float(x), 2) for x in body["removed"]})[:2000]
            except (TypeError, ValueError):
                raise HTTPException(400, "Ungültige Wortauswahl")
        if "style" in body:
            job["settings"] = clean_settings({**job["settings"], **body["style"]})
        if job["status"] != "prüfen":
            clip["status"] = "wartet"
            job["status"] = "wartet"
            job["message"] = "Clip %d wird neu gerendert" % (idx + 1)
            work.put(("render", jid, idx))
        save_job(job)
    return public_job(job, with_text=True)


@app.get("/api/jobs/{jid}/transcript")
def transcript(jid: str, request: Request):
    """All words of the video for marking clips by hand: [[word, start, end], ...]."""
    require_auth(request)
    segments = load_transcript(jid) or []
    return {"words": [[w["w"], round(w["s"], 2), round(w["e"], 2)] for w in pipeline.all_words(segments)]}


def series_parts(words, start, end, part_len):
    """Split [start, end] at sentence ends into parts of roughly part_len seconds (never shorter than 8 s)."""
    sel = [w for w in words if w["s"] >= start - 0.01 and w["e"] <= end + 0.01]
    parts, cur = [], []
    for w in sel:
        cur.append(w)
        if cur[-1]["e"] - cur[0]["s"] >= part_len and w["w"][-1:] in ".!?":
            parts.append(cur)
            cur = []
        elif cur[-1]["e"] - cur[0]["s"] >= part_len * 1.5:  # no sentence end in sight: cut at the next word
            parts.append(cur)
            cur = []
    if cur:
        if parts and cur[-1]["e"] - cur[0]["s"] < 8:
            parts[-1] += cur
        else:
            parts.append(cur)
    return [(round(max(0, p_[0]["s"] - 0.15), 2), round(p_[-1]["e"] + 0.35, 2)) for p_ in parts]


def new_clip(words, start, end, extra=None):
    text = " ".join(w["w"] for w in words if w["s"] >= start - 0.01 and w["e"] <= end + 0.01)
    first = re.split(r"(?<=[.!?])\s", text)[0] if text else ""
    title = (first[:70] + "…") if len(first) > 70 else first
    c = {"start": start, "end": end, "title": title, "hook": "Von Hand markiert", "caption": "", "score": 0,
         "hooks": [title], "keywords": pipeline.guess_keywords(text), "hook_text": title, "cover_text": title,
         "manual": True, "uid": uuid.uuid4().hex[:6]}
    c.update(extra or {})
    return c


def add_clips(jid, clips):
    """Append clips to a finished (or reviewed) project and render them unless the project is in review."""
    with lock:
        job = load_job(jid)
        if job["status"] not in ("fertig", "fehler", "prüfen", "abgebrochen"):
            raise HTTPException(400, "Bitte warten, bis das Projekt fertig ist.")
        require_source(job)
        if len(job["clips"]) + len(clips) > 40:
            raise HTTPException(400, "Zu viele Clips in diesem Projekt (max. 40)")
        first = len(job["clips"])
        for c in clips:
            c["status"] = "entwurf" if job["status"] == "prüfen" else "wartet"
            job["clips"].append(c)
        if job["status"] != "prüfen":
            job["status"], job["message"] = "wartet", "%d neue Clips werden gerendert" % len(clips)
            for i in range(first, first + len(clips)):
                work.put(("render", jid, i))
        save_job(job)
    return public_job(job, with_text=True)


@app.post("/api/jobs/{jid}/clips")
async def create_clip(jid: str, request: Request):
    """Clip from a stretch marked in the transcript, or a series of parts when `series` is set."""
    require_auth(request)
    body = await request.json()
    try:
        start, end = float(body["start"]), float(body["end"])
        part_len = max(20.0, min(180.0, float(body.get("part_len", 60))))
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "Bitte Anfang und Ende markieren")
    if end - start < 3:
        raise HTTPException(400, "Die Markierung ist zu kurz (mind. 3 Sekunden)")
    if body.get("series"):
        return add_series(jid, start, end, part_len)
    words = pipeline.all_words(load_transcript(jid) or [])
    if end - start > 180:
        raise HTTPException(400, "Ein Clip darf höchstens 3 Minuten lang sein – für Längeres „Als Serie“ wählen.")
    return add_clips(jid, [new_clip(words, start, end)])


@app.post("/api/jobs/{jid}/clips/{idx}/split")
async def split_clip(jid: str, idx: int, request: Request):
    """Turn one clip into a series of parts (the original clip stays)."""
    require_auth(request)
    body = await request.json()
    job = load_job(jid)
    if idx >= len(job["clips"]):
        raise HTTPException(404, "Unbekannt")
    c = job["clips"][idx]
    return add_series(jid, c["start"], c["end"], body.get("part_len", 30))


def add_series(jid, start, end, part_len):
    try:
        part_len = max(15.0, min(180.0, float(part_len)))
    except (TypeError, ValueError):
        raise HTTPException(400, "Ungültige Teillänge")
    words = pipeline.all_words(load_transcript(jid) or [])
    parts = series_parts(words, start, end, part_len)
    if len(parts) < 2:
        raise HTTPException(400, "Der Clip ist zu kurz für mehrere Teile – wähle eine kürzere Teillänge.")
    sid = uuid.uuid4().hex[:6]
    clips = [new_clip(words, a, b, {"series": {"id": sid, "part": i + 1, "total": len(parts)}}) for i, (a, b) in enumerate(parts)]
    for c in clips:
        c["hook_text"] = "Teil %d/%d: %s" % (c["series"]["part"], c["series"]["total"], c["title"])
    return add_clips(jid, clips)


@app.delete("/api/jobs/{jid}/clips/{idx}")
def delete_clip(jid: str, idx: int, request: Request):
    require_auth(request)
    with lock:
        job = load_job(jid)
        if job["status"] in ("wartet", "läuft"):
            raise HTTPException(400, "Bitte warten, bis das Projekt fertig ist.")
        if idx >= len(job["clips"]):
            raise HTTPException(404, "Unbekannt")
        clip = job["clips"].pop(idx)
        trash_add(job, "clip", clip, idx)  # files stay until the trash is emptied
        for c in job["clips"]:  # variants point at positions
            if c.get("variant_of") is not None:
                c["variant_of"] = None if c["variant_of"] == idx else c["variant_of"] - (c["variant_of"] > idx)
        save_job(job)
    return public_job(job, with_text=True)


# ---------------------------------------------------------------- trash: deleted clips and replaced versions

TRASH_DAYS = 14


def keep_previous(clip):
    """Remember the rendered version before a change, so it lands in the trash instead of being overwritten."""
    if clip.get("file") and not clip.get("prev"):
        clip["prev"] = json.loads(json.dumps({k: v for k, v in clip.items() if k != "prev"}))


def trash_add(job, kind, clip, idx):
    clip = {k: v for k, v in clip.items() if k != "prev"}
    if not clip.get("file") and kind == "version":
        return
    job.setdefault("trash", []).insert(0, {"id": uuid.uuid4().hex[:8], "kind": kind, "deleted": time.time(),
                                           "index": idx, "clip": clip})


def trash_purge(job, which):
    """Remove trash entries for good, with every file that no clip or other entry still uses."""
    gone = [t for t in job.get("trash", []) if which(t)]
    if not gone:
        return False
    job["trash"] = [t for t in job["trash"] if t not in gone]
    used = {f for c in job["clips"] for f in clip_files(c)} | \
           {f for c in job["clips"] if c.get("prev") for f in clip_files(c["prev"])} | \
           {f for t in job["trash"] for f in clip_files(t["clip"])}
    for t in gone:
        for f in clip_files(t["clip"]):
            if f not in used:
                try:
                    os.remove(os.path.join(JOBS, job["id"], f))
                except OSError:
                    pass
    return True


def trash_entry(job, tid):
    for t in job.get("trash", []):
        if t["id"] == tid:
            return t
    raise HTTPException(404, "Nicht mehr im Papierkorb")


@app.post("/api/jobs/{jid}/trash/{tid}/restore")
def trash_restore(jid: str, tid: str, request: Request):
    """A deleted clip comes back at the end of the list; an older version swaps places with the current one."""
    require_auth(request)
    with lock:
        job = load_job(jid)
        if job["status"] in ("wartet", "läuft"):
            raise HTTPException(400, "Bitte warten, bis das Projekt fertig ist.")
        t = trash_entry(job, tid)
        job["trash"].remove(t)
        clip = dict(t["clip"], status="fertig")
        current = next((i for i, c in enumerate(job["clips"]) if t["kind"] == "version" and c.get("uid") == clip.get("uid")), None)
        if current is not None:
            trash_add(job, "version", job["clips"][current], current)
            clip["variant_of"] = job["clips"][current].get("variant_of")
            job["clips"][current] = clip
        else:
            if len(job["clips"]) >= 30:
                raise HTTPException(400, "Zu viele Clips in diesem Projekt")
            clip["variant_of"] = None
            job["clips"].append(clip)
        save_job(job)
    return public_job(job, with_text=True)


@app.delete("/api/jobs/{jid}/trash/{tid}")
def trash_delete(jid: str, tid: str, request: Request):
    require_auth(request)
    with lock:
        job = load_job(jid)
        if tid == "all":
            trash_purge(job, lambda t: True)
        else:
            trash_entry(job, tid)
            trash_purge(job, lambda t: t["id"] == tid)
        save_job(job)
    return public_job(job, with_text=True)


# ---------------------------------------------------------------- timeline: waveform and audio of the source

def source_window(jid, start, end):
    job = load_job(jid)
    require_source(job)
    dur = (job.get("info") or {}).get("duration") or 0
    t0 = max(0.0, float(start))
    t1 = min(float(end), dur) if dur else float(end)
    if t1 - t0 < 0.5 or t1 - t0 > 420:
        raise HTTPException(400, "Ungültiger Ausschnitt")
    return os.path.join(job_dir(jid), job["source"]), t0, t1


@app.get("/api/jobs/{jid}/waveform")
def waveform(jid: str, request: Request, start: float, end: float):
    """Peak levels of the source audio, 50 per second, scaled to 0..100."""
    require_auth(request)
    import numpy as np

    src, t0, t1 = source_window(jid, start, end)
    p = subprocess.run([pipeline.FFMPEG, "-v", "error", "-ss", "%.3f" % t0, "-t", "%.3f" % (t1 - t0), "-i", src, "-vn",
                        "-ac", "1", "-ar", "8000", "-f", "s16le", "-"], capture_output=True)
    a = np.abs(np.frombuffer(p.stdout, np.int16).astype(np.float32))
    n = len(a) // 160
    if p.returncode != 0 or not n:
        raise HTTPException(400, "Ton konnte nicht gelesen werden")
    peaks = a[:n * 160].reshape(n, 160).max(axis=1)
    peaks = np.minimum(100, peaks / max(1.0, float(np.percentile(peaks, 99))) * 100)
    return {"t0": round(t0, 3), "rate": 50, "peaks": [int(x) for x in peaks]}


@app.get("/api/jobs/{jid}/listen")
def listen(jid: str, request: Request, start: float, end: float):
    """The source audio of a stretch as MP3, for playing along the timeline."""
    require_auth(request)
    src, t0, t1 = source_window(jid, start, end)
    p = subprocess.run([pipeline.FFMPEG, "-v", "error", "-ss", "%.3f" % t0, "-t", "%.3f" % (t1 - t0), "-i", src, "-vn",
                        "-ac", "1", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "96k", "-f", "mp3", "-"], capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise HTTPException(400, "Ton konnte nicht gelesen werden")
    return Response(p.stdout, media_type="audio/mpeg", headers={"Cache-Control": "private, max-age=3600"})


@app.post("/api/jobs/{jid}/clips/{idx}/duplicate")
async def duplicate_clip(jid: str, idx: int, request: Request):
    """Second version of a clip with another hook text, e.g. to compare two hooks (A/B)."""
    require_auth(request)
    body = await request.json()
    with lock:
        job = load_job(jid)
        if job["status"] not in ("fertig", "fehler", "prüfen") or idx >= len(job["clips"]):
            raise HTTPException(400, "Clip kann gerade nicht kopiert werden")
        if len(job["clips"]) >= 30:
            raise HTTPException(400, "Zu viele Clips in diesem Projekt")
        src = job["clips"][idx]
        copy = {k: v for k, v in src.items() if k not in ("file", "thumb", "srt", "cover", "formats", "version", "postiz",
                                                          "translations", "uid", "checks", "spelling", "prev")}
        copy["hook_text"] = str(body.get("hook_text") or src.get("hook_text", ""))[:300]
        copy["variant_of"] = idx
        copy["status"] = "entwurf" if job["status"] == "prüfen" else "wartet"
        job["clips"].append(copy)
        new = len(job["clips"]) - 1
        if job["status"] != "prüfen":
            job["status"], job["message"] = "wartet", "Variante wird gerendert"
            work.put(("render", jid, new))
        save_job(job)
    return public_job(job, with_text=True)


@app.get("/api/jobs/{jid}/frame")
def source_frame(jid: str, request: Request, t: float = 0.0):
    """Still image of the source video, for setting the crop by hand."""
    require_auth(request)
    job = load_job(jid)
    try:
        img = pipeline.frame_jpeg(os.path.join(job_dir(jid), job["source"]), t)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return Response(img, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


def queue_extra(jid, kind, arg, message):
    with lock:
        job = load_job(jid)
        if kind != "texts":
            require_source(job)
        if job["status"] not in ("fertig", "fehler"):
            raise HTTPException(400, "Bitte warten, bis das Projekt fertig ist.")
        job["status"], job["message"] = "wartet", message
        save_job(job)
    work.put((kind, jid, arg))
    return public_job(job, with_text=True)


@app.post("/api/jobs/{jid}/clips/{idx}/translate")
async def translate_clip(jid: str, idx: int, request: Request):
    require_auth(request)
    body = await request.json()
    lang = str(body.get("lang") or "en")
    if lang not in pipeline.LANGUAGES:
        raise HTTPException(400, "Unbekannte Sprache")
    if not load_config().get("anthropic_key"):
        raise HTTPException(400, "Für Übersetzungen wird ein Claude-API-Schlüssel benötigt (Einstellungen).")
    job = load_job(jid)
    if idx >= len(job["clips"]) or not job["clips"][idx].get("file"):
        raise HTTPException(400, "Clip ist noch nicht fertig")
    return queue_extra(jid, "translate", (idx, lang), "Übersetze Clip %d (%s)" % (idx + 1, pipeline.LANGUAGES[lang]))


@app.post("/api/jobs/{jid}/bestof")
async def bestof(jid: str, request: Request):
    require_auth(request)
    body = await request.json()
    try:
        target = max(15, min(90, int(body.get("target", 45))))
    except (TypeError, ValueError):
        raise HTTPException(400, "Ungültige Länge")
    with lock:
        job = load_job(jid)
        if not job.get("clips"):
            raise HTTPException(400, "Noch keine Clips vorhanden")
        job["bestof_hook"] = str(body.get("hook_text") or "")[:200]
        save_job(job)
    return queue_extra(jid, "bestof", target, "Erstelle Best-of-Trailer")


@app.post("/api/jobs/{jid}/clips/{idx}/fix")
async def fix_spelling(jid: str, idx: int, request: Request):
    """Take over a spelling suggestion: correct the transcript and render the clip again."""
    require_auth(request)
    body = await request.json()
    wrong, good = str(body.get("wrong") or ""), str(body.get("suggestion") or "")
    with lock:
        job = load_job(jid)
        if job["status"] not in ("fertig", "fehler", "prüfen") or idx >= len(job["clips"]) or not wrong:
            raise HTTPException(400, "Clip kann gerade nicht bearbeitet werden")
        clip = job["clips"][idx]
        segments = load_transcript(jid)
        old = [w for w in pipeline.all_words(segments) if w["e"] > clip["start"] and w["s"] < clip["end"]]
        text = " ".join(w["w"] for w in old)
        if wrong not in text:
            raise HTTPException(400, "Text wurde inzwischen geändert")
        save_transcript(jid, pipeline.replace_words(segments, clip["start"], clip["end"],
                                                    pipeline.realign(old, text.replace(wrong, good, 1))))
        clip["spelling"] = [x for x in clip.get("spelling", []) if x.get("wrong") != wrong]
        if job["status"] != "prüfen":
            keep_previous(clip)
            clip["status"], job["status"], job["message"] = "wartet", "wartet", "Clip %d wird korrigiert" % (idx + 1)
            work.put(("render", jid, idx))
        save_job(job)
    return public_job(job, with_text=True)


@app.post("/api/jobs/{jid}/texts")
def create_texts(jid: str, request: Request):
    require_auth(request)
    if not load_config().get("anthropic_key"):
        raise HTTPException(400, "Für die Texte wird ein Claude-API-Schlüssel benötigt (Einstellungen).")
    return queue_extra(jid, "texts", None, "Erstelle Texte zum Video")


@app.post("/api/jobs/{jid}/video_url")
async def set_video_url(jid: str, request: Request):
    """Link to the published long video; clip texts then point to the matching moment in it."""
    require_auth(request)
    url = str((await request.json()).get("url") or "").strip()
    if url and not re.fullmatch(r"https?://\S{4,300}", url):
        raise HTTPException(400, "Bitte einen vollständigen Link angeben (https://…)")
    try:
        load_job(jid)
    except FileNotFoundError:
        raise HTTPException(404, "Unbekannt")
    update(jid, video_url=url)
    return public_job(load_job(jid), with_text=True)


def eta_for(job):
    """Rough remaining seconds, from measured speeds of earlier runs on this server."""
    sp = load_config().get("speed", {})
    s = job.get("settings", {})
    dur = (job.get("info") or {}).get("duration") or 0
    clip_len = (s.get("min_len", 20) + s.get("max_len", 60)) / 2
    formats = 1 + sum(1 for f in ("fmt_4x5", "fmt_1x1") if s.get(f))
    render = sp.get("render", 4.0) * clip_len * s.get("clips", 5) * formats
    transcribe = 0 if os.path.exists(os.path.join(JOBS, job["id"], "transcript.json")) else \
        sp.get("transcribe_" + s.get("quality", "ausgewogen"), {"schnell": 0.35, "ausgewogen": 0.9, "genau": 2.2}.get(s.get("quality"), 0.9)) * dur
    total = transcribe + render + 30
    if job["status"] == "läuft" and job.get("started") and job.get("progress", 0) > 3:
        elapsed = time.time() - job["started"]
        by_progress = elapsed / job["progress"] * (100 - job["progress"])
        return int(max(10, 0.5 * by_progress + 0.5 * max(0, total - elapsed)))
    return int(total) if dur else None


def learn_speed(key, seconds, media_seconds):
    if media_seconds <= 0:
        return
    with lock:
        cfg = load_config()
        sp = cfg.setdefault("speed", {})
        rate = seconds / media_seconds
        sp[key] = round(0.7 * sp[key] + 0.3 * rate, 3) if key in sp else round(rate, 3)
        save_config(cfg)


@app.get("/api/queue")
def queue_state(request: Request):
    require_auth(request)
    out, seen = [], set()
    for kind, jid, _ in ([("cur", CURRENT["jid"], None)] if CURRENT["jid"] else []) + work.snapshot():
        if jid in seen:
            continue
        seen.add(jid)
        try:
            job = load_job(jid)
        except Exception:
            continue
        out.append({"id": jid, "name": job["name"], "status": job["status"], "progress": job.get("progress", 0),
                    "message": job.get("message", ""), "running": jid == CURRENT["jid"], "eta": eta_for(job)})
    # the waiting time of each entry includes everything in front of it
    acc = 0
    for q in out:
        acc += q["eta"] or 0
        q["eta_total"] = acc
    return out


@app.post("/api/queue/move")
async def queue_move(request: Request):
    require_auth(request)
    body = await request.json()
    if not work.move_job(str(body.get("id")), -1 if body.get("dir") == "up" else 1):
        raise HTTPException(400, "Projekt wartet nicht in der Warteschlange")
    return {"ok": True}


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


@app.get("/api/storage")
def storage(request: Request):
    require_auth(request)
    du = shutil.disk_usage(DATA)
    projects = []
    for jid in os.listdir(JOBS):
        try:
            job = load_job(jid)
        except Exception:
            continue
        src = os.path.join(JOBS, jid, job.get("source") or "")
        src_size = os.path.getsize(src) if job.get("source") and os.path.isfile(src) else 0
        projects.append({"id": jid, "name": job["name"], "status": job["status"], "created": job["created"],
                         "source": src_size, "total": dir_size(os.path.join(JOBS, jid)),
                         "source_deleted": bool(job.get("source_deleted"))})
    projects.sort(key=lambda x: -x["total"])
    models = os.environ.get("HF_HOME", "/models")
    return {"disk_total": du.total, "disk_free": du.free, "disk_used": du.used, "projects": projects,
            "projects_total": sum(x["total"] for x in projects), "music": dir_size(MUSIC),
            "models": dir_size(models) if os.path.isdir(models) else 0, "retention_days": RETENTION_DAYS}


def delete_source(jid):
    with lock:
        job = load_job(jid)
        if job["status"] in ("wartet", "läuft"):
            raise HTTPException(400, "Projekt läuft noch")
        if job.get("source"):
            try:
                os.remove(os.path.join(JOBS, jid, job["source"]))
            except FileNotFoundError:
                pass
        job["source_deleted"] = True
        save_job(job)
    log(jid, "Originalvideo gelöscht (Clips bleiben erhalten)")


@app.post("/api/jobs/{jid}/delete-source")
def delete_source_endpoint(jid: str, request: Request):
    require_auth(request)
    delete_source(jid)
    return public_job(load_job(jid))


@app.post("/api/storage/cleanup")
def storage_cleanup(request: Request):
    """Delete the originals of all finished projects; clips, subtitles and texts stay."""
    require_auth(request)
    freed = 0
    for jid in os.listdir(JOBS):
        try:
            job = load_job(jid)
        except Exception:
            continue
        if job["status"] == "fertig" and not job.get("source_deleted") and job.get("source"):
            path = os.path.join(JOBS, jid, job["source"])
            freed += os.path.getsize(path) if os.path.isfile(path) else 0
            delete_source(jid)
    return {"freed": freed}


COSTS = os.path.join(DATA, "costs.json")
cost_lock = threading.Lock()


def record_usage(feature, model, tin, tout, usd):
    """Called by the pipeline after every Claude request: totals per project and per month."""
    jid = CURRENT["jid"]
    with cost_lock:
        try:
            with open(COSTS, encoding="utf-8") as f:
                costs = json.load(f)
        except (FileNotFoundError, ValueError):
            costs = {}
        month = costs.setdefault(time.strftime("%Y-%m"), {})
        e = month.setdefault(feature, {"usd": 0.0, "calls": 0, "in": 0, "out": 0})
        e["usd"] = round(e["usd"] + usd, 6)
        e["calls"] += 1
        e["in"] += tin
        e["out"] += tout
        with open(COSTS + ".tmp", "w", encoding="utf-8") as f:
            json.dump(costs, f)
        os.replace(COSTS + ".tmp", COSTS)
    if jid:
        with lock:
            try:
                job = load_job(jid)
            except Exception:
                return
            c = job.setdefault("costs", {}).setdefault(feature, {"usd": 0.0, "calls": 0})
            c["usd"] = round(c["usd"] + usd, 6)
            c["calls"] += 1
            save_job(job)


pipeline.USAGE_HOOK = record_usage


@app.get("/api/costs")
def costs(request: Request):
    require_auth(request)
    try:
        with open(COSTS, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, ValueError):
        data = {}
    months = [{"month": m, "usd": round(sum(v["usd"] for v in feats.values()), 4),
               "features": {k: {"usd": round(v["usd"], 4), "calls": v["calls"]} for k, v in feats.items()}}
              for m, feats in sorted(data.items(), reverse=True)]
    return {"model": pipeline.CLAUDE_MODEL, "prices": pipeline.PRICES.get(pipeline.CLAUDE_MODEL), "months": months[:12]}


@app.get("/api/jobs/{jid}/export")
def export_project(jid: str, request: Request):
    """ZIP for editing software: EDL per clip (Premiere, DaVinci), one FCPXML (Final Cut, DaVinci), SRTs, notes."""
    require_auth(request)
    job = load_job(jid)
    if not job.get("clips") or not job.get("info"):
        raise HTTPException(400, "Noch keine Clips vorhanden")
    info = job["info"]
    segments = load_transcript(jid)
    src_name = job["name"] if os.path.splitext(job["name"])[1] else (job["name"] + ".mp4")
    base = re.sub(r"[^\w\- ]+", "_", os.path.splitext(job["name"])[0])[:50].strip() or "projekt"
    buf = io.BytesIO()
    entries = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, c in enumerate(job["clips"]):
            dur = c["end"] - c["start"]
            words = pipeline.clip_words(segments, c["start"], c["end"])
            iv = pipeline.keep_intervals(words, dur, job["settings"].get("remove_pauses", True),
                                         [r - c["start"] for r in c.get("removed", [])],
                                         natural=job["settings"].get("natural_pauses", True))
            iv = pipeline.add_spans(iv, reaction_spans(jid, c["start"], c["end"]), dur)
            name = "clip%02d" % (i + 1)
            title = "%s – %s" % (name, c.get("title", ""))
            entries.append((title[:80], c["start"], iv))
            z.writestr(name + ".edl", pipeline.edl(title, src_name, c["start"], iv, info.get("fps", 30)))
            if c.get("srt") and os.path.exists(os.path.join(JOBS, jid, c["srt"])):
                z.write(os.path.join(JOBS, jid, c["srt"]), name + ".srt")
        z.writestr(base + ".fcpxml", pipeline.fcpxml(base, src_name, info, entries))
        z.writestr("LIESMICH.txt", (
            "Export aus ShortCut – %s\n\n"
            "Die Schnittlisten verweisen auf das Originalvideo „%s“ (%.3g fps). Lege es in denselben Ordner oder "
            "verknüpfe es nach dem Import neu (Premiere: Medien verknüpfen, DaVinci: Relink, Final Cut: Neu verbinden).\n\n"
            "Premiere Pro:   Datei > Importieren > clipNN.edl\n"
            "DaVinci Resolve: Datei > Importieren > Timeline > clipNN.edl oder %s.fcpxml\n"
            "Final Cut Pro:   Ablage > Importieren > XML > %s.fcpxml (ein Projekt je Clip)\n\n"
            "Enthalten sind alle Schnitte (Pausen, Füllwörter, gestrichene Wörter) und die Untertitel als SRT "
            "(clipNN.srt, passend zur geschnittenen Zeitachse). Hochformat-Zuschnitt, Zooms, Untertitel-Design und "
            "Musik werden nicht übertragen – die Sequenz ist 1080×1920, den Bildausschnitt setzt du im Schnittprogramm.\n"
        ) % (job["name"], src_name, info.get("fps", 30), base, base))
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="%s_schnitt.zip"' % base})


@app.post("/api/jobs/{jid}/render")
def render_all(jid: str, request: Request):
    """Leave review mode and render every clip."""
    require_auth(request)
    with lock:
        job = load_job(jid)
        if job["status"] != "prüfen":
            raise HTTPException(400, "Projekt ist nicht im Prüfmodus")
        require_source(job)
        job["status"], job["message"] = "wartet", "Clips werden gerendert"
        save_job(job)
    work.put(("render_all", jid, None))
    return public_job(job, with_text=True)


@app.get("/api/postiz/integrations")
def postiz_integrations(request: Request):
    require_auth(request)
    try:
        return integrations.Postiz(load_config()).integrations()
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, "Postiz nicht erreichbar: " + str(e)[:200])


@app.post("/api/jobs/{jid}/clips/{idx}/postiz")
async def send_to_postiz(jid: str, idx: int, request: Request):
    require_auth(request)
    body = await request.json()
    job = load_job(jid)
    if idx >= len(job["clips"]) or not job["clips"][idx].get("file"):
        raise HTTPException(400, "Clip ist noch nicht fertig")
    clip = job["clips"][idx]
    channels = [c for c in body.get("channels", []) if isinstance(c, dict) and c.get("id")]
    if not channels:
        raise HTTPException(400, "Bitte mindestens einen Kanal wählen.")
    now = body.get("when") == "now"
    try:
        date = datetime.fromisoformat(str(body.get("date"))).astimezone(timezone.utc) if not now else datetime.now(timezone.utc)
    except ValueError:
        raise HTTPException(400, "Ungültiges Datum")
    content = str(body.get("content") or (clip.get("title", "") + "\n\n" + clip.get("caption", ""))).strip()
    try:
        integrations.Postiz(load_config()).post(os.path.join(job_dir(jid), clip["file"]), content, clip.get("title", ""),
                                                 channels, date.strftime("%Y-%m-%dT%H:%M:%S.000Z"), now)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, "Postiz nicht erreichbar: " + str(e)[:200])
    with lock:
        job = load_job(jid)
        job["clips"][idx]["postiz"] = {"at": time.time(), "date": date.timestamp(), "now": now,
                                       "channels": [c.get("name", "") for c in channels]}
        save_job(job)
    return {"ok": True}


def music_path(name):
    name = os.path.basename(name)
    if not re.fullmatch(r"[\w\-. äöüÄÖÜß()]{1,100}\.(mp3|m4a|wav|ogg|aac)", name, re.I):
        raise HTTPException(400, "Bitte eine Audiodatei (MP3, M4A, WAV, OGG) wählen.")
    return os.path.join(MUSIC, name)


@app.get("/api/music")
def list_music(request: Request):
    require_auth(request)
    return [{"name": n, "size": os.path.getsize(os.path.join(MUSIC, n))}
            for n in sorted(os.listdir(MUSIC)) if not n.endswith(".part")]


@app.put("/api/music")
async def upload_music(request: Request, name: str = ""):
    require_auth(request)
    path = music_path(name)
    size = 0
    with open(path + ".part", "wb") as f:
        async for chunk in request.stream():
            size += len(chunk)
            if size > 40 << 20:
                f.close()
                os.remove(path + ".part")
                raise HTTPException(400, "Musikdatei ist zu groß (max. 40 MB).")
            f.write(chunk)
    os.replace(path + ".part", path)
    return {"ok": True}


@app.delete("/api/music/{name}")
def delete_music(name: str, request: Request):
    require_auth(request)
    path = music_path(name)
    if os.path.exists(path):
        os.remove(path)
    return {"ok": True}


@app.get("/music/{name}")
def get_music(name: str, request: Request):
    require_auth(request)
    path = music_path(name)
    if not os.path.exists(path):
        raise HTTPException(404, "Unbekannt")
    return FileResponse(path)


@app.get("/files/{jid}/{name}")
def get_file(jid: str, name: str, request: Request, dl: int = 0):
    require_auth(request)
    if not re.fullmatch(r"((clip\d{1,2}|c[0-9a-f]{6})(_v\d+)?(_4x5|_1x1|_cover|_[a-z]{2})?\.(mp4|jpg|srt)|bestof_\d+\.(mp4|jpg))", name):
        raise HTTPException(404, "Unbekannt")
    path = os.path.join(job_dir(jid), name)
    if not os.path.exists(path):
        raise HTTPException(404, "Unbekannt")
    headers = {}
    if dl:
        job = load_job(jid)
        base = re.sub(r"[^\w\-]+", "_", os.path.splitext(job["name"])[0])[:40]
        headers["Content-Disposition"] = 'attachment; filename="%s_%s"' % (base, name)
    return FileResponse(path, headers=headers)


# ---------------------------------------------------------------- worker

class Cancelled(Exception):
    pass


def update(jid, **fields):
    with lock:
        job = load_job(jid)
        if job.get("cancel"):
            raise Cancelled()
        job.update(fields)
        save_job(job)
        return job


def log(jid, msg):
    with lock:
        try:
            job = load_job(jid)
        except FileNotFoundError:
            return
        job["log"].append(time.strftime("%H:%M:%S ") + msg)
        save_job(job)


FORMATS = {"4x5": (1080, 1350), "1x1": (1080, 1080)}


def clip_files(clip):
    """Every file that belongs to the current version of a clip."""
    files = [clip.get(k) for k in ("file", "thumb", "srt", "cover")]
    files += list((clip.get("formats") or {}).values())
    for t in (clip.get("translations") or {}).values():
        files += [t.get("file"), t.get("srt")]
    return [f for f in files if f]


def manual_crop(clip, info):
    """Crop positions set by hand in the editor: [(t, centre 0..1)] -> [(t0, t1, centre_x)] in clip time."""
    keys = sorted(clip.get("crop_keys") or [])
    if not keys:
        return None
    duration = clip["end"] - clip["start"]
    keys[0] = [0.0, keys[0][1]]
    segs = []
    for (t, c), nxt in zip(keys, keys[1:] + [[duration, None]]):
        if nxt[0] > t:
            segs.append((t, nxt[0], int(c * info["width"])))
    return segs or None


def snap_to_cuts(det, duration, limit=0.9):
    """A camera cut just after the start or just before the end would flash a frame of the other shot:
    move the clip boundary onto the cut. Returns (cut off at the start, cut off at the end) in seconds."""
    head = tail = 0.0
    for c in (det or {}).get("cuts", []):
        if 0.05 < c < limit:
            head = max(head, c + 0.05)
        if duration - limit < c < duration - 0.05:
            tail = max(tail, duration - c + 0.05)
    if duration - head - tail < 3:
        return 0.0, 0.0
    return round(head, 2), round(tail, 2)


def shift_segs(segs, offset, duration):
    """Framing computed for the original range, moved to a clip that now starts `offset` seconds later."""
    if not segs:
        return segs
    out = []
    for seg in segs:
        t0, t1 = max(0.0, seg[0] - offset), min(duration, seg[1] - offset)
        if t1 <= t0:
            continue
        pts = [(min(max(t - offset, t0), t1), x) for t, x in (seg[3] if len(seg) > 3 and seg[3] else [])]
        out.append((t0, t1, seg[2], pts))
    return out or None


def prepare(jid, idx, info):
    """Everything a render of this clip needs: cut intervals, words on the output timeline, zooms, framing, music."""
    job = load_job(jid)
    clip = dict(job["clips"][idx])
    s = dict(job["settings"])
    src = os.path.join(JOBS, jid, job["source"])
    manual = manual_crop(clip, info)
    split_wanted = s.get("mode") == "split"
    # faces are only needed for automatic framing; scene cuts are always looked for
    det = pipeline.detect_faces(src, info, clip["start"], clip["end"], faces=not manual and s.get("mode") in ("auto", "split"))
    head, tail = snap_to_cuts(det, clip["end"] - clip["start"]) if not manual else (0.0, 0.0)
    clip["start"], clip["end"] = round(clip["start"] + head, 2), round(clip["end"] - tail, 2)
    duration = clip["end"] - clip["start"]
    words = pipeline.clip_words(load_transcript(jid), clip["start"], clip["end"])
    removed = [r - clip["start"] for r in clip.get("removed", [])]
    beeped = {round(b, 2) for b in clip.get("beeped", [])}
    for w in words:  # clip_words gives clip-relative times; the marks are absolute word starts
        if round(w["s"] + clip["start"], 2) in beeped:
            w["beep"] = True
    intervals = pipeline.keep_intervals(words, duration, s.get("remove_pauses", True), removed,
                                        natural=s.get("natural_pauses", True))
    intervals = pipeline.add_spans(intervals, reaction_spans(jid, clip["start"], clip["end"]), duration)
    if s.get("loop_end", True):
        intervals = pipeline.tighten_ends(intervals, [w for w in words if round(w["s"], 2) not in {round(r, 2) for r in removed}])
    fps = pipeline.out_fps(info)
    if s.get("blink_safe", True):
        try:
            intervals = pipeline.refine_cuts(src, info, clip["start"], intervals, words, fps)
        except Exception as e:
            log(jid, "Schnittpunkt-Feinschliff übersprungen: " + str(e)[:200])
    intervals = pipeline.quantize(intervals, fps)
    out_words = pipeline.remap_words(words, intervals)
    beeps = [(w["s"], w["e"]) for w in out_words if w.get("beep")]
    for w in out_words:  # blacked out in the subtitles too
        if w.get("beep"):
            w["w"] = w["w"][:1] + "***" + ("".join(ch for ch in w["w"][-1:] if ch in ".,!?;:"))
    out_len = sum(b - a for a, b in intervals)
    series = clip.get("series")
    end_text = ("Teil %d folgt →" % (series["part"] + 1)) if series and series["part"] < series["total"] else ""
    crop_segs, split = manual, None
    if manual:
        s["mode"] = "auto"  # hand-set framing always crops, whatever the project mode
    elif s.get("mode") in ("auto", "split"):
        if split_wanted:
            split = pipeline.split_pair(det)
        crop_segs = None if split else shift_segs(pipeline.speaker_segments(det, duration + head + tail), head, duration)
        crop_segs = pipeline.align_switches(crop_segs, words, [c - head for c in (det or {}).get("cuts", [])])
        s["mode"] = "auto"
        s["split_layout"] = bool(split)
    music = os.path.join(MUSIC, s["music"]) if s.get("music") and os.path.exists(os.path.join(MUSIC, s["music"])) else None

    # picture rhythm: speaker switches count as a change; zooms at jump cuts and to keep long takes alive
    switches = [pipeline.remap(seg[0], intervals) for seg in (crop_segs or [])[1:]]
    zoom_on, jumpcuts = s.get("zoom", True), s.get("jumpcut_zoom", True)
    zooms = pipeline.zoom_spans(out_words, intervals, out_len, jumpcuts=jumpcuts, rhythm=zoom_on, switches=switches,
                                loop=s.get("loop_end", True)) if (zoom_on or jumpcuts) and not split else None
    changes = sorted(set([0.0, out_len] + switches + pipeline.cut_points(intervals) +
                         [e for z in (zooms or []) for e in z[:2]]))
    static_max = max((b - a for a, b in zip(changes, changes[1:])), default=out_len)
    tracks = [t for t in (det or {}).get("tracks", []) if t.get("eye")]
    eye = sum(t["eye"] * t["seen"] for t in tracks) / sum(t["seen"] for t in tracks) if tracks and crop_segs else 0.42

    quiet, roomtone_at = [], None
    if s.get("roomtone", True):
        for a, b in zip(out_words, out_words[1:]):  # pauses that stay: breaths and noises go down
            if b["s"] - a["e"] > 0.25:
                quiet.append((round(a["e"] + 0.06, 3), round(b["s"] - 0.06, 3)))
        if s.get("enhance_audio", True):
            try:
                near = [w for w in pipeline.all_words(load_transcript(jid)) if w["e"] > clip["start"] - 25 and w["s"] < clip["end"] + 25]
                roomtone_at = pipeline.roomtone_spot(src, clip["start"], clip["end"], near)
            except Exception:
                roomtone_at = None
    return {"job": job, "clip": clip, "s": s, "src": src, "duration": duration, "intervals": intervals,
            "out_words": out_words, "out_len": out_len, "crop_segs": crop_segs, "music": music,
            "det": det, "split": split, "split_wanted": split_wanted, "snapped": (head, tail), "beeps": beeps,
            "end_text": end_text, "zooms": zooms, "static_max": static_max, "eye": eye, "quiet": quiet,
            "roomtone_at": roomtone_at}


def render_variant(jid, p, base, words, hook, keywords, size, srt=True):
    d = os.path.join(JOBS, jid)
    clip = p["clip"]
    ass = base + ".ass"
    burn = pipeline.write_subtitles(words, os.path.join(d, ass), os.path.join(d, base + ".srt") if srt else None,
                                    p["s"], hook, p["out_len"], keywords, size, p["end_text"])
    out_len = pipeline.render_clip(p["src"], p["job"]["info"], clip["start"], clip["end"], d, base, p["s"], ass, burn,
                                   p["intervals"], p["crop_segs"], size, p["zooms"], p["music"], p["split"], p["beeps"],
                                   p["quiet"], p["roomtone_at"], p["eye"])
    os.remove(os.path.join(d, ass))
    return out_len


def job_settings(jid):
    return load_job(jid).get("settings", {})


def ensure_aligned(jid, idx):
    """Word-exact timing for the stretch of one clip (once per stretch; the rest of the transcript stays as is)."""
    job = load_job(jid)
    if not job["settings"].get("precise_timing", True) or job.get("source_deleted"):
        return
    clip = job["clips"][idx]
    lo, hi = max(0.0, clip["start"] - 1), clip["end"] + 1
    done = job.get("aligned", [])
    if any(a <= lo and hi <= b for a, b in done):
        return
    segments = load_transcript(jid)
    words = [w for w in pipeline.all_words(segments) if w["e"] > lo and w["s"] < hi]
    try:
        new = pipeline.align_words(os.path.join(JOBS, jid, job["source"]), words, lo, hi)
    except Exception as e:
        log(jid, "Wortgenaue Zeitstempel nicht möglich: " + str(e)[:200])
        return
    moved = sum(1 for a, b in zip(words, new) if abs(a["s"] - b["s"]) > 0.04)
    save_transcript(jid, pipeline.replace_words(segments, lo, hi, new))
    update(jid, aligned=done + [[round(lo, 2), round(hi, 2)]])
    log(jid, "Clip %d: %d von %d Wörtern zeitlich nachjustiert" % (idx + 1, moved, len(words)))


def render_one(jid, idx, info):
    """Render one clip: speaker framing, cuts (pauses, fillers, struck-out words), zooms, subtitles, hook, audio,
    extra formats and the cover image. Translations of an older version are dropped."""
    d = os.path.join(JOBS, jid)
    ensure_aligned(jid, idx)
    p = prepare(jid, idx, info)
    clip, s = p["clip"], p["s"]
    uid = clip.get("uid") or uuid.uuid4().hex[:6]
    # never reuse a file name that an older version in the trash still holds
    version = max([clip.get("version", 0)] + [t["clip"].get("version", 0) for t in load_job(jid).get("trash", [])
                                              if t["clip"].get("uid") == uid]) + 1
    name = "c" + uid + ("_v%d" % version if version > 1 else "")
    hook = clip.get("hook_text", clip.get("title", ""))
    keywords = clip.get("keywords", [])

    outputs = {}
    out_len = p["out_len"]
    for fmt, size in [("9x16", (1080, 1920))] + [(f, FORMATS[f]) for f in FORMATS if s.get("fmt_" + f)]:
        base = name if fmt == "9x16" else name + "_" + fmt
        out_len = render_variant(jid, p, base, p["out_words"], hook, keywords, size, srt=fmt == "9x16")
        outputs[fmt] = base + ".mp4"
    pipeline.render_thumb(d, name + ".mp4", name, min(1.0, out_len / 2))
    cover = None
    if s.get("cover", True):
        try:
            at = pipeline.best_cover_time(p["src"], info, clip["start"], clip["end"])
        except Exception:
            at = None
        pipeline.render_cover(p["src"], info, clip["start"], at if at is not None else min(p["duration"] / 2, 1.5), d,
                              name + "_cover", s, p["crop_segs"], clip.get("cover_text") or clip.get("title", ""))
        cover = name + "_cover.jpg"

    checks = pipeline.quality_checks(d, outputs["9x16"], info, load_transcript(jid), clip["start"], clip["end"],
                                     p["det"], p["crop_segs"], s.get("mode"), p["split_wanted"], p["split"], out_len)
    if p["static_max"] > 6 and not s.get("zoom", True):
        checks.append({"level": "info", "msg": "Das Bild steht bis zu %.0f s ohne Veränderung – mit „Dynamische Zooms“ "
                                               "bleibt es lebendiger." % p["static_max"]})
    key = load_config().get("anthropic_key", "")
    if key and job_settings(jid).get("ai_check", True):
        try:
            issues = pipeline.visual_check(key, os.path.join(d, outputs["9x16"]), out_len, hook, clip.get("title", ""),
                                           " ".join(w["w"] for w in p["out_words"]))
            checks += [{"level": "warn", "msg": "KI-Kontrolle: " + i} for i in issues] or                 [{"level": "info", "msg": "KI-Endkontrolle ohne Befund."}]
        except Exception as e:
            log(jid, "KI-Endkontrolle fehlgeschlagen: " + str(e)[:200])

    keep = {name + ".mp4", name + ".jpg", name + ".srt", cover} | set(outputs.values())
    if not clip.get("prev"):  # with a kept previous version, its files go to the trash instead
        for old in clip_files(clip):
            if old not in keep:
                try:
                    os.remove(os.path.join(d, old))
                except OSError:
                    pass
    with lock:
        job = load_job(jid)
        prev = job["clips"][idx].pop("prev", None)
        if prev and prev.get("file"):
            trash_add(job, "version", prev, idx)
        job["clips"][idx].update({
            "file": outputs["9x16"], "thumb": name + ".jpg", "srt": name + ".srt", "cover": cover,
            "formats": {k: v for k, v in outputs.items() if k != "9x16"}, "version": version, "status": "fertig",
            "duration_out": round(out_len, 1), "speakers": len({seg[2] for seg in p["crop_segs"] or []}),
            "translations": {}, "checks": checks, "split": bool(p["split"]), "uid": uid,
        })
        save_job(job)
    export_files(jid, clip_files(load_job(jid)["clips"][idx]), "Clip %d" % (idx + 1))


def render_translation(jid, idx, lang):
    """Same clip with translated subtitles, hook and texts (the voice stays German)."""
    key = load_config().get("anthropic_key", "")
    if not key:
        raise RuntimeError("Für Übersetzungen wird ein Claude-API-Schlüssel benötigt (Einstellungen).")
    job = load_job(jid)
    p = prepare(jid, idx, job["info"])
    clip = p["clip"]
    meta = {"hook": clip.get("hook_text", clip.get("title", "")), "title": clip.get("title", ""),
            "caption": clip.get("caption", ""), "cover": clip.get("cover_text", ""), "keywords": clip.get("keywords", [])}
    words, tmeta = pipeline.translate_clip(p["out_words"], key, lang, meta)
    base = os.path.splitext(clip["file"])[0] + "_" + lang
    render_variant(jid, p, base, words, tmeta.get("hook", ""), tmeta.get("keywords") or [], (1080, 1920))
    with lock:
        job = load_job(jid)
        c = job["clips"][idx]
        c.setdefault("translations", {})[lang] = {
            "file": base + ".mp4", "srt": base + ".srt", "title": tmeta.get("title", ""),
            "caption": tmeta.get("caption", ""), "hook": tmeta.get("hook", ""), "version": c.get("version", 1),
        }
        save_job(job)
    export_files(jid, [base + ".mp4", base + ".srt"], "Clip %d (%s)" % (idx + 1, lang))


def render_bestof(jid, target):
    """Teaser from the strongest sentence of several clips, joined into one video."""
    d = os.path.join(JOBS, jid)
    job = load_job(jid)
    info, s = job["info"], job["settings"]
    segments = load_transcript(jid)
    picks = pipeline.bestof_segments(segments, job["clips"], target)
    if not picks:
        raise RuntimeError("Keine passenden Sätze für ein Best-of gefunden.")
    stamp = int(time.time())
    parts, total = [], 0.0
    for n, (a, b, c) in enumerate(picks):
        clip = {"start": a, "end": b, "removed": [], "crop_keys": c.get("crop_keys")}
        words = pipeline.clip_words(segments, a, b)
        intervals = pipeline.keep_intervals(words, b - a, s.get("remove_pauses", True))
        intervals = pipeline.add_spans(intervals, reaction_spans(jid, a, b), b - a)
        intervals = pipeline.quantize(intervals, pipeline.out_fps(info))
        out_words = pipeline.remap_words(words, intervals)
        out_len = sum(y - x for x, y in intervals)
        crop = manual_crop(clip, info) or (pipeline.track_speaker(os.path.join(d, job["source"]), info, a, b)
                                           if s.get("mode") in ("auto", "split") else None)
        ps = dict(s, hook=bool(n == 0 and s.get("hook", True)),
                  mode="auto" if crop else ("crop" if s.get("mode") == "split" else s.get("mode")))
        base = "bestof_part%02d" % n
        ass = base + ".ass"
        burn = pipeline.write_subtitles(out_words, os.path.join(d, ass), None, ps,
                                        job.get("bestof_hook") or picks[0][2].get("hook_text") or "Das Beste aus der Folge", out_len, c.get("keywords", []))
        pipeline.render_clip(os.path.join(d, job["source"]), info, a, b, d, base, ps, ass, burn, intervals, crop,
                             (1080, 1920), pipeline.zoom_spans(out_words, intervals, out_len) if s.get("zoom", True) else None)
        os.remove(os.path.join(d, ass))
        parts.append(base + ".mp4")
        total += out_len
    music = os.path.join(MUSIC, s["music"]) if s.get("music") and os.path.exists(os.path.join(MUSIC, s["music"])) else None
    out = "bestof_%d.mp4" % stamp
    pipeline.concat_parts(d, parts, out, music, int(s.get("music_volume", 15)))
    for f in parts:
        os.remove(os.path.join(d, f))
    pipeline.render_thumb(d, out, "bestof_%d" % stamp, 1.0)
    with lock:
        job = load_job(jid)
        old = job.get("bestof") or {}
        for k in ("file", "thumb"):
            if old.get(k):
                try:
                    os.remove(os.path.join(d, old[k]))
                except OSError:
                    pass
        job["bestof"] = {"file": out, "thumb": "bestof_%d.jpg" % stamp, "duration": round(total, 1),
                         "parts": [[a, b] for a, b, _ in picks], "status": "fertig"}
        save_job(job)
    export_files(jid, [out], "Best-of")


def export_files(jid, files, label):
    cfg = load_config()
    if not cfg.get("nc_export"):
        return
    job = load_job(jid)
    try:
        nc = integrations.Nextcloud(cfg)
        folder = "Shorts/" + re.sub(r"[^\w\- ]+", "_", os.path.splitext(job["name"])[0])[:60].strip()
        nc.mkdirs(folder)
        for f in files:
            if not f.endswith(".jpg") or "_cover" in f:
                nc.upload(folder + "/" + f, os.path.join(JOBS, jid, f))
        log(jid, "%s in Nextcloud gespeichert (%s/%s)" % (label, nc.folder, folder))
    except Exception as e:
        log(jid, "Nextcloud-Export fehlgeschlagen: " + str(e)[:200])


WATCH = {"at": None, "ok": None, "message": "", "seen": {}}


def watcher():
    """Poll the Nextcloud folder <folder>/Eingang; a video whose size stayed the same for one round is moved to
    Eingang/Verarbeitet and processed with the default settings."""
    while True:
        cfg = load_config()
        if cfg.get("nc_watch") and cfg.get("nc_pass"):
            watch_once(cfg)
        time.sleep(int(os.environ.get("WATCH_INTERVAL", "120")))


def watch_once(cfg):
    try:
        nc = integrations.Nextcloud(cfg)
        nc.mkdirs("Eingang/Verarbeitet")
        started = 0
        for f in nc.list_videos("Eingang"):
            name, size = f["name"], f["size"]
            if WATCH["seen"].get(name) != size:  # new or still uploading
                WATCH["seen"][name] = size
                continue
            target = "Eingang/Verarbeitet/" + time.strftime("%Y%m%d-%H%M%S_") + name
            nc.move("Eingang/" + name, target)
            WATCH["seen"].pop(name, None)
            new_job(name, clean_settings(cfg.get("defaults")), "", None, {"type": "nextcloud", "name": target})
            started += 1
        WATCH.update(at=time.time(), ok=True,
                     message=("%d neue Videos gestartet" % started) if started else "Ordner geprüft, nichts Neues")
    except Exception as e:
        WATCH.update(at=time.time(), ok=False, message=str(e)[:200])


def fetch_source(jid, job):
    """Download the video for jobs imported from Nextcloud."""
    d = os.path.join(JOBS, jid)
    remote = job.get("remote") or {}

    def prog(p, msg):
        update(jid, message=msg, **({"progress": round(p * 0.03, 1)} if p is not None else {}))

    if remote.get("type") == "nextcloud":
        update(jid, message="Lade Video aus Nextcloud …")
        src = source_name(remote["name"])
        integrations.Nextcloud(load_config()).download(remote["name"], os.path.join(d, src), prog)
        return update(jid, source=src, size=os.path.getsize(os.path.join(d, src)))
    raise RuntimeError("Keine Videodatei vorhanden.")


def load_audio(jid):
    path = os.path.join(JOBS, jid, "audio.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_audio_analysis(jid, wav, segments):
    """Loudness, liveliness and speech rate over the whole video; kept in audio.json, a small curve goes to the job."""
    update(jid, progress=71, message="Analysiere Ton (Lautstärke, Stimme, Tempo, Reaktionen) …")
    try:
        audio = pipeline.analyze_audio(wav, pipeline.all_words(segments))
    except Exception as e:
        log(jid, "Tonanalyse fehlgeschlagen: " + str(e)[:200])
        return
    if not audio:
        return
    with open(os.path.join(JOBS, jid, "audio.json"), "w", encoding="utf-8") as f:
        json.dump(audio, f)
    counts = {}
    for ev in audio["events"]:
        counts[ev["type"]] = counts.get(ev["type"], 0) + 1
    update(jid, audio_curve=pipeline.audio_curve(audio), audio_events=audio["events"][:300])
    log(jid, "Tonanalyse: " + (", ".join("%d× %s" % (n, pipeline.TAG_LABELS[k]) for k, n in counts.items()) or "keine Auffälligkeiten"))


def reaction_spans(jid, start, end):
    """Laughter/applause inside a clip (clip time), so the pause cutter keeps them."""
    audio = load_audio(jid)
    if not audio:
        return []
    return [(ev["s"] - start, ev["e"] - start) for ev in audio["events"]
            if ev["type"] == "reaktion" and ev["e"] > start and ev["s"] < end]


QUALITY_RANK = {"schnell": 0, "ausgewogen": 1, "genau": 2}


def reuse_transcript(jid, job):
    """Same video processed before (same fingerprint, at least the same recognition quality): copy its transcript
    and audio analysis instead of transcribing again."""
    want = QUALITY_RANK.get(job["settings"].get("quality"), 1)
    best = None
    for other in os.listdir(JOBS):
        if other == jid:
            continue
        try:
            o = load_job(other)
        except Exception:
            continue
        if o.get("hash") != job["hash"] or not os.path.exists(os.path.join(JOBS, other, "transcript.json")):
            continue
        if QUALITY_RANK.get(o.get("settings", {}).get("quality"), 1) < want:
            continue
        if best is None or o["created"] > best["created"]:
            best = o
    if not best:
        return None
    for f in ("transcript.json", "audio.json"):
        if os.path.exists(os.path.join(JOBS, best["id"], f)):
            shutil.copy(os.path.join(JOBS, best["id"], f), os.path.join(JOBS, jid, f))
    audio = load_audio(jid)
    if audio:
        update(jid, audio_curve=pipeline.audio_curve(audio), audio_events=audio["events"][:300])
    update(jid, vocabulary=best.get("vocabulary", ""), reused_from=best["name"])
    note = ""
    if best.get("vocabulary", "") != load_config().get("vocabulary", ""):
        note = " Die Fachbegriffe wurden seitdem geändert – für eine neue Erkennung den Haken „Transkript wiederverwenden“ abwählen."
    log(jid, "Transkript vom Projekt „%s“ übernommen – spart die Spracherkennung.%s" % (best["name"], note))
    return load_transcript(jid)


def process(jid):
    d = os.path.join(JOBS, jid)
    job = update(jid, status="läuft", progress=1, message="Analysiere Video …", started=time.time())
    if not job.get("source") or not os.path.exists(os.path.join(d, job["source"])):
        job = fetch_source(jid, job)
    src = os.path.join(d, job["source"])
    info = pipeline.probe(src)
    job = update(jid, info=info)
    log(jid, "Video: %dx%d, %s" % (info["width"], info["height"], pipeline.fmt_time(info["duration"])))

    if not job.get("hash"):
        job = update(jid, hash=pipeline.quick_hash(src))
    segments = load_transcript(jid)
    if segments is None and job["settings"].get("reuse_transcript", True):
        segments = reuse_transcript(jid, job)
    if segments is None:
        t0 = time.time()
        update(jid, progress=3, message="Extrahiere Audio …")
        wav = os.path.join(d, "audio.wav")
        pipeline.extract_audio(src, wav)

        def prog(p, msg):
            fields = {"message": msg}
            if p is not None:
                fields["progress"] = round(5 + p * 0.65, 1)
            update(jid, **fields)

        segments = pipeline.transcribe(wav, info["duration"], job["settings"]["quality"], prog,
                                       load_config().get("vocabulary", ""))
        save_transcript(jid, segments)
        learn_speed("transcribe_" + job["settings"].get("quality", "ausgewogen"), time.time() - t0, info["duration"])
        update(jid, vocabulary=load_config().get("vocabulary", ""))
        log(jid, "Transkript: %d Abschnitte" % len(segments))
        if job["settings"].get("audio_analysis", True):
            run_audio_analysis(jid, wav, segments)
        os.remove(wav)
    if not segments:
        raise RuntimeError("Im Video wurde keine Sprache erkannt.")
    if job["settings"].get("audio_analysis", True) and load_audio(jid) is None:
        wav = os.path.join(d, "audio.wav")
        pipeline.extract_audio(src, wav)
        run_audio_analysis(jid, wav, segments)
        os.remove(wav)

    s = job["settings"]
    update(jid, progress=72, message="Suche die besten Highlights …")
    key = load_config().get("anthropic_key", "")
    audio = load_audio(jid) if s.get("audio_analysis", True) else None
    clips, source = pipeline.find_highlights(segments, key, s["clips"], s["min_len"], s["max_len"], job.get("context", ""),
                                             lambda m: log(jid, m), audio, s.get("two_stage", True))
    if not clips:
        raise RuntimeError("Keine passenden Highlights gefunden. Versuche eine kürzere Mindestlänge.")
    for c in clips:
        c["status"] = "entwurf" if s.get("review") else "wartet"
        c["hook_text"] = c.get("title", "")
        c["uid"] = uuid.uuid4().hex[:6]
        c["cover_text"] = c.get("title", "")
    update(jid, clips=clips, selector=source)
    log(jid, "%d Highlights gefunden (%s)" % (len(clips), "Claude KI" if source == "claude" else "Automatik"))
    if s.get("review"):
        update(jid, status="prüfen", progress=75, message="Bitte Clips prüfen und dann rendern")
        return
    render_clips(jid, info)


def spelling(jid, indices):
    cfg = load_config()
    key = cfg.get("anthropic_key", "")
    if not key:
        return
    job = load_job(jid)
    words = pipeline.all_words(load_transcript(jid))
    texts = {i: " ".join(w["w"] for w in words if w["e"] > job["clips"][i]["start"] and w["s"] < job["clips"][i]["end"])
             for i in indices if i < len(job["clips"])}
    try:
        found = pipeline.spelling_check(key, texts, cfg.get("vocabulary", ""))
    except Exception as e:
        log(jid, "Rechtschreibprüfung fehlgeschlagen: " + str(e)[:200])
        return
    with lock:
        job = load_job(jid)
        for i in texts:
            job["clips"][i]["spelling"] = found.get(i, [])
        save_job(job)
    n = sum(len(v) for v in found.values())
    log(jid, "Rechtschreibprüfung: %s" % ("%d mögliche Fehler gefunden" % n if n else "keine Auffälligkeiten"))


def render_clips(jid, info):
    job = load_job(jid)
    n = len(job["clips"])
    for i in range(n):
        update(jid, status="läuft", progress=round(75 + i / n * 24, 1), message="Rendere Clip %d von %d …" % (i + 1, n))
        t0 = time.time()
        render_one(jid, i, info)
        c = load_job(jid)["clips"][i]
        formats = 1 + len(c.get("formats") or {})
        learn_speed("render", (time.time() - t0) / formats, c.get("duration_out") or 0)
    update(jid, message="Prüfe Rechtschreibung …")
    pipeline.release_aligner()
    spelling(jid, range(n))
    update(jid, status="fertig", progress=100, message="Fertig – %d Clips" % n, finished=time.time())


def rerender(jid, idx):
    job = update(jid, status="läuft", message="Rendere Clip %d neu …" % (idx + 1))
    render_one(jid, idx, job["info"])
    pipeline.release_aligner()
    spelling(jid, [idx])
    update(jid, status="fertig", progress=100, message="Fertig – %d Clips" % len(job["clips"]))


def extra_task(jid, kind, arg):
    """Translation or best-of on a finished project; a failure is reported but leaves the clips usable."""
    update(jid, status="läuft", progress=50)
    n = len(load_job(jid)["clips"])
    label = {"translate": "Übersetzung", "bestof": "Best-of", "texts": "Texterstellung"}[kind]
    try:
        if kind == "texts":
            job = load_job(jid)
            texts = pipeline.generate_texts(load_config().get("anthropic_key", ""), load_transcript(jid), job.get("context", ""))
            update(jid, texts=texts)
            msg = "Fertig – Texte zum Video erstellt"
        elif kind == "translate":
            render_translation(jid, arg[0], arg[1])
            msg = "Fertig – Übersetzung von Clip %d erstellt" % (arg[0] + 1)
        else:
            render_bestof(jid, arg)
            msg = "Fertig – Best-of-Trailer erstellt"
    except Cancelled:
        raise
    except Exception as e:
        traceback.print_exc()
        log(jid, label + " fehlgeschlagen: " + str(e)[:300])
        msg = "Fertig – %d Clips (%s fehlgeschlagen: %s)" % (n, label, str(e)[:150])
    update(jid, status="fertig", progress=100, message=msg)


def worker():
    while True:
        kind, jid, idx = work.get()
        CURRENT.update(jid=jid, kind=kind, since=time.time())
        try:
            if not os.path.exists(os.path.join(JOBS, jid, "job.json")):
                continue
            if kind == "process":
                process(jid)
            elif kind == "render_all":
                render_clips(jid, load_job(jid)["info"])
            elif kind in ("translate", "bestof", "texts"):
                extra_task(jid, kind, idx)
            else:
                rerender(jid, idx)
        except Cancelled:
            with lock:
                try:
                    job = load_job(jid)
                    job.pop("cancel", None)
                    job.update(status="abgebrochen", message="Abgebrochen")
                    job["clips"] = [dict(c, status="fertig") if c.get("file") else c for c in job.get("clips", [])]
                    save_job(job)
                except Exception:
                    pass
        except Exception as e:
            traceback.print_exc()
            try:
                update(jid, status="fehler", message=str(e)[:500])
            except Exception:
                pass
        finally:
            CURRENT.update(jid=None, kind=None, since=None)


def cleanup():
    while True:
        cutoff = time.time() - RETENTION_DAYS * 86400
        for jid in os.listdir(JOBS):
            try:
                if load_job(jid)["created"] < cutoff:
                    shutil.rmtree(os.path.join(JOBS, jid), ignore_errors=True)
                    continue
                with lock:
                    job = load_job(jid)
                    if trash_purge(job, lambda t: t["deleted"] < time.time() - TRASH_DAYS * 86400):
                        save_job(job)
            except Exception:
                pass
        for name in os.listdir(UPLOADS):  # abandoned uploads
            path = os.path.join(UPLOADS, name)
            if os.path.getmtime(path) < time.time() - 86400:
                os.remove(path)
        time.sleep(3600)


@app.on_event("startup")
def start():
    # re-queue jobs interrupted by a restart
    pending = []
    for jid in os.listdir(JOBS):
        try:
            job = load_job(jid)
        except Exception:
            continue
        if job["status"] in ("wartet", "läuft"):
            pending.append(job)
    for job in sorted(pending, key=lambda j: j["created"]):
        if job.get("info") and job["clips"] and all(c.get("file") for c in job["clips"]):
            job["status"], job["message"] = "fertig", "Fertig – %d Clips" % len(job["clips"])
            save_job(job)
        elif job.get("info") and job["clips"]:
            work.put(("render_all", job["id"], None))
        else:
            work.put(("process", job["id"], None))
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=cleanup, daemon=True).start()
    threading.Thread(target=watcher, daemon=True).start()


# ---------------------------------------------------------------- installable web app (PWA) with share target

MANIFEST = {
    "name": "ShortCut", "short_name": "ShortCut", "start_url": "/", "scope": "/", "display": "standalone",
    "background_color": "#0f1115", "theme_color": "#0f1115", "lang": "de",
    "description": "Aus langen Videos werden Shorts und Reels",
    "icons": [{"src": "/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
              {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"}],
    "share_target": {"action": "/share-target", "method": "POST", "enctype": "multipart/form-data",
                     "params": {"title": "title", "files": [{"name": "video", "accept": ["video/*"]}]}},
}

SERVICE_WORKER = """
// ShortCut service worker: makes the app installable and receives videos shared from other apps.
self.addEventListener('install', e => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (e.request.method === 'POST' && url.pathname === '/share-target') {
    e.respondWith((async () => {
      const form = await e.request.formData();
      const file = form.get('video');
      if (file) {
        const cache = await caches.open('shortcut-share');
        await cache.put('/shared-video', new Response(file, {headers: {
          'content-type': file.type || 'video/mp4', 'x-name': encodeURIComponent(file.name || 'video.mp4')}}));
      }
      return Response.redirect('/?shared=1', 303);
    })());
  }
});
"""

_ICONS = {}


def app_icon(size):
    """App icon in the ShortCut colours (drawn once with OpenCV, no image files needed)."""
    if size not in _ICONS:
        import cv2
        import numpy as np

        img = np.zeros((size, size, 3), np.uint8)
        t = np.linspace(0, 1, size)[None, :, None] * 0.5 + np.linspace(0, 1, size)[:, None, None] * 0.5
        a, b = np.array([110, 61, 255], np.float32), np.array([46, 176, 255], np.float32)  # BGR of #ff3d6e, #ffb02e
        img[:] = (a * (1 - t) + b * t).astype(np.uint8)
        w, h = int(size * 0.3), int(size * 0.56)
        x0, y0 = (size - w) // 2, (size - h) // 2
        r = max(2, size // 24)
        cv2.rectangle(img, (x0 + r, y0), (x0 + w - r, y0 + h), (255, 255, 255), -1)
        cv2.rectangle(img, (x0, y0 + r), (x0 + w, y0 + h - r), (255, 255, 255), -1)
        for cx, cy in ((x0 + r, y0 + r), (x0 + w - r, y0 + r), (x0 + r, y0 + h - r), (x0 + w - r, y0 + h - r)):
            cv2.circle(img, (cx, cy), r, (255, 255, 255), -1)
        _ICONS[size] = cv2.imencode(".png", img)[1].tobytes()
    return _ICONS[size]


@app.get("/manifest.webmanifest")
def manifest():
    return JSONResponse(MANIFEST, media_type="application/manifest+json")


@app.get("/sw.js")
def service_worker():
    return Response(SERVICE_WORKER, media_type="application/javascript", headers={"Cache-Control": "no-cache"})


@app.get("/icon-{size}.png")
def icon(size: int):
    if size not in (180, 192, 512):
        raise HTTPException(404, "Unbekannt")
    return Response(app_icon(size), media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@app.post("/share-target")
def share_target_fallback():
    """Normally answered by the service worker; without it, just open the app."""
    return Response(status_code=303, headers={"Location": "/"})
