"""ShortCut – erstellt aus langen Videos Shorts/Reels mit Highlights und Untertiteln."""
import hashlib
import hmac
import json
import os
import queue
import re
import secrets
import shutil
import threading
import time
import traceback
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

import pipeline

DATA = os.environ.get("DATA_DIR", "/data")
JOBS = os.path.join(DATA, "jobs")
UPLOADS = os.path.join(DATA, "uploads")
CONFIG = os.path.join(DATA, "config.json")
RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "14"))
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(JOBS, exist_ok=True)
os.makedirs(UPLOADS, exist_ok=True)

DEFAULTS = {
    "clips": 5, "min_len": 20, "max_len": 60, "mode": "crop", "quality": "ausgewogen",
    "highlight": "#FFE600", "uppercase": True, "font_size": 78, "position": "unten", "words_per_line": 3,
}

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
lock = threading.Lock()
work = queue.Queue()


# ---------------------------------------------------------------- config & auth

def load_config():
    try:
        with open(CONFIG, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


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
    return {"has_key": bool(key), "key_hint": ("…" + key[-4:]) if key else "", "defaults": {**DEFAULTS, **cfg.get("defaults", {})}}


@app.post("/api/settings")
async def set_settings(request: Request):
    require_auth(request)
    body = await request.json()
    with lock:
        cfg = load_config()
        if "anthropic_key" in body:
            key = (body["anthropic_key"] or "").strip()
            if key:
                cfg["anthropic_key"] = key
            else:
                cfg.pop("anthropic_key", None)
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
    if out["mode"] not in ("crop", "blur"):
        out["mode"] = "crop"
    if out["quality"] not in pipeline.WHISPER_PRESETS:
        out["quality"] = "ausgewogen"
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", out["highlight"]):
        out["highlight"] = DEFAULTS["highlight"]
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


def public_job(job):
    return {k: v for k, v in job.items() if k not in ("log",)} | {"log": job.get("log", [])[-5:]}


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
        return public_job(load_job(jid))
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


@app.post("/api/jobs")
async def create_job(request: Request):
    require_auth(request)
    body = await request.json()
    part = upload_path(str(body.get("upload_id", "")))
    if not os.path.exists(part) or os.path.getsize(part) < 1000:
        raise HTTPException(400, "Upload fehlt oder ist leer")
    filename = os.path.basename(str(body.get("filename") or "video.mp4"))[:120] or "video.mp4"
    context = str(body.get("context") or "")[:1000]
    settings = clean_settings(body.get("settings"))
    jid = uuid.uuid4().hex[:12]
    d = os.path.join(JOBS, jid)
    os.makedirs(d)
    ext = os.path.splitext(filename)[1].lower()
    if not re.fullmatch(r"\.[a-z0-9]{2,5}", ext):
        ext = ".mp4"
    src = os.path.join(d, "source" + ext)
    os.replace(part, src)
    size = os.path.getsize(src)
    job = {
        "id": jid, "name": filename, "created": time.time(), "status": "wartet", "progress": 0,
        "message": "In der Warteschlange", "settings": settings, "context": context, "source": os.path.basename(src),
        "size": size, "clips": [], "log": [], "selector": None,
    }
    save_job(job)
    work.put(("process", jid, None))
    return public_job(job)


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str, request: Request):
    require_auth(request)
    d = job_dir(jid)
    job = load_job(jid)
    if job["status"] not in ("fertig", "fehler"):
        job["cancel"] = True
        save_job(job)
    shutil.rmtree(d, ignore_errors=True)
    return {"ok": True}


@app.post("/api/jobs/{jid}/clips/{idx}")
async def rerender_clip(jid: str, idx: int, request: Request):
    """Adjust start/end/title of a clip and render it again."""
    require_auth(request)
    body = await request.json()
    job = load_job(jid)
    if job["status"] not in ("fertig", "fehler") or idx >= len(job["clips"]):
        raise HTTPException(400, "Clip kann gerade nicht bearbeitet werden")
    clip = job["clips"][idx]
    try:
        s, e = float(body.get("start", clip["start"])), float(body.get("end", clip["end"]))
    except (TypeError, ValueError):
        raise HTTPException(400, "Ungültige Zeit")
    if e - s < 3 or e - s > 180 or s < 0:
        raise HTTPException(400, "Clip muss zwischen 3 und 180 Sekunden lang sein")
    clip.update({"start": round(s, 2), "end": round(e, 2), "title": str(body.get("title", clip["title"]))[:200], "status": "wartet"})
    if "style" in body:
        job["settings"] = clean_settings({**job["settings"], **body["style"]})
    job["status"] = "wartet"
    job["message"] = "Clip %d wird neu gerendert" % (idx + 1)
    save_job(job)
    work.put(("render", jid, idx))
    return public_job(job)


@app.get("/files/{jid}/{name}")
def get_file(jid: str, name: str, request: Request, dl: int = 0):
    require_auth(request)
    if not re.fullmatch(r"clip\d{1,2}(_v\d+)?\.(mp4|jpg|srt)", name):
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


def render_one(jid, job, idx, info):
    d = os.path.join(JOBS, jid)
    clip = job["clips"][idx]
    with open(os.path.join(d, "transcript.json"), encoding="utf-8") as f:
        segments = json.load(f)
    version = clip.get("version", 0) + 1
    name = "clip%02d" % (idx + 1) + ("_v%d" % version if version > 1 else "")
    words = pipeline.clip_words(segments, clip["start"], clip["end"])
    ass = name + ".ass"
    pipeline.write_subtitles(words, os.path.join(d, ass), os.path.join(d, name + ".srt"), job["settings"])
    pipeline.render_clip(os.path.join(d, job["source"]), info, clip["start"], clip["end"], d, name,
                         job["settings"]["mode"], ass, burn=bool(words))
    for old in (clip.get("file"), clip.get("thumb"), clip.get("srt")):
        if old and old.split(".")[0] != name:
            try:
                os.remove(os.path.join(d, old))
            except OSError:
                pass
    try:
        os.remove(os.path.join(d, ass))
    except OSError:
        pass
    clip.update({"file": name + ".mp4", "thumb": name + ".jpg", "srt": name + ".srt", "version": version,
                 "status": "fertig", "text": " ".join(w["w"] for w in words)})


def process(jid):
    d = os.path.join(JOBS, jid)
    job = update(jid, status="läuft", progress=1, message="Analysiere Video …", started=time.time())
    src = os.path.join(d, job["source"])
    info = pipeline.probe(src)
    job = update(jid, info=info)
    log(jid, "Video: %dx%d, %s" % (info["width"], info["height"], pipeline.fmt_time(info["duration"])))

    tpath = os.path.join(d, "transcript.json")
    if not os.path.exists(tpath):
        update(jid, progress=3, message="Extrahiere Audio …")
        wav = os.path.join(d, "audio.wav")
        pipeline.extract_audio(src, wav)

        def prog(p, msg):
            fields = {"message": msg}
            if p is not None:
                fields["progress"] = round(5 + p * 0.65, 1)
            update(jid, **fields)

        segments = pipeline.transcribe(wav, info["duration"], job["settings"]["quality"], prog)
        os.remove(wav)
        with open(tpath, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False)
        log(jid, "Transkript: %d Abschnitte" % len(segments))
    else:
        with open(tpath, encoding="utf-8") as f:
            segments = json.load(f)
    if not segments:
        raise RuntimeError("Im Video wurde keine Sprache erkannt.")

    s = job["settings"]
    update(jid, progress=72, message="Suche die besten Highlights …")
    key = load_config().get("anthropic_key", "")
    clips, source = pipeline.find_highlights(segments, key, s["clips"], s["min_len"], s["max_len"], job.get("context", ""),
                                             lambda m: log(jid, m))
    if not clips:
        raise RuntimeError("Keine passenden Highlights gefunden. Versuche eine kürzere Mindestlänge.")
    for c in clips:
        c["status"] = "wartet"
    job = update(jid, clips=clips, selector=source)
    log(jid, "%d Highlights gefunden (%s)" % (len(clips), "Claude KI" if source == "claude" else "Automatik"))

    for i in range(len(clips)):
        job = update(jid, progress=round(75 + i / len(clips) * 24, 1), message="Rendere Clip %d von %d …" % (i + 1, len(clips)))
        render_one(jid, job, i, info)
        update(jid, clips=job["clips"])
    update(jid, status="fertig", progress=100, message="Fertig – %d Clips" % len(clips), finished=time.time())


def rerender(jid, idx):
    job = update(jid, status="läuft", message="Rendere Clip %d neu …" % (idx + 1))
    render_one(jid, job, idx, job["info"])
    job["clips"][idx]["status"] = "fertig"
    update(jid, clips=job["clips"], status="fertig", progress=100, message="Fertig – %d Clips" % len(job["clips"]))


def worker():
    while True:
        kind, jid, idx = work.get()
        try:
            if not os.path.exists(os.path.join(JOBS, jid, "job.json")):
                continue
            process(jid) if kind == "process" else rerender(jid, idx)
        except Cancelled:
            pass
        except Exception as e:
            traceback.print_exc()
            try:
                update(jid, status="fehler", message=str(e)[:500])
            except Exception:
                pass


def cleanup():
    while True:
        cutoff = time.time() - RETENTION_DAYS * 86400
        for jid in os.listdir(JOBS):
            try:
                if load_job(jid)["created"] < cutoff:
                    shutil.rmtree(os.path.join(JOBS, jid), ignore_errors=True)
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
        else:
            work.put(("process", job["id"], None))
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=cleanup, daemon=True).start()
