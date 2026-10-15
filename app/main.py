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
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

import integrations
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
    "clips": 5, "min_len": 20, "max_len": 60, "mode": "auto", "quality": "ausgewogen",
    "highlight": "#FFE600", "uppercase": True, "font_size": 78, "position": "unten", "words_per_line": 3,
    "remove_pauses": True, "hook": True, "hook_seconds": 3, "review": False,
}
# settings stored in config.json; secrets are never sent back to the browser
CONFIG_TEXT = ("vocabulary", "nc_url", "nc_user", "nc_folder", "pz_url")
CONFIG_SECRET = ("anthropic_key", "nc_pass", "pz_key")
CONFIG_DEFAULTS = {"nc_url": "https://cloud.paetze.digital", "nc_folder": "ShortCut", "nc_export": False,
                   "pz_url": "https://postiz.srv2011660.hstgr.cloud/api/public/v1", "vocabulary": ""}

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
lock = threading.Lock()
work = queue.Queue()


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
           "defaults": {**DEFAULTS, **cfg.get("defaults", {})}, "nc_export": bool(cfg.get("nc_export"))}
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
        if "nc_export" in body:
            cfg["nc_export"] = bool(body["nc_export"])
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
    if out["mode"] not in ("auto", "crop", "blur"):
        out["mode"] = "auto"
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
    if with_text and job.get("clips"):
        segments = load_transcript(job["id"])
        if segments:
            out["clips"] = [dict(c, text=" ".join(w["w"] for w in pipeline.clip_words(segments, c["start"], c["end"])))
                            for c in job["clips"]]
    return out


def new_job(name, settings, context, source=None, remote=None, size=0):
    jid = uuid.uuid4().hex[:12]
    os.makedirs(os.path.join(JOBS, jid))
    job = {
        "id": jid, "name": name, "created": time.time(), "status": "wartet", "progress": 0,
        "message": "In der Warteschlange", "settings": settings, "context": context, "source": source,
        "remote": remote, "size": size, "clips": [], "log": [], "selector": None,
    }
    save_job(job)
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
    job = new_job(filename, clean_settings(body.get("settings")), str(body.get("context") or "")[:1000], src, None, size)
    os.replace(part, os.path.join(JOBS, job["id"], src))
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
    if job["status"] not in ("fertig", "fehler", "prüfen"):
        job["cancel"] = True
        save_job(job)
    shutil.rmtree(d, ignore_errors=True)
    return {"ok": True}


@app.post("/api/jobs/{jid}/clips/{idx}")
async def edit_clip(jid: str, idx: int, request: Request):
    """Change timing, titles or transcript of a clip; renders it again unless the job is still in review."""
    require_auth(request)
    body = await request.json()
    with lock:
        job = load_job(jid)
        if job["status"] not in ("fertig", "fehler", "prüfen") or idx >= len(job["clips"]):
            raise HTTPException(400, "Clip kann gerade nicht bearbeitet werden")
        clip = job["clips"][idx]
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
        for k in ("title", "hook_text", "caption"):
            if k in body:
                clip[k] = str(body[k])[:2000]
        if "style" in body:
            job["settings"] = clean_settings({**job["settings"], **body["style"]})
        if job["status"] != "prüfen":
            clip["status"] = "wartet"
            job["status"] = "wartet"
            job["message"] = "Clip %d wird neu gerendert" % (idx + 1)
            work.put(("render", jid, idx))
        save_job(job)
    return public_job(job, with_text=True)


@app.post("/api/jobs/{jid}/render")
def render_all(jid: str, request: Request):
    """Leave review mode and render every clip."""
    require_auth(request)
    with lock:
        job = load_job(jid)
        if job["status"] != "prüfen":
            raise HTTPException(400, "Projekt ist nicht im Prüfmodus")
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


def render_one(jid, idx, info):
    """Render one clip: subtitles, speaker framing, pause cuts and the hook text."""
    d = os.path.join(JOBS, jid)
    job = load_job(jid)
    clip = job["clips"][idx]
    s = job["settings"]
    segments = load_transcript(jid)
    version = clip.get("version", 0) + 1
    name = "clip%02d" % (idx + 1) + ("_v%d" % version if version > 1 else "")
    duration = clip["end"] - clip["start"]
    words = pipeline.clip_words(segments, clip["start"], clip["end"])
    intervals = pipeline.keep_intervals(words, duration, s.get("remove_pauses", True))
    out_words = pipeline.remap_words(words, intervals)
    out_len = sum(b - a for a, b in intervals)
    ass = name + ".ass"
    burn = pipeline.write_subtitles(out_words, os.path.join(d, ass), os.path.join(d, name + ".srt"), s,
                                    clip.get("hook_text", clip.get("title", "")), out_len)
    crop_segs = None
    if s.get("mode") == "auto":
        crop_segs = pipeline.track_speaker(os.path.join(d, job["source"]), info, clip["start"], clip["end"])
    out_len = pipeline.render_clip(os.path.join(d, job["source"]), info, clip["start"], clip["end"], d, name,
                                   s["mode"], ass, burn, intervals, crop_segs)
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
    with lock:
        job = load_job(jid)
        job["clips"][idx].update({
            "file": name + ".mp4", "thumb": name + ".jpg", "srt": name + ".srt", "version": version,
            "status": "fertig", "duration_out": round(out_len, 1), "speakers": len({x for _, _, x in crop_segs or []}),
        })
        save_job(job)
    export_clip(jid, idx)


def export_clip(jid, idx):
    cfg = load_config()
    if not cfg.get("nc_export"):
        return
    job = load_job(jid)
    clip = job["clips"][idx]
    try:
        nc = integrations.Nextcloud(cfg)
        folder = "Shorts/" + re.sub(r"[^\w\- ]+", "_", os.path.splitext(job["name"])[0])[:60].strip()
        nc.mkdirs(folder)
        for k in ("file", "srt"):
            nc.upload(folder + "/" + clip[k], os.path.join(JOBS, jid, clip[k]))
        log(jid, "Clip %d in Nextcloud gespeichert (%s/%s)" % (idx + 1, nc.folder, folder))
    except Exception as e:
        log(jid, "Nextcloud-Export fehlgeschlagen: " + str(e)[:200])


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


def process(jid):
    d = os.path.join(JOBS, jid)
    job = update(jid, status="läuft", progress=1, message="Analysiere Video …", started=time.time())
    if not job.get("source") or not os.path.exists(os.path.join(d, job["source"])):
        job = fetch_source(jid, job)
    src = os.path.join(d, job["source"])
    info = pipeline.probe(src)
    job = update(jid, info=info)
    log(jid, "Video: %dx%d, %s" % (info["width"], info["height"], pipeline.fmt_time(info["duration"])))

    segments = load_transcript(jid)
    if segments is None:
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
        os.remove(wav)
        save_transcript(jid, segments)
        log(jid, "Transkript: %d Abschnitte" % len(segments))
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
        c["status"] = "entwurf" if s.get("review") else "wartet"
        c["hook_text"] = c.get("title", "")
    update(jid, clips=clips, selector=source)
    log(jid, "%d Highlights gefunden (%s)" % (len(clips), "Claude KI" if source == "claude" else "Automatik"))
    if s.get("review"):
        update(jid, status="prüfen", progress=75, message="Bitte Clips prüfen und dann rendern")
        return
    render_clips(jid, info)


def render_clips(jid, info):
    job = load_job(jid)
    n = len(job["clips"])
    for i in range(n):
        update(jid, status="läuft", progress=round(75 + i / n * 24, 1), message="Rendere Clip %d von %d …" % (i + 1, n))
        render_one(jid, i, info)
    update(jid, status="fertig", progress=100, message="Fertig – %d Clips" % n, finished=time.time())


def rerender(jid, idx):
    job = update(jid, status="läuft", message="Rendere Clip %d neu …" % (idx + 1))
    render_one(jid, idx, job["info"])
    update(jid, status="fertig", progress=100, message="Fertig – %d Clips" % len(job["clips"]))


def worker():
    while True:
        kind, jid, idx = work.get()
        try:
            if not os.path.exists(os.path.join(JOBS, jid, "job.json")):
                continue
            if kind == "process":
                process(jid)
            elif kind == "render_all":
                render_clips(jid, load_job(jid)["info"])
            else:
                rerender(jid, idx)
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
        elif job.get("info") and job["clips"]:
            work.put(("render_all", job["id"], None))
        else:
            work.put(("process", job["id"], None))
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=cleanup, daemon=True).start()
