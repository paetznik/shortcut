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
    "fmt_4x5": False, "fmt_1x1": False, "cover": True,
}
# settings stored in config.json; secrets are never sent back to the browser
CONFIG_TEXT = ("vocabulary", "nc_url", "nc_user", "nc_folder", "pz_url")
CONFIG_SECRET = ("anthropic_key", "nc_pass", "pz_key")
CONFIG_DEFAULTS = {"nc_url": "https://cloud.paetze.digital", "nc_folder": "ShortCut", "nc_export": False, "nc_watch": False,
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
    if out["mode"] not in ("auto", "crop", "blur"):
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
    if with_text and job.get("clips"):
        segments = load_transcript(job["id"])
        if segments:
            words = pipeline.all_words(segments)
            out["clips"] = []
            for c in job["clips"]:
                cw = [w for w in words if w["e"] > c["start"] and w["s"] < c["end"]]
                out["clips"].append(dict(c, text=" ".join(w["w"] for w in cw), words=[[w["w"], w["s"]] for w in cw]))
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
        copy = {k: v for k, v in src.items() if k not in ("file", "thumb", "srt", "cover", "formats", "version", "postiz", "translations")}
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
    if not re.fullmatch(r"(clip\d{1,2}(_v\d+)?(_4x5|_1x1|_cover|_[a-z]{2})?\.(mp4|jpg|srt)|bestof_\d+\.(mp4|jpg))", name):
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


def prepare(jid, idx, info):
    """Everything a render of this clip needs: cut intervals, words on the output timeline, zooms, framing, music."""
    job = load_job(jid)
    clip = job["clips"][idx]
    s = dict(job["settings"])
    src = os.path.join(JOBS, jid, job["source"])
    duration = clip["end"] - clip["start"]
    words = pipeline.clip_words(load_transcript(jid), clip["start"], clip["end"])
    removed = [r - clip["start"] for r in clip.get("removed", [])]
    intervals = pipeline.keep_intervals(words, duration, s.get("remove_pauses", True), removed)
    out_words = pipeline.remap_words(words, intervals)
    out_len = sum(b - a for a, b in intervals)
    crop_segs = manual_crop(clip, info)
    if crop_segs:
        s["mode"] = "auto"  # hand-set framing always crops, whatever the project mode
    elif s.get("mode") == "auto":
        crop_segs = pipeline.track_speaker(src, info, clip["start"], clip["end"])
    music = os.path.join(MUSIC, s["music"]) if s.get("music") and os.path.exists(os.path.join(MUSIC, s["music"])) else None
    return {"job": job, "clip": clip, "s": s, "src": src, "duration": duration, "intervals": intervals,
            "out_words": out_words, "out_len": out_len, "crop_segs": crop_segs, "music": music,
            "zooms": pipeline.zoom_spans(out_words, intervals, out_len) if s.get("zoom", True) else None}


def render_variant(jid, p, base, words, hook, keywords, size, srt=True):
    d = os.path.join(JOBS, jid)
    clip = p["clip"]
    ass = base + ".ass"
    burn = pipeline.write_subtitles(words, os.path.join(d, ass), os.path.join(d, base + ".srt") if srt else None,
                                    p["s"], hook, p["out_len"], keywords, size)
    out_len = pipeline.render_clip(p["src"], p["job"]["info"], clip["start"], clip["end"], d, base, p["s"], ass, burn,
                                   p["intervals"], p["crop_segs"], size, p["zooms"], p["music"])
    os.remove(os.path.join(d, ass))
    return out_len


def render_one(jid, idx, info):
    """Render one clip: speaker framing, cuts (pauses, fillers, struck-out words), zooms, subtitles, hook, audio,
    extra formats and the cover image. Translations of an older version are dropped."""
    d = os.path.join(JOBS, jid)
    p = prepare(jid, idx, info)
    clip, s = p["clip"], p["s"]
    version = clip.get("version", 0) + 1
    name = "clip%02d" % (idx + 1) + ("_v%d" % version if version > 1 else "")
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
        pipeline.render_cover(p["src"], info, clip["start"], min(p["duration"] / 2, 1.5), d, name + "_cover", s,
                              p["crop_segs"], clip.get("cover_text") or clip.get("title", ""))
        cover = name + "_cover.jpg"

    keep = {name + ".mp4", name + ".jpg", name + ".srt", cover} | set(outputs.values())
    for old in clip_files(clip):
        if old not in keep:
            try:
                os.remove(os.path.join(d, old))
            except OSError:
                pass
    with lock:
        job = load_job(jid)
        job["clips"][idx].update({
            "file": outputs["9x16"], "thumb": name + ".jpg", "srt": name + ".srt", "cover": cover,
            "formats": {k: v for k, v in outputs.items() if k != "9x16"}, "version": version, "status": "fertig",
            "duration_out": round(out_len, 1), "speakers": len({x for _, _, x in p["crop_segs"] or []}),
            "translations": {},
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
        out_words = pipeline.remap_words(words, intervals)
        out_len = sum(y - x for x, y in intervals)
        crop = manual_crop(clip, info) or (pipeline.track_speaker(os.path.join(d, job["source"]), info, a, b)
                                           if s.get("mode") == "auto" else None)
        ps = dict(s, hook=bool(n == 0 and s.get("hook", True)), mode="auto" if crop else s.get("mode"))
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
        c["cover_text"] = c.get("title", "")
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


def extra_task(jid, kind, arg):
    """Translation or best-of on a finished project; a failure is reported but leaves the clips usable."""
    update(jid, status="läuft", progress=50)
    n = len(load_job(jid)["clips"])
    try:
        if kind == "translate":
            render_translation(jid, arg[0], arg[1])
            msg = "Fertig – Übersetzung von Clip %d erstellt" % (arg[0] + 1)
        else:
            render_bestof(jid, arg)
            msg = "Fertig – Best-of-Trailer erstellt"
    except Cancelled:
        raise
    except Exception as e:
        traceback.print_exc()
        log(jid, ("Übersetzung" if kind == "translate" else "Best-of") + " fehlgeschlagen: " + str(e)[:300])
        msg = "Fertig – %d Clips (%s fehlgeschlagen: %s)" % (n, "Übersetzung" if kind == "translate" else "Best-of", str(e)[:150])
    update(jid, status="fertig", progress=100, message=msg)


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
            elif kind in ("translate", "bestof"):
                extra_task(jid, kind, idx)
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
    threading.Thread(target=watcher, daemon=True).start()
