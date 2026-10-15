"""Connections to Nextcloud (WebDAV) and Postiz (public API)."""
import os
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote, unquote

import httpx

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi")


# ---------------------------------------------------------------- Nextcloud

class Nextcloud:
    def __init__(self, cfg):
        self.base = (cfg.get("nc_url") or "").rstrip("/")
        self.user = cfg.get("nc_user") or ""
        self.auth = (self.user, cfg.get("nc_pass") or "")
        self.folder = (cfg.get("nc_folder") or "ShortCut").strip("/")
        if not (self.base and self.user and cfg.get("nc_pass")):
            raise RuntimeError("Nextcloud ist nicht eingerichtet (Einstellungen → Nextcloud).")

    def url(self, path=""):
        parts = [p for p in (self.folder + "/" + path).split("/") if p]
        return "%s/remote.php/dav/files/%s/%s" % (self.base, quote(self.user), "/".join(quote(p) for p in parts))

    def check(self, r):
        if r.status_code == 401:
            raise RuntimeError("Nextcloud: Benutzername oder App-Passwort falsch.")
        if r.status_code == 404:
            raise RuntimeError("Nextcloud: Ordner „%s“ nicht gefunden." % self.folder)
        if r.status_code >= 400:
            raise RuntimeError("Nextcloud-Fehler %d" % r.status_code)
        return r

    def list_videos(self):
        body = ('<?xml version="1.0"?><d:propfind xmlns:d="DAV:"><d:prop><d:getcontentlength/>'
                '<d:getlastmodified/><d:resourcetype/></d:prop></d:propfind>')
        r = self.check(httpx.request("PROPFIND", self.url() + "/", auth=self.auth, headers={"Depth": "1"},
                                     content=body, timeout=30))
        ns = {"d": "DAV:"}
        out = []
        for resp in ET.fromstring(r.content).findall("d:response", ns):
            href = unquote(resp.findtext("d:href", "", ns))
            prop = resp.find("d:propstat/d:prop", ns)
            if prop is None or prop.find("d:resourcetype/d:collection", ns) is not None:
                continue
            name = href.rstrip("/").split("/")[-1]
            if not name.lower().endswith(VIDEO_EXT):
                continue
            out.append({"name": name, "size": int(prop.findtext("d:getcontentlength", "0", ns) or 0),
                        "modified": prop.findtext("d:getlastmodified", "", ns)})
        return sorted(out, key=lambda f: f["name"].lower())

    def download(self, name, dest, progress):
        with httpx.stream("GET", self.url(name), auth=self.auth, timeout=httpx.Timeout(60, read=300)) as r:
            self.check(r)
            total = int(r.headers.get("content-length") or 0)
            done, last = 0, 0
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total and time.time() - last > 2:
                        last = time.time()
                        progress(done / total * 100, "Lade aus Nextcloud … %d %%" % (done / total * 100))

    def mkdirs(self, path):
        cur = ""
        for part in [p for p in path.split("/") if p]:
            cur += "/" + part
            r = httpx.request("MKCOL", self.url(cur), auth=self.auth, timeout=30)
            if r.status_code not in (201, 405):  # 405 = exists
                self.check(r)

    def upload(self, path, local):
        with open(local, "rb") as f:
            self.check(httpx.put(self.url(path), auth=self.auth, content=f, timeout=httpx.Timeout(60, write=600)))


# ---------------------------------------------------------------- Postiz

class Postiz:
    def __init__(self, cfg):
        self.base = (cfg.get("pz_url") or "").rstrip("/")
        self.key = cfg.get("pz_key") or ""
        if not (self.base and self.key):
            raise RuntimeError("Postiz ist nicht eingerichtet (Einstellungen → Postiz).")

    def req(self, method, path, **kw):
        r = httpx.request(method, self.base + path, headers={"Authorization": self.key}, timeout=300, **kw)
        if r.status_code in (401, 403):
            raise RuntimeError("Postiz: API-Schlüssel ungültig.")
        if r.status_code >= 400:
            raise RuntimeError("Postiz-Fehler %d: %s" % (r.status_code, r.text[:300]))
        return r.json() if r.content else {}

    def integrations(self):
        out = []
        for i in self.req("GET", "/integrations"):
            if i.get("disabled"):
                continue
            out.append({"id": i.get("id"), "name": i.get("name") or i.get("identifier"),
                        "provider": i.get("identifier") or i.get("providerIdentifier") or "",
                        "picture": i.get("picture") or ""})
        return out

    @staticmethod
    def settings_for(provider, title):
        p = provider or ""
        if p == "youtube":
            return {"__type": p, "title": title[:100], "type": "public"}
        if p == "tiktok":
            return {"__type": p, "title": title[:90], "privacy_level": "PUBLIC_TO_EVERYONE", "duet": False,
                    "stitch": False, "comment": True, "autoAddMusic": "no", "brand_content_toggle": False,
                    "brand_organic_toggle": False, "content_posting_method": "DIRECT_POST"}
        if p.startswith("instagram"):
            return {"__type": p, "post_type": "post", "collaborators": []}
        if p == "x":
            return {"__type": p, "who_can_reply_post": "everyone"}
        if p.startswith("linkedin"):
            return {"__type": p, "post_as_images_carousel": False}
        return {"__type": p}

    def post(self, video_path, content, title, channels, date_iso, now=False):
        with open(video_path, "rb") as f:
            media = self.req("POST", "/upload", files={"file": (os.path.basename(video_path), f, "video/mp4")})
        image = [{"id": media.get("id"), "path": media.get("path")}]
        body = {
            "type": "now" if now else "schedule",
            "date": date_iso,
            "shortLink": False,
            "tags": [],
            "posts": [{"integration": {"id": c["id"]}, "value": [{"content": content, "image": image}],
                       "settings": self.settings_for(c.get("provider"), title)} for c in channels],
        }
        return self.req("POST", "/posts", json=body)
