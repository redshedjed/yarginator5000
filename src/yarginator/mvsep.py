"""MVSEP (mvsep.com) API client, guard rails first.

The rules, all enforced here rather than left to callers:

- One job at a time, across processes: a lock file (~/.yarginator/mvsep/lock) makes a second
  `yarginator` wait/stop instead of submitting in parallel (free accounts get one concurrent job
  anyway).
- Never submit the same work twice: every job is keyed by (file content hash, algorithm, options,
  format) in a ledger (~/.yarginator/mvsep/ledger.json). A finished job's files are reused; a job
  left running by an interrupted run is resumed (polled), not resubmitted.
- Caps: ``max_jobs_per_run`` and ``max_jobs_per_day`` (rolling 24 h, from the ledger). Hitting one
  stops with an error; it never queues more.
- Pace: at least ``min_interval_s`` between any two requests; polling starts at ``poll_start_s`` and
  backs off to ``poll_max_s``; a job is abandoned after ``job_timeout_s``.
- Retries only where they're safe: status polls and downloads (idempotent), a few times with
  backoff. Creating a job is never retried automatically (a lost response could mean it was created).
  HTTP 429 / "too many" stops the run and reports the wait the server asked for.
- Uploads are checked before anything is sent: under ``max_upload_mb``, at least ``min_upload_kb``,
  and (when an audio library is available) decodable audio at least ``min_audio_s`` long. Downloads only from mvsep.com
  hosts over HTTPS.
- The API token comes from the MVSEP_API_TOKEN environment variable (or the Windows user
  environment, or ``[mvsep] api_token`` in ~/.yarginator.toml). It's never logged, printed or put in
  an exception message.
- Clean up: after downloading, the job is deleted on the server. Ctrl+C cancels a job that hasn't
  started processing (MVSEP refunds it).
- ``dry_run`` reports what would be sent and makes no request at all.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

log = logging.getLogger(__name__)

API = "https://mvsep.com/api"
ALLOWED_HOSTS = ("mvsep.com",)            # and its subdomains (mirrors, file servers)
FLAC = 2                                  # output_format: 2 = flac 16-bit
USER_AGENT = "yarginator (YARG charting tool)"


class MvsepError(RuntimeError):
    pass


@dataclass
class Limits:
    min_interval_s: float = 3.0
    poll_start_s: float = 15.0
    poll_max_s: float = 60.0
    poll_backoff: float = 1.5
    job_timeout_s: float = 3600.0
    max_jobs_per_run: int = 4
    max_jobs_per_day: int = 20
    max_upload_mb: float = 150.0
    min_upload_kb: float = 64.0           # anything smaller isn't a song
    min_audio_s: float = 5.0
    max_retries: int = 3
    upload_timeout_s: float = 600.0       # sending a 30-40 MB FLAC takes a while on home upload speeds
    lock_wait_s: float = 0.0              # 0 = don't wait for another run's lock, stop

    @classmethod
    def from_config(cls, cfg: dict) -> "Limits":
        known = {k: type(getattr(cls, k)) for k in cls.__dataclass_fields__}
        return cls(**{k: known[k](v) for k, v in cfg.items() if k in known})


def home() -> Path:
    return Path.home() / ".yarginator" / "mvsep"


def find_token(config: dict | None = None) -> str | None:
    tok = os.environ.get("MVSEP_API_TOKEN")
    if not tok and sys.platform == "win32":  # set in Windows but this process started before
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                tok = winreg.QueryValueEx(k, "MVSEP_API_TOKEN")[0]
        except OSError:
            tok = None
    if not tok and config:
        tok = config.get("api_token")
    return tok.strip() if tok else None


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- process lock ----------------------------------------------------------------------------------
class Lock:
    def __init__(self, path: Path, stale_s: float, wait_s: float = 0.0):
        self.path, self.stale_s, self.wait_s = path, stale_s, wait_s

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.wait_s
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()} {time.time():.0f}".encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age > self.stale_s:
                    log.warning("mvsep: removing a stale lock (%.0f min old)", age / 60)
                    self.path.unlink(missing_ok=True)
                    continue
                if time.monotonic() >= deadline:
                    raise MvsepError(f"another yarginator run is using MVSEP (lock {self.path}); "
                                     "wait for it, or delete the lock if no run is active")
                time.sleep(5)

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


# --- ledger ----------------------------------------------------------------------------------------
class Ledger:
    """Every job ever created: key, server hash, when, status, and the files it produced."""

    def __init__(self, path: Path):
        self.path = path
        self.jobs: list[dict] = json.loads(path.read_text("utf-8")) if path.exists() else []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.jobs, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def find(self, key: str) -> dict | None:
        return next((j for j in reversed(self.jobs) if j["key"] == key and j["status"] not in ("failed", "cancelled")), None)

    def created_since(self, seconds: float, now: float) -> int:
        return sum(1 for j in self.jobs if now - j["created_at"] < seconds)

    def add(self, key: str, job_hash: str, now: float, info: dict) -> dict:
        j = {"key": key, "hash": job_hash, "created_at": now, "status": "waiting", "files": [], **info}
        self.jobs.append(j)
        self.save()
        return j


def file_name(entry: dict) -> str:
    """A result file's name: MVSEP gives it as "download" (e.g. "mix._bs6stem_mt_0_vocals.flac");
    reduced to a bare file name so nothing from the server can write outside the folder."""
    raw = entry.get("download") or entry.get("name") or urlparse(entry.get("url", "")).path
    name = Path(str(raw).replace("\\", "/")).name
    if not name or name in (".", ".."):
        raise MvsepError("mvsep: a result file has no usable name")
    return name


# --- client ----------------------------------------------------------------------------------------
@dataclass
class Job:
    sep_type: int
    options: dict = field(default_factory=dict)   # add_opt1 / add_opt2 / add_opt3
    output_format: int = FLAC


class Client:
    def __init__(self, token: str | None, limits: Limits | None = None, session=None,
                 root: Path | None = None, dry_run: bool = False):
        self._token = token
        self.limits = limits or Limits()
        self.dry_run = dry_run
        self.root = root or home()
        self.ledger = Ledger(self.root / "ledger.json")
        self._session = session
        self._last = 0.0
        self.jobs_this_run = 0

    def __repr__(self) -> str:                    # never show the token
        return f"Client(token={'set' if self._token else 'missing'}, dry_run={self.dry_run})"

    # -- plumbing --
    @property
    def session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers["User-Agent"] = USER_AGENT
        return self._session

    def _scrub(self, text: str) -> str:
        return text.replace(self._token, "<token>") if self._token else text

    def _pace(self) -> None:
        wait = self._last + self.limits.min_interval_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def _request(self, method: str, path: str, *, retry: bool, **kw):
        if self.dry_run:
            raise MvsepError("dry run: no requests")
        url = f"{API}/{path}"
        attempts = self.limits.max_retries if retry else 1
        delay = 5.0
        for attempt in range(1, attempts + 1):
            self._pace()
            # requests applies the first timeout to sending the body as well as connecting
            timeout = (self.limits.upload_timeout_s, 300) if "files" in kw else (15, 300)
            try:
                r = self.session.request(method, url, timeout=timeout, **kw)
            except Exception as e:  # network trouble: retry only idempotent calls
                if attempt < attempts:
                    log.warning("mvsep: %s %s failed (%s); retrying in %.0f s", method, path,
                                self._scrub(type(e).__name__), delay)
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise MvsepError(f"mvsep: {method} {path} failed: {self._scrub(str(e))}") from None
            if r.status_code == 429:
                raise MvsepError(f"mvsep: the server says too many requests (retry after "
                                 f"{r.headers.get('Retry-After', 'a while')}); stopping")
            if r.status_code >= 500 and attempt < attempts:
                log.warning("mvsep: %s %s -> HTTP %s; retrying in %.0f s", method, path, r.status_code, delay)
                time.sleep(delay)
                delay *= 2
                continue
            try:
                data = r.json()
            except ValueError:
                raise MvsepError(f"mvsep: {method} {path} -> HTTP {r.status_code}, not JSON") from None
            if r.status_code >= 400 or data.get("success") is False:
                msg = (data.get("data") or {}).get("message") or data.get("message") or f"HTTP {r.status_code}"
                if "too many" in str(msg).lower():
                    raise MvsepError(f"mvsep: {self._scrub(str(msg))}; stopping")
                raise MvsepError(f"mvsep: {method} {path}: {self._scrub(str(msg))}")
            return data
        raise MvsepError(f"mvsep: {method} {path} kept failing")

    # -- api --
    def _create(self, audio: Path, job: Job) -> str:
        if not self._token:
            raise MvsepError("no MVSEP API token: set the MVSEP_API_TOKEN environment variable")
        data = {"api_token": self._token, "sep_type": str(job.sep_type), "output_format": str(job.output_format),
                **{k: str(v) for k, v in job.options.items()}}
        with open(audio, "rb") as f:
            res = self._request("POST", "separation/create", retry=False, data=data,
                                files={"audiofile": (audio.name, f)})
        h = (res.get("data") or {}).get("hash")
        if not h:
            raise MvsepError("mvsep: create returned no job hash")
        return h

    def _status(self, job_hash: str) -> dict:
        return self._request("GET", "separation/get", retry=True, params={"hash": job_hash})

    def _delete(self, job_hash: str) -> None:
        try:
            self._request("POST", "separation/delete", retry=False, data={"hash": job_hash, "api_token": self._token})
        except MvsepError as e:
            log.info("mvsep: couldn't delete job on the server (%s)", e)

    def _cancel(self, job_hash: str) -> None:
        try:
            self._request("POST", "separation/cancel", retry=False, data={"hash": job_hash, "api_token": self._token})
            log.info("mvsep: cancelled the job")
        except MvsepError as e:
            log.info("mvsep: couldn't cancel (%s)", e)

    def _download(self, url: str, dest: Path) -> Path:
        u = urlparse(url)
        if u.scheme != "https" or not any(u.hostname == h or (u.hostname or "").endswith("." + h) for h in ALLOWED_HOSTS):
            raise MvsepError(f"mvsep: refusing to download from {u.scheme}://{u.hostname}")
        delay = 5.0
        for attempt in range(1, self.limits.max_retries + 1):
            self._pace()
            try:
                with self.session.get(url, stream=True, timeout=(15, 300)) as r:
                    r.raise_for_status()
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                    tmp.replace(dest)
                    return dest
            except Exception as e:
                if attempt == self.limits.max_retries:
                    raise MvsepError(f"mvsep: download failed: {self._scrub(str(e))}") from None
                log.warning("mvsep: download failed; retrying in %.0f s", delay)
                time.sleep(delay)
                delay *= 2
        raise MvsepError("mvsep: download failed")

    def _check_upload(self, audio: Path, size_mb: float) -> None:
        if size_mb > self.limits.max_upload_mb:
            raise MvsepError(f"mvsep: {audio.name} is {size_mb:.0f} MB, over max_upload_mb "
                             f"({self.limits.max_upload_mb:.0f}); not uploading")
        if size_mb * 1000 < self.limits.min_upload_kb:
            raise MvsepError(f"mvsep: {audio.name} is only {size_mb * 1000:.0f} KB; that isn't a song, not uploading")
        try:
            import soundfile
        except ImportError:
            return
        try:
            seconds = soundfile.info(str(audio)).duration
        except Exception:
            raise MvsepError(f"mvsep: {audio.name} isn't readable audio; not uploading") from None
        if seconds < self.limits.min_audio_s:
            raise MvsepError(f"mvsep: {audio.name} is {seconds:.1f} s long; not uploading")

    # -- the one entry point --
    def separate(self, audio: Path, job: Job, out_dir: Path) -> list[Path]:
        """Run ``job`` on ``audio`` (or reuse / resume an identical one) and download its files to
        ``out_dir``. Returns the downloaded paths, named as MVSEP names them."""
        audio = Path(audio)
        size_mb = audio.stat().st_size / 1e6
        key = hashlib.sha1(json.dumps([file_sha1(audio), job.sep_type, sorted(job.options.items()),
                                       job.output_format]).encode()).hexdigest()
        existing = self.ledger.find(key)
        if existing and existing["status"] == "downloaded" and all(Path(p).exists() for p in existing["files"]):
            log.info("mvsep: already done (job %s), reusing its files", existing["hash"][:8])
            return [Path(p) for p in existing["files"]]
        if self.dry_run:
            what = "resume" if existing else "submit"
            log.info("mvsep (dry run): would %s sep_type %s %s on %s (%.1f MB)", what, job.sep_type,
                     job.options, audio.name, size_mb)
            return []
        self._check_upload(audio, size_mb)
        with Lock(self.root / "lock", stale_s=self.limits.job_timeout_s + 600, wait_s=self.limits.lock_wait_s):
            self.ledger = Ledger(self.ledger.path)           # re-read: another run may have added jobs
            existing = self.ledger.find(key)
            if existing and existing["status"] in ("waiting", "processing", "done", "distributing", "merging"):
                entry = existing
                log.info("mvsep: resuming job %s from an earlier run", entry["hash"][:8])
            else:
                now = time.time()
                if self.jobs_this_run >= self.limits.max_jobs_per_run:
                    raise MvsepError(f"mvsep: max_jobs_per_run ({self.limits.max_jobs_per_run}) reached; stopping")
                if self.ledger.created_since(86400, now) >= self.limits.max_jobs_per_day:
                    raise MvsepError(f"mvsep: max_jobs_per_day ({self.limits.max_jobs_per_day}) reached; stopping")
                log.info("mvsep: uploading %s (%.1f MB) for sep_type %s", audio.name, size_mb, job.sep_type)
                job_hash = self._create(audio, job)
                self.jobs_this_run += 1
                entry = self.ledger.add(key, job_hash, now, {"audio": str(audio), "sep_type": job.sep_type,
                                                            "options": job.options})
            files = self._wait(entry)
            out_dir.mkdir(parents=True, exist_ok=True)
            got = []
            for f in files:
                dest = out_dir / file_name(f)
                self._download(f["url"], dest)
                want = str(f.get("bytes") or "")
                if want.isdigit() and dest.stat().st_size != int(want):
                    raise MvsepError(f"mvsep: {dest.name} downloaded {dest.stat().st_size} bytes, expected {want}; "
                                     "run again to retry the download")
                got.append(dest)
            entry["status"], entry["files"] = "downloaded", [str(p) for p in got]
            self.ledger.save()
            self._delete(entry["hash"])
            return got

    def _wait(self, entry: dict) -> list[dict]:
        delay = self.limits.poll_start_s
        start = time.monotonic()
        try:
            while True:
                res = self._status(entry["hash"])
                status = res.get("status")
                if status != entry["status"]:
                    entry["status"] = status
                    self.ledger.save()
                data = res.get("data") or {}
                if status == "done":
                    files = data.get("files") or []
                    if not files:
                        raise MvsepError("mvsep: job finished with no files")
                    return files
                if status == "failed":
                    raise MvsepError(f"mvsep: job failed: {self._scrub(str(data.get('message', '')))}")
                where = f" (queue position {data.get('current_order')})" if status == "waiting" and data.get("current_order") else ""
                log.info("mvsep: %s%s", status, where)
                if time.monotonic() - start > self.limits.job_timeout_s:
                    raise MvsepError(f"mvsep: job not done after {self.limits.job_timeout_s / 60:.0f} min; "
                                     "run again later to resume it")
                time.sleep(delay)
                delay = min(self.limits.poll_max_s, delay * self.limits.poll_backoff)
        except KeyboardInterrupt:
            if entry["status"] == "waiting":
                self._cancel(entry["hash"])
                entry["status"] = "cancelled"
                self.ledger.save()
            raise
