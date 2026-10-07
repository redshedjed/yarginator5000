"""MVSEP guard rails, against a fake server and a fake clock (no network)."""
import json
import types

import pytest

from yarginator import mvsep

TOKEN = "SECRET-TOKEN-1234567890abcdef"
REAL_CHECK = mvsep.Client._check_upload


class Clock:
    def __init__(self):
        self.t = 1000.0
        self.sleeps = []

    def monotonic(self):
        return self.t

    def time(self):
        return 1.7e9 + self.t

    def sleep(self, d):
        self.sleeps.append(d)
        self.t += d


class Resp:
    def __init__(self, status=200, data=None, headers=None, body=b"FLACDATA"):
        self.status_code, self._data, self.headers, self._body = status, data, headers or {}, body

    def json(self):
        if self._data is None:
            raise ValueError
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, n):
        yield self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class FakeServer:
    """Records every request with the fake time it was made at."""

    def __init__(self, clock, polls_until_done=2, files=("mix_vocals.flac", "mix_drums.flac")):
        self.clock, self.calls, self.polls_until_done, self.files = clock, [], polls_until_done, files
        self.created = 0
        self.fail_create = None
        self.create_status = 200

    def request(self, method, url, **kw):
        path = url.rsplit("/api/", 1)[1]
        self.calls.append((self.clock.t, method, path, kw))
        if path == "separation/create":
            if self.fail_create:
                raise self.fail_create
            if self.create_status != 200:
                return Resp(self.create_status, {"success": False, "data": {"message": "Too many requests"}},
                            {"Retry-After": "60"})
            self.created += 1
            return Resp(200, {"success": True, "data": {"hash": f"h{self.created}", "link": "x"}})
        if path == "separation/get":
            n = sum(1 for c in self.calls if c[2] == "separation/get")
            if n < self.polls_until_done:
                return Resp(200, {"success": True, "status": "processing", "data": {}})
            return Resp(200, {"success": True, "status": "done", "data": {
                "files": [{"type": f.split("_")[-1].split(".")[0].title(), "download": f, "bytes": str(len(b"FLACDATA")),
                           "url": f"https://de2.mvsep.com/storage/processed/55/x-{f[:20]}"} for f in self.files]}})
        return Resp(200, {"success": True, "data": {}})

    def get(self, url, **kw):
        self.calls.append((self.clock.t, "GET", url, kw))
        return Resp()


@pytest.fixture
def env(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(mvsep, "time", types.SimpleNamespace(monotonic=clock.monotonic, time=clock.time,
                                                             sleep=clock.sleep))
    server = FakeServer(clock)
    audio = tmp_path / "mix.flac"
    audio.write_bytes(b"audio" * 100)
    # these tests use stand-in bytes; the real-audio checks have their own test below
    monkeypatch.setattr(mvsep.Client, "_check_upload", lambda self, a, mb: None)

    def client(**kw):
        limits = mvsep.Limits(**kw)
        return mvsep.Client(TOKEN, limits, session=server, root=tmp_path / "mvsep")
    return types.SimpleNamespace(clock=clock, server=server, audio=audio, client=client, tmp=tmp_path)


JOB = mvsep.Job(63)


def api_calls(server, path):
    return [c for c in server.calls if c[2] == path]


def test_happy_path_paced_and_cleaned_up(env):
    c = env.client()
    got = c.separate(env.audio, JOB, env.tmp / "out")
    assert [p.name for p in got] == ["mix_vocals.flac", "mix_drums.flac"] and all(p.exists() for p in got)
    times = [t for t, *_ in env.server.calls]
    assert all(b - a >= c.limits.min_interval_s - 1e-9 for a, b in zip(times, times[1:]))   # paced
    assert len(api_calls(env.server, "separation/create")) == 1
    assert len(api_calls(env.server, "separation/delete")) == 1                            # cleaned up
    assert not (env.tmp / "mvsep" / "lock").exists()
    assert env.clock.sleeps.count(c.limits.poll_start_s) >= 1                               # poll interval
    create = api_calls(env.server, "separation/create")[0][3]
    assert create["timeout"][0] == c.limits.upload_timeout_s                                # uploads get time


def test_same_work_is_never_submitted_twice(env):
    env.client().separate(env.audio, JOB, env.tmp / "out")
    env.client().separate(env.audio, JOB, env.tmp / "out")
    assert len(api_calls(env.server, "separation/create")) == 1                           # reused
    # different options are different work
    env.client().separate(env.audio, mvsep.Job(63, {"add_opt1": 1}), env.tmp / "out2")
    assert len(api_calls(env.server, "separation/create")) == 2


def test_interrupted_job_is_resumed_not_resubmitted(env):
    c = env.client()
    key_job = mvsep.Job(40)
    c2 = env.client()
    # simulate a previous run that created the job then died while waiting
    import hashlib
    key = hashlib.sha1(json.dumps([mvsep.file_sha1(env.audio), 40, [], mvsep.FLAC]).encode()).hexdigest()
    c2.ledger.add(key, "old-hash", env.clock.time(), {})
    got = c.separate(env.audio, key_job, env.tmp / "out")
    assert got and not api_calls(env.server, "separation/create")
    assert api_calls(env.server, "separation/get")[0][3]["params"] == {"hash": "old-hash"}


def test_create_is_never_retried(env):
    env.server.fail_create = ConnectionError("reset")
    with pytest.raises(mvsep.MvsepError):
        env.client().separate(env.audio, JOB, env.tmp / "out")
    assert len(api_calls(env.server, "separation/create")) == 1
    assert not (env.tmp / "mvsep" / "lock").exists()


def test_too_many_requests_stops(env):
    env.server.create_status = 429
    with pytest.raises(mvsep.MvsepError, match="too many"):
        env.client().separate(env.audio, JOB, env.tmp / "out")
    assert len(api_calls(env.server, "separation/create")) == 1


def test_caps(env):
    c = env.client(max_jobs_per_run=1)
    c.separate(env.audio, JOB, env.tmp / "out")
    other = env.tmp / "b.flac"
    other.write_bytes(b"other")
    with pytest.raises(mvsep.MvsepError, match="max_jobs_per_run"):
        c.separate(other, JOB, env.tmp / "out")
    c2 = env.client(max_jobs_per_day=1)                   # the ledger already has one job today
    with pytest.raises(mvsep.MvsepError, match="max_jobs_per_day"):
        c2.separate(other, JOB, env.tmp / "out")
    assert len(api_calls(env.server, "separation/create")) == 1


def test_upload_checks_send_nothing(env, monkeypatch):
    monkeypatch.setattr(mvsep.Client, "_check_upload", REAL_CHECK)   # the real checks, guards stay on
    sf = pytest.importorskip("soundfile")
    import numpy as np
    c = env.client(min_audio_s=5)
    for path, why in ((env.audio, "isn't a song"), ):
        with pytest.raises(mvsep.MvsepError, match=why):
            c.separate(path, JOB, env.tmp / "out")
    junk = env.tmp / "junk.flac"
    junk.write_bytes(b"x" * 100_000)
    with pytest.raises(mvsep.MvsepError, match="readable audio"):
        c.separate(junk, JOB, env.tmp / "out")
    short = env.tmp / "short.wav"
    sf.write(short, np.zeros(44100 * 2), 44100)
    with pytest.raises(mvsep.MvsepError, match="long"):
        c.separate(short, JOB, env.tmp / "out")
    with pytest.raises(mvsep.MvsepError, match="max_upload_mb"):
        env.client(max_upload_mb=0.0001).separate(short, JOB, env.tmp / "out")
    assert env.server.calls == []


def test_dry_run_sends_nothing(env):
    dry = mvsep.Client(TOKEN, mvsep.Limits(), session=env.server, root=env.tmp / "mvsep", dry_run=True)
    assert dry.separate(env.audio, JOB, env.tmp / "out") == []
    assert env.server.calls == []


def test_another_run_holding_the_lock_stops_us(env):
    lock = env.tmp / "mvsep" / "lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("123")
    with pytest.raises(mvsep.MvsepError, match="another yarginator run"):
        env.client().separate(env.audio, JOB, env.tmp / "out")
    assert env.server.calls == []


def test_token_never_leaks(env):
    c = env.client()
    assert TOKEN not in repr(c)
    env.server.fail_create = ConnectionError(f"bad url ...?api_token={TOKEN}")
    with pytest.raises(mvsep.MvsepError) as e:
        c.separate(env.audio, JOB, env.tmp / "out")
    assert TOKEN not in str(e.value) and "<token>" in str(e.value)
    assert TOKEN not in (env.tmp / "mvsep" / "ledger.json").read_text() if (env.tmp / "mvsep" / "ledger.json").exists() else True


def test_downloads_only_from_mvsep_over_https(env):
    c = env.client()
    for url in ("http://files.mvsep.com/x.flac", "https://evil.example.com/x.flac", "https://mvsep.com.evil.io/x"):
        with pytest.raises(mvsep.MvsepError, match="refusing"):
            c._download(url, env.tmp / "x.flac")
    assert c._download("https://de.mvsep.com/x.flac", env.tmp / "x.flac").exists()


def test_mvsep_is_opt_in_only(tmp_path, monkeypatch):
    from yarginator import separate
    monkeypatch.setattr(mvsep, "find_token", lambda config=None: "a-token")   # even with a token set up
    with pytest.raises(ValueError):
        separate.pick_engine("auto")
    from yarginator.project import SongProject
    from yarginator.workflow import separate_song
    proj = SongProject.init(tmp_path / "Band - Tune", workspace=True, stems={})
    called = []
    monkeypatch.setattr(separate, "separate_mvsep", lambda *a, **k: called.append(1) or {})
    monkeypatch.setattr(separate, "separate", lambda *a, **k: {})
    (proj.work / "source").mkdir()
    (proj.work / "source" / "mix.flac").write_bytes(b"")
    separate_song(proj)
    assert not called                                  # default engine: local
    separate_song(proj, engine="mvsep")
    assert called


def test_tests_cannot_reach_the_network():
    with pytest.raises(RuntimeError, match="must not"):
        mvsep.Client("tok").session


def test_result_file_names_are_bare():
    assert mvsep.file_name({"download": "mix._bs6stem_mt_0_vocals.flac", "url": "https://x"}) == "mix._bs6stem_mt_0_vocals.flac"
    assert mvsep.file_name({"download": "..\..\evil.flac"}) == "evil.flac"
    assert mvsep.file_name({"url": "https://de2.mvsep.com/a/b/c.flac"}) == "c.flac"
    with pytest.raises(mvsep.MvsepError):
        mvsep.file_name({"download": ".."})


def test_short_download_is_caught(env):
    env.server.get = lambda url, **kw: Resp(body=b"FLAC")          # truncated
    with pytest.raises(mvsep.MvsepError, match="expected"):
        env.client().separate(env.audio, JOB, env.tmp / "out")
