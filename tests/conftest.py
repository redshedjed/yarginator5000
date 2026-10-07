import pytest


@pytest.fixture(autouse=True)
def no_user_config(tmp_path, monkeypatch):
    """Keep the developer's ~/.yarginator.toml out of tests."""
    monkeypatch.setenv("YARGINATOR_CONFIG", str(tmp_path / "no-user-config.toml"))


@pytest.fixture(autouse=True)
def no_mvsep(tmp_path, monkeypatch):
    """Tests must never reach mvsep.com: no token (not even the Windows user environment's), no real
    HTTP session, and the ledger/lock in a temp folder. Fakes pass their own session explicitly."""
    from yarginator import mvsep
    monkeypatch.delenv("MVSEP_API_TOKEN", raising=False)
    monkeypatch.setattr(mvsep, "find_token", lambda config=None: None)
    monkeypatch.setattr(mvsep, "home", lambda: tmp_path / "mvsep-home")

    def blocked(self):
        if self._session is None:
            raise RuntimeError("tests must not make real MVSEP requests")
        return self._session
    monkeypatch.setattr(mvsep.Client, "session", property(blocked))
