import pytest


@pytest.fixture(autouse=True)
def no_user_config(tmp_path, monkeypatch):
    """Keep the developer's ~/.yarginator.toml out of tests."""
    monkeypatch.setenv("YARGINATOR_CONFIG", str(tmp_path / "no-user-config.toml"))
