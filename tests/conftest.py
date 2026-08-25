import pytest


@pytest.fixture(scope="session")
def am():
    """The engine, booted once with the standard library, for the whole session.

    Session-scoped on purpose: the engine has one global namespace and no
    teardown path, so a per-test engine is not a thing that exists. Tests that
    need isolation use their own global names.
    """
    import amber
    amber.init()
    if amber.home() is None:
        pytest.skip("no Amber standard library found (set $AMBER_HOME)")
    return amber


@pytest.fixture(scope="session")
def taq(am):
    """A generated trades/quotes universe -- the engine's own sample data."""
    am.gentq(20_000)
    return am
