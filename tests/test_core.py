"""Evaluation, errors, conversion and dynamic dispatch."""
import pytest


def test_version_and_abi(am):
    major, minor = (int(x) for x in am.version().split(".")[:2])
    assert (major, minor) >= (2, 9)
    assert am.abi() >= 1


def test_scalars(am):
    assert am.eval("2+3") == 5
    assert am.eval("2.5*2") == 5.0
    assert am.eval('"hello"') == "hello"
    assert am.eval("`aapl") == "aapl"


def test_vectors(am):
    import numpy as np
    v = am.eval("1 2 3 4 5")
    assert isinstance(v, np.ndarray)
    assert list(v) == [1, 2, 3, 4, 5]
    f = am.eval("1.5 2.5 3.5")
    assert f.dtype == np.float64
    assert list(am.eval("`a`b`c")) == ["a", "b", "c"]


def test_narrow_integer_storage(am):
    """Amber stores integers in the narrowest type that holds them. The binding
    surfaces that rather than hiding it behind a widening copy."""
    import numpy as np
    assert am.eval("1 2 3").dtype == np.int8
    assert am.eval("1 2 300").dtype == np.int16
    assert am.eval("1 2 3000000000").dtype == np.int64


def test_lazy_range_is_materialised(am):
    import numpy as np
    v = am.eval("!10")
    assert isinstance(v, np.ndarray)
    assert list(v) == list(range(10))


def test_errors_are_exceptions(am):
    with pytest.raises(am.AmberError) as excinfo:
        am.eval("noSuchNameAtAll+1")
    assert "noSuchNameAtAll" in str(excinfo.value)
    # ... and the engine is still usable afterwards
    assert am.eval("1+1") == 2


def test_general_list(am):
    out = am.eval('(1;"two";`three)')
    assert out[0] == 1
    assert out[1] == "two"
    assert out[2] == "three"


def test_dict(am):
    out = am.eval("`a`b`c!1 2 3")
    assert out == {"a": 1, "b": 2, "c": 3}


def test_boolean_vector_widens(am):
    out = am.eval("1 2 3 4>2")
    assert list(out) == [0, 0, 1, 1]


def test_dynamic_dispatch(am):
    am.eval("dbl:{2*x}")
    assert am.dbl(21) == 42
    assert am.dbl(1.5) == 3.0


def test_dispatch_on_dotted_names(am):
    """`arrow.export` and friends live in a namespace; the global table keys
    those on (namespace, name), so a dotted name must not be interned as one
    symbol. This is the regression test for that."""
    am.eval("myns.twice:{2*x}")
    assert am.myns.twice(7) == 14
    assert am.get("myns.twice") is not None


def test_call_arity_limit(am):
    am.eval("dbl:{2*x}")
    with pytest.raises(TypeError):
        am.call("dbl", *range(9))


def test_globals_roundtrip(am):
    import numpy as np
    am.set("pyvec", np.arange(5, dtype=np.int64))
    assert am.eval("sum pyvec") == 10
    assert list(am.get("pyvec")) == [0, 1, 2, 3, 4]


def test_set_from_python_types(am):
    am.set("s1", "text")
    assert am.eval("s1") == "text"
    am.set("f1", 2.5)
    assert am.eval("f1") == 2.5
    am.set("l1", [1, 2, 3])
    assert am.eval("sum l1") == 6


def test_set_dict_of_sequences_becomes_a_table(am):
    """`set` with a dict of equal-length sequences builds a table (via
    make_table); a dict of scalars stays an Amber dictionary."""
    am.set("book", {"sym": ["AAPL", "MSFT"], "px": [187.2, 411.1]})
    assert am.get("book", raw=True).is_table()
    assert am.eval("#book") == 2
    assert am.eval("sum book`px") == pytest.approx(598.3)

    am.set("scalars", {"a": 1, "b": 2})
    assert not am.get("scalars", raw=True).is_table()
    assert am.get("scalars") == {"a": 1, "b": 2}


def test_raw_values_are_opaque_handles(am):
    v = am.eval("1 2 3", raw=True)
    assert isinstance(v, am.Value)
    assert v.count() == 3
    assert "vector" in v.type_name()



def test_old_python_amber_alongside_is_refused(am, tmp_path):
    """amber 1.x and amber-lang share the `amber` folder: importing must say so,
    and only when it really is the same folder."""
    import os, shutil, subprocess, sys, textwrap
    pkg = os.path.dirname(am.__file__)
    shutil.copytree(pkg, tmp_path / "amber", ignore=shutil.ignore_patterns("__pycache__"))
    dist = tmp_path / "amber-1.9.6.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: amber\nVersion: 1.9.6\n")
    (dist / "RECORD").write_text("amber/__init__.py,,\n")
    code = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {str(tmp_path)!r})
        try:
            import amber
            print("imported", amber.__file__)
        except ImportError as e:
            print("refused:", e)
    """)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-s", "-c", code], capture_output=True, text=True, timeout=60, env=env)
    assert "refused:" in r.stdout and "pip uninstall -y amber amber-lang" in r.stdout, r.stdout + r.stderr
    shutil.rmtree(dist)                     # no old install: the same copy imports fine
    r = subprocess.run([sys.executable, "-s", "-c", code], capture_output=True, text=True, timeout=60, env=env)
    assert "imported" in r.stdout, r.stdout + r.stderr[-500:]
