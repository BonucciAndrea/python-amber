"""Evaluation, errors, conversion and dynamic dispatch."""
import pytest


def test_version_and_abi(am):
    assert am.version().startswith("1.")
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
    assert "value" in str(excinfo.value)
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


def test_raw_values_are_opaque_handles(am):
    v = am.eval("1 2 3", raw=True)
    assert isinstance(v, am.Value)
    assert v.count() == 3
    assert "vector" in v.type_name()
