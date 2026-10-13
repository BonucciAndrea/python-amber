"""Property tests: whatever goes in comes back out, bit for bit."""
import math

import pytest

hypothesis = pytest.importorskip("hypothesis")
np = pytest.importorskip("numpy")
from hypothesis import given, settings, strategies as st  # noqa: E402

FAST = settings(max_examples=150, deadline=None)
INT64 = st.integers(min_value=-(2**63) + 1, max_value=2**63 - 1)


@FAST
@given(INT64)
def test_ints(am, x):
    assert am.to_python(am.convert(x)) == x


@FAST
@given(st.floats(allow_nan=False))
def test_floats(am, x):
    assert am.to_python(am.convert(x)) == x


@FAST
@given(st.lists(INT64, min_size=1, max_size=50))
def test_int_lists(am, xs):
    assert am.to_python(am.convert(xs)).tolist() == xs


@FAST
@given(st.lists(st.floats(allow_nan=True), min_size=1, max_size=50))
def test_float_arrays(am, xs):
    a = np.array(xs, dtype=np.float64)
    back = am.to_python(am.convert(a))
    np.testing.assert_array_equal(back, a)


@FAST
@given(st.text(max_size=200))
def test_strings(am, s):
    out = am.to_python(am.convert(s))
    assert (out if isinstance(out, str) else "".join(out)) == s


@FAST
@given(st.lists(st.text(alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\0"),
                        min_size=1, max_size=12), min_size=1, max_size=20))
def test_symbols(am, names):
    v = am._amber.symbols(names)
    assert v.to_symbols() == names


@FAST
@given(st.lists(INT64, min_size=0, max_size=30))
def test_pickle_and_bytes(am, xs):
    import pickle
    v = am.convert(np.array(xs, dtype=np.int64))
    assert pickle.loads(pickle.dumps(v)).equals(v)


@FAST
@given(st.lists(st.integers(min_value=-10**6, max_value=10**6), min_size=1, max_size=30))
def test_dates(am, days):
    a = np.array(days, dtype="datetime64[D]")
    np.testing.assert_array_equal(am.to_python(am.convert(a)), a)


@FAST
@given(st.lists(st.integers(min_value=-(2**62), max_value=2**62), min_size=1, max_size=30))
def test_timestamps_and_spans(am, ns):
    a = np.array(ns, dtype="datetime64[ns]")
    np.testing.assert_array_equal(am.to_python(am.convert(a)), a)
    b = np.array(ns, dtype="timedelta64[ns]")
    np.testing.assert_array_equal(am.to_python(am.convert(b)), b)


@FAST
@given(st.lists(st.tuples(INT64, st.floats(allow_nan=False, width=64), st.sampled_from(["a", "bb", "c"])),
                min_size=1, max_size=20))
def test_tables_through_arrow(am, rows):
    pytest.importorskip("pyarrow")
    cols = {"i": [r[0] for r in rows], "f": [r[1] for r in rows], "s": [r[2] for r in rows]}
    t = am.make_table(cols)
    back = am.Table.from_arrow(t.to_arrow())
    assert back["i"].tolist() == cols["i"]
    assert all(a == b or (math.isnan(a) and math.isnan(b)) for a, b in zip(back["f"].tolist(), cols["f"]))
    assert list(back["s"]) == cols["s"]
