"""The claim this package is built around: results do not get copied.

These tests do not check that the numbers match -- that would pass just as well
against a copy. They check that the ADDRESS the engine handed out is the address
NumPy and Arrow are reading from. An Amber value's handle IS its payload
pointer, which makes the comparison exact rather than circumstantial.
"""
import pytest

np = pytest.importorskip("numpy")


def test_numpy_view_aliases_engine_memory(taq):
    am = taq
    t = am.q("select from trades")
    raw = t.column("px", raw=True)
    arr = np.asarray(raw)
    assert arr.ctypes.data == raw.handle(), "NumPy array is not aliasing the engine buffer"
    assert arr.dtype == np.float64
    assert arr.nbytes == len(t) * 8


def test_view_keeps_the_engine_value_alive(taq):
    """The dangerous case: the Python handle goes out of scope while an array
    made from it is still live. The buffer protocol pins the Amber refcount, so
    the payload must stay readable."""
    am = taq
    arr = np.asarray(am.eval("100000#1.5", raw=True))
    import gc
    gc.collect()
    assert arr[0] == 1.5
    assert arr[-1] == 1.5
    assert float(arr.sum()) == pytest.approx(150000.0)


def test_memoryview_is_readonly(am):
    v = am.eval("1.0 2.0 3.0", raw=True)
    mv = memoryview(v)
    assert mv.readonly
    with pytest.raises(TypeError):
        mv[0] = 9.0


def test_table_has_no_buffer(am):
    v = am.eval("([]a:1 2 3)", raw=True)
    with pytest.raises(BufferError) as excinfo:
        memoryview(v)
    # The hint must name methods that actually exist on Value, not the
    # nonexistent .to_list()/.columns() the message used to suggest.
    msg = str(excinfo.value)
    assert ".to_list()" not in msg and ".columns()" not in msg
    for method in ("items", "to_str", "scalar"):
        assert hasattr(v, method)


def test_arrow_buffer_aliases_engine_memory(taq):
    pa = pytest.importorskip("pyarrow")
    am = taq
    t = am.q("select from trades")
    raw = t.column("px", raw=True)
    at = t.to_arrow()
    buf = at.column("px").chunk(0).buffers()[1]
    assert buf.address == raw.handle(), "Arrow buffer is not aliasing the engine buffer"


def test_symbol_columns_are_materialised_not_aliased(taq):
    """Symbols are the one column type that MUST copy: Amber stores interned
    int32 ids, Arrow wants utf8 offsets and bytes. Assert the values survive."""
    pytest.importorskip("pyarrow")
    am = taq
    at = am.q("select from trades").to_arrow()
    syms = at.column("sym").chunk(0).to_pylist()[:5]
    assert all(isinstance(s, str) and s for s in syms)
