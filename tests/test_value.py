"""The Value handle: lifetime, release, buffers, equality, pickling."""
import gc
import pickle

import pytest

np = pytest.importorskip("numpy")


def test_release_frees_and_is_idempotent(am):
    v = am.eval("1 2 3", raw=True)
    v.release()
    assert v.is_released()
    v.release()                                        # twice is fine
    assert repr(v) == "<amber.Value released>"
    assert v.handle() == 0


@pytest.mark.parametrize("method", ["type", "type_name", "count", "is_table", "flatten",
                                    "to_symbols", "scalar", "ncols", "keys", "values", "items",
                                    "format", "to_bytes"])
def test_released_value_refuses_everything(am, method):
    v = am.eval("1 2 3", raw=True)
    v.release()
    with pytest.raises(ValueError, match="released"):
        getattr(v, method)()


def test_released_value_has_no_buffer(am):
    v = am.eval("1 2 3", raw=True)
    v.release()
    with pytest.raises(BufferError):
        memoryview(v)


def test_release_refused_while_a_view_lives(am):
    v = am.eval("1.0 2.0", raw=True)
    a = np.asarray(v)
    with pytest.raises(BufferError, match="export"):
        v.release()
    assert a.tolist() == [1.0, 2.0]                    # still readable
    del a
    gc.collect()
    v.release()


def test_every_buffer_request_is_counted(am):
    v = am.eval("1 2 3", raw=True)
    m1 = memoryview(v)
    m2 = memoryview(v)
    m1.release()
    with pytest.raises(BufferError):
        v.release()                                    # m2 still open
    m2.release()
    v.release()


def test_writable_buffer_refused(am):
    v = am.eval("1 2 3", raw=True)
    import ctypes
    with pytest.raises((BufferError, TypeError)):
        (ctypes.c_byte * 3).from_buffer(v)             # asks for a writable buffer


def test_buffer_layout(am):
    v = am.eval("1.5 2.5 3.5", raw=True)
    m = memoryview(v)
    assert m.format == "d" and m.itemsize == 8 and m.ndim == 1
    assert m.shape == (3,) and m.strides == (8,) and m.readonly
    assert m.nbytes == 24


def test_symbol_buffer_is_ids(am):
    v = am.eval("`a`b`a", raw=True)
    ids = np.asarray(v)
    assert ids.dtype == np.int32
    assert ids[0] == ids[2] != ids[1]
    assert [am._amber.symbol_name(int(i)) for i in ids] == ["a", "b", "a"]


def test_symbol_name_refuses_negative(am):
    with pytest.raises(ValueError):
        am._amber.symbol_name(-1)


def test_symbol_codes(am):
    codes, cats = am.eval("`b`a`b`c`a", raw=True).symbol_codes()
    assert np.frombuffer(codes, dtype=np.int32).tolist() == [0, 1, 0, 2, 1]
    assert cats == ["b", "a", "c"]
    codes, cats = am.eval("0#`", raw=True).symbol_codes()
    assert codes == b"" and cats == []


def test_large_symbol_column_decodes_with_shared_strings(am):
    am.eval("bigsyms:1000000#`aapl`msft`goog")
    out = am.get("bigsyms")
    assert len(out) == 1_000_000 and out[999_999] == "aapl"
    assert out[0] is out[3]                            # one str object per distinct name


def test_equals(am):
    a = am.eval("([]x:1 2;y:`a`b)", raw=True)
    b = am.eval("([]x:1 2;y:`a`b)", raw=True)
    c = am.eval("([]x:1 2;y:`a`c)", raw=True)
    assert a.equals(b) and not a.equals(c)
    with pytest.raises(TypeError):
        a.equals(5)


@pytest.mark.parametrize("src", ["42", "1 2 3", "3.5 0n 0w", "\"text\"", "`a`b`c", "`a`b!1 2",
                                 "([]a:1 2;b:`x`y;c:2026.01.01 2026.01.02)", "(1;\"x\";`y;2.5)",
                                 "2026.01.01D09:30 0Np", "0#0", "()"])
def test_pickle_roundtrip(am, src):
    v = am.eval(src, raw=True)
    back = pickle.loads(pickle.dumps(v))
    assert isinstance(back, am.Value)
    assert back.equals(v)


def test_from_bytes_rejects_garbage(am):
    with pytest.raises(am.AmberError):
        am._amber.from_bytes(b"not amber at all")
    with pytest.raises(am.AmberError):
        am._amber.from_bytes(b"")


def test_len_and_count(am):
    assert len(am.eval("1 2 3 4", raw=True)) == 4
    assert len(am.eval("([]a:1 2 3)", raw=True)) == 3


def test_keys_on_a_non_dict(am):
    with pytest.raises(TypeError, match="not a dictionary"):
        am.eval("1 2", raw=True).keys()


def test_column_index_bounds(am):
    t = am.eval("([]a:1 2)", raw=True)
    with pytest.raises(IndexError):
        t.column(5)
    with pytest.raises(IndexError):
        t.column(-1)
    with pytest.raises(IndexError):
        t.colname(2**40)


def test_function_values_come_back_as_handles(am):
    f = am.eval("{x+1}")
    assert isinstance(f, am.Value)
    assert am.call("pyamber.at", f, 41) == 42           # and it can be passed back in
