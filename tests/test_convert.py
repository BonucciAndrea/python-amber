"""Python -> Amber conversion: every buffer shape, scalar edge and container."""
import array
import math

import pytest

np = pytest.importorskip("numpy")


def roundtrip(am, obj):
    return am.to_python(am.convert(obj))


# ---- buffers ---------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
                                   "float16", "float32", "float64"])
def test_numeric_dtypes_keep_their_values(am, dtype):
    if dtype == "uint8":
        pytest.skip("uint8 buffers are bytes: a char vector (see test_uint8_is_bytes)")
    a = np.array([0, 1, 2, 100], dtype=dtype)
    out = roundtrip(am, a)
    assert out.tolist() == a.astype("float64" if a.dtype.kind == "f" else "int64").tolist()


def test_uint8_is_bytes(am):
    v = am.convert(np.frombuffer(b"abc", dtype=np.uint8))
    assert v.type_name() == "char vector"
    assert am.to_python(v) == "abc"


def test_big_endian_is_byteswapped(am):
    a = np.array([1, 256, -2], dtype=">i4")
    assert roundtrip(am, a).tolist() == [1, 256, -2]
    f = np.array([1.5, -0.25], dtype=">f8")
    assert roundtrip(am, f).tolist() == [1.5, -0.25]


def test_strided_views(am):
    a = np.arange(20, dtype=np.int64)[::3]
    assert roundtrip(am, a).tolist() == list(range(0, 20, 3))
    b = np.arange(10, dtype=np.float64)[::-1]
    assert roundtrip(am, b).tolist() == list(range(9, -1, -1))


def test_unsigned_ranges(am):
    assert roundtrip(am, np.array([2**32 - 1], dtype=np.uint32)).tolist() == [2**32 - 1]
    assert roundtrip(am, np.array([2**63 - 1], dtype=np.uint64)).tolist() == [2**63 - 1]
    with pytest.raises(OverflowError):
        am.convert(np.array([2**63], dtype=np.uint64))


def test_bool_arrays(am):
    assert roundtrip(am, np.array([True, False, True])).tolist() == [1, 0, 1]


def test_two_d_is_refused(am):
    with pytest.raises(TypeError, match="1-D"):
        am.convert(np.zeros((2, 2)))


def test_empty_arrays(am):
    for dtype in ("int64", "float64", "int32"):
        v = am.convert(np.array([], dtype=dtype))
        assert v.count() == 0


def test_array_array_and_memoryview(am):
    assert roundtrip(am, array.array("q", [1, 2, 3])).tolist() == [1, 2, 3]
    assert roundtrip(am, array.array("d", [0.5])).tolist() == [0.5]
    assert roundtrip(am, memoryview(array.array("i", [7, 8]))).tolist() == [7, 8]
    assert roundtrip(am, bytearray(b"xy")) == "xy"


def test_unsupported_buffer_format(am):
    with pytest.raises(TypeError, match="no Amber equivalent"):
        am.convert(np.array([1 + 2j]))


def test_numpy_scalars(am):
    assert am.to_python(am.convert(np.int64(5))) == 5
    assert am.to_python(am.convert(np.float64(2.5))) == 2.5


# ---- scalars ---------------------------------------------------------------

def test_int_range(am):
    assert roundtrip(am, 2**63 - 1) == 2**63 - 1
    assert roundtrip(am, -(2**63) + 1) == -(2**63) + 1
    with pytest.raises(OverflowError):
        am.convert(2**63)


def test_float_specials(am):
    assert math.isnan(roundtrip(am, float("nan")))
    assert roundtrip(am, float("inf")) == float("inf")
    assert roundtrip(am, -float("inf")) == -float("inf")
    assert math.copysign(1.0, roundtrip(am, -0.0)) in (1.0, -1.0)   # either sign is a zero


def test_bools_are_ints(am):
    assert roundtrip(am, True) == 1 and roundtrip(am, False) == 0


@pytest.mark.parametrize("text", ["", "a", "héllo wörld", "日本語", "emoji 🎉", "tab\tnew\nline", "x" * 100_000])
def test_strings_roundtrip(am, text):
    assert roundtrip(am, text) == text


def test_embedded_nul(am):
    assert roundtrip(am, "a\0b") == "a\0b"


def test_bytes_are_char_vectors(am):
    assert roundtrip(am, b"raw") == "raw"


def test_none_is_generic_null(am):
    assert roundtrip(am, None) is None


# ---- containers ------------------------------------------------------------

def test_list_squeezes_to_a_vector(am):
    v = am.convert([1, 2, 3])
    assert "vector" in v.type_name()
    assert am.to_python(v).tolist() == [1, 2, 3]


def test_mixed_list_stays_general(am):
    out = roundtrip(am, [1, "two", 3.0, None])
    assert out[0] == 1 and out[1] == "two" and out[2] == 3.0 and out[3] is None


def test_nested_lists(am):
    out = roundtrip(am, [[1, 2], [3, 4, 5]])
    assert [x.tolist() for x in out] == [[1, 2], [3, 4, 5]]


def test_tuples_are_lists(am):
    assert roundtrip(am, (1, 2)).tolist() == [1, 2]


def test_dicts(am):
    assert roundtrip(am, {"a": 1, "b": 2}) == {"a": 1, "b": 2}
    assert roundtrip(am, {1: "x"}) == {"1": "x"}          # keys become symbols, via str()


def test_empty_containers(am):
    assert len(roundtrip(am, [])) == 0
    assert am.convert({}).is_dict()


def test_unconvertible_object(am):
    class Thing:
        pass
    with pytest.raises(TypeError, match="cannot convert"):
        am.convert(Thing())


def test_error_inside_a_list_leaves_nothing_behind(am):
    with pytest.raises(OverflowError):
        am.convert([1, 2, 2**70])
    assert am.eval("1+1") == 2


def test_convert_hook_errors_propagate(am):
    class Bad:
        dtype = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        am.convert(Bad())


# ---- pandas nullable --------------------------------------------------------

def test_pandas_nullable_ints_and_floats(am):
    pd = pytest.importorskip("pandas")
    v = am.convert(pd.Series([1, None, 3], dtype="Int64"))
    assert v.format() == "1 0N 3"
    f = am.to_python(am.convert(pd.Series([1.5, None], dtype="Float64")))
    assert f[0] == 1.5 and math.isnan(f[1])
    b = am.convert(pd.Series([True, None], dtype="boolean"))
    assert b.format() == "1 0N"


def test_pandas_numeric_series(am):
    pd = pytest.importorskip("pandas")
    assert roundtrip(am, pd.Series([1.0, 2.0])).tolist() == [1.0, 2.0]
    assert roundtrip(am, pd.Index([5, 6])).tolist() == [5, 6]
