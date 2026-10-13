"""Table: access, slicing, and every way in and out (pandas, Arrow, Polars)."""
import datetime as dt
import math

import pytest

np = pytest.importorskip("numpy")


@pytest.fixture()
def t(am):
    return am.q("([]sym:`a`b`a`c;px:1.5 2.5 3.5 4.5;sz:10 20 30 40;f:1010b)")


# ---- access ------------------------------------------------------------------

def test_shape_and_columns(t):
    assert t.shape == (4, 4)
    assert t.columns == ["sym", "px", "sz", "f"]
    assert list(t) == t.columns
    assert "px" in t and "nope" not in t


def test_column_by_name_and_position(t):
    assert t["px"].tolist() == [1.5, 2.5, 3.5, 4.5]
    assert t[1].tolist() == t["px"].tolist()
    assert t[-1].tolist() == t["f"].tolist()


def test_bad_column_lookups(t):
    with pytest.raises(KeyError):
        t["nope"]
    with pytest.raises(IndexError):
        t[9]
    with pytest.raises(TypeError):
        t[1.5]
    with pytest.raises(TypeError):
        t[True]


def test_head_tail(t):
    assert t.head(2)["sz"].tolist() == [10, 20]
    assert t.tail(1)["sz"].tolist() == [40]
    assert len(t.head(0)) == 0
    assert len(t.head(100)) == 100                     # take wraps, as # does in Amber


def test_dtypes(t):
    assert t.dtypes["px"] == "float vector"
    assert t.dtypes["sym"] == "symbol vector"


def test_to_dict_and_numpy(t, am):
    d = t.to_dict()
    assert list(d) == t.columns
    num = am.q("([]a:1 2;b:3 4)").to_numpy()
    assert num.shape == (2, 2) and num.tolist() == [[1, 3], [2, 4]]


def test_empty_table(am):
    e = am.q("0#([]a:`long$();b:`float$())")
    assert e.shape == (0, 2)
    assert len(e.to_pandas()) == 0
    pa = pytest.importorskip("pyarrow")
    assert e.to_arrow().num_rows == 0


def test_table_needs_a_table(am):
    with pytest.raises(TypeError):
        am.Table(am.eval("1 2 3", raw=True))
    with pytest.raises(TypeError):
        am.Table([1, 2])


def test_repr_and_html_stay_small(am):
    am.eval("bigt:([]a:til 100000)")
    t = am.get("bigt")
    assert len(repr(t)) < 5000
    assert len(t._repr_html_()) < 20000


def test_keyed_result(am):
    am.eval("kt:([]s:`a`b`a;v:1 2 3)")
    k = am.q("select sum v by s from kt")
    assert k.key_columns == ["s"]
    df = k.to_pandas()
    assert list(df.index) == ["a", "b"] and df["v"].tolist() == [4, 2]
    assert list(k.to_pandas(index=False).columns) == ["s", "v"]


# ---- pandas ------------------------------------------------------------------

def test_to_pandas_categorical(t):
    pd = pytest.importorskip("pandas")
    df = t.to_pandas(categorical=True)
    assert isinstance(df["sym"].dtype, pd.CategoricalDtype)
    assert df["sym"].tolist() == ["a", "b", "a", "c"]


def test_to_pandas_copy_is_independent(am):
    am.eval("cp:([]a:1 2 3)")
    df = am.get("cp").to_pandas(copy=True)
    am.eval("cp:0#cp")
    assert df["a"].tolist() == [1, 2, 3]


def test_from_pandas_mixed(am):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "i": [1, 2, 3],
        "f": [0.5, float("nan"), 2.0],
        "s": ["x", None, "z"],
        "c": pd.Categorical(["u", "v", "u"]),
        "n": pd.array([1, None, 3], dtype="Int64"),
        "b": [True, False, True],
    })
    t = am.Table.from_pandas(df)
    assert t.dtypes["s"] == "symbol vector" and list(t["s"]) == ["x", "", "z"]
    assert list(t["c"]) == ["u", "v", "u"]
    assert t.column("n", raw=True).format() == "1 0N 3"
    assert math.isnan(t["f"][1])


def test_from_pandas_strings_as_char(am):
    pd = pytest.importorskip("pandas")
    t = am.Table.from_pandas(pd.DataFrame({"s": ["ab", "c"]}), strings="char")
    assert t["s"][0] == "ab"
    with pytest.raises(ValueError):
        am.Table.from_pandas(pd.DataFrame({"s": ["a"]}), strings="bogus")


def test_from_pandas_index(am):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"v": [1, 2]}, index=pd.Index(["a", "b"], name="k"))
    t = am.Table.from_pandas(df, index=True)
    assert t.columns == ["k", "v"]


def test_from_pandas_duplicate_names(am):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(ValueError, match="duplicate"):
        am.Table.from_pandas(df)


def test_pandas_roundtrip(t, am):
    pytest.importorskip("pandas")
    back = am.Table.from_pandas(t.to_pandas())
    assert back.value.equals(t.value) or back.to_dict().keys() == t.to_dict().keys()
    assert back["px"].tolist() == t["px"].tolist()
    assert list(back["sym"]) == list(t["sym"])


# ---- Arrow -------------------------------------------------------------------

def test_to_arrow_types(t):
    pa = pytest.importorskip("pyarrow")
    at = t.to_arrow()
    assert at.schema.field("px").type == pa.float64()
    assert pa.types.is_integer(at.schema.field("sz").type)   # Amber stores small ints narrow
    assert at.column("sym").to_pylist() == ["a", "b", "a", "c"]


def test_to_arrow_bytes_are_signed(am):
    pa = pytest.importorskip("pyarrow")
    at = am.q("([]b:`byte$0 -1 127)").to_arrow()
    assert at.schema.field("b").type == pa.int8()
    assert at.column("b").to_pylist() == [0, -1, 127]


def test_to_arrow_mixed_column_falls_back(am):
    pytest.importorskip("pyarrow")
    at = am.q("([]s:(\"ab\";\"cde\"))").to_arrow()
    assert at.column("s").to_pylist() == ["ab", "cde"]


@pytest.mark.parametrize("typ, values, expected_type", [
    ("int8", [1, None, -3], "long vector"),
    ("int16", [1, 2], "long vector"),
    ("int32", [1, 2], "int vector"),
    ("int64", [1, None], "long vector"),
    ("uint8", [255, 0], "long vector"),
    ("uint32", [2**32 - 1], "long vector"),
    ("uint64", [2**63 - 1], "long vector"),
    ("float16", [1.5], "float vector"),
    ("float32", [1.5, None], "float vector"),
    ("float64", [1.5, None], "float vector"),
    ("bool_", [True, False], None),
    ("string", ["a", None, "c"], "symbol vector"),
    ("large_string", ["a", "b"], "symbol vector"),
])
def test_from_arrow_type_matrix(am, typ, values, expected_type):
    pa = pytest.importorskip("pyarrow")
    if typ == "float16":
        arr = pa.array(np.array(values, dtype=np.float16))
    else:
        arr = pa.array(values, type=getattr(pa, typ)())
    t = am.Table.from_arrow(pa.table({"c": arr}))
    if expected_type:
        assert t.dtypes["c"] == expected_type
    out = t["c"]
    for got, want in zip(list(out), values):
        if want is None:
            assert got in ("", None) or (isinstance(got, float) and math.isnan(got)) or got == -(2**63) \
                or got == -(2**31)
        else:
            assert got == want


def test_from_arrow_uint64_overflow(am):
    pa = pytest.importorskip("pyarrow")
    with pytest.raises(OverflowError):
        am.Table.from_arrow(pa.table({"c": pa.array([2**64 - 1], pa.uint64())}))


def test_from_arrow_bool_with_nulls(am):
    pa = pytest.importorskip("pyarrow")
    t = am.Table.from_arrow(pa.table({"b": pa.array([True, None, False])}))
    assert t.column("b", raw=True).format() == "1 0N 0"


def test_from_arrow_dictionary(am):
    pa = pytest.importorskip("pyarrow")
    arr = pa.array(["x", "y", "x"]).dictionary_encode()
    assert list(am.Table.from_arrow(pa.table({"d": arr}))["d"]) == ["x", "y", "x"]


def test_from_arrow_decimal_and_null(am):
    pa = pytest.importorskip("pyarrow")
    import decimal
    t = am.Table.from_arrow(pa.table({
        "dec": pa.array([decimal.Decimal("1.25")], pa.decimal128(5, 2)),
        "nul": pa.array([None], pa.null()),
    }))
    assert t["dec"].tolist() == [1.25]
    assert math.isnan(t["nul"][0])


def test_from_arrow_sliced_and_chunked(am):
    pa = pytest.importorskip("pyarrow")
    base = pa.table({"a": list(range(10)), "s": [str(i) for i in range(10)]})
    sliced = base.slice(3, 4)
    t = am.Table.from_arrow(sliced)
    assert t["a"].tolist() == [3, 4, 5, 6]
    assert list(t["s"]) == ["3", "4", "5", "6"]
    chunked = pa.concat_tables([base.slice(0, 2), base.slice(5, 2)])
    assert am.Table.from_arrow(chunked)["a"].tolist() == [0, 1, 5, 6]


def test_from_arrow_record_batch_and_reader(am):
    pa = pytest.importorskip("pyarrow")
    rb = pa.record_batch({"a": [1, 2]})
    assert am.Table.from_arrow(rb)["a"].tolist() == [1, 2]
    reader = pa.RecordBatchReader.from_batches(rb.schema, [rb, rb])
    assert am.Table.from_arrow(reader)["a"].tolist() == [1, 2, 1, 2]


def test_from_arrow_nested_and_binary_refused(am):
    pa = pytest.importorskip("pyarrow")
    with pytest.raises(TypeError):
        am.Table.from_arrow(pa.table({"l": pa.array([[1, 2]])}))
    with pytest.raises(TypeError):
        am.Table.from_arrow(pa.table({"b": pa.array([b"\xff\xfe"], pa.binary())}))


def test_from_arrow_garbage(am):
    pytest.importorskip("pyarrow")
    with pytest.raises(TypeError):
        am.Table.from_arrow(42)


def test_arrow_roundtrip_is_exact(t, am):
    pytest.importorskip("pyarrow")
    back = am.Table.from_arrow(t.to_arrow())
    assert back["px"].tolist() == t["px"].tolist()
    assert list(back["sym"]) == list(t["sym"])
    assert back["sz"].tolist() == t["sz"].tolist()


def test_to_arrow_outlives_the_table(am):
    pa = pytest.importorskip("pyarrow")
    import gc
    at = am.q("([]x:til 1000)").to_arrow()
    gc.collect()
    am.eval("junk:til 100000")
    assert at.column("x").to_pylist()[-1] == 999


# ---- Polars ------------------------------------------------------------------

def test_polars_roundtrip(t, am):
    pl = pytest.importorskip("polars")
    pytest.importorskip("pyarrow")
    df = t.to_polars()
    assert isinstance(df, pl.DataFrame)
    assert df["px"].to_list() == [1.5, 2.5, 3.5, 4.5]
    back = am.Table.from_polars(df)
    assert list(back["sym"]) == ["a", "b", "a", "c"]


def test_polars_temporal(am):
    pl = pytest.importorskip("polars")
    pytest.importorskip("pyarrow")
    df = pl.DataFrame({"d": [dt.date(2026, 1, 1)], "ts": [dt.datetime(2026, 1, 1, 9, 30)]})
    t = am.Table.from_polars(df)
    assert t.dtypes == {"d": "date vector", "ts": "timestamp vector"}
    assert t.column("d", raw=True).format() == ",2026.01.01"


# ---- from_dict / make_table --------------------------------------------------

def test_make_table(am):
    t = am.make_table({"a": [1, 2], "b": ["x", "y"], "c": np.array([0.5, 1.5])})
    assert t.shape == (2, 3)
    with pytest.raises(TypeError):
        am.make_table([("a", [1])])


def test_make_table_ragged_columns(am):
    with pytest.raises((am.AmberError, ValueError)):
        am.make_table({"a": [1, 2], "b": [1]})
