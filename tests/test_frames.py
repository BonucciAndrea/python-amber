"""pandas / Arrow / Polars bridges, and the keyed-table mapping."""
import pytest


def test_to_pandas(taq):
    pd = pytest.importorskip("pandas")
    am = taq
    df = am.q("select from trades").to_pandas()
    assert list(df.columns) == ["sym", "time", "px", "sz"]
    assert len(df) == am.eval("#trades")
    assert df["px"].dtype.kind == "f"


def test_keyed_result_becomes_an_index(taq):
    pytest.importorskip("pandas")
    am = taq
    t = am.q("select vwap:wavg[sz;px], n:#px by sym from trades")
    assert t.key_columns == ["sym"]
    assert t.columns == ["sym", "vwap", "n"]
    df = t.to_pandas()
    assert df.index.name == "sym"
    assert list(df.columns) == ["vwap", "n"]
    flat = t.to_pandas(index=False)
    assert "sym" in flat.columns


def test_arrow_roundtrip(taq):
    pa = pytest.importorskip("pyarrow")
    am = taq
    original = am.q("select from trades")
    at = original.to_arrow()
    assert at.num_rows == len(original)
    assert at.column_names == original.columns
    back = am.Table.from_arrow(at)
    assert len(back) == len(original)
    assert back.columns == original.columns
    assert list(back["px"][:5]) == list(original["px"][:5])


def test_pandas_roundtrip(taq):
    pd = pytest.importorskip("pandas")
    am = taq
    df = am.q("select from trades").to_pandas().head(100)
    back = am.Table.from_pandas(df)
    assert len(back) == 100
    assert set(back.columns) == set(df.columns)


def test_make_table(am):
    import numpy as np
    t = am.make_table({"q": np.arange(4, dtype=np.int64),
                       "p": np.array([1.0, 2.0, 3.0, 4.0])})
    assert t.columns == ["q", "p"]
    assert len(t) == 4
    am.set("mt", t.value)
    assert am.eval("sum mt`q") == 6


def test_repr_html(taq):
    am = taq
    html = am.q("5#select from trades")._repr_html_()
    assert "<table" in html and "amber-table" in html


def test_table_indexing(taq):
    am = taq
    t = am.q("select from trades")
    assert "px" in t
    assert len(t.columns) == 4
    assert len(list(t.items())) == 4


def test_from_pandas_dtype_mapping(am):
    pd = pytest.importorskip("pandas")
    import numpy as np
    df = pd.DataFrame({
        "sym": ["AAPL", "MSFT", "AAPL"],
        "ts": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "flag": [True, False, True],
        "px": [1.5, 2.5, 3.5],
    })
    t = am.Table.from_pandas(df)
    assert t.columns == ["sym", "ts", "flag", "px"]
    assert list(t["sym"]) == ["AAPL", "MSFT", "AAPL"]     # symbol vector
    assert t["ts"].dtype == np.dtype("datetime64[ns]")     # an Amber timestamp column (2.9)
    assert t.column("ts", raw=True).type_name() == "timestamp vector"
    assert list(t["ts"]) == list(df["ts"].to_numpy())
    assert list(t["flag"]) == [1, 0, 1]
    am.set("dfin", t.value)
    assert am.eval("`AAPL=first dfin`sym") == 1


def test_from_pandas_with_index(am):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"v": [1, 2, 3]}, index=pd.Index(["a", "b", "c"], name="k"))
    t = am.Table.from_pandas(df, index=True)
    assert t.columns == ["k", "v"]


def test_arrow_export_containers_are_freed_by_the_high_level_path(taq):
    """The low-level `arrow_export` hands back two heap structs the caller owns.

    An Arrow "move" transfers contents, never containers, so importing from them
    leaves the containers behind -- 152 bytes per export, invisible in a notebook
    and fatal in a Flight server that exports a table per request. `Table.to_arrow`
    uses the caller-allocates form instead, so repeating it must not grow the
    process's allocation count without bound.
    """
    pytest.importorskip("pyarrow")
    import gc
    import tracemalloc

    am = taq
    table = am.q("select from trades")
    table.to_arrow()                       # warm every lazy import
    gc.collect()

    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    for _ in range(30):
        table.to_arrow()
    gc.collect()
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    grew = sum(stat.size_diff for stat in after.compare_to(before, "filename"))
    # 30 exports leaking two Arrow structs would be ~4.5 kB of C allocations,
    # which tracemalloc does not see -- but the ctypes structs and the PyArrow
    # objects it DOES see would grow in step with them. A flat Python-side
    # footprint is the observable half of "nothing is accumulating".
    assert grew < 2_000_000, f"to_arrow() accumulated {grew:,} bytes over 30 calls"
