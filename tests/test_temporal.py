"""Amber 2.9's temporal types, both ways: atoms, vectors, nulls, and the bridges.

Amber counts from 2000.01.01, NumPy and Arrow from 1970.01.01, so every one of
these is a place a silent thirty-year shift could hide. Each test pins an exact
value, not just a type.
"""
import datetime as dt

import pytest

np = pytest.importorskip("numpy")


# ---- Amber -> Python -------------------------------------------------------

@pytest.mark.parametrize("src, dtype, expected", [
    ("2026.01.01 2026.03.15", "datetime64[D]", ["2026-01-01", "2026-03-15"]),
    ("2000.01.01 1999.12.31", "datetime64[D]", ["2000-01-01", "1999-12-31"]),
    ("2026.03 2000.01m", "datetime64[M]", ["2026-03", "2000-01"]),
    ("2026.01.01D09:30:00.000000001 2000.01.01D00:00:00.000000000", "datetime64[ns]",
     ["2026-01-01T09:30:00.000000001", "2000-01-01T00:00:00.000000000"]),
])
def test_absolute_vectors_shift_the_epoch(am, src, dtype, expected):
    out = am.eval(src)
    assert out.dtype == np.dtype(dtype)
    assert list(out) == list(np.array(expected, dtype=dtype))


@pytest.mark.parametrize("src, dtype, expected", [
    ("0D01:00:00.000000000 0D00:00:00.000000005", "timedelta64[ns]", [3_600_000_000_000, 5]),
    ("09:30:00.000 00:00:00.001", "timedelta64[ms]", [34_200_000, 1]),
    ("09:30:15 00:00:01", "timedelta64[s]", [34_215, 1]),
    ("09:30 00:01", "timedelta64[m]", [570, 1]),
])
def test_relative_vectors(am, src, dtype, expected):
    out = am.eval(src)
    assert out.dtype == np.dtype(dtype)
    assert out.astype("int64").tolist() == expected


@pytest.mark.parametrize("src", ["2026.01.01 0Nd", "2026.03 0Nm", "2026.01.01D09:30 0Np",
                                 "0D01:00 0Nn", "09:30:00.000 0Nt", "09:30:15 0Nv", "09:30 0Nu"])
def test_nulls_become_nat(am, src):
    out = am.eval(src)
    assert not np.isnat(out[0])
    assert np.isnat(out[1])


def test_timespan_vectors_are_zero_copy(am):
    raw = am.eval("0D01:00 0D02:00", raw=True)
    out = am.to_python(raw)
    assert out.dtype == np.dtype("timedelta64[ns]")
    assert out.ctypes.data == raw.handle()          # the engine's own buffer


def test_raw_numbers_stay_reachable_zero_copy(am):
    raw = am.eval("2026.01.01 2026.01.02", raw=True)
    nums = np.asarray(raw)
    assert nums.ctypes.data == raw.handle()
    assert nums.tolist() == [9497, 9498]             # days since 2000.01.01


@pytest.mark.parametrize("src, expected", [
    ("2026.01.01", np.datetime64("2026-01-01")),
    ("2026.03m", np.datetime64("2026-03")),
    ("2026.01.01D09:30:00.000000007", np.datetime64("2026-01-01T09:30:00.000000007")),
    ("0D00:00:01", np.timedelta64(1_000_000_000, "ns")),
    ("09:30:00.000", np.timedelta64(34_200_000, "ms")),
    ("09:30:15", np.timedelta64(34_215, "s")),
    ("09:30", np.timedelta64(570, "m")),
])
def test_atoms(am, src, expected):
    out = am.eval(src)
    assert out == expected
    assert out.dtype == expected.dtype


@pytest.mark.parametrize("src", ["0Nd", "0Np", "0Nn", "0Nt", "0Nu", "0Nv", "0Nm"])
def test_null_atoms(am, src):
    assert np.isnat(am.eval(src))


# ---- Python -> Amber -------------------------------------------------------

@pytest.mark.parametrize("obj, type_name, text", [
    (dt.date(2026, 1, 1), "date", "2026.01.01"),
    (dt.date(1999, 12, 31), "date", "1999.12.31"),
    (dt.datetime(2026, 1, 1, 9, 30, 0, 123456), "timestamp", "2026.01.01D09:30:00.123456000"),
    (dt.timedelta(hours=1, microseconds=5), "timespan", "0D01:00:00.000005000"),
    (dt.timedelta(days=-1), "timespan", "-1D00:00:00.000000000"),
    (dt.time(9, 30, 15, 250000), "time", "09:30:15.250"),
    (np.datetime64("2026-01-01"), "date", "2026.01.01"),
    (np.datetime64("2026-03", "M"), "month", "2026.03m"),
    (np.datetime64("2026-01-01T09:30:00.000000001"), "timestamp", "2026.01.01D09:30:00.000000001"),
    (np.timedelta64(5, "ns"), "timespan", "0D00:00:00.000000005"),
])
def test_python_scalars(am, obj, type_name, text):
    v = am.convert(obj)
    assert v.type_name() == type_name
    assert v.format() == text


def test_aware_datetime_is_utc(am):
    aware = dt.datetime(2026, 1, 1, 10, 30, tzinfo=dt.timezone(dt.timedelta(hours=1)))
    assert am.convert(aware).format() == "2026.01.01D09:30:00.000000000"


@pytest.mark.parametrize("arr, type_name", [
    (np.array(["2026-01-01", "NaT"], "M8[D]"), "date vector"),
    (np.array(["2026-01", "NaT"], "M8[M]"), "month vector"),
    (np.array(["2026-01-01T09:30", "NaT"], "M8[ms]"), "timestamp vector"),
    (np.array([5, "NaT"], "m8[s]"), "timespan vector"),
])
def test_numpy_arrays(am, arr, type_name):
    v = am.convert(arr)
    assert v.type_name() == type_name
    back = am.to_python(v)
    assert np.isnat(back[1])
    assert back[0] == arr[0]


def test_list_of_dates_is_a_date_vector(am):
    v = am.convert([dt.date(2026, 1, 1), dt.date(2026, 1, 2)])
    assert v.type_name() == "date vector"


def test_roundtrip_through_a_global(am):
    stamps = np.array(["2026-01-01T09:30:00.000000001", "1990-05-17T00:00", "NaT"], "M8[ns]")
    am.set("pystamps", stamps)
    assert am.eval("`timestamp~type pystamps") == 1 or am.get("pystamps", raw=True).type_name() == "timestamp vector"
    back = am.get("pystamps")
    np.testing.assert_array_equal(back, stamps)
    assert am.eval("1+first pystamps") == np.datetime64("2026-01-01T09:30:00.000000002")


def test_dates_before_2000_and_far_future(am):
    days = np.array(["1700-01-01", "2290-12-31"], "M8[D]")
    np.testing.assert_array_equal(am.to_python(am.convert(days)), days)


# ---- pandas ---------------------------------------------------------------

def test_from_pandas_timestamps_are_amber_timestamps(am):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"ts": pd.to_datetime(["2026-01-01 09:30:00.000000005", None]),
                       "td": pd.to_timedelta(["1h", None])})
    t = am.Table.from_pandas(df)
    assert t.dtypes == {"ts": "timestamp vector", "td": "timespan vector"}
    out = t.to_pandas()
    # pandas 3 parses "1h" to microseconds; Amber's are always nanoseconds
    pd.testing.assert_series_equal(out["ts"], df["ts"].astype("datetime64[ns]"), check_names=False)
    pd.testing.assert_series_equal(out["td"], df["td"].astype("timedelta64[ns]"), check_names=False)


def test_from_pandas_tz_aware_is_utc(am):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"ts": pd.to_datetime(["2026-07-01 11:30"]).tz_localize("Europe/Amsterdam")})
    t = am.Table.from_pandas(df)
    assert t.column("ts", raw=True).format() == ",2026.07.01D09:30:00.000000000"


# ---- Arrow -----------------------------------------------------------------

def test_to_arrow_temporal_types(am):
    pa = pytest.importorskip("pyarrow")
    t = am.q("([]d:2026.01.01 0Nd;m:2026.03 2026.04m;p:2026.01.01D09:30 0Np;n:0D01:00 0Nn;"
             "t:09:30:00.000 0Nt;v:09:30:15 0Nv;u:09:30 0Nu)")
    at = t.to_arrow()
    assert at.schema.types == [pa.date32(), pa.date32(), pa.timestamp("ns"), pa.duration("ns"),
                               pa.time32("ms"), pa.time32("s"), pa.time32("s")]
    row0 = at.slice(0, 1).to_pylist()[0]
    assert row0["d"] == dt.date(2026, 1, 1)
    assert row0["m"] == dt.date(2026, 3, 1)
    assert row0["t"] == dt.time(9, 30)
    assert row0["v"] == dt.time(9, 30, 15)
    assert row0["u"] == dt.time(9, 30)
    assert at.column("p").null_count == 1 and at.column("d").null_count == 1
    assert at.column("u").null_count == 1


def test_from_arrow_temporal_types(am):
    pa = pytest.importorskip("pyarrow")
    tbl = pa.table({
        "d32": pa.array([dt.date(2026, 1, 1), None], pa.date32()),
        "d64": pa.array([dt.date(2026, 1, 2), None], pa.date64()),
        "ts_s": pa.array([dt.datetime(2026, 1, 1, 9, 30), None], pa.timestamp("s")),
        "ts_utc": pa.array([dt.datetime(2026, 1, 1, 9, 30), None], pa.timestamp("us", tz="UTC")),
        "dur": pa.array([dt.timedelta(seconds=90), None], pa.duration("ms")),
        "t32ms": pa.array([dt.time(9, 30), None], pa.time32("ms")),
        "t32s": pa.array([dt.time(9, 30, 15), None], pa.time32("s")),
        "t64": pa.array([dt.time(9, 30, 0, 1), None], pa.time64("us")),
    })
    t = am.Table.from_arrow(tbl)
    assert t.dtypes == {"d32": "date vector", "d64": "date vector", "ts_s": "timestamp vector",
                        "ts_utc": "timestamp vector", "dur": "timespan vector", "t32ms": "time vector",
                        "t32s": "second vector", "t64": "timespan vector"}
    assert t.column("d32", raw=True).format() == "2026.01.01 0N"
    assert t.column("ts_s", raw=True).format() == "2026.01.01D09:30:00.000000000 0N"
    assert t.column("dur", raw=True).format() == "0D00:01:30.000000000 0N"
    assert t.column("t32ms", raw=True).format() == "09:30:00.000 0N"
    assert t.column("t64", raw=True).format() == "0D09:30:00.000001000 0N"


def test_arrow_temporal_roundtrip(am):
    pytest.importorskip("pyarrow")
    t = am.q("([]d:2026.01.01 2026.01.02;p:2026.01.01D09:30 2026.01.02D10:00;n:0D01:00 0D00:00:00.000000001)")
    back = am.Table.from_arrow(t.to_arrow())
    am.set("rt1", t.value)
    am.set("rt2", back.value)
    assert am.eval("rt1~rt2") == 1
