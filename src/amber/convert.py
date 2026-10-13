"""Amber value <-> Python object.

The rule this module follows is: *convert the shape, alias the bytes.*

A table becomes a :class:`amber.Table`, a dictionary becomes a ``dict``, a
symbol vector becomes a list of ``str`` -- those are all structural conversions
and they copy, because there is no way not to. But a numeric column becomes a
NumPy array that points straight at the engine's own memory, with the Amber
value pinned as the array's ``base``. Nothing is copied and nothing can be freed
underneath it.

Integer width is surfaced, not hidden. Amber stores an integer vector in the
*narrowest* type that holds every element, so ``1 2 3`` is physically ``int8``
and ``1 2 3000000000`` is ``int64`` -- both are "long vectors" as far as the
language is concerned. The arrays you get back carry the real dtype;
``arr.astype('int64')`` is one call away when you want uniformity.

Temporal values (2.9) map to NumPy's own types:

=============  ==========================  ===================================
Amber          NumPy                       note
=============  ==========================  ===================================
date           ``datetime64[D]``           days since 2000.01.01, shifted
month          ``datetime64[M]``           months since 2000.01, shifted
timestamp      ``datetime64[ns]``          ns since 2000.01.01, shifted
timespan       ``timedelta64[ns]``         **zero-copy** (same numbers)
time           ``timedelta64[ms]``         a time of day, as its offset
second         ``timedelta64[s]``          a time of day, as its offset
minute         ``timedelta64[m]``          a time of day, as its offset
=============  ==========================  ===================================

Amber counts dates and timestamps from 2000.01.01 and NumPy from 1970.01.01,
so those columns are a copy with the epoch shifted; a null (``0Nd``, ``0Np``,
...) becomes ``NaT``. The raw numbers are always available zero-copy as
``numpy.asarray(value)`` on the :class:`~amber.Value`. Without NumPy installed,
temporal values come back as ``datetime`` objects (microsecond precision).

The other direction -- Python to Amber -- is the C converter in
``py_amber.c``; it hands anything date- or time-like to :func:`_to_amber_hook`
below, which builds the matching typed Amber value.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from . import _amber
from ._amber import Value

#: Amber element type -> NumPy dtype string, for the plain numeric vectors.
VECTOR_DTYPES = {
    _amber.AMBER_T_BYTE:  "int8",
    _amber.AMBER_T_SHORT: "int16",
    _amber.AMBER_T_INT:   "int32",
    _amber.AMBER_T_LONG:  "int64",
    _amber.AMBER_T_FLOAT: "float64",
}

#: Amber epoch offsets, in each kind's own unit, from 1970-01-01.
EPOCH_DAYS = 10957                       # 2000-01-01 - 1970-01-01
EPOCH_MONTHS = 360                       # 2000-01 - 1970-01
EPOCH_NS = 946_684_800_000_000_000       # 2000-01-01T00:00:00 in Unix ns

_INT32_NULL = -(2 ** 31)
_INT64_NULL = -(2 ** 63)

# Amber temporal kind -> (NumPy dtype, epoch shift in that unit, raw width)
_TEMPORAL_VECTORS = {
    _amber.AMBER_T_DATE_LIST:      ("datetime64[D]",   EPOCH_DAYS,   32),
    _amber.AMBER_T_MONTH_LIST:     ("datetime64[M]",   EPOCH_MONTHS, 32),
    _amber.AMBER_T_TIMESTAMP_LIST: ("datetime64[ns]",  EPOCH_NS,     64),
    _amber.AMBER_T_TIMESPAN_LIST:  ("timedelta64[ns]", 0,            64),
    _amber.AMBER_T_TIME_LIST:      ("timedelta64[ms]", 0,            32),
    _amber.AMBER_T_SECOND_LIST:    ("timedelta64[s]",  0,            32),
    _amber.AMBER_T_MINUTE_LIST:    ("timedelta64[m]",  0,            32),
}
_TEMPORAL_ATOMS = {
    _amber.AMBER_T_DATE:      _amber.AMBER_T_DATE_LIST,
    _amber.AMBER_T_MONTH:     _amber.AMBER_T_MONTH_LIST,
    _amber.AMBER_T_TIMESTAMP: _amber.AMBER_T_TIMESTAMP_LIST,
    _amber.AMBER_T_TIMESPAN:  _amber.AMBER_T_TIMESPAN_LIST,
    _amber.AMBER_T_TIME:      _amber.AMBER_T_TIME_LIST,
    _amber.AMBER_T_SECOND:    _amber.AMBER_T_SECOND_LIST,
    _amber.AMBER_T_MINUTE:    _amber.AMBER_T_MINUTE_LIST,
}
#: every temporal type code, atom or vector
TEMPORAL_TYPES = frozenset(_TEMPORAL_VECTORS) | frozenset(_TEMPORAL_ATOMS)

_PLAIN_ATOMS = {
    _amber.AMBER_T_INT_ATOM, _amber.AMBER_T_LONG_ATOM, _amber.AMBER_T_FLOAT_ATOM,
    _amber.AMBER_T_CHAR_ATOM, _amber.AMBER_T_SYM_ATOM,
}


def _numpy():
    try:
        import numpy  # noqa: PLC0415
        return numpy
    except ImportError:
        return None


# --------------------------------------------------------------------------
# Amber -> Python
# --------------------------------------------------------------------------

def as_array(value: Value, *, copy: bool = False):
    """A NumPy view of a flat Amber vector -- **zero-copy** unless ``copy``.

    The returned array's ``base`` is the Amber value, so the engine cannot
    reclaim the payload while the array is alive. For a temporal vector this is
    the *raw* numbers (days, ns, ...); :func:`to_python` gives the
    ``datetime64`` form. Raises :class:`TypeError` for a value that has no flat
    payload (a table, a general list, an atom).
    """
    np = _numpy()
    if np is None:
        raise ImportError("amber: NumPy is required for as_array(); "
                          "use memoryview(value) or list(memoryview(value)) instead")
    value = value.flatten()
    try:
        arr = np.asarray(value)          # PEP 3118 -- no copy, `value` becomes base
    except BufferError as exc:
        raise TypeError(str(exc)) from None
    return arr.copy() if copy else arr


def _temporal_array(value: Value, kind: int):
    """A temporal vector as datetime64/timedelta64, nulls as NaT."""
    np = _numpy()
    dtype, shift, bits = _TEMPORAL_VECTORS[kind]
    raw = np.asarray(value)                      # zero-copy raw numbers
    if bits == 64 and shift == 0:
        return raw.view(dtype)                   # timespan: the same numbers, NaT included
    wide = raw.astype("int64")                   # a copy, and room for NaT
    if bits == 32:
        nulls = raw == _INT32_NULL
        if shift:
            wide += shift
        wide[nulls] = _INT64_NULL
    else:
        nulls = raw == _INT64_NULL
        wide += shift
        wide[nulls] = _INT64_NULL
    return wide.view(dtype)


def _temporal_scalar(kind: int, raw: int):
    """A temporal atom from its raw number: NumPy's scalar when NumPy is here."""
    vec = _TEMPORAL_ATOMS[kind]
    dtype, shift, bits = _TEMPORAL_VECTORS[vec]
    null = _INT32_NULL if bits == 32 else _INT64_NULL
    np = _numpy()
    if np is not None:
        if raw == null:
            return np.datetime64("NaT") if dtype.startswith("datetime") else np.timedelta64("NaT")
        unit = dtype[dtype.index("[") + 1:-1]
        cls = np.datetime64 if dtype.startswith("datetime") else np.timedelta64
        return cls(raw + shift, unit)
    if raw == null:
        return None
    if kind == _amber.AMBER_T_DATE:
        return _dt.date(2000, 1, 1) + _dt.timedelta(days=raw)
    if kind == _amber.AMBER_T_MONTH:
        y, m = divmod(raw, 12)
        return _dt.date(2000 + y, m + 1, 1)
    if kind == _amber.AMBER_T_TIMESTAMP:
        return _dt.datetime(2000, 1, 1) + _dt.timedelta(microseconds=raw // 1000)
    unit = {_amber.AMBER_T_TIMESPAN: 1e-3, _amber.AMBER_T_TIME: 1e3,
            _amber.AMBER_T_SECOND: 1e6, _amber.AMBER_T_MINUTE: 6e7}[kind]
    return _dt.timedelta(microseconds=raw * unit)


def _temporal_list_no_numpy(value: Value, kind: int):
    raw = list(memoryview(value))
    atom = next(a for a, v in _TEMPORAL_ATOMS.items() if v == kind)
    return [_temporal_scalar(atom, r) for r in raw]


def to_python(value: Value | None) -> Any:
    """Convert an Amber value to the natural Python object.

    ``None`` in, ``None`` out. A function value converts to itself: there is no
    Python object that means "the Amber verb ``+``", and inventing one would be
    worse than handing back the handle.
    """
    if value is None:
        return None
    if value.handle() == _amber.NULL_HANDLE:
        # The generic null `::`. This is what an assignment evaluates to, and
        # what any statement with no value evaluates to, so a REPL or a notebook
        # that renders it prints noise after every `t:1`. None is the honest
        # Python equivalent, and callers already treat None as "nothing to show".
        return None
    from .table import Table          # local import: table.py imports this module

    kind = value.type()

    if kind == _amber.AMBER_T_TABLE:
        return Table(value)

    if kind == _amber.AMBER_T_DICT:
        keys_value = value.keys()
        if keys_value.is_table():
            # A KEYED table -- what `select ... by sym from t` returns. Amber
            # models it as a dictionary from a key table to a value table, which
            # is exactly right for the language and exactly wrong as a Python
            # dict: the keys are rows, not scalars. Flatten it to an ordinary
            # table with the key columns in front, and remember which they were
            # so `.to_pandas()` can put them back on the index.
            flat = _amber.call("pyamber.unkey", (value,))
            key_names = to_python(_amber.call("pyamber.keycols", (value,)))
            if isinstance(key_names, str):
                key_names = [key_names]
            return Table(flat, key_columns=list(key_names))
        keys = to_python(keys_value)
        if isinstance(keys, (str, bytes)) or not hasattr(keys, "__len__"):
            keys = [keys]             # a single-key dict decodes its key to a scalar
        vals = value.values()
        if vals.type() == _amber.AMBER_T_LIST:
            items = [to_python(v) for v in vals.items()]
        else:
            # A uniformly-typed value vector: split it elementwise, so each key
            # maps to its own scalar rather than to one array shared by all.
            converted = to_python(vals)
            items = list(converted) if not isinstance(converted, str) else list(converted)
        if hasattr(keys, "tolist"):
            keys = keys.tolist()
        return dict(zip(keys, items))

    if kind in (_amber.AMBER_T_CHAR, _amber.AMBER_T_CHAR_ATOM):
        return value.to_str()

    if kind in (_amber.AMBER_T_SYM, _amber.AMBER_T_SYM_ATOM):
        return value.to_symbols()

    if kind in _TEMPORAL_ATOMS:
        return _temporal_scalar(kind, value.scalar())

    if kind in _TEMPORAL_VECTORS:
        if _numpy() is None:
            return _temporal_list_no_numpy(value, kind)
        return _temporal_array(value, kind)

    if kind in _PLAIN_ATOMS:
        return value.scalar()

    if kind == _amber.AMBER_T_RANGE:
        return to_python(value.flatten())

    if kind in VECTOR_DTYPES:
        np = _numpy()
        if np is not None:
            return np.asarray(value)          # zero-copy
        return list(memoryview(value))        # NumPy-free fallback

    if kind == _amber.AMBER_T_LIST:
        return [to_python(item) for item in value.items()]

    if kind == _amber.AMBER_T_BOOL:
        # The one vector type with no PEP 3118 representation: it is genuinely
        # bit-packed, one bit per element. Widening happens inside the engine
        # (`pyamber.widen`), and the result comes back as NumPy booleans.
        widened = to_python(_amber.call("pyamber.widen", (value,)))
        np = _numpy()
        return widened.astype(bool) if np is not None else [bool(x) for x in widened]

    # Functions, projections, compiled closures: hand back the handle unchanged.
    return value


# --------------------------------------------------------------------------
# Python -> Amber: the temporal hook the C converter calls
# --------------------------------------------------------------------------

def _mk(helper: str, raw):
    """Raw numbers (a Python int, or an int64 array) -> a typed Amber value."""
    return _amber.call(helper, (raw,))


def _ns_since_2000(dt: _dt.datetime) -> int:
    if dt.tzinfo is not None:
        dt = dt.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    delta = dt - _dt.datetime(2000, 1, 1)
    return ((delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds) * 1000 \
        + int(getattr(dt, "nanosecond", 0))


def _timedelta_ns(td: _dt.timedelta) -> int:
    return ((td.days * 86400 + td.seconds) * 1_000_000 + td.microseconds) * 1000 \
        + int(getattr(td, "nanoseconds", 0))


def _datetime64_array(np, arr):
    """A datetime64/timedelta64 array -> a typed Amber vector (NaT -> 0N)."""
    unit = np.datetime_data(arr.dtype)[0]
    if arr.dtype.kind == "M":
        if unit == "D":
            raw = arr.astype("int64") - EPOCH_DAYS
            helper = "pyamber.mkd"
        elif unit in ("M", "Y"):
            months = arr.astype("datetime64[M]")
            raw = months.astype("int64") - EPOCH_MONTHS
            helper = "pyamber.mkm"
            arr = months
        else:
            arr = arr.astype("datetime64[ns]")
            raw = arr.astype("int64") - EPOCH_NS
            helper = "pyamber.mkp"
    else:
        arr = arr.astype("timedelta64[ns]")
        raw = arr.astype("int64")
        helper = "pyamber.mkn"
    nat = np.isnat(arr)
    if nat.any():
        raw = raw.copy()
        raw[nat] = _INT64_NULL
    return _mk(helper, np.ascontiguousarray(raw, dtype="int64"))


def _to_amber_hook(obj):
    """What the C converter calls for an object it has no rule for.

    Returns an :class:`amber.Value` for anything date- or time-like, or ``None``
    to let the C side carry on (sequence, mapping, error). Kept cheap for the
    objects it declines: a couple of isinstance checks and an attribute probe.
    """
    if isinstance(obj, _dt.datetime):            # before date: datetime is a date
        if getattr(obj, "value", None) is not None and type(obj).__name__ == "NaTType":
            return _mk("pyamber.mkp", _INT64_NULL)
        return _mk("pyamber.mkp", _ns_since_2000(obj))
    if isinstance(obj, _dt.date):
        return _mk("pyamber.mkd", (obj - _dt.date(2000, 1, 1)).days)
    if isinstance(obj, _dt.timedelta):
        return _mk("pyamber.mkn", _timedelta_ns(obj))
    if isinstance(obj, _dt.time):
        ms = ((obj.hour * 60 + obj.minute) * 60 + obj.second) * 1000 + obj.microsecond // 1000
        return _mk("pyamber.mkt", ms)
    if type(obj).__name__ == "NaTType":          # pandas.NaT
        return _mk("pyamber.mkp", _INT64_NULL)
    dtype = getattr(obj, "dtype", None)
    kind = getattr(dtype, "kind", None)
    if type(obj).__name__ == "NAType":           # pandas.NA: Amber's long null
        return _amber.convert(_INT64_NULL)
    if kind in ("b", "i", "u", "f") and getattr(obj, "ndim", None) == 0 and hasattr(obj, "item"):
        return _amber.convert(obj.item())        # a NumPy scalar (np.int32(5), np.float32(...))
    if dtype is not None and hasattr(obj, "to_numpy") and kind not in ("M", "m"):
        np = _numpy()
        name = str(dtype)
        if np is not None and not isinstance(obj, np.ndarray):
            # pandas: a nullable Int/UInt/Float/boolean column holds pd.NA, which
            # to_numpy() would make an object array of; give NA Amber's nulls
            # (0N for the integer kinds, 0n for floats) and convert the numbers.
            if name.startswith(("Int", "UInt")) or name == "boolean":
                return _amber.convert(obj.to_numpy(dtype="int64", na_value=_INT64_NULL))
            if name.startswith("Float"):
                return _amber.convert(obj.to_numpy(dtype="float64", na_value=np.nan))
            if name == "category":
                return _amber.symbols(["" if v is None else str(v) for v in obj.astype(object).tolist()])
            if kind in ("b", "i", "u", "f"):
                return _amber.convert(np.ascontiguousarray(obj.to_numpy()))
    if kind in ("M", "m"):
        np = _numpy()
        if np is None:
            return None
        if hasattr(obj, "to_numpy") and not isinstance(obj, np.ndarray):
            tz = getattr(dtype, "tz", None)
            if tz is not None:                   # a tz-aware pandas column: as UTC
                obj = obj.dt.tz_convert("UTC").dt.tz_localize(None) if hasattr(obj, "dt") \
                    else obj.tz_convert("UTC").tz_localize(None)
            obj = obj.to_numpy()
        arr = np.asarray(obj)
        if arr.ndim == 0:                        # a datetime64/timedelta64 scalar
            out = _datetime64_array(np, arr.reshape(1))
            return _amber.call("pyamber.at", (out, 0))
        if arr.ndim != 1:
            raise TypeError("amber: only 1-D datetime arrays convert to an Amber vector")
        return _datetime64_array(np, arr)
    if dtype is not None and str(dtype) == "datetime64[ns, UTC]":   # pragma: no cover
        return None
    return None
