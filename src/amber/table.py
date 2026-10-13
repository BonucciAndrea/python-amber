"""``amber.Table`` -- an Amber table, and the bridges out of it.

A ``Table`` owns an engine handle and nothing else. Columns are fetched lazily
and, for numeric types, aliased rather than copied: ``t["px"]`` is a NumPy view
of the engine's own buffer.

Three exits are provided, in decreasing order of how much they preserve:

``to_arrow()``    the Apache Arrow C Data Interface. Numeric columns are
                  genuinely zero-copy -- the ``ArrowArray``'s data buffer *is*
                  the Amber column's payload, and the release callback drops the
                  Amber refcount when Arrow is done. Symbol columns are
                  materialised to utf8, which cannot be avoided: Amber stores
                  interned int32 ids, Arrow wants offsets and bytes. Temporal
                  columns arrive as Arrow's own date/timestamp/time/duration
                  types (a copy, for the epoch shift), and anything the engine's
                  exporter has no layout for (strings held as char vectors,
                  nested lists) is converted on the Python side.

``to_pandas()``   a DataFrame built from those same views. Whether pandas keeps
                  the views or consolidates them into fresh blocks is pandas'
                  decision and it varies by version, so this is documented as
                  "zero-copy where pandas permits" rather than promised.

``to_dict()``     plain Python. Always copies. Useful for JSON, and for tests
                  that should not depend on NumPy being installed.

The way in -- :meth:`Table.from_arrow`, :meth:`Table.from_pandas`,
:meth:`Table.from_polars`, :meth:`Table.from_dict` -- copies, and normalises
every column to a type the engine stores natively first (see
:func:`_arrow_normalise`), so no Arrow type reaches the engine's importer that
it would misread.
"""

from __future__ import annotations

from typing import Any, Iterator

from . import _amber
from ._amber import Value
from .convert import (to_python, EPOCH_DAYS, EPOCH_NS, TEMPORAL_TYPES,
                      _temporal_array, _TEMPORAL_VECTORS, _INT32_NULL, _INT64_NULL)

#: column types the engine's own Arrow exporter lays out (zero-copy for numbers)
_EXPORTABLE = {
    _amber.AMBER_T_BOOL, _amber.AMBER_T_BYTE, _amber.AMBER_T_SHORT, _amber.AMBER_T_INT,
    _amber.AMBER_T_LONG, _amber.AMBER_T_FLOAT, _amber.AMBER_T_SYM, _amber.AMBER_T_RANGE,
}


class Table:
    """A columnar table living inside the Amber engine."""

    __slots__ = ("_value", "_names", "_keys")

    def __init__(self, value: Value, key_columns: list[str] | None = None) -> None:
        if not isinstance(value, Value):
            raise TypeError(f"amber.Table: expected an amber.Value, got {type(value).__name__}")
        if not value.is_table():
            raise TypeError(f"amber.Table: expected a table, got {value.type_name()}")
        self._value = value
        self._names = [value.colname(i) for i in range(value.ncols())]
        self._keys = list(key_columns or ())

    # ---- identity ---------------------------------------------------------

    @property
    def value(self) -> Value:
        """The underlying engine handle, for passing back into Amber."""
        return self._value

    @property
    def columns(self) -> list[str]:
        return list(self._names)

    @property
    def key_columns(self) -> list[str]:
        """The grouping columns, when this table came from a keyed result.

        `select ... by sym from t` produces a KEYED table: Amber models it as a
        dictionary from a key table to a value table. Converting that to a
        Python dict would be wrong -- the keys are rows, not scalars -- so it
        arrives here as an ordinary table with the key columns in front, and
        this is the record of which ones they were. :meth:`to_pandas` uses it as
        the DataFrame index.
        """
        return list(self._keys)

    def __len__(self) -> int:
        return int(self._value.count())

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self), len(self._names))

    @property
    def dtypes(self) -> dict[str, str]:
        """Each column's Amber type name, e.g. ``{'px': 'float vector'}``."""
        return {n: self._value.column(i).type_name() for i, n in enumerate(self._names)}

    # ---- column access ----------------------------------------------------

    def _index(self, name: str | int) -> int:
        if isinstance(name, bool) or not isinstance(name, (str, int)):
            raise TypeError(f"amber.Table: a column is a name or a position, not {type(name).__name__}")
        if isinstance(name, int):
            n = len(self._names)
            if not -n <= name < n:
                raise IndexError(f"amber.Table: column {name} out of range for {n} columns")
            return name % n
        try:
            return self._names.index(name)
        except ValueError:
            raise KeyError(name) from None

    def column(self, name: str | int, *, raw: bool = False):
        """One column. ``raw=True`` gives the :class:`Value` rather than a
        converted object, which is what you want when handing it straight back
        to the engine."""
        value = self._value.column(self._index(name))
        return value if raw else to_python(value)

    def __getitem__(self, name: str | int):
        return self.column(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __contains__(self, name: object) -> bool:
        return name in self._names

    def items(self):
        for name in self._names:
            yield name, self.column(name)

    def head(self, n: int = 5) -> "Table":
        """The first ``n`` rows (the last ``-n`` when negative), as a Table."""
        return Table(_amber.call("pyamber.take", (int(n), self._value)), self._keys)

    def tail(self, n: int = 5) -> "Table":
        """The last ``n`` rows, as a Table."""
        return self.head(-int(n))

    # ---- exits ------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """``{name: column}``, columns converted to Python objects."""
        return {name: self.column(name) for name in self._names}

    def to_pandas(self, *, copy: bool = False, index: bool = True, categorical: bool = False):
        """A pandas ``DataFrame``.

        Numeric columns arrive as NumPy views of engine memory, so the DataFrame
        holds a reference to this table's data for as long as it lives. Pass
        ``copy=True`` to take an independent snapshot -- worth doing if the
        DataFrame will outlive the query that produced it and you would rather
        not pin the engine's buffers. Temporal columns arrive as
        ``datetime64``/``timedelta64``.

        ``index=False`` keeps the grouping columns of a keyed result (see
        :attr:`key_columns`) as ordinary columns instead of moving them to the
        index. ``categorical=True`` brings symbol columns in as
        ``pandas.Categorical`` -- the engine's interned ids become the codes, so a
        ticker column costs four bytes a row instead of a Python str each.
        """
        import pandas as pd  # noqa: PLC0415
        data = {}
        for i, name in enumerate(self._names):
            raw = self._value.column(i)
            if categorical and raw.type() == _amber.AMBER_T_SYM:
                import numpy as np  # noqa: PLC0415
                codes, cats = raw.symbol_codes()
                data[name] = pd.Categorical.from_codes(np.frombuffer(codes, dtype=np.int32), cats)
            else:
                data[name] = to_python(raw)
        frame = pd.DataFrame(data, copy=False)
        if not self._names:
            frame = pd.DataFrame(index=range(len(self)))
        if index and self._keys:
            frame = frame.set_index(self._keys)
        return frame.copy() if copy else frame

    def to_numpy(self):
        """The columns as one 2-D NumPy array (a copy; NumPy picks the dtype)."""
        import numpy as np  # noqa: PLC0415
        return np.column_stack([np.asarray(self.column(n)) for n in self._names]) \
            if self._names else np.empty((len(self), 0))

    def to_arrow(self):
        """A ``pyarrow.Table`` over the Amber columns, zero-copy for numerics.

        The numeric and symbol columns go through the engine's own exporter,
        which moves its export into two Arrow structs allocated here on the
        Python side (see :mod:`amber.arrow_abi`); PyArrow then moves the
        contents out of those into its own. Two moves, no copies of the numbers
        -- and because Python owns both containers throughout, nothing allocated
        by one allocator is ever handed to another's ``free()``.

        Columns the exporter has no layout for are built here instead: temporal
        columns as ``date32``, ``timestamp[ns]``, ``time32``/``duration[ns]``
        (``month`` as the ``date32`` of its first day), and anything else
        (strings held as char vectors, nested lists) from their Python values.
        """
        import pyarrow as pa  # noqa: PLC0415
        from .arrow_abi import new_pair  # noqa: PLC0415

        raws = [self._value.column(i) for i in range(len(self._names))]
        exported_names, exported = [], []
        for name, raw in zip(self._names, raws):
            if raw.type() in _EXPORTABLE:
                exported_names.append(name)
                exported.append(raw)

        batch = None
        if exported:
            sub = _amber.make_table(exported_names, exported)
            schema, array, schema_addr, array_addr = new_pair()
            _amber.arrow_export_into(sub, schema_addr, array_addr)
            batch = pa.RecordBatch._import_from_c(array_addr, schema_addr)
            # `schema` and `array` stay referenced until here on purpose: their
            # storage must outlive the import that reads through their addresses.
            del schema, array

        arrays, k = [], 0
        for name, raw in zip(self._names, raws):
            kind = raw.type()
            if kind in _EXPORTABLE:
                arr = batch.column(k)
                k += 1
                if kind == _amber.AMBER_T_BYTE and pa.types.is_uint8(arr.type):
                    # Amber's byte vectors are signed ints the exporter labels
                    # uint8: relabel the same buffers (zero-copy) so -1 stays -1
                    arr = pa.Array.from_buffers(pa.int8(), len(arr), arr.buffers(), arr.null_count)
                arrays.append(arr)
            elif kind in _TEMPORAL_VECTORS:
                arrays.append(_temporal_to_arrow(pa, raw, kind))
            else:
                arrays.append(pa.array(_python_column(to_python(raw))))
        if not arrays:
            return pa.table({})
        return pa.Table.from_arrays(arrays, names=list(self._names))

    def to_polars(self):
        """A ``polars.DataFrame``, via Arrow. Zero-copy for numeric columns."""
        import polars as pl  # noqa: PLC0415
        return pl.from_arrow(self.to_arrow())

    # ---- construction from the outside ------------------------------------

    @classmethod
    def from_dict(cls, columns: dict[str, Any]) -> "Table":
        """``{name: sequence}`` -> a Table (copies)."""
        if not isinstance(columns, dict):
            raise TypeError("amber.Table.from_dict: expected a dict of columns")
        return cls(_amber.make_table([str(k) for k in columns], list(columns.values())))

    @classmethod
    def from_arrow(cls, data) -> "Table":
        """Import Arrow data into the engine: a ``pyarrow.Table``, a
        ``RecordBatch``, a ``RecordBatchReader``, or anything with the Arrow
        PyCapsule interface (``__arrow_c_stream__``: Polars, DuckDB, ...).

        This direction copies. Amber stores a 32-byte object header immediately
        before every vector payload, so it cannot adopt a foreign allocation as
        a native column. Every column is first normalised to a type the engine
        stores natively (see :func:`_arrow_normalise`): strings become symbols,
        dates and timestamps become Amber's own temporal types, unsigned and
        narrow integers widen, float32 becomes float64. Nested types (lists,
        structs, maps) raise ``TypeError``.
        """
        import pyarrow as pa  # noqa: PLC0415
        from .arrow_abi import new_pair  # noqa: PLC0415

        table = _as_arrow_table(pa, data)
        names, arrays, fixes = [], [], []
        for i, (name, col) in enumerate(zip(table.column_names, table.columns)):
            arr = col.combine_chunks() if col.num_chunks != 1 else col.chunk(0)
            if col.num_chunks == 0:
                arr = pa.array([], type=col.type)
            arr, helper = _arrow_normalise(pa, arr, name)
            if arr.offset:
                # the engine's importer reads a utf8 or numeric buffer from its
                # start: a sliced array is copied to one that starts at 0
                arr = pa.concat_arrays([arr, pa.array([], type=arr.type)])
            names.append(str(name))
            arrays.append(arr)
            if helper:
                fixes.append((i, helper))

        if not arrays:
            raise ValueError("amber.Table.from_arrow: a table needs at least one column")
        batch = pa.RecordBatch.from_arrays(arrays, names=names)
        schema, array, schema_addr, array_addr = new_pair()
        batch._export_to_c(array_addr, schema_addr)
        value = _amber.arrow_import(schema_addr, array_addr)
        del schema, array
        if fixes:
            cols = [value.column(i) for i in range(len(names))]
            for i, helper in fixes:
                cols[i] = _amber.call(helper, (cols[i],))
            value = _amber.make_table(names, cols)
        return cls(value)

    @classmethod
    def from_polars(cls, frame) -> "Table":
        """Import a ``polars.DataFrame`` (copies), via Arrow."""
        return cls.from_arrow(frame.to_arrow())

    @classmethod
    def from_pandas(cls, frame, *, index: bool = False, strings: str = "symbol") -> "Table":
        """Import a pandas ``DataFrame`` into the engine (copies).

        Column dtypes map as follows:

        * **object / string / category** columns become Amber **symbol** vectors
          (``strings="char"`` keeps them as strings instead). Symbols are
          interned int32 ids, so a ticker column costs four bytes a row and joins
          and groups on integer comparisons -- the reason Amber queries are fast
          on ``sym``. Missing values become the empty symbol.
        * **datetime64** columns become Amber **timestamps** (time-zone-aware
          ones as UTC), **timedelta64** become **timespans**, ``NaT`` the null.
        * pandas' nullable **Int/UInt/Float/boolean** columns keep their numbers,
          with ``NA`` as Amber's null (``0N``, or ``0n`` for floats).
        * **bool** becomes 0/1 (Amber's booleans are small ints).

        ``index=True`` brings the DataFrame index in as leading column(s).
        """
        if strings not in ("symbol", "char"):
            raise ValueError("amber.Table.from_pandas: strings must be 'symbol' or 'char'")
        source = frame.reset_index() if index else frame
        names, columns = [], []
        seen = set()
        for i, name in enumerate(source.columns):
            key = str(name)
            if key in seen:
                raise ValueError(f"amber.Table.from_pandas: duplicate column name {key!r}")
            seen.add(key)
            series = source.iloc[:, i]            # by position: source[name] is a frame when names repeat
            dtype = series.dtype
            kind = getattr(dtype, "kind", "O")
            if str(dtype) in ("category", "string", "str") or kind in ("O", "U", "S"):
                values = ["" if _is_missing(v) else (v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v))
                          for v in series.astype(object).tolist()]
                columns.append(_amber.symbols(values) if strings == "symbol" else values)
            else:
                columns.append(series)            # the convert hook does datetimes and nullables
            names.append(key)
        return cls(_amber.make_table(names, columns))

    # ---- rendering --------------------------------------------------------

    def __repr__(self) -> str:
        text = self._value.format(plain=True)
        rows, cols = self.shape
        return f"amber.Table[{rows} rows x {cols} cols]\n{text}"

    def _repr_html_(self) -> str:
        """Rich rendering in Jupyter. The heavy version -- sortable, sticky
        header, type badges -- lives in the ``amber-jupyter`` satellite; this is
        the plain table so that a bare ``pip install`` still renders something
        readable in a notebook. Only the rows shown are converted."""
        from html import escape

        nrows = len(self)
        limit = min(nrows, 100)
        shown = self.head(limit) if nrows > limit else self
        head = "".join(f"<th>{escape(str(n))}</th>" for n in self._names)
        columns = [shown.column(n) for n in self._names]
        body = []
        for r in range(limit):
            cells = "".join(f"<td>{escape(str(c[r]))}</td>" for c in columns)
            body.append(f"<tr>{cells}</tr>")
        more = ("" if nrows <= limit
                else f"<tfoot><tr><td colspan='{len(self._names)}'>"
                     f"… {nrows - limit:,} more rows</td></tr></tfoot>")
        return (
            "<table class='amber-table'>"
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody>{more}</table>"
            f"<div style='font:11px monospace;opacity:.6'>"
            f"amber.Table &middot; {nrows:,} rows &times; {len(self._names)} cols</div>"
        )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and v != v:
        return True
    return type(v).__name__ in ("NAType", "NaTType")


def _python_column(values):
    """A converted column as something pa.array accepts (lists of arrays become
    lists of lists)."""
    if isinstance(values, list):
        return [v.tolist() if hasattr(v, "tolist") else v for v in values]
    return values


def _temporal_to_arrow(pa, raw: Value, kind: int):
    import numpy as np  # noqa: PLC0415
    values = _temporal_array(raw, kind)               # datetime64/timedelta64, NaT for nulls
    if kind == _amber.AMBER_T_DATE_LIST:
        return pa.array(values, type=pa.date32())
    if kind == _amber.AMBER_T_MONTH_LIST:
        return pa.array(values.astype("datetime64[D]"), type=pa.date32())
    if kind == _amber.AMBER_T_TIMESTAMP_LIST:
        return pa.array(values, type=pa.timestamp("ns"))
    if kind == _amber.AMBER_T_TIMESPAN_LIST:
        return pa.array(values, type=pa.duration("ns"))
    # times of day: Arrow's time32 counts from midnight, as Amber's do
    rawnums = np.asarray(raw).astype("int64")
    mask = rawnums == _INT32_NULL
    if kind == _amber.AMBER_T_TIME_LIST:
        return pa.array(rawnums, type=pa.int64(), mask=mask).cast(pa.int32()).cast(pa.time32("ms"))
    if kind == _amber.AMBER_T_SECOND_LIST:
        return pa.array(rawnums, type=pa.int64(), mask=mask).cast(pa.int32()).cast(pa.time32("s"))
    # minutes: Arrow has no minute unit, so seconds
    return pa.array(rawnums * 60, type=pa.int64(), mask=mask).cast(pa.int32()).cast(pa.time32("s"))


def _as_arrow_table(pa, data):
    if isinstance(data, pa.Table):
        return data.combine_chunks()
    if isinstance(data, pa.RecordBatch):
        return pa.Table.from_batches([data])
    if isinstance(data, pa.RecordBatchReader):
        return data.read_all().combine_chunks()
    if hasattr(data, "__arrow_c_stream__") or hasattr(data, "__arrow_c_array__"):
        return pa.table(data).combine_chunks()
    if hasattr(data, "to_arrow"):                     # polars, duckdb relations, ...
        return _as_arrow_table(pa, data.to_arrow())
    raise TypeError(f"amber.Table.from_arrow: cannot read Arrow data from {type(data).__name__}")


def _arrow_normalise(pa, arr, name):
    """One Arrow array -> (an array of a type the engine's importer handles,
    the pyamber helper that gives the column its Amber type afterwards, or None).

    The engine's importer understands utf8, bool, int16, int32, int64, float64
    and uint8; any other format it would read as int64 -- garbage at best, and
    past the end of a 4-byte buffer at worst -- so everything is mapped onto
    those first, here, where PyArrow does the casting.
    """
    import pyarrow.compute as pc  # noqa: PLC0415
    t = arr.type
    if pa.types.is_dictionary(t):
        return _arrow_normalise(pa, arr.dictionary_decode(), name)
    if pa.types.is_string(t) or pa.types.is_large_string(t) or str(t) == "string_view":
        return arr.cast(pa.string()), None
    if pa.types.is_binary(t) or pa.types.is_large_binary(t) or pa.types.is_fixed_size_binary(t):
        try:
            return arr.cast(pa.string()), None
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
            raise TypeError(f"amber.Table.from_arrow: column {name!r} holds binary that is not utf8") from exc
    if pa.types.is_boolean(t):
        return (arr.cast(pa.int64()) if arr.null_count else arr), None
    if pa.types.is_integer(t):
        if t in (pa.int32(), pa.int64()):
            return arr, None
        if pa.types.is_uint64(t):
            try:
                return arr.cast(pa.int64()), None          # a safe cast: fails past int64
            except pa.ArrowInvalid as exc:
                raise OverflowError(f"amber.Table.from_arrow: column {name!r} has uint64 values "
                                    "past int64's range") from exc
        # int8 (no format in the importer), int16 and uint8 (no nulls there),
        # uint16 and uint32: all fit int64
        return arr.cast(pa.int64()), None
    if pa.types.is_floating(t) or pa.types.is_decimal(t):
        return (arr if t == pa.float64() else arr.cast(pa.float64())), None
    if pa.types.is_null(t):
        return arr.cast(pa.float64()), None
    if pa.types.is_date(t):
        days = arr.cast(pa.date32()).cast(pa.int32()).cast(pa.int64())
        return pc.subtract(days, EPOCH_DAYS), "pyamber.mkd"
    if pa.types.is_timestamp(t):
        ns = arr.cast(pa.timestamp("ns", tz=t.tz)).cast(pa.int64())   # tz-aware: UTC, as Arrow stores it
        return pc.subtract(ns, EPOCH_NS), "pyamber.mkp"
    if pa.types.is_duration(t):
        return arr.cast(pa.duration("ns")).cast(pa.int64()), "pyamber.mkn"
    if pa.types.is_time32(t):
        if t.unit == "s":
            return arr.cast(pa.int32()).cast(pa.int64()), "pyamber.mkv"
        return arr.cast(pa.int32()).cast(pa.int64()), "pyamber.mkt"
    if pa.types.is_time64(t):
        return arr.cast(pa.time64("ns")).cast(pa.int64()), "pyamber.mkn"
    raise TypeError(f"amber.Table.from_arrow: column {name!r} has Arrow type {t}, "
                    "which has no Amber column type (nested types are not supported)")
