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
                  interned int32 ids, Arrow wants offsets and bytes.

``to_pandas()``   a DataFrame built from those same views. Whether pandas keeps
                  the views or consolidates them into fresh blocks is pandas'
                  decision and it varies by version, so this is documented as
                  "zero-copy where pandas permits" rather than promised.

``to_dict()``     plain Python. Always copies. Useful for JSON, and for tests
                  that should not depend on NumPy being installed.
"""

from __future__ import annotations

from typing import Any, Iterator

from . import _amber
from ._amber import Value
from .convert import to_python, VECTOR_DTYPES


class Table:
    """A columnar table living inside the Amber engine."""

    __slots__ = ("_value", "_names", "_keys")

    def __init__(self, value: Value, key_columns: list[str] | None = None) -> None:
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

    # ---- column access ----------------------------------------------------

    def column(self, name: str | int, *, raw: bool = False):
        """One column. ``raw=True`` gives the :class:`Value` rather than a
        converted object, which is what you want when handing it straight back
        to the engine."""
        index = name if isinstance(name, int) else self._names.index(name)
        value = self._value.column(index)
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

    # ---- exits ------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """``{name: column}``, columns converted to Python objects."""
        return {name: self.column(name) for name in self._names}

    def to_pandas(self, *, copy: bool = False, index: bool = True):
        """A pandas ``DataFrame``.

        Numeric columns arrive as NumPy views of engine memory, so the DataFrame
        holds a reference to this table's data for as long as it lives. Pass
        ``copy=True`` to take an independent snapshot -- worth doing if the
        DataFrame will outlive the query that produced it and you would rather
        not pin the engine's buffers.

        ``index=False`` keeps the grouping columns of a keyed result (see
        :attr:`key_columns`) as ordinary columns instead of moving them to the
        index.
        """
        import pandas as pd  # noqa: PLC0415
        frame = pd.DataFrame(self.to_dict(), copy=False)
        if index and self._keys:
            frame = frame.set_index(self._keys)
        return frame.copy() if copy else frame

    def to_arrow(self):
        """A ``pyarrow.Table`` over the Amber columns, zero-copy for numerics.

        The engine moves its export into two Arrow structs allocated here on the
        Python side (see :mod:`amber.arrow_abi`), and PyArrow then moves the
        contents out of those into its own. Two moves, no copies of the data
        itself, and -- because Python owns both containers throughout -- nothing
        allocated by one allocator is ever handed to another's ``free()``.
        """
        import pyarrow as pa  # noqa: PLC0415
        from .arrow_abi import new_pair  # noqa: PLC0415

        schema, array, schema_addr, array_addr = new_pair()
        _amber.arrow_export_into(self._value, schema_addr, array_addr)
        batch = pa.RecordBatch._import_from_c(array_addr, schema_addr)
        # `schema` and `array` stay referenced until here on purpose: their
        # storage must outlive the import that reads through their addresses.
        del schema, array
        return pa.Table.from_batches([batch])

    def to_polars(self):
        """A ``polars.DataFrame``, via Arrow. Zero-copy for numeric columns."""
        import polars as pl  # noqa: PLC0415
        return pl.from_arrow(self.to_arrow())

    # ---- construction from the outside ------------------------------------

    @classmethod
    def from_arrow(cls, table) -> "Table":
        """Import a ``pyarrow.Table`` (or ``RecordBatch``) into the engine.

        This direction copies. Amber stores a 32-byte object header immediately
        before every vector payload, so it cannot adopt a foreign allocation as
        a native column -- see the note at the top of the engine's ``src/ar.c``.
        """
        import pyarrow as pa  # noqa: PLC0415
        from .arrow_abi import new_pair  # noqa: PLC0415

        if isinstance(table, pa.Table):
            batches = table.combine_chunks().to_batches()
            if not batches:
                raise ValueError("amber.Table.from_arrow: table has no record batches")
            batch = batches[0]
        else:
            batch = table

        schema, array, schema_addr, array_addr = new_pair()
        batch._export_to_c(array_addr, schema_addr)
        value = _amber.arrow_import(schema_addr, array_addr)
        del schema, array
        return cls(value)

    @classmethod
    def from_pandas(cls, frame, *, index: bool = False) -> "Table":
        """Import a pandas ``DataFrame`` into the engine (copies).

        Column dtypes map as follows, and the two non-obvious ones are non-obvious
        for a reason:

        * **object / string** columns become Amber **symbol** vectors, not lists
          of char vectors. Symbols are interned int32 ids, so a ticker column
          costs four bytes a row and joins and groups on integer comparisons --
          which is the entire reason Amber queries are fast on ``sym``. A column
          of long, high-cardinality free text is the case where that is the wrong
          trade; convert it yourself if you hit it.
        * **datetime64** columns become int64 nanoseconds since the epoch, which
          is the representation, not a lossy one. Amber's own timestamp type
          counts nanoseconds from 2000.01.01 rather than 1970.01.01, so the epoch
          is left as pandas had it instead of silently re-basing it under you.

        ``index=True`` brings the DataFrame index in as leading column(s).
        """
        import numpy as np  # noqa: PLC0415

        source = frame.reset_index() if index else frame
        names, columns = [], []
        for name in source.columns:
            series = source[name]
            values = series.to_numpy()
            if values.dtype.kind in ("O", "U", "S") or str(series.dtype) in ("string", "category"):
                columns.append(_amber.symbols([("" if v is None else v) for v in values.tolist()]))
            elif values.dtype.kind == "M":
                columns.append(values.astype("datetime64[ns]").astype(np.int64))
            elif values.dtype.kind == "m":
                columns.append(values.astype("timedelta64[ns]").astype(np.int64))
            elif values.dtype.kind == "b":
                columns.append(values.astype(np.int64))
            else:
                columns.append(values)
            names.append(str(name))
        return cls(_amber.make_table(names, columns))

    # ---- rendering --------------------------------------------------------

    def __repr__(self) -> str:
        text = self._value.format(plain=True)
        rows, cols = self.shape
        return f"amber.Table[{rows} rows x {cols} cols]\n{text}"

    def _repr_html_(self) -> str:
        """Rich rendering in Jupyter. The heavy version -- sortable, sticky
        header, type badges -- lives in the ``amber-jupyter`` satellite; this is
        the plain table so that a bare ``pip install amber`` still renders
        something readable in a notebook."""
        from html import escape

        head = "".join(f"<th>{escape(str(n))}</th>" for n in self._names)
        columns = [self.column(n) for n in self._names]
        nrows = len(self)
        limit = min(nrows, 100)
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
