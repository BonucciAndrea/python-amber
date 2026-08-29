"""Amber value -> Python object.

The rule this module follows is: *convert the shape, alias the bytes.*

A table becomes a :class:`amber.Table`, a dictionary becomes a ``dict``, a
symbol vector becomes a list of ``str`` -- those are all structural conversions
and they copy, because there is no way not to. But a numeric column becomes a
NumPy array that points straight at the engine's own memory, with the Amber
value pinned as the array's ``base``. Nothing is copied and nothing can be freed
underneath it.

The one place that gets surprising is integer width. Amber stores an integer
vector in the *narrowest* type that holds every element (see ``tZ()`` in the
engine's ``src/m.c``), so ``1 2 3`` is physically ``int8`` and
``1 2 3000000000`` is ``int64`` -- both are "long vectors" as far as the
language is concerned. Rather than hide that behind a widening copy, the arrays
you get back carry the real dtype. ``arr.astype('int64')`` is one call away when
you want uniformity, and it is your call to make, not this module's.
"""

from __future__ import annotations

from typing import Any

from . import _amber
from ._amber import Value

#: Amber element type -> NumPy dtype string, for the types with a flat payload.
VECTOR_DTYPES = {
    _amber.AMBER_T_BYTE:  "int8",
    _amber.AMBER_T_SHORT: "int16",
    _amber.AMBER_T_INT:   "int32",
    _amber.AMBER_T_LONG:  "int64",
    _amber.AMBER_T_FLOAT: "float64",
}

_ATOMS = {
    _amber.AMBER_T_INT_ATOM, _amber.AMBER_T_LONG_ATOM, _amber.AMBER_T_FLOAT_ATOM,
    _amber.AMBER_T_CHAR_ATOM, _amber.AMBER_T_SYM_ATOM,
    _amber.AMBER_T_DATE, _amber.AMBER_T_TIME, _amber.AMBER_T_TIMESTAMP,
}


def _numpy():
    try:
        import numpy  # noqa: PLC0415
        return numpy
    except ImportError:
        return None


def as_array(value: Value, *, copy: bool = False):
    """A NumPy view of a flat Amber vector -- **zero-copy** unless ``copy``.

    The returned array's ``base`` is the Amber value, so the engine cannot
    reclaim the payload while the array is alive. Raises :class:`TypeError` for
    a value that has no flat payload (a table, a general list, an atom).
    """
    np = _numpy()
    if np is None:
        raise ImportError("amber: NumPy is required for as_array(); "
                          "use memoryview(value) or list(memoryview(value)) instead")
    value = value.flatten()
    arr = np.asarray(value)          # PEP 3118 -- no copy, `value` becomes base
    return arr.copy() if copy else arr


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
        if isinstance(keys, str):     # a single-key dict decodes its key to str
            keys = [keys]
        vals = value.values()
        if vals.type() == _amber.AMBER_T_LIST:
            items = [to_python(v) for v in vals.items()]
        else:
            # A uniformly-typed value vector: split it elementwise, so each key
            # maps to its own scalar rather than to one array shared by all.
            items = list(to_python(vals))
        return dict(zip(keys, items))

    if kind == _amber.AMBER_T_CHAR:
        return value.to_str()

    if kind == _amber.AMBER_T_CHAR_ATOM:
        return value.to_str()

    if kind in (_amber.AMBER_T_SYM, _amber.AMBER_T_SYM_ATOM):
        return value.to_symbols()

    if kind in _ATOMS:
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
        # bit-packed, one bit per element. Widening has to happen inside the
        # engine, so it does -- `pyamber.widen` is installed at boot for exactly
        # this. The result is an ordinary narrow int vector and converts on the
        # normal path.
        return to_python(_amber.call("pyamber.widen", (value,)))

    # Functions, projections, compiled closures: hand back the handle unchanged.
    return value
