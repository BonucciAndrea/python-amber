"""ctypes mirrors of the Apache Arrow C Data Interface structs.

Arrow's whole point is that these two struct layouts are frozen: they are the
ABI, and any two libraries that agree on them can hand columnar data to each
other with no shared build, no shared allocator and no copy. So this module
declares them once, in ctypes, and nothing in python-amber ever needs
``pyarrow`` at build time -- only at the moment you actually ask for a
``pyarrow.Table``.

Ownership follows Arrow's rules, which are worth stating because getting them
wrong is silent rather than loud:

* a struct is "moved" by copying its contents and clearing the source's
  ``release`` pointer -- the CONTENTS transfer, the container does not;
* whoever allocated a container frees it (here: Python, via ctypes);
* ``release`` is called exactly once, by whoever ends up owning the contents.

Both directions in :mod:`amber.table` allocate the containers here, on the
Python side, and let the engine and PyArrow move contents in and out of them.
That is why neither direction has to free anything the other allocated.

Reference: https://arrow.apache.org/docs/format/CDataInterface.html
"""

from __future__ import annotations

import ctypes


class ArrowSchema(ctypes.Structure):
    pass


ArrowSchema._fields_ = [
    ("format", ctypes.c_char_p),
    ("name", ctypes.c_char_p),
    ("metadata", ctypes.c_char_p),
    ("flags", ctypes.c_int64),
    ("n_children", ctypes.c_int64),
    ("children", ctypes.POINTER(ctypes.POINTER(ArrowSchema))),
    ("dictionary", ctypes.POINTER(ArrowSchema)),
    ("release", ctypes.CFUNCTYPE(None, ctypes.POINTER(ArrowSchema))),
    ("private_data", ctypes.c_void_p),
]


class ArrowArray(ctypes.Structure):
    pass


ArrowArray._fields_ = [
    ("length", ctypes.c_int64),
    ("null_count", ctypes.c_int64),
    ("offset", ctypes.c_int64),
    ("n_buffers", ctypes.c_int64),
    ("n_children", ctypes.c_int64),
    ("buffers", ctypes.POINTER(ctypes.c_void_p)),
    ("children", ctypes.POINTER(ctypes.POINTER(ArrowArray))),
    ("dictionary", ctypes.POINTER(ArrowArray)),
    ("release", ctypes.CFUNCTYPE(None, ctypes.POINTER(ArrowArray))),
    ("private_data", ctypes.c_void_p),
]


def new_pair():
    """A fresh, zeroed (schema, array) pair plus their addresses.

    Returns ``(schema, array, schema_addr, array_addr)``. Keep the first two
    alive for as long as the addresses are in use -- ctypes frees the storage
    when they are collected, and an address that outlives its object is exactly
    the bug this whole module exists to avoid.
    """
    schema = ArrowSchema()
    array = ArrowArray()
    return schema, array, ctypes.addressof(schema), ctypes.addressof(array)
