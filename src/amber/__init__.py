"""python-amber -- the Amber columnar time-series engine, in process.

    >>> import amber as am
    >>> am.gentq(10_000)                    # generate trades + quotes
    >>> am.q("select vwap:wavg[sz;px] by sym from trades").to_pandas()

The package is a thin, honest layer over ``libamber.so``. Three ideas carry
everything else:

**One engine, in your process.** There is no server, no socket and no
serialisation between Python and Amber. ``am.eval("2+3")`` calls the same C
function the REPL calls. That also means the engine is single-threaded and the
GIL is what serialises access to it -- see the threading note in
``src/py_amber.c``.

**Zero-copy out, copy in.** An Amber vector handed to Python is exposed through
the buffer protocol, so ``numpy.asarray(v)`` aliases engine memory and keeps the
engine value alive as long as the array exists. The reverse direction copies:
Amber stores a 32-byte object header immediately before every payload, so it
cannot adopt a foreign allocation. That asymmetry matches where the volume is --
query results are large, query parameters are small.

**Dynamic dispatch, not a wrapper per verb.** ``am.<name>`` resolves to whatever
``<name>`` is in the engine's global namespace, at the moment you touch it.
Nothing in this package enumerates Amber's vocabulary, so it cannot fall behind
it, and a function you define yourself in a ``.k`` file is reachable the same way
a built-in is.

Amber - GNU AGPLv3 - see LICENSE and NOTICE.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Iterable, Sequence

def _old_install_check() -> None:
    """The first python-amber was published as `amber` 1.x. It and amber-lang both
    install the `amber` package directory, so having both leaves a mix of their
    files: an engine from one, a standard library or extension from the other,
    and errors that make no sense ('ge from inside gentq). Say what it is."""
    try:
        from importlib.metadata import PackageNotFoundError, distribution
        old = distribution("amber")
    except (ImportError, PackageNotFoundError):
        return
    if not any(str(f).replace("\\", "/") == "amber/__init__.py" for f in (old.files or ())):
        return                                   # some other project called "amber"
    try:                                         # only when it is THIS folder it wrote into
        theirs = os.path.realpath(os.path.dirname(str(old.locate_file("amber/__init__.py"))))
    except Exception:                            # noqa: BLE001
        return
    if theirs != os.path.realpath(os.path.dirname(os.path.abspath(__file__))):
        return
    raise ImportError(
        f"the old python-amber ({old.metadata['Name']} {old.version}) is installed next to amber-lang, and they "
        "share the `amber` folder. Remove both, then install amber-lang again:\n"
        "    pip uninstall -y amber amber-lang\n"
        "    pip install amber-lang")


_old_install_check()

from . import _amber  # noqa: E402
from ._amber import AmberError, Value  # noqa: F401,E402  (re-exported)

from .table import Table
from .convert import to_python, as_array, VECTOR_DTYPES, _to_amber_hook

# dates, times and pandas' nullable columns: the C converter's fallback (convert.py)
_amber._set_convert_hook(_to_amber_hook)

__all__ = [
    "AmberError", "Value", "Table",
    "init", "home", "version", "abi",
    "eval", "q", "call", "get", "set", "delete", "globals_",
    "convert", "to_python", "as_array", "make_table", "load", "plugin_load",
    "diagnostics", "shutdown",
]

__version__ = "2.9.4"

# --------------------------------------------------------------------------
# boot
# --------------------------------------------------------------------------

_HOME: str | None = None
_READY = False


def _bundled_home() -> str | None:
    """The .k standard library shipped inside the wheel, if there is one.

    A binary wheel carries both ``libamber.so`` and the eight ``.k`` modules the
    engine's full vocabulary lives in (amber.k, fin.k, std.k, qsql.k,
    temporal.k, sys.k, hdb.k, ipc.k). Without them ``import amber`` would give
    you the bare engine -- primitives only, no ``gentq``, no ``select`` -- which
    looks like a broken install rather than a deliberate one. With them,
    ``pip install amber`` is genuinely all you need.
    """
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kmodules")
    return here if os.path.isfile(os.path.join(here, "amber.k")) else None


def _discover_home() -> str | None:
    """Where to load the standard library from.

    The copy bundled with the engine comes FIRST: it is the one built with this
    libamber, and a standard library from another release does not fit it (2.9's
    parser turns `>=` into `Q.ge`, which 2.4's amber.k has never heard of, so
    gentq died with 'ge). $AMBER_HOME is what the REPL uses, so it is set on most
    machines that have Amber at all -- it is only a fallback here, for builds
    without a bundled copy. $AMBER_PY_HOME, or init(home=...), overrides on purpose.
    """
    env = os.environ.get("AMBER_PY_HOME")
    if env and os.path.isfile(os.path.join(env, "amber.k")):
        return env
    bundled = _bundled_home()
    if bundled:
        return bundled
    env = os.environ.get("AMBER_HOME")
    if env and os.path.isfile(os.path.join(env, "amber.k")):
        return env
    # A developer checkout sitting next to this one, or the current directory --
    # the two places someone hacking on the engine actually runs from.
    for candidate in (os.getcwd(),
                      os.path.join(os.getcwd(), "amber"),
                      os.path.join(sys.prefix, "share", "amber")):
        if os.path.isfile(os.path.join(candidate, "amber.k")):
            return candidate
    return None


def init(home: str | None = None, *, bare: bool = False) -> None:
    """Boot the engine. Called automatically on first use; call it yourself only
    to choose a different ``home`` or to boot ``bare``.

    ``home``  directory holding the ``.k`` standard library. ``None`` takes
              ``$AMBER_PY_HOME``, then the copy bundled with the engine, then
              ``$AMBER_HOME``, then the working directory.
    ``bare``  skip the standard library entirely: primitives only. Faster to
              start and useful when you are embedding Amber as a kernel rather
              than using it as an analytics language.
    """
    global _HOME, _READY
    if bare:
        _amber.init(None)
    else:
        _HOME = home if home is not None else _discover_home()
        _amber.init(_HOME)
    _READY = True


def _ensure() -> None:
    if not _READY:
        init()


def _unpickle(data: bytes) -> Value:
    """What a pickled Value calls to come back: boots the engine if this process
    has not yet, then reads the engine's own serialisation."""
    _ensure()
    return _amber.from_bytes(data)


def home() -> str | None:
    """The directory the standard library was loaded from, or None if bare."""
    return _HOME


def version() -> str:
    """The engine version, e.g. ``'2.9.0'``."""
    return _amber.version()


def abi() -> int:
    """The engine's C-API ABI number."""
    return _amber.abi()


def shutdown() -> None:
    """Release the engine's scratch arena. The object heap is kept for process
    lifetime by design; Amber has no teardown path and has never needed one."""
    _amber.shutdown()


def diagnostics(on: bool) -> bool:
    """Turn the engine's rich stderr diagnostics on or off; returns the previous
    setting. Off by default here: an error becomes an :class:`AmberError`, and
    printing it to stderr as well would be noise."""
    _ensure()
    return bool(_amber.set_diagnostics(bool(on)))


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def eval(src: str, *, raw: bool = False):          # noqa: A001 (shadowing is the API)
    """Evaluate Amber source and return the last statement's value.

    ``raw=True`` returns the :class:`Value` handle instead of converting to a
    Python object -- which is what you want when the next thing you do is hand
    it back to the engine, since conversion is the only step that costs
    anything.
    """
    _ensure()
    if not isinstance(src, str):
        raise TypeError(f"amber.eval: expected str source, got {type(src).__name__}")
    value = _amber.eval(src)
    return value if raw else to_python(value)


def q(line: str, *, raw: bool = False):
    """Evaluate one line of qSQL, with the bare-query syntax accepted verbatim::

        am.q("select vwap:wavg[sz;px] by sym from trades where px>0")

    Amber's SQL surface is a rewrite (``qsql.k``'s ``qrw``) that the REPL
    applies to interactive lines and that script execution deliberately does
    not see. This is the entry point that applies it, so a query you can type
    at the ``amber>`` prompt is a query you can pass here. Plain expressions
    pass through untouched, so ``q`` is always safe to use.

    Text over several lines is rewritten the way a script is (``qrwf``): a query
    may run over indented continuation lines, as in q, and the value is the last
    statement's.
    """
    _ensure()
    if not isinstance(line, str):
        raise TypeError(f"amber.q: expected str, got {type(line).__name__}")
    if "\n" in line.strip("\n") and _HOME is not None:
        src = to_python(_amber.call("qrwf", (line,)))
        # qrwf pads a query it joins with the lines it took (for a script's line numbers): the value is the
        # query's, not the empty line's after it
        value = _amber.eval((src if isinstance(src, str) else line).rstrip())
    else:
        value = _amber.eval_qsql(line.strip("\n"))
    return value if raw else to_python(value)


def call(name: str, *args: Any, raw: bool = False):
    """Apply the global function ``name`` to ``args`` (0..8 of them).

    Faster than composing source text: the arguments cross as native values
    rather than being formatted into a string and re-parsed.
    """
    _ensure()
    value = _amber.call(name, args)
    return value if raw else to_python(value)


def convert(obj: Any) -> Value:
    """Convert a Python object into an Amber :class:`Value`."""
    _ensure()
    return _amber.convert(obj)


def get(name: str, *, raw: bool = False):
    """Read an Amber global."""
    _ensure()
    value = _amber.get_global(name)
    return value if raw else to_python(value)


def _is_column_dict(obj: Any) -> bool:
    """True for a non-empty dict whose values are all equal-length sequences.

    That shape is a table written column-wise -- ``{"sym": [...], "px": [...]}``
    -- and :func:`set` routes it through :func:`make_table`. A ``str``/``bytes``
    value counts as a scalar, not a sequence, so ``{"a": "x", "b": "y"}`` stays
    an ordinary dictionary.
    """
    if not isinstance(obj, dict) or not obj:
        return False
    length = None
    for v in obj.values():
        if isinstance(v, (str, bytes)) or not hasattr(v, "__len__"):
            return False
        if length is None:
            length = len(v)
        elif len(v) != length:
            return False
    return True


def set(name: str, obj: Any) -> None:              # noqa: A001 (shadowing is the API)
    """Bind an Amber global from a Python object.

    NumPy arrays, pandas Series, lists, dicts, ``str``, ``bytes`` and scalars all
    convert. A dict of equal-length sequences becomes a table via
    :func:`make_table`; a dict of anything else becomes an Amber dictionary.
    """
    _ensure()
    if _is_column_dict(obj):
        _amber.set_global(name, make_table(obj).value)
    else:
        _amber.set_global(name, obj)


def delete(name: str) -> None:
    """Unbind an Amber global."""
    _ensure()
    _amber.set_global(name, None)


def globals_() -> list[str]:
    """Every global name currently defined in the engine's root namespace."""
    _ensure()
    names = _amber.eval("!`")
    return [n for n in to_python(names) if not n.startswith("pyamber")]


def make_table(columns: dict[str, Any]) -> Table:
    """Build an Amber table from a mapping of column name to sequence."""
    _ensure()
    if not isinstance(columns, dict):
        raise TypeError("amber.make_table: expected a dict of column name -> sequence")
    return Table.from_dict(columns)


def load(path) -> None:
    """Load a script (``.k`` or ``.q``) or a database directory into the engine,
    exactly as ``\\l path`` does: through the engine's own loader, so bare qSQL,
    queries over several lines and CRLF files all behave as they do at the
    prompt."""
    _ensure()
    path = os.fspath(path)
    if "\n" in path or "\r" in path or "\0" in path:
        raise ValueError("amber.load: a path cannot hold a newline or NUL")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    _amber.eval("\\l " + path)


def plugin_load(path: str) -> None:
    """``dlopen`` a native Amber plugin and run its ``amber_plugin_init``."""
    _ensure()
    _amber.plugin_load(path)


# --------------------------------------------------------------------------
# dynamic dispatch:  am.gentq(10_000)
# --------------------------------------------------------------------------

class Verb:
    """A lazily-resolved handle on an Amber global.

    ``am.gentq`` produces one of these. Calling it applies the Amber function;
    using it in any value context resolves the global. Nothing is looked up
    until you do one of those two things, so a ``Verb`` for a name that does not
    exist yet is not an error -- which matters, because in Amber you routinely
    define the function after you have written the code that uses it.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_name", name)

    # ---- application -----------------------------------------------------
    def __call__(self, *args: Any, raw: bool = False):
        return call(self._name, *args, raw=raw)

    # ---- resolution ------------------------------------------------------
    @property
    def value(self):
        """The current value of the global, converted to Python."""
        return get(self._name)

    def raw(self) -> Value:
        """The current value of the global as an unconverted :class:`Value`."""
        return get(self._name, raw=True)

    # ---- namespacing:  am.u.pub(...)  ->  the Amber global `u.pub` --------
    def __getattr__(self, attr: str) -> "Verb":
        if attr.startswith("_"):
            raise AttributeError(attr)
        return Verb(f"{self._name}.{attr}")

    # ---- value-context conveniences --------------------------------------
    def __repr__(self) -> str:
        try:
            return repr(self.value)
        except AmberError:
            return f"<amber verb `{self._name} (undefined)>"

    def __len__(self):
        return len(self.value)

    def __iter__(self):
        return iter(self.value)

    def __getitem__(self, key):
        return self.value[key]

    def __array__(self, dtype=None, copy=None):
        import numpy as np
        arr = np.asarray(self.value)
        return arr if dtype is None else arr.astype(dtype, copy=False)

    def _repr_html_(self):
        value = self.value
        return value._repr_html_() if hasattr(value, "_repr_html_") else None


def __getattr__(name: str) -> Verb:
    """Module-level dynamic dispatch: ``am.<anything>`` is an Amber global.

    Dunder and private names are refused rather than dispatched. That is not
    tidiness: import machinery, ``copy``, ``pickle``, IPython's feature probes
    and ``typing`` all interrogate modules for attributes like ``__path__``,
    ``__all__`` and ``_ipython_canary_method_should_not_exist_``, and a module
    whose ``__getattr__`` cheerfully returns an object for every one of those
    breaks in ways that are extremely hard to attribute.
    """
    if name.startswith("_"):
        raise AttributeError(f"module 'amber' has no attribute {name!r}")
    return Verb(name)


def __dir__() -> list[str]:
    """Tab completion in a REPL or notebook lists the engine's real globals."""
    base = sorted(__all__)
    try:
        return base + globals_()
    except Exception:
        return base
