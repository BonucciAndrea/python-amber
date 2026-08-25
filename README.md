# python-amber

**The [Amber](https://github.com/BonucciAndrea/amber) columnar time-series engine, in your Python process, with zero-copy NumPy / Arrow / pandas interop.**

```python
import amber as am

am.gentq(10_000_000)                                     # 10M synthetic trades + quotes
df = am.q("select vwap:wavg[sz;px], n:#px by sym from trades").to_pandas()
```

No server. No socket. No serialisation. `am.eval("2+3")` calls the same C
function the Amber REPL calls, in the same address space, and a query result
that comes back as a NumPy array is *pointing at the engine's own memory* —
not at a copy of it.

```python
>>> col = am.q("select from trades").column("px", raw=True)
>>> arr = numpy.asarray(col)
>>> hex(arr.ctypes.data) == hex(col.handle())            # the engine's buffer
True
```

---

## Contents

- [Install](#install)
- [How it fits together](#how-it-fits-together)
- [Quick start](#quick-start)
- [The API](#the-api)
- [Zero-copy, precisely](#zero-copy-precisely)
- [Type mapping](#type-mapping)
- [Building from source](#building-from-source)
- [Building wheels](#building-wheels)
- [Threading](#threading)
- [Troubleshooting](#troubleshooting)
- [Licence](#licence)

---

## Install

```bash
pip install amber
```

A binary wheel carries `libamber.so` **and** the eight `.k` standard-library
modules, so this is genuinely all you need — no engine checkout, no
`$AMBER_HOME`, no build toolchain.

From source, against an engine checkout:

```bash
git clone https://github.com/BonucciAndrea/amber
git clone https://github.com/BonucciAndrea/python-amber
cd python-amber
AMBER_SRC=../amber pip install .
```

Optional extras, none of them required:

```bash
pip install "amber[numpy]"     # NumPy views of columns
pip install "amber[pandas]"    # .to_pandas()
pip install "amber[arrow]"     # .to_arrow(), .to_polars()
pip install "amber[all]"
```

They are optional because they are reached through **ABI contracts, not Python
imports**: the buffer protocol (PEP 3118) for NumPy, the Arrow C Data Interface
for PyArrow. This package pins neither and imports both lazily, at the moment
you call the method that needs them.

---

## How it fits together

```
   your Python process
   ┌──────────────────────────────────────────────────────────────┐
   │                                                              │
   │   import amber as am                                         │
   │        │                                                     │
   │        ▼                                                     │
   │   amber/__init__.py       dynamic dispatch, conversion       │
   │   amber/table.py          pandas / Arrow / Polars bridges    │
   │        │                                                     │
   │        ▼                                                     │
   │   amber/_amber.so         CPython extension  (src/py_amber.c)│
   │        │                  ~800 lines, no NumPy/Arrow headers │
   │        ▼                                                     │
   │   libamber.so             the engine, exporting 57 symbols   │
   │                           all named amber_* or am_ext_*      │
   │        │                                                     │
   │        ├──── column payload ────────────┐                    │
   │        │                                │  same address      │
   │        ▼                                ▼                    │
   │   numpy.ndarray                  pyarrow.Table               │
   └──────────────────────────────────────────────────────────────┘
```

The engine repository is not modified by this package and does not know it
exists. The entire contract between them is section 6 of the engine's
`src/ext.h` — mirrored here in `include/amber_capi.h` and checked at import
time via `amber_abi_version()`.

---

## Quick start

```python
import amber as am

# --- evaluate ------------------------------------------------------------
am.eval("2+3")                       # 5
am.eval("1 2 3 4 5")                 # array([1,2,3,4,5], dtype=int8)
am.eval('"hello"')                   # 'hello'
am.eval("`aapl`msft")                # ['aapl', 'msft']

# --- dynamic dispatch: any Amber global is an attribute -------------------
am.gentq(1_000_000)                  # runs `gentq 1000000` in the engine
am.wavg([1, 2, 3], [10.0, 20.0, 30.0])
am.u.pub                             # dotted names work: the `u.pub` global

# --- qSQL, exactly as you'd type it at the prompt -------------------------
t = am.q("select vwap:wavg[sz;px], n:#px by sym from trades where px>0")
t.to_pandas()

# --- as-of join, the thing Amber is actually for --------------------------
am.eval("taq:aj[`sym`time; trades; quotes]")
am.q("select from taq where sym=`AAPL").to_arrow()

# --- push data back in ----------------------------------------------------
import numpy as np
am.set("signal", np.random.randn(1_000_000))
am.eval("avg signal")

am.set("book", {"sym": ["AAPL", "MSFT"], "px": [187.2, 411.1]})
```

---

## The API

### Module level

| | |
|---|---|
| `am.init(home=None, *, bare=False)` | Boot the engine. Automatic on first use. `bare=True` skips the `.k` standard library. |
| `am.eval(src, *, raw=False)` | Evaluate Amber source; returns the last statement's value. |
| `am.q(line, *, raw=False)` | Evaluate one line **after the bare-qSQL rewrite** — `select … from …` works verbatim. |
| `am.call(name, *args, raw=False)` | Apply a global function to 0–8 native arguments (no string round-trip). |
| `am.get(name)` / `am.set(name, obj)` | Read / bind an Amber global. |
| `am.make_table({name: seq, …})` | Build a table from Python columns. |
| `am.load(path)` | Load a `.k` script, as `\l` does in the REPL. |
| `am.convert(obj)` | Python object → `Value`. |
| `am.plugin_load(path)` | `dlopen` a native Amber plugin. |
| `am.diagnostics(bool)` | Re-enable the engine's rich stderr diagnostics (off by default here). |
| `am.<anything>` | A `Verb` bound to that Amber global. |

`raw=True` skips conversion and hands back a `Value`. Use it whenever the next
thing you do is give the result back to the engine — conversion is the only step
in the whole path that costs anything.

### `am.Value`

An owned handle on an engine value. Flat vectors export the **buffer protocol**,
so `numpy.asarray(v)`, `memoryview(v)` and `pyarrow.py_buffer(v)` are all
zero-copy and all keep the engine value alive for exactly as long as they exist.

`.type()` `.type_name()` `.count()` `.is_table()` `.is_dict()` `.is_atom()`
`.flatten()` `.format(plain=True)` `.to_str()` `.to_symbols()` `.scalar()`
`.ncols()` `.colname(i)` `.column(i)` `.keys()` `.values()` `.items()`
`.handle()`

### `am.Table`

| | |
|---|---|
| `t.columns` | Column names. |
| `t.key_columns` | Grouping columns of a keyed (`by`) result. |
| `t[name]` / `t.column(name, raw=…)` | One column. |
| `t.to_pandas(copy=False, index=True)` | DataFrame; key columns become the index. |
| `t.to_arrow()` | `pyarrow.Table`, zero-copy for numerics. |
| `t.to_polars()` | `polars.DataFrame`, via Arrow. |
| `t.to_dict()` | Plain Python (always copies). |
| `am.Table.from_pandas(df, index=False)` | Import a DataFrame. |
| `am.Table.from_arrow(at)` | Import a `pyarrow.Table` or `RecordBatch`. |

### A REPL

```
$ python -m amber
amber> gentq 1000
amber> select vwap:wavg[sz;px] by sym from trades
amber> \py df = am.q("select from trades").to_pandas()
```

---

## Zero-copy, precisely

Claims about zero copy are cheap, so here is exactly what is and is not copied.

**Engine → Python is zero-copy for numeric columns.** `amber_get_vector_ptr()`
returns the payload pointer; the buffer protocol wraps it with `view->obj` set
to the owning `Value`, INCREF'd. NumPy stores that as the array's `base`, so the
Amber refcount cannot reach zero while the array is alive. The test suite
asserts pointer equality, not value equality:

```python
assert arr.ctypes.data == raw.handle()
assert arrow_buffer.address == raw.handle()
```

**Engine → Arrow is zero-copy for numeric columns.** The `ArrowArray`'s data
buffer *is* the Amber column's payload, and its release callback drops the Amber
refcount. Symbol columns are materialised to utf8 — Amber stores interned int32
ids, Arrow wants offsets and bytes, and no pointer arithmetic bridges that.

**Python → engine copies. Always.** Amber stores a 32-byte object header
immediately *before* every vector payload, so it cannot adopt a foreign
allocation as a native vector. That asymmetry is not a limitation anyone is
working around; it matches where the volume is. Query results are large and
query parameters are small.

**pandas is "where pandas permits".** `to_pandas()` builds the DataFrame from
the same views. Whether pandas keeps them or consolidates them into fresh blocks
is pandas' decision and it varies by version, so this package documents that
honestly rather than promising something it does not control. Pass `copy=True`
to take a deliberate snapshot and stop pinning the engine's buffers.

---

## Type mapping

| Amber | Python | Notes |
|---|---|---|
| int atom, long atom | `int` | |
| float atom | `float` | |
| char atom, char vector | `str` | UTF-8 decoded |
| symbol atom | `str` | |
| symbol vector | `list[str]` | interned int32 ids resolved through the engine's symbol table |
| byte / short / int / long vector | `numpy.ndarray` | **zero-copy**, dtype `int8`/`int16`/`int32`/`int64` |
| float vector | `numpy.ndarray` `float64` | **zero-copy** |
| bool vector | `numpy.ndarray` | bit-packed in the engine, widened by it on the way out |
| lazy range (`!n`) | `numpy.ndarray` | materialised first |
| general list | `list` | recursive |
| dict | `dict` | |
| keyed table (`by`) | `amber.Table` | flattened, `key_columns` recorded |
| table | `amber.Table` | |
| function / projection | `amber.Value` | there is no Python object that means "the Amber verb `+`" |

**Integer width is real, not hidden.** Amber stores an integer vector in the
narrowest type that holds every element, so `1 2 3` is physically `int8` and
`1 2 3000000000` is `int64` — both are "long vectors" as far as the language is
concerned. The arrays you get back carry the true dtype rather than being
widened behind your back; `arr.astype("int64")` is one call away when you want
uniformity, and it should be your call.

Going the other way, `Table.from_pandas` maps object/string columns to **symbol**
vectors (four bytes a row, integer comparisons — the reason Amber groups and
joins fast on `sym`) and datetime64 columns to int64 nanoseconds **without
re-basing the epoch**, since Amber's own timestamps count from 2000.01.01 and
silently shifting your data by thirty years would be worse than leaving it
where pandas had it.

---

## Building from source

```bash
AMBER_SRC=/path/to/amber pip install -e .
python -m pytest tests -q
```

`setup.py` looks for the engine in this order:

1. `$AMBER_LIB_DIR` — a directory containing a prebuilt `libamber.so`
2. `$AMBER_SRC` — an engine checkout; runs its own `./build.sh --shared`
3. `vendor/amber/` — a vendored checkout or git submodule
4. `../amber/` — a sibling checkout
5. the system library path

It runs the **engine's** build script rather than reimplementing it, because
that script already knows how to probe this toolchain for LTO, OpenMP and the
right export-map flag, and a second copy of that logic here would rot.

Environment variables:

| | |
|---|---|
| `AMBER_SRC` | Engine checkout to build. |
| `AMBER_LIB_DIR` | Directory holding a prebuilt `libamber.so`. |
| `AMBER_BUNDLE=0` | Do **not** copy the library and `.k` modules into the package. Right for a distro package, wrong for a wheel. |
| `AMBER_FORCE_REBUILD=1` | Rebuild the engine even if a library is already there. |
| `AMBER_HOME` | Where to load the `.k` standard library from at **run** time. |

---

## Building wheels

```bash
scripts/build_wheel.sh                    # a local wheel with the engine bundled
scripts/build_wheel.sh --manylinux        # auditwheel-repaired, redistributable
```

The plain build copies `libamber.so` into `amber/` and links the extension with
an `$ORIGIN` rpath, so the library is found next to the extension wherever the
wheel lands. `auditwheel repair` then does the rest of what a redistributable
wheel needs — hashing the library's name, fixing the rpath, and checking the
glibc symbols the engine actually references.

---

## Threading

The engine is single-threaded at its API boundary. **Every call in this package
runs with the GIL held and never releases it**, which makes the GIL the engine's
lock for free.

That is deliberate, and it is why you will not find `Py_BEGIN_ALLOW_THREADS`
around long evaluations. Releasing the GIL there would let a second Python
thread re-enter the interpreter mid-evaluation, and the resulting corruption
would be intermittent, timing-dependent and effectively unattributable — the
worst possible failure mode in exchange for parallelism you can get safely
another way. If you need concurrency, use several Amber **processes**, which is
what the engine's own `peach` does (fork-based, copy-on-write, no shared heap).

---

## Troubleshooting

**`ImportError: libamber.so: cannot open shared object file`** — the wheel was
built with `AMBER_BUNDLE=0`, or the library was moved after installation. Either
reinstall with bundling on, or put `libamber.so` on `LD_LIBRARY_PATH`.

**`AmberError: 'value` on a name that clearly exists** — the standard library
did not load. Check `am.home()`; if it is `None` you have a bare engine
(primitives only, no `gentq`, no `select`). Set `$AMBER_HOME` to a directory
containing `amber.k`.

**`ImportError: libamber ABI mismatch`** — the extension and the library came
from different engine versions. Rebuild one of them; the message names both ABI
numbers.

**Underscores in Amber names** — `_` is a *verb* in Amber (drop, and floor), so
`my_name` does not lex as one identifier. Use dotted names (`my.name`) or camel
case. This bites everyone once.

**`BufferError: … has no flat buffer to export`** — you asked for a memoryview
of a table, a general list or an atom. Use `.columns()`, `.items()` or
`.scalar()`.

---

## Licence

GNU AGPLv3, matching the engine. See `LICENSE` and `NOTICE`.
