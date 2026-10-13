# Changelog

## 2.9.0

- **`pip install amber-lang`**: it's on PyPI now, still `import amber`. Wheels for Linux (x86_64, aarch64) and
  Apple silicon on Python 3.10 to 3.14, and the sdist carries the engine's source, so it builds anywhere with a C
  compiler. Releases go out from a tag, through trusted publishing.
- **Amber 2.9's temporals**: dates, months, timestamps, timespans, times, seconds and minutes come back as
  `datetime64`/`timedelta64` with the epoch moved right (Amber counts from 2000, NumPy from 1970) and nulls as `NaT`.
  Timespans stay zero-copy. `datetime`, `date`, `time`, `timedelta`, NumPy and pandas datetimes go in as the real
  thing, aware ones as UTC. They used to go in as plain nanosecond numbers.
- **Arrow both ways, properly**: `from_arrow` takes sliced and chunked tables, readers, dictionaries, decimals, every
  int width, bools with nulls and every temporal type. Things it can't hold (nested, binary, a uint64 past the int64
  range) say so, they used to read garbage. `to_arrow` gives temporal columns their Arrow types and bytes as `int8`.
- **pandas**: nullable `Int64`/`Float64`/`boolean`, categoricals and tz-aware columns import, duplicate column names
  are an error, and `to_pandas(categorical=True)` gives symbols as a `Categorical` straight from the engine's ids.
- **Sturdier values**: `release()`, `equals()`, pickling, `symbol_codes()`. A released value raises instead of
  crashing, views can't be written through, and every NumPy view keeps its value alive.
- **Conversions in**: big-endian, strided, `float16` and unsigned buffers, NumPy scalars, and ints past int64 are
  an `OverflowError`, they used to wrap.
- `am.q` takes a query over several lines, `am.load` loads `.q` scripts and databases as `\l` does, and
  `Table` got `head`, `tail`, `dtypes`, `to_numpy` and `from_polars`.
- 255 tests, up from 30: Hypothesis round trips, leak and thread checks, and the CLI.

## 1.9.6 — first release

The Amber engine in your Python process, with zero-copy NumPy, pandas and Arrow interop.

Tracks Amber engine release 1.9.6, which added the `libamber.so` dynamic C API
this package is built on. Everything here lives outside the engine repository and
reaches it only through that documented seam.

- **Zero-copy out.** An Amber vector is exposed through PEP 3118, so
  `numpy.asarray(v)`, `memoryview(v)` and `pyarrow.py_buffer(v)` all alias the
  engine's own column payload and all keep it alive for as long as they exist.
  One implementation, no NumPy or Arrow header anywhere in the build.
- **Dynamic dispatch.** `am.gentq(10_000_000)` resolves whatever `gentq` is in
  the engine's namespace at the moment you touch it, dotted names included.
  Nothing in the package enumerates Amber's vocabulary, so it cannot fall behind.
- **Bare qSQL.** `am.q("select vwap:wavg[sz;px] by sym from trades")` — the
  rewrite the REPL applies to interactive lines, applied where the REPL cannot
  reach.
- **Self-contained wheels.** `libamber.so` and the eight `.k` standard-library
  modules ship inside the wheel, so `pip install amber` gives a working engine
  rather than a working binding that cannot find one.
- **Keyed results** (`select … by sym`) arrive as a table with the grouping
  columns recorded, and `to_pandas()` puts them on the index.
- 30 tests, including pointer-equality proofs that the NumPy array and the Arrow
  buffer are the engine's own memory.

### Licence

GNU AGPLv3, matching the engine.
