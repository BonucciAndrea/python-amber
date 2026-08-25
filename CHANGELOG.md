# Changelog

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
