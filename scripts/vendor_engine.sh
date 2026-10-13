#!/usr/bin/env bash
# scripts/vendor_engine.sh  -  copy the engine's sources into vendor/amber.
#
#   scripts/vendor_engine.sh [ENGINE_DIR]     default: $AMBER_SRC, else ../amber
#
# The sdist on PyPI has to build anywhere pip can run a C compiler, with no
# Amber checkout next to it. So the release pipeline vendors exactly what
# `./build.sh --shared` needs (the C sources, the export map, the build script)
# plus the eight .k standard-library modules and the licence, and setup.py
# finds them in vendor/amber. Nothing compiled is copied.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd -P)"

SRC="${1:-${AMBER_SRC:-../amber}}"
[ -f "$SRC/build.sh" ] && [ -f "$SRC/src/ext.h" ] || { echo "vendor_engine: $SRC is not an Amber checkout" >&2; exit 1; }

rm -rf vendor/amber
mkdir -p vendor/amber/src
cp "$SRC"/build.sh vendor/amber/
cp "$SRC"/src/*.c "$SRC"/src/*.h vendor/amber/src/
[ -f "$SRC/src/libamber.map" ] && cp "$SRC/src/libamber.map" vendor/amber/src/
for k in amber.k fin.k std.k qsql.k temporal.k sys.k hdb.k ipc.k; do cp "$SRC/$k" vendor/amber/; done
for f in LICENSE NOTICE; do [ -f "$SRC/$f" ] && cp "$SRC/$f" vendor/amber/; done
chmod +x vendor/amber/build.sh

# the engine's ABI and this package's copy of it must agree
diff -q <(sed -n '/#define AMBER_CAPI_ABI/p' "$SRC/src/ext.h" | head -1) \
        <(sed -n '/#define AMBER_CAPI_ABI/p' include/amber_capi.h | head -1) >/dev/null \
  || echo "vendor_engine: WARNING the C API ABI line differs between ext.h and include/amber_capi.h" >&2

rev=$(git -C "$SRC" describe --tags --always 2>/dev/null || echo unknown)
echo "$rev" > vendor/amber/ENGINE_VERSION
echo "vendor_engine: vendored Amber $rev ($(du -sh vendor/amber | cut -f1))"
