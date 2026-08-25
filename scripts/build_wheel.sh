#!/usr/bin/env bash
# scripts/build_wheel.sh  -  build a python-amber wheel with the engine bundled.
#
#   scripts/build_wheel.sh                 a local wheel for this interpreter
#   scripts/build_wheel.sh --manylinux     additionally run auditwheel repair
#   scripts/build_wheel.sh --sdist         source distribution only (no binary)
#
# The plain build copies libamber.so and the eight .k standard-library modules
# into the package directory and links the extension with an $ORIGIN rpath, so
# the wheel is self-contained and relocatable on the machine that built it.
#
# `--manylinux` adds the step that makes it redistributable: auditwheel inspects
# every external library the extension pulls in, copies them into the wheel under
# hashed names, rewrites the rpath to match, and FAILS if the engine references a
# glibc symbol newer than the target tag allows. Run it inside a manylinux
# container -- running it on a modern host produces a wheel that installs
# everywhere and imports nowhere, which is worse than not producing one.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd -P)"

MANYLINUX=0; SDIST_ONLY=0
for a in "$@"; do case "$a" in --manylinux) MANYLINUX=1;; --sdist) SDIST_ONLY=1;; esac; done

: "${AMBER_SRC:=../amber}"
export AMBER_SRC AMBER_BUNDLE=1

echo "python-amber: engine source = $AMBER_SRC"
rm -rf build dist src/amber/libamber.so src/amber/kmodules src/*.egg-info

python3 -m pip install --quiet --upgrade build auditwheel 2>/dev/null || true

if [ "$SDIST_ONLY" = 1 ]; then
  python3 -m build --sdist
  echo "python-amber: sdist in dist/"
  exit 0
fi

python3 -m build --wheel
echo
echo "python-amber: wheel contents"
python3 - <<'PY'
import glob, zipfile
for whl in sorted(glob.glob("dist/*.whl")):
    print(f"  {whl}")
    with zipfile.ZipFile(whl) as z:
        for info in sorted(z.infolist(), key=lambda i: -i.file_size)[:12]:
            print(f"    {info.file_size:>10,}  {info.filename}")
PY

if [ "$MANYLINUX" = 1 ]; then
  echo
  echo "python-amber: auditwheel repair"
  for whl in dist/*.whl; do
    auditwheel show "$whl" || true
    auditwheel repair "$whl" -w dist/manylinux/
  done
  echo "python-amber: repaired wheels in dist/manylinux/"
fi
