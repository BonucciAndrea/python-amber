#!/usr/bin/env python3
"""Build python-amber: the CPython extension plus, optionally, the engine itself.

There are two things a wheel of this package has to get right, and both of them
are about a shared library that is not written in Python.

**Finding libamber.so.** In descending order of explicitness this script looks
at ``$AMBER_LIB_DIR`` (a prebuilt library), ``$AMBER_SRC`` (an engine checkout to
build), a vendored ``vendor/amber`` tree, a sibling ``../amber`` checkout, and
finally the system library path. When it finds sources rather than a library it
runs the engine's own ``./build.sh --shared`` -- not a reimplementation of it,
because the engine's build script already knows how to probe for LTO, OpenMP and
the right export-map flag on this toolchain, and a second copy of that logic
here would rot.

**Making the wheel self-contained.** ``pip install amber`` should give you a
working engine, not a working binding that then cannot find an engine. So the
build copies both ``libamber.so`` and the eight ``.k`` standard-library modules
into the package directory, and links the extension with an ``$ORIGIN`` rpath so
the library is found next to it at import time regardless of where the wheel was
installed. Set ``AMBER_BUNDLE=0`` to link against a system-wide engine instead --
the right choice for a distro package, the wrong one for a wheel.

For a redistributable manylinux wheel, run ``auditwheel repair`` afterwards: it
will see the bundled library, rename it with a hash, and fix the rpath. That
step is deliberately left to the release pipeline (scripts/build_wheel.sh)
rather than done here, because it needs a manylinux container to be meaningful.
"""

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

HERE = Path(__file__).parent.resolve()
PKG = HERE / "src" / "amber"

#: The engine's standard library, in the order repl.k loads it. Order matters:
#: sys.k needs temporal.k, hdb.k and ipc.k need the rest.
K_MODULES = ["amber.k", "fin.k", "std.k", "qsql.k",
             "temporal.k", "sys.k", "hdb.k", "ipc.k"]

IS_MAC = sys.platform == "darwin"
SOEXT = "dylib" if IS_MAC else "so"
SONAME = f"libamber.{SOEXT}"


def _log(msg):
    print(f"python-amber: {msg}", flush=True)


def _engine_source_dir():
    """A directory that looks like an Amber engine checkout, or None."""
    candidates = []
    if os.environ.get("AMBER_SRC"):
        candidates.append(Path(os.environ["AMBER_SRC"]))
    candidates += [HERE / "vendor" / "amber", HERE.parent / "amber"]
    for c in candidates:
        if c and (c / "build.sh").is_file() and (c / "src" / "ext.h").is_file():
            return c.resolve()
    return None


def _build_engine(src_dir: Path) -> Path:
    """Run the engine's own build script and return the library it produced."""
    lib = src_dir / SONAME
    if lib.is_file() and not os.environ.get("AMBER_FORCE_REBUILD"):
        _log(f"using existing {lib}")
        return lib
    _log(f"building the engine in {src_dir} (./build.sh --shared)")
    subprocess.check_call(["bash", "./build.sh", "--shared"], cwd=str(src_dir))
    if not lib.is_file():
        raise SystemExit(f"python-amber: {src_dir}/build.sh --shared produced no {SONAME}")
    return lib


def _find_engine():
    """Return ``(library_path_or_None, k_module_dir_or_None, link_dir)``."""
    lib_dir = os.environ.get("AMBER_LIB_DIR")
    if lib_dir:
        lib = Path(lib_dir) / SONAME
        if not lib.is_file():
            raise SystemExit(f"python-amber: $AMBER_LIB_DIR is set but {lib} does not exist")
        src = _engine_source_dir()
        return lib, (src if src else None), str(Path(lib_dir).resolve())

    src = _engine_source_dir()
    if src:
        lib = _build_engine(src)
        return lib, src, str(src)

    # Nothing local: assume a system-installed engine and let the linker find it.
    _log("no engine checkout found; linking against a system libamber "
         "(set $AMBER_SRC or $AMBER_LIB_DIR to change that)")
    return None, None, None


_BUNDLED = {}


def _bundle_engine():
    """Copy libamber.so and the .k standard library into the package directory.

    Returns ``(library_path_or_None, link_dir_or_None)``.

    Called from BOTH build_py and build_ext, and memoised, because of an
    ordering trap that produces a wheel which looks fine and is broken:
    setuptools runs build_py (which collects package_data) BEFORE build_ext. Do
    the bundling only in build_ext and the .k modules land in the build tree
    after the file list was taken -- so libamber.so ships (it is copied straight
    into build_lib) and the eight .k modules silently do not. The wheel then
    installs, imports, and boots a BARE engine: no gentq, no select, and an
    error message pointing at the user's query rather than at the packaging.
    """
    if "result" in _BUNDLED:
        return _BUNDLED["result"]

    lib, ksrc, link_dir = _find_engine()
    bundle = os.environ.get("AMBER_BUNDLE", "1") != "0"

    if bundle and lib:
        PKG.mkdir(parents=True, exist_ok=True)
        dest = PKG / SONAME
        if not dest.exists() or dest.stat().st_mtime < lib.stat().st_mtime:
            shutil.copy2(lib, dest)
            _log(f"bundled {lib.name} -> {dest.relative_to(HERE)}")
    if bundle and ksrc:
        kdir = PKG / "kmodules"
        kdir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for name in K_MODULES:
            source = ksrc / name
            if source.is_file():
                shutil.copy2(source, kdir / name)
                copied += 1
        _log(f"bundled {copied}/{len(K_MODULES)} standard-library modules "
             f"-> {kdir.relative_to(HERE)}")
        if copied != len(K_MODULES):
            _log("WARNING: some .k modules were missing; the installed package "
                 "will boot with a partial vocabulary")

    _BUNDLED["result"] = (lib if bundle else None, link_dir)
    return _BUNDLED["result"]


class AmberBuildPy(build_py):
    """Bundle before the package file list is taken."""

    def run(self):
        _bundle_engine()
        super().run()


class AmberBuildExt(build_ext):
    """Resolve the engine, link against it, and keep the bundled copy in step."""

    def run(self):
        lib, link_dir = _bundle_engine()
        if link_dir:
            self.extensions[0].library_dirs.append(link_dir)
        super().run()
        # Also place the library beside the freshly built extension, so an
        # in-place / editable build works without a separate step.
        if lib:
            out = Path(self.build_lib) / "amber"
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(lib, out / SONAME)
            kdir_src = PKG / "kmodules"
            if kdir_src.is_dir():
                kdir_dst = out / "kmodules"
                kdir_dst.mkdir(parents=True, exist_ok=True)
                for k in kdir_src.glob("*.k"):
                    shutil.copy2(k, kdir_dst / k.name)


def _extension():
    rpath = []
    extra_link = []
    if sys.platform.startswith("linux"):
        # $ORIGIN is resolved by the dynamic loader relative to the .so being
        # loaded -- i.e. the installed package directory -- which is what makes
        # the wheel relocatable. It has to be escaped from the shell AND from
        # setuptools, hence the literal.
        extra_link.append("-Wl,-rpath,$ORIGIN")
    elif IS_MAC:
        extra_link.append("-Wl,-rpath,@loader_path")

    return Extension(
        name="amber._amber",
        sources=["src/py_amber.c"],
        include_dirs=["include"],
        libraries=["amber"],
        library_dirs=[],
        runtime_library_dirs=rpath,
        extra_link_args=extra_link,
        extra_compile_args=["-std=c99"] if not sys.platform.startswith("win") else [],
        define_macros=[("PY_SSIZE_T_CLEAN", "1")],
    )


setup(
    ext_modules=[_extension()],
    cmdclass={"build_py": AmberBuildPy, "build_ext": AmberBuildExt},
)
