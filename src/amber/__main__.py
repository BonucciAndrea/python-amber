"""``python -m amber`` -- a minimal REPL over the in-process engine.

Not a replacement for the engine's own REPL (``./a``), which has line editing,
completion, syntax highlighting and the ``\\``-command surface. This exists for
the case the native REPL cannot cover: you are inside a Python environment,
possibly a container with only a wheel installed, and you want to poke at the
engine and at Python objects in the same session.

    $ python -m amber
    amber> gentq 1000
    amber> select vwap:wavg[sz;px] by sym from trades
    amber> \\py trades_df = am.q("select from trades").to_pandas()
"""

from __future__ import annotations

import sys

import amber as am


BANNER = """Amber {version} (python-amber, in-process)
  bare qSQL works verbatim:   select vwap:wavg[sz;px] by sym from trades
  \\py <code>                  run a line of Python; `am` is bound
  \\\\  or Ctrl-D               exit
"""


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("-v", "--version"):
        am.init()
        print(am.version())
        return 0
    if argv and argv[0] in ("-c", "--command"):
        am.init()
        print(am.q(" ".join(argv[1:])))
        return 0

    am.init()
    print(BANNER.format(version=am.version()), end="")
    if am.home():
        print(f"  standard library: {am.home()}\n")
    else:
        print("  standard library: NOT LOADED (bare engine)\n")

    scope = {"am": am, "amber": am}
    while True:
        try:
            line = input("amber> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        line = line.strip()
        if not line:
            continue
        if line in ("\\\\", "\\q", "exit", "quit"):
            return 0
        if line.startswith("\\py "):
            try:
                exec(compile(line[4:], "<amber>", "single"), scope)
            except Exception as exc:                      # noqa: BLE001
                print(f"python: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        try:
            result = am.q(line)
        except am.AmberError as exc:
            print(exc, file=sys.stderr)
            continue
        if result is not None:
            print(result)


if __name__ == "__main__":
    raise SystemExit(main())
