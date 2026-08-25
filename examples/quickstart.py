#!/usr/bin/env python3
"""python-amber in about forty lines: generate a market, query it, get a frame.

    python3 examples/quickstart.py
"""
import amber as am


def main() -> None:
    am.init()
    print(f"Amber {am.version()}  (stdlib: {am.home() or 'not loaded'})\n")

    # 1. A synthetic trades/quotes universe, generated inside the engine.
    print("generating 1,000,000 trades + quotes ...")
    am.gentq(1_000_000)
    print(f"  trades: {am.eval('#trades'):,} rows")
    print(f"  quotes: {am.eval('#quotes'):,} rows\n")

    # 2. qSQL, typed the way you would type it at the amber> prompt.
    vwap = am.q("select vwap:wavg[sz;px], n:#px, notional:sum px*sz by sym from trades")
    print("VWAP by symbol")
    print(vwap.to_pandas().head(10), "\n")

    # 3. The thing Amber is actually for: an as-of join of trades to the
    #    prevailing quote, then the spread each trade crossed.
    am.eval("taq:aj[`sym`time; trades; quotes]")
    spread = am.q(
        "select trades:#px, avgSpread:avg ask-bid, avgMid:avg 0.5*bid+ask by sym from taq"
    )
    print("spread paid, by symbol")
    print(spread.to_pandas().head(10), "\n")

    # 4. Zero-copy out. The NumPy array points at the engine's own buffer.
    column = am.q("select from trades").column("px", raw=True)
    try:
        import numpy as np
        prices = np.asarray(column)
        print(f"px column: {prices.dtype}, {prices.nbytes:,} bytes")
        print(f"  engine buffer  {hex(column.handle())}")
        print(f"  numpy buffer   {hex(prices.ctypes.data)}")
        print(f"  same address:  {prices.ctypes.data == column.handle()}\n")
    except ImportError:
        print("(install numpy to see the zero-copy view)\n")

    # 5. And back in. This direction copies -- see the README on why.
    try:
        import numpy as np
        am.set("signal", np.random.default_rng(0).standard_normal(1_000_000))
        print(f"pushed 1M floats back in; engine says avg = {am.eval('avg signal'):.6f}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
