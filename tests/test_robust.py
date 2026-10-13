"""Lifetime, leaks, threads, errors and the command line."""
import gc
import os
import subprocess
import sys
import threading

import pytest

np = pytest.importorskip("numpy")


def _rss_mb():
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except (OSError, ValueError, AttributeError):
        pytest.skip("RSS needs /proc")


# ---- lifetime ----------------------------------------------------------------

def test_view_outlives_every_python_reference(am):
    a = am.eval("1000?1.0")
    am.eval("junk:10000?1.0")
    gc.collect()
    assert a.shape == (1000,) and np.isfinite(a).all()


def test_view_survives_the_global_being_rebound(am):
    am.eval("rb:til 1000")
    a = am.get("rb")
    am.eval("rb:0")
    am.eval("junk2:til 100000")
    assert a[-1] == 999


def test_column_views_survive_the_table(am):
    col = am.q("([]x:til 100)")["x"]
    gc.collect()
    am.eval("junk3:til 100000")
    assert col.sum() == 4950


@pytest.mark.parametrize("src", ["til 200000", "200000?1.0", "200000#`a`b", "([]a:til 50000;b:50000?`x`y)",
                                 "200000#2026.01.01", "20000#enlist \"str\""])
def test_no_leak(am, src):
    for _ in range(20):                                # warm the allocator first
        am.to_python(am.eval(src, raw=True))
    gc.collect()
    before = _rss_mb()
    for _ in range(150):
        v = am.eval(src, raw=True)
        am.to_python(v)
        v.release() if not v.is_table() else None
        del v
    gc.collect()
    assert _rss_mb() - before < 40, f"RSS grew by {_rss_mb() - before:.0f} MB"


def test_no_leak_converting_in(am):
    data = np.arange(200_000, dtype=np.int64)
    for _ in range(20):
        am.convert(data)
    gc.collect()
    before = _rss_mb()
    for _ in range(200):
        am.convert(data)
        am.convert(list(range(1000)))
        am.convert({"a": 1, "b": "x"})
    gc.collect()
    assert _rss_mb() - before < 40


def test_no_leak_on_errors(am):
    for _ in range(50):
        with pytest.raises(am.AmberError):
            am.eval("1+`a")
    gc.collect()
    before = _rss_mb()
    for _ in range(2000):
        with pytest.raises(am.AmberError):
            am.eval("1+`a")
        with pytest.raises(OverflowError):
            am.convert([1, 2**70])
    gc.collect()
    assert _rss_mb() - before < 20


# ---- errors ------------------------------------------------------------------

@pytest.mark.parametrize("src, word", [("1+`a", "type"), ("undefinedname42", "undefinedname42"),
                                      ("1 2+1 2 3", "length"), ("'custom", "custom")])
def test_errors_name_the_problem(am, src, word):
    with pytest.raises(am.AmberError, match=word):
        am.eval(src)
    assert am.eval("2+2") == 4                         # the engine is still fine


def test_syntax_errors(am):
    with pytest.raises(am.AmberError):
        am.eval("(1;2")
    with pytest.raises(am.AmberError):
        am.q("select from where")
    assert am.eval("2+2") == 4


def test_type_errors_on_api_misuse(am):
    with pytest.raises(TypeError):
        am.eval(42)
    with pytest.raises(TypeError):
        am.q(b"1+1")


def test_deep_recursion_is_an_error_not_a_crash(am):
    code = "import amber as am; am.init()\ntry:\n    am.eval('f:{1+f x}; f 1')\nexcept am.AmberError as e:\n    print('err', e)\nprint('alive', am.eval('1+1'))"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert "alive 2" in r.stdout or r.returncode != 0 and r.returncode > 0, r.stderr[-500:]


# ---- set / get / call --------------------------------------------------------

def test_set_get_call(am):
    am.set("pyx", np.arange(5))
    assert am.get("pyx").tolist() == [0, 1, 2, 3, 4]
    am.eval("pyf:{x*y}")
    assert am.call("pyf", 6, 7) == 42
    with pytest.raises(am.AmberError):
        am.get("not_defined_anywhere")


def test_call_arity(am):
    am.eval("f9:{[a;b;c;d;e;f;g;h]a+b+c+d+e+f+g+h}")
    assert am.call("f9", *range(8)) == 28
    with pytest.raises((TypeError, ValueError, am.AmberError)):
        am.call("f9", *range(9))


def test_load_script(am, tmp_path):
    p = tmp_path / "s.q"
    p.write_text("tbl9:([]a:1 2 3;b:`x`y`z)\nr9:select a\n  from tbl9 where a>1\n", encoding="utf-8")
    am.load(str(p))
    assert am.get("r9")["a"].tolist() == [2, 3]
    with pytest.raises(FileNotFoundError):
        am.load(str(tmp_path / "missing.k"))
    with pytest.raises(ValueError):
        am.load("bad\nname.k")


def test_multiline_q(am):
    am.eval("ml:([]a:1 2 3)")
    out = am.q("select a\n  from ml where a<3")
    assert out["a"].tolist() == [1, 2]


@pytest.mark.xfail(strict=True, reason="engine: qjg stops a query at the line that brings its from (Amber 2.9.0)")
def test_multiline_q_where_on_its_own_line(am):
    am.eval("ml2:([]a:1 2 3)")
    assert am.q("select a\n  from ml2\n  where a<3")["a"].tolist() == [1, 2]


# ---- threads -----------------------------------------------------------------

def test_threads_hammering_the_engine(am):
    errors = []

    def work(i):
        try:
            for j in range(200):
                am.set(f"th{i}", np.full(10, j, dtype=np.int64))
                assert am.eval(f"sum th{i}") == 10 * j
                assert am.q(f"select sum a from ([]a:th{i})")["a"][0] == 10 * j
        except Exception as e:                         # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors[:3]


def test_views_shared_across_threads(am):
    a = am.eval("til 100000")
    sums = []
    threads = [threading.Thread(target=lambda: sums.append(int(a.sum()))) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sums == [4999950000] * 8


# ---- command line ------------------------------------------------------------

def _cli(*args, stdin=None):
    return subprocess.run([sys.executable, "-m", "amber", *args], input=stdin,
                          capture_output=True, text=True, timeout=60)


def test_cli_version(am):
    r = _cli("--version")
    assert r.returncode == 0 and r.stdout.strip() == am.version()


def test_cli_command(am):
    r = _cli("-c", "1+1")
    assert r.returncode == 0 and r.stdout.strip() == "2"


def test_cli_repl(am):
    r = _cli(stdin="x:6*7\nx\n1+`a\n\\py print('py', am.eval('x'))\n\\\\\n")
    assert r.returncode == 0
    assert "42" in r.stdout and "py 42" in r.stdout
    assert "type" in r.stderr


def test_shutdown_then_keep_going():
    code = ("import amber as am; am.init(); a = am.eval('til 10'); am.shutdown()\n"
            "print(am.eval('1+1'), int(a.sum()), am.q('select sum x from ([]x:1 2 3)')['x'][0])")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.split() == ["2", "45", "6"]


def test_pickle_across_processes(am):
    import pickle
    blob = pickle.dumps(am.eval("([]a:1 2;s:`x`y;d:2026.01.01 2026.01.02)", raw=True))
    code = ("import pickle,sys,amber as am; v = pickle.loads(sys.stdin.buffer.read());"
            "print(am.Table(v)['d'].astype(str).tolist())")
    r = subprocess.run([sys.executable, "-c", code], input=blob, capture_output=True, timeout=20)
    assert r.returncode == 0, r.stderr[-500:]
    assert b"2026-01-02" in r.stdout
