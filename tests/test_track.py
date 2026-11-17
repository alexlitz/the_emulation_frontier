"""serving.track on small synthetic snapshots (no network)."""
import datetime as dt
import os

from serving import arbitrage, track


def books(b1=(), b3=()):
    return {"book1": [dict(target=t, spread=s) for t, s in b1], "book2": [],
            "book3": [dict(target=t, workloads={w: dict(spread=s) for w, s in ws.items()}) for t, ws in b3]}


def snap(history, name, b, report=None):
    d = os.path.join(history, name)
    os.makedirs(d)
    track.write_gz(b, os.path.join(d, track.BOOKS))
    if report:
        open(os.path.join(d, "report.md"), "w").write(report)
    return d


def test_diff_categories(tmp_path):
    h = str(tmp_path)
    snap(h, "2026-10-01", books(b1=[("a", 0.30), ("gone", 0.2), ("steady", 0.5)],
                                b3=[("x", {"chat": 0.2, "rag": -0.1, "agent": 0.4})]))
    snap(h, "2026-10-02", books(b1=[("a", 0.45), ("new", 0.1), ("steady", 0.55)],
                                b3=[("x", {"chat": -0.05, "rag": 0.3, "agent": 0.25}), ("y", {"chat": -0.2, "rag": 0.1})]))
    res = track.compare("2026-10-01", "2026-10-02", history=h)
    keys = {c: {k for k, _, _ in v} for c, v in res.items()}
    assert keys["new"] == {("book1", "new", track.BLEND), ("book3", "y", "rag")}       # y/chat is negative: not new
    assert keys["gone"] == {("book1", "gone", track.BLEND)}
    assert keys["flipped"] == {("book3", "x", "chat"), ("book3", "x", "rag")}
    assert keys["moved"] == {("book1", "a", track.BLEND), ("book3", "x", "agent")}     # steady moved only 5 pts
    rep = track.report(res, "2026-10-02", "2026-10-01")
    assert "## chat" in rep and "## rag" in rep and "### flipped" in rep and "| book3 | x | +20.0% | -5.0% |" in rep


def test_run_writes_report_and_gz(tmp_path, monkeypatch):
    h = str(tmp_path / "history")
    snap(h, "2026-10-01", books(b1=[("a", 0.3)]))
    monkeypatch.setattr(arbitrage, "read_json", lambda f, data=None: {"data": [], "f": f})
    monkeypatch.setattr(arbitrage, "books", lambda d: books(b1=[("a", 0.1), ("b", 0.2)]))
    res = track.run(history=h, today=dt.date(2026, 10, 2), do_fetch=False)
    d = os.path.join(h, "2026-10-02")
    assert sorted(os.listdir(d)) == sorted([f + ".gz" for f in track.RAW] + [track.BOOKS, "report.md"])
    assert track.read_gz(os.path.join(d, "or_models.json.gz"))["f"] == "or_models.json"
    assert [k[1] for k, _, _ in res["new"]] == ["b"] and [k[1] for k, _, _ in res["moved"]] == ["a"]
    assert "vs 2026-10-01" in open(os.path.join(d, "report.md")).read()


def test_prune_keeps_reports(tmp_path):
    h = str(tmp_path)
    snap(h, "2026-08-01", books(), report="old report")
    snap(h, "2026-08-19", books(), report="61 days")
    snap(h, "2026-08-20", books(), report="60 days")
    snap(h, "2026-10-19", books())
    os.makedirs(os.path.join(h, "reports"))
    gone = track.prune(h, today=dt.date(2026, 10, 19))
    assert gone == ["2026-08-01", "2026-08-19"]
    assert track.snapshots(h) == ["2026-08-20", "2026-10-19"]
    assert open(os.path.join(h, "reports", "2026-08-01.md")).read() == "old report"
    assert open(os.path.join(h, "reports", "2026-08-19.md")).read() == "61 days"


def test_arbitrage_reads_gz_snapshot(tmp_path):
    d = str(tmp_path)
    track.write_gz({"data": [{"id": "o/m", "hugging_face_id": "Org/M"}]}, os.path.join(d, "or_models.json.gz"))
    track.write_gz({"o/m": {"endpoints": [{"pricing": {"prompt": "1e-6", "completion": "1e-6"},
                                           "provider_name": "P"}]}}, os.path.join(d, "or_endpoints.json.gz"))
    track.write_gz({"data": [{"id": "org/m", "model_class": "c", "context_length": 8,
                              "pricing": {"input": 0.5, "output": 0.5}}]}, os.path.join(d, "fl_models.json.gz"))
    sell, fl, _ = arbitrage.load(d)
    assert arbitrage.book1(sell, fl)[0]["spread"] == 0.5
