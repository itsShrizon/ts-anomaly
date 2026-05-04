"""Scheduled-report builder. Emits CSV + a single self-contained HTML summary
that the MIS/FIS team can open without any tooling. Pure-stdlib output.
"""
from __future__ import annotations

import csv
import html
from collections.abc import Sequence
from pathlib import Path

from . import queries


def write_daily_csv(conn, run_id: int, out: Path, paramstyle: str = "qmark") -> int:
    rows = queries.daily_anomaly_counts(conn, run_id, paramstyle)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["full_date", "n_windows", "n_anomaly_windows", "mean_score", "peak_score"])
        w.writerows(rows)
    return len(rows)


def write_top_n_csv(conn, run_id: int, out: Path, n: int = 50,
                    paramstyle: str = "qmark") -> int:
    rows = queries.top_n_anomaly_windows(conn, run_id, n, paramstyle)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts", "window_start", "window_end", "score", "label"])
        w.writerows(rows)
    return len(rows)


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def cell(x: object) -> str:
        if isinstance(x, float):
            return f"{x:.4f}"
        return html.escape("" if x is None else str(x))

    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell(c)}</td>" for c in r) + "</tr>" for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


_HTML_TPL = """<!doctype html>
<html><head><meta charset="utf-8"><title>Anomaly Report — run {run_id}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
h1 {{ margin-bottom: .25rem; }}
.section {{ margin: 1.5rem 0; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: .35rem .5rem; text-align: left; }}
th {{ background: #f4f4f4; }}
.kpi {{ display: inline-block; margin-right: 1.5rem; }}
.kpi b {{ font-size: 1.4rem; }}
</style></head><body>
<h1>Anomaly detection — run {run_id}</h1>
<div class="section">
  <span class="kpi">Precision <b>{precision:.3f}</b></span>
  <span class="kpi">Recall    <b>{recall:.3f}</b></span>
  <span class="kpi">F1        <b>{f1:.3f}</b></span>
  <span class="kpi">TP/FP/FN/TN <b>{tp}/{fp}/{fn}/{tn}</b></span>
</div>
<div class="section"><h2>Daily counts</h2>{daily}</div>
<div class="section"><h2>Top {top_n} anomaly windows</h2>{top}</div>
</body></html>
"""


def write_html(conn, run_id: int, out: Path, top_n: int = 20,
               paramstyle: str = "qmark") -> Path:
    daily = queries.daily_anomaly_counts(conn, run_id, paramstyle)
    top = queries.top_n_anomaly_windows(conn, run_id, top_n, paramstyle)
    tp, fp, fn, tn = queries.confusion(conn, run_id, paramstyle)
    p, r, f1 = queries.precision_recall_f1(conn, run_id, paramstyle)
    daily_html = _html_table(
        ["full_date", "n_windows", "n_anomaly_windows", "mean_score", "peak_score"], daily
    )
    top_html = _html_table(
        ["ts", "window_start", "window_end", "score", "label"], top
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        _HTML_TPL.format(
            run_id=run_id, precision=p, recall=r, f1=f1,
            tp=tp, fp=fp, fn=fn, tn=tn,
            daily=daily_html, top_n=top_n, top=top_html,
        ),
        encoding="utf-8",
    )
    return out


def write_bundle(conn, run_id: int, out_dir: Path, *,
                 top_n: int = 50, paramstyle: str = "qmark") -> dict[str, Path]:
    """Produce the standard per-run report bundle: daily.csv, top.csv, summary.html."""
    out_dir.mkdir(parents=True, exist_ok=True)
    daily_csv = out_dir / "daily.csv"
    top_csv = out_dir / "top.csv"
    summary = out_dir / "summary.html"
    write_daily_csv(conn, run_id, daily_csv, paramstyle)
    write_top_n_csv(conn, run_id, top_csv, top_n, paramstyle)
    write_html(conn, run_id, summary, min(top_n, 20), paramstyle)
    return {"daily": daily_csv, "top": top_csv, "summary": summary}
