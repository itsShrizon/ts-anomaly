"""Pick the right DDL flavour for the configured warehouse backend.

Backends supported:
  - postgres  -> schema/postgres.sql
  - sqlserver -> schema/sqlserver.sql   (T-SQL)
  - sqlite    -> schema/sqlite.sql      (used as the test backend)
"""
from __future__ import annotations

from pathlib import Path

_SCHEMA_DIR = Path(__file__).parent / "schema"
_FILES = {
    "postgres": "postgres.sql",
    "sqlserver": "sqlserver.sql",
    "sqlite": "sqlite.sql",
}


def ddl_for(backend: str) -> str:
    backend = backend.lower()
    if backend not in _FILES:
        raise ValueError(f"unknown backend {backend!r}; expected one of {sorted(_FILES)}")
    return (_SCHEMA_DIR / _FILES[backend]).read_text(encoding="utf-8")


def split_statements(sql: str) -> list[str]:
    """Naive splitter for stdlib sqlite3, which does not accept multi-statement strings.

    Splits on `;` at end-of-line and drops empties / pure comments. Sufficient for the
    DDL files we ship — no procedures or string literals containing `;`.
    """
    out: list[str] = []
    buf: list[str] = []
    for raw in sql.splitlines():
        line = raw.split("--", 1)[0].rstrip()
        if not line:
            continue
        buf.append(line)
        if line.endswith(";"):
            stmt = " ".join(buf).strip().rstrip(";").strip()
            if stmt:
                out.append(stmt)
            buf = []
    tail = " ".join(buf).strip().rstrip(";").strip()
    if tail:
        out.append(tail)
    return out
