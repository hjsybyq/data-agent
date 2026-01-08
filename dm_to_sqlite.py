import argparse
import os
import sqlite3
import sys
from decimal import Decimal

try:
    import dmPython  # type: ignore
except Exception as exc:
    dmPython = None
    _dm_import_error = exc


TYPE_MAP = {
    "CHAR": "TEXT",
    "NCHAR": "TEXT",
    "VARCHAR": "TEXT",
    "VARCHAR2": "TEXT",
    "NVARCHAR2": "TEXT",
    "TEXT": "TEXT",
    "CLOB": "TEXT",
    "NCLOB": "TEXT",
    "LONG": "TEXT",
    "INT": "INTEGER",
    "INTEGER": "INTEGER",
    "BIGINT": "INTEGER",
    "SMALLINT": "INTEGER",
    "TINYINT": "INTEGER",
    "NUMBER": "REAL",
    "DECIMAL": "REAL",
    "NUMERIC": "REAL",
    "FLOAT": "REAL",
    "DOUBLE": "REAL",
    "REAL": "REAL",
    "DATE": "TEXT",
    "TIME": "TEXT",
    "TIMESTAMP": "TEXT",
    "DATETIME": "TEXT",
    "BLOB": "BLOB",
    "BINARY": "BLOB",
    "VARBINARY": "BLOB",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Dameng to SQLite.")
    parser.add_argument("--host", default=os.getenv("DM_HOST", "10.100.15.201"))
    parser.add_argument("--port", type=int, default=int(os.getenv("DM_PORT", "15236")))
    parser.add_argument("--user", default=os.getenv("DM_USER", "sgsmcp"))
    parser.add_argument("--password", default=os.getenv("DM_PASSWORD", "sgsmcp@2017"))
    parser.add_argument("--schema", default=os.getenv("DM_SCHEMA", ""))
    parser.add_argument("--sqlite", default="dm_export.sqlite")
    parser.add_argument("--chunk", type=int, default=2000)
    parser.add_argument(
        "--tables",
        default=os.getenv("DM_TABLES", ""),
        help="Comma-separated table list. If empty, export all tables.",
    )
    return parser.parse_args()


def dm_connect(host: str, port: int, user: str, password: str):
    if dmPython is None:
        raise RuntimeError(
            "dmPython is not installed. Install it first, e.g. `pip install dmPython`."
        ) from _dm_import_error
    return dmPython.connect(
        user=user,
        password=password,
        server=host,
        port=port,
    )


def list_tables(cur, schema: str):
    if schema:
        cur.execute(
            "SELECT TABLE_NAME FROM ALL_TABLES WHERE OWNER = ? ORDER BY TABLE_NAME",
            (schema.upper(),),
        )
    else:
        cur.execute("SELECT TABLE_NAME FROM USER_TABLES ORDER BY TABLE_NAME")
    return [row[0] for row in cur.fetchall()]


def list_columns(cur, table: str, schema: str):
    if schema:
        cur.execute(
            """
            SELECT COLUMN_NAME, DATA_TYPE, DATA_LENGTH, DATA_PRECISION, DATA_SCALE
            FROM ALL_TAB_COLUMNS
            WHERE OWNER = ? AND TABLE_NAME = ?
            ORDER BY COLUMN_ID
            """,
            (schema.upper(), table),
        )
    else:
        cur.execute(
            """
            SELECT COLUMN_NAME, DATA_TYPE, DATA_LENGTH, DATA_PRECISION, DATA_SCALE
            FROM USER_TAB_COLUMNS
            WHERE TABLE_NAME = ?
            ORDER BY COLUMN_ID
            """,
            (table,),
        )
    return cur.fetchall()


def map_type(dm_type: str, precision, scale) -> str:
    if not dm_type:
        return "TEXT"
    base = dm_type.upper()
    sqlite_type = TYPE_MAP.get(base, "TEXT")
    if base in ("NUMBER", "DECIMAL", "NUMERIC"):
        if scale is None or scale == 0:
            return "INTEGER"
    return sqlite_type


def ensure_table(sqlite_cur, table: str, columns):
    col_defs = []
    for name, dm_type, _length, precision, scale in columns:
        sqlite_type = map_type(dm_type, precision, scale)
        col_defs.append(f'"{name}" {sqlite_type}')
    ddl = f'CREATE TABLE IF NOT EXISTS "{table}" ({", ".join(col_defs)});'
    sqlite_cur.execute(ddl)


def copy_table(dm_cur, sqlite_cur, table: str, schema: str, chunk: int):
    columns = list_columns(dm_cur, table, schema)
    if not columns:
        return 0
    ensure_table(sqlite_cur, table, columns)

    col_names = [c[0] for c in columns]
    col_sql = ", ".join(f'"{name}"' for name in col_names)
    if schema:
        select_sql = f'SELECT {col_sql} FROM "{schema}"."{table}"'
    else:
        select_sql = f'SELECT {col_sql} FROM "{table}"'

    dm_cur.execute(select_sql)

    placeholders = ", ".join("?" for _ in col_names)
    insert_sql = f'INSERT INTO "{table}" ({col_sql}) VALUES ({placeholders})'

    total = 0
    while True:
        rows = dm_cur.fetchmany(chunk)
        if not rows:
            break
        normalized_rows = []
        for row in rows:
            normalized = []
            for value in row:
                if isinstance(value, Decimal):
                    normalized.append(str(value))
                else:
                    normalized.append(value)
            normalized_rows.append(tuple(normalized))
        sqlite_cur.executemany(insert_sql, normalized_rows)
        total += len(rows)
    return total


def main() -> int:
    args = parse_args()
    schema = args.schema.strip()
    table_filter = [t.strip().upper() for t in args.tables.split(",") if t.strip()]

    dm_conn = dm_connect(args.host, args.port, args.user, args.password)
    dm_cur = dm_conn.cursor()

    sqlite_conn = sqlite3.connect(args.sqlite)
    sqlite_cur = sqlite_conn.cursor()

    try:
        tables = list_tables(dm_cur, schema)
        if table_filter:
            tables = [t for t in tables if t.upper() in table_filter]
        if not tables:
            print("No tables found.")
            return 1
        for table in tables:
            row_count = copy_table(dm_cur, sqlite_cur, table, schema, args.chunk)
            sqlite_conn.commit()
            print(f"{table}: {row_count} rows")
    finally:
        sqlite_conn.close()
        dm_cur.close()
        dm_conn.close()

    print(f"Done. SQLite file: {args.sqlite}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
