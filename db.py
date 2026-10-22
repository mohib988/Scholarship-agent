"""SQLite storage for student records and their scholarship runs."""
import os, json, sqlite3, hashlib
from datetime import datetime

DB_PATH = os.getenv("SCHOLARSHIP_DB", "students.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    email          TEXT UNIQUE NOT NULL,
    name           TEXT,
    degree         TEXT,
    field          TEXT,
    nationality    TEXT,
    target_country TEXT,
    profile_json   TEXT NOT NULL,
    cv_text        TEXT,
    cv_hash        TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_students_cv_hash ON students(cv_hash);
CREATE TABLE IF NOT EXISTS runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id     INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    created_at     TEXT NOT NULL,
    degree         TEXT,
    target_country TEXT,
    n_scholarships INTEGER,
    result_json    TEXT NOT NULL
);
"""


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


def _now():
    return datetime.now().isoformat(timespec="seconds")


def init():
    with _conn() as c:
        c.executescript(SCHEMA)


def cv_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _student(row):
    if not row:
        return None
    d = dict(row)
    d["profile"] = json.loads(d.pop("profile_json"))
    return d


def upsert_student(profile, cv_text=None, cv_hash=None):
    """Insert or update by email. Keeps the stored CV if no new one is given. Returns the student id."""
    email = (profile.get("email") or "").strip().lower()
    if not email:
        raise ValueError("email is required to save a student")
    profile = {**profile, "email": email}
    cols = dict(name=profile.get("name"), degree=profile.get("degree"), field=profile.get("field"),
                nationality=profile.get("nationality"), target_country=profile.get("target_country"),
                profile_json=json.dumps(profile, default=str))
    now = _now()
    with _conn() as c:
        row = c.execute("SELECT id FROM students WHERE email = ?", (email,)).fetchone()
        if row:
            sets = ", ".join(f"{k} = ?" for k in cols) + ", updated_at = ?"
            vals = list(cols.values()) + [now]
            if cv_text:
                sets += ", cv_text = ?, cv_hash = ?"
                vals += [cv_text, cv_hash]
            c.execute(f"UPDATE students SET {sets} WHERE id = ?", vals + [row["id"]])
            return row["id"]
        cur = c.execute(
            "INSERT INTO students (email, name, degree, field, nationality, target_country, profile_json, "
            "cv_text, cv_hash, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (email, cols["name"], cols["degree"], cols["field"], cols["nationality"], cols["target_country"],
             cols["profile_json"], cv_text, cv_hash, now, now))
        return cur.lastrowid


def get_student(student_id):
    with _conn() as c:
        return _student(c.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone())


def find_by_email(email):
    with _conn() as c:
        return _student(c.execute("SELECT * FROM students WHERE email = ?",
                                  ((email or "").strip().lower(),)).fetchone())


def find_by_cv_hash(h):
    with _conn() as c:
        return _student(c.execute("SELECT * FROM students WHERE cv_hash = ? ORDER BY updated_at DESC",
                                  (h,)).fetchone())


def list_students():
    with _conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT id, email, name, degree, field, target_country, updated_at FROM students "
            "ORDER BY updated_at DESC").fetchall()]


def delete_student(student_id):
    with _conn() as c:
        c.execute("DELETE FROM students WHERE id = ?", (student_id,))


def save_run(student_id, result):
    p = result.get("profile") or {}
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO runs (student_id, created_at, degree, target_country, n_scholarships, result_json) "
            "VALUES (?,?,?,?,?,?)",
            (student_id, _now(), p.get("degree"), p.get("target_country"), len(result.get("ranked") or []),
             json.dumps(result, default=str)))
        return cur.lastrowid


def list_runs(student_id, limit=20):
    with _conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT id, created_at, degree, target_country, n_scholarships FROM runs WHERE student_id = ? "
            "ORDER BY id DESC LIMIT ?", (student_id, limit)).fetchall()]


def get_run(run_id):
    with _conn() as c:
        r = c.execute(
            "SELECT * FROM runs WHERE id = ?",
            (run_id,)
        ).fetchone()

        if not r:
            return None

        d = dict(r)
        d["result"] = json.loads(d.pop("result_json"))
        return d