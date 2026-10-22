"""Conservative, transactional episode/series generation budgets.

Amounts are integer USD microdollars. Every attempt reserves the entire model
serving ceiling before dispatch. Retries are new attempts. Unknown outcomes hold
their reservation until explicit billing reconciliation.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path

PRICE_CHECKED = "2026-10-22"
PRICE_VALID_THROUGH = date(2026, 12, 31)
PRICE_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"
MAX_INPUT_TOKENS = 8192
MAX_OUTPUT_TOKENS = 16384
RATES = {
    "gemini-3.8-flash-tts": (Decimal("0.5"), Decimal("9")),
    "gemini-3.8-flash-lite-tts": (Decimal("0.5"), Decimal("6")),
}


class BudgetError(ValueError):
    pass


def dollars_to_microdollars(value: str) -> int:
    try:
        number = Decimal(value)
        if not number.is_finite() or number <= 0 or number > 10000:
            raise BudgetError("Budget must be greater than zero and at most $10,000.")
        units = number * 1_000_000
        if units != units.to_integral_value():
            raise BudgetError("Use at most six decimal places for USD amounts.")
        return int(units)
    except InvalidOperation as exc:
        raise BudgetError("Enter a numeric USD budget.") from exc


def reservation_cost(model: str, today: date | None = None) -> int:
    if model not in RATES:
        raise BudgetError("Choose a supported Gemini TTS model.")
    if (today or date.today()) > PRICE_VALID_THROUGH:
        raise BudgetError("The stored price schedule expired; update and verify rates before generation.")
    input_rate, output_rate = RATES[model]
    return int((MAX_INPUT_TOKENS * input_rate + MAX_OUTPUT_TOKENS * output_rate).to_integral_value(rounding=ROUND_CEILING))


def usage_cost(model: str, usage: dict, reserved: int) -> tuple[int, str]:
    """Settle only complete, plausible token counts; otherwise retain full charge."""
    if not isinstance(usage, dict):
        return reserved, "conservative-ceiling"
    input_tokens = usage.get("total_input_tokens")
    output_tokens = usage.get("total_output_tokens")
    if (
        type(input_tokens) is not int
        or type(output_tokens) is not int
        or not 0 <= input_tokens <= MAX_INPUT_TOKENS
        or not 0 < output_tokens <= MAX_OUTPUT_TOKENS
        or usage.get("total_thought_tokens", 0) != 0
        or usage.get("total_tool_use_tokens", 0) != 0
    ):
        return reserved, "conservative-ceiling"
    input_rate, output_rate = RATES[model]
    amount = int((input_tokens * input_rate + output_tokens * output_rate).to_integral_value(rounding=ROUND_CEILING))
    if amount > reserved:
        return reserved, "conservative-ceiling"
    return amount, "provider-token-usage"


class Budget:
    def __init__(self, database: Path):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS budget_limits (
                    scope TEXT PRIMARY KEY, limit_micro INTEGER NOT NULL CHECK(limit_micro > 0)
                );
                CREATE TABLE IF NOT EXISTS reservations (
                    attempt_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL, series_id TEXT NOT NULL,
                    reserved_micro INTEGER NOT NULL CHECK(reserved_micro > 0),
                    charged_micro INTEGER NOT NULL DEFAULT 0 CHECK(charged_micro >= 0),
                    state TEXT NOT NULL CHECK(state IN ('reserved','spent','released')),
                    accounting TEXT NOT NULL DEFAULT 'pending'
                );
            """)

    def connect(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA busy_timeout=30000")
        return db

    @contextmanager
    def transaction(self):
        db = self.connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def totals(db, column: str, identity: str) -> tuple[int, int]:
        if column not in {"episode_id", "series_id"}:
            raise ValueError("Invalid budget scope.")
        row = db.execute(
            f"""SELECT COALESCE(SUM(CASE WHEN state='spent' THEN charged_micro ELSE 0 END),0) spent,
            COALESCE(SUM(CASE WHEN state='reserved' THEN reserved_micro ELSE 0 END),0) held
            FROM reservations WHERE {column}=?""", (identity,)
        ).fetchone()
        return int(row["spent"]), int(row["held"])

    def configure(self, episode_id: str, series_id: str, episode_limit: int, series_limit: int):
        if type(episode_limit) is not int or type(series_limit) is not int or min(episode_limit, series_limit) <= 0:
            raise BudgetError("Episode and series limits must be positive integer microdollars.")
        with self.transaction() as db:
            for prefix, column, identity, limit in (
                ("episode", "episode_id", episode_id, episode_limit),
                ("series", "series_id", series_id, series_limit),
            ):
                spent, held = self.totals(db, column, identity)
                if limit < spent + held:
                    raise BudgetError(f"The {prefix} limit cannot be below spent plus reserved allocations.")
                db.execute(
                    "INSERT INTO budget_limits(scope,limit_micro) VALUES(?,?) "
                    "ON CONFLICT(scope) DO UPDATE SET limit_micro=excluded.limit_micro",
                    (f"{prefix}:{identity}", limit),
                )

    def reserve(self, attempt_id: str, episode_id: str, series_id: str, amount: int, connection=None):
        if type(amount) is not int or amount <= 0:
            raise BudgetError("Reservation must be positive integer microdollars.")
        def insert(db):
            for prefix, column, identity in (
                ("episode", "episode_id", episode_id), ("series", "series_id", series_id)
            ):
                limit = db.execute("SELECT limit_micro FROM budget_limits WHERE scope=?", (f"{prefix}:{identity}",)).fetchone()
                if limit is None:
                    raise BudgetError(f"Set an explicit {prefix} generation budget first.")
                spent, held = self.totals(db, column, identity)
                if spent + held + amount > limit["limit_micro"]:
                    raise BudgetError(f"The {prefix} budget cannot cover this request's maximum allocation.")
            db.execute(
                "INSERT INTO reservations(attempt_id,episode_id,series_id,reserved_micro,state) VALUES(?,?,?,?, 'reserved')",
                (attempt_id, episode_id, series_id, amount),
            )

        if connection is not None:
            if not connection.in_transaction:
                raise BudgetError("An existing reservation transaction is required.")
            insert(connection)
        else:
            with self.transaction() as db:
                insert(db)

    def settle(self, attempt_id: str, charged: int, accounting: str, connection=None):
        def update(db):
            row = db.execute("SELECT * FROM reservations WHERE attempt_id=?", (attempt_id,)).fetchone()
            if row is None or row["state"] != "reserved":
                raise BudgetError("No pending reservation to settle.")
            if type(charged) is not int or not 0 <= charged <= row["reserved_micro"]:
                raise BudgetError("Invalid reconciled charge.")
            db.execute(
                "UPDATE reservations SET state='spent',charged_micro=?,accounting=? WHERE attempt_id=?",
                (charged, accounting, attempt_id),
            )
        if connection is not None:
            if not connection.in_transaction:
                raise BudgetError("An existing accounting transaction is required.")
            update(connection)
        else:
            with self.transaction() as db:
                update(db)

    def release_unsent(self, attempt_id: str):
        """Only for a request known never to have reached dispatch."""
        with self.transaction() as db:
            db.execute(
                "UPDATE reservations SET state='released',accounting='never-dispatched' "
                "WHERE attempt_id=? AND state='reserved'", (attempt_id,)
            )

    def summary(self, episode_id: str, series_id: str) -> dict:
        result = {}
        with self.connect() as db:
            for prefix, column, identity in (
                ("episode", "episode_id", episode_id), ("series", "series_id", series_id)
            ):
                limit = db.execute("SELECT limit_micro FROM budget_limits WHERE scope=?", (f"{prefix}:{identity}",)).fetchone()
                spent, held = self.totals(db, column, identity)
                value = limit["limit_micro"] if limit else None
                result[prefix] = {
                    "id": identity, "limit_microdollars": value,
                    "spent_microdollars": spent, "reserved_microdollars": held,
                    "available_microdollars": None if value is None else value - spent - held,
                }
        return result
