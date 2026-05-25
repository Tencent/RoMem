from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional


_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_MONTH_NAME_RE = (
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_SEASON_RE = re.compile(r"\b(spring|summer|fall|autumn|winter)\s+((?:19|20)\d{2})\b", re.I)
_QUARTER_RE = re.compile(r"\bq([1-4])\s*((?:19|20)\d{2})\b", re.I)


def _parse_iso_like(text: str) -> Optional[datetime]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        # Handles full ISO strings and "YYYY-MM-DD".
        if _ISO_RE.match(text):
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None
    return None


def _month_to_int(token: str) -> Optional[int]:
    if not token:
        return None
    token = token.lower()
    if token.startswith("jan"):
        return 1
    if token.startswith("feb"):
        return 2
    if token.startswith("mar"):
        return 3
    if token.startswith("apr"):
        return 4
    if token == "may":
        return 5
    if token.startswith("jun"):
        return 6
    if token.startswith("jul"):
        return 7
    if token.startswith("aug"):
        return 8
    if token.startswith("sep"):
        return 9
    if token.startswith("oct"):
        return 10
    if token.startswith("nov"):
        return 11
    if token.startswith("dec"):
        return 12
    return None


def _parse_numeric_date(text: str, reference_time: Optional[datetime]) -> Optional[datetime]:
    text = (text or "").strip()
    if not text:
        return None
    m = re.search(r"\b(?P<year>(?:19|20)\d{2})[/-](?P<month>\d{1,2})(?:[/-](?P<day>\d{1,2}))?\b", text)
    if m:
        year = int(m.group("year"))
        month = int(m.group("month"))
        day = int(m.group("day")) if m.group("day") else 1
        try:
            return datetime(year, month, day, tzinfo=timezone.utc)
        except ValueError:
            return None
    m = re.search(r"\b(?P<month>\d{1,2})[/-](?P<day>\d{1,2})[/-](?P<year>(?:19|20)\d{2})\b", text)
    if m:
        year = int(m.group("year"))
        month = int(m.group("month"))
        day = int(m.group("day"))
        try:
            return datetime(year, month, day, tzinfo=timezone.utc)
        except ValueError:
            return None
    m = re.search(r"\b(?P<month>\d{1,2})[/-](?P<year>(?:19|20)\d{2})\b", text)
    if m:
        year = int(m.group("year"))
        month = int(m.group("month"))
        try:
            return datetime(year, month, 1, tzinfo=timezone.utc)
        except ValueError:
            return None
    m = re.search(r"\b(?P<month>\d{1,2})[/-](?P<day>\d{1,2})\b", text)
    if m:
        ref = _normalize_reference(reference_time)
        year = ref.year
        month = int(m.group("month"))
        day = int(m.group("day"))
        try:
            return datetime(year, month, day, tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _parse_month_name_date(text: str, reference_time: Optional[datetime]) -> Optional[datetime]:
    text = (text or "").strip()
    if not text:
        return None
    m = re.search(
        rf"\b(?P<month>{_MONTH_NAME_RE})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?(?:,)?\s+(?P<year>(?:19|20)\d{{2}})\b",
        text,
        re.I,
    )
    if m:
        month = _month_to_int(m.group("month"))
        day = int(m.group("day"))
        year = int(m.group("year"))
        if month:
            try:
                return datetime(year, month, day, tzinfo=timezone.utc)
            except ValueError:
                return None
    m = re.search(
        rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<month>{_MONTH_NAME_RE})(?:,)?\s+(?P<year>(?:19|20)\d{{2}})\b",
        text,
        re.I,
    )
    if m:
        month = _month_to_int(m.group("month"))
        day = int(m.group("day"))
        year = int(m.group("year"))
        if month:
            try:
                return datetime(year, month, day, tzinfo=timezone.utc)
            except ValueError:
                return None
    m = re.search(
        rf"\b(?P<month>{_MONTH_NAME_RE})\s+(?P<year>(?:19|20)\d{{2}})\b",
        text,
        re.I,
    )
    if m:
        month = _month_to_int(m.group("month"))
        year = int(m.group("year"))
        if month:
            return datetime(year, month, 1, tzinfo=timezone.utc)
    m = re.search(
        rf"\b(?P<year>(?:19|20)\d{{2}})\s+(?P<month>{_MONTH_NAME_RE})\b",
        text,
        re.I,
    )
    if m:
        month = _month_to_int(m.group("month"))
        year = int(m.group("year"))
        if month:
            return datetime(year, month, 1, tzinfo=timezone.utc)
    m = re.search(
        rf"\b(?P<month>{_MONTH_NAME_RE})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\b",
        text,
        re.I,
    )
    if m:
        ref = _normalize_reference(reference_time)
        month = _month_to_int(m.group("month"))
        day = int(m.group("day"))
        if month:
            try:
                return datetime(ref.year, month, day, tzinfo=timezone.utc)
            except ValueError:
                return None
    m = re.search(
        rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<month>{_MONTH_NAME_RE})\b",
        text,
        re.I,
    )
    if m:
        ref = _normalize_reference(reference_time)
        month = _month_to_int(m.group("month"))
        day = int(m.group("day"))
        if month:
            try:
                return datetime(ref.year, month, day, tzinfo=timezone.utc)
            except ValueError:
                return None
    return None


def _parse_quarter(text: str) -> Optional[datetime]:
    text = (text or "").strip()
    if not text:
        return None
    m = _QUARTER_RE.search(text)
    if not m:
        return None
    quarter = int(m.group(1))
    year = int(m.group(2))
    month = 1 + (quarter - 1) * 3
    return datetime(year, month, 1, tzinfo=timezone.utc)


def _parse_season(text: str) -> Optional[datetime]:
    text = (text or "").strip()
    if not text:
        return None
    m = _SEASON_RE.search(text)
    if not m:
        return None
    season = m.group(1).lower()
    year = int(m.group(2))
    if season == "spring":
        month = 3
    elif season == "summer":
        month = 6
    elif season in ("fall", "autumn"):
        month = 9
    else:
        month = 12
    return datetime(year, month, 1, tzinfo=timezone.utc)


def _normalize_reference(reference_time: Optional[datetime]) -> datetime:
    if reference_time is None:
        reference_time = datetime.now(tz=timezone.utc)
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)
    return reference_time


def _shift_months(dt: datetime, months: int) -> datetime:
    month_index = dt.month - 1 + months
    year = dt.year + (month_index // 12)
    month = month_index % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def _shift_years(dt: datetime, years: int) -> datetime:
    year = dt.year + years
    day = min(dt.day, calendar.monthrange(year, dt.month)[1])
    return dt.replace(year=year, day=day)


def _parse_relative(text: str, reference_time: Optional[datetime]) -> Optional[datetime]:
    text = (text or "").strip().lower()
    if not text:
        return None
    ref = _normalize_reference(reference_time)
    if "today" in text:
        return ref
    if "yesterday" in text:
        return ref - timedelta(days=1)
    if "tomorrow" in text:
        return ref + timedelta(days=1)

    m = re.search(r"\b(last|this|next)\s+(year|month|week|day)\b", text)
    if m:
        direction = m.group(1)
        unit = m.group(2)
        if unit == "year":
            if direction == "last":
                return _shift_years(ref, -1)
            if direction == "next":
                return _shift_years(ref, 1)
            return ref
        if unit == "month":
            if direction == "last":
                return _shift_months(ref, -1)
            if direction == "next":
                return _shift_months(ref, 1)
            return ref
        if unit == "week":
            delta = 0
            if direction == "last":
                delta = -7
            elif direction == "next":
                delta = 7
            return ref + timedelta(days=delta)
        if unit == "day":
            delta = 0
            if direction == "last":
                delta = -1
            elif direction == "next":
                delta = 1
            return ref + timedelta(days=delta)

    m = re.search(r"\b(\d+)\s+(years?|yrs?|months?|mos?|weeks?|wks?|days?)\s+ago\b", text)
    if m:
        value = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("year") or unit.startswith("yr"):
            return _shift_years(ref, -value)
        if unit.startswith("month") or unit.startswith("mo"):
            return _shift_months(ref, -value)
        if unit.startswith("week") or unit.startswith("wk"):
            return ref - timedelta(weeks=value)
        return ref - timedelta(days=value)

    m = re.search(r"\bin\s+(\d+)\s+(years?|yrs?|months?|mos?|weeks?|wks?|days?)\b", text)
    if m:
        value = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("year") or unit.startswith("yr"):
            return _shift_years(ref, value)
        if unit.startswith("month") or unit.startswith("mo"):
            return _shift_months(ref, value)
        if unit.startswith("week") or unit.startswith("wk"):
            return ref + timedelta(weeks=value)
        return ref + timedelta(days=value)

    m = re.search(r"\b(\d+)\s+(years?|yrs?|months?|mos?|weeks?|wks?|days?)\s+later\b", text)
    if m:
        value = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("year") or unit.startswith("yr"):
            return _shift_years(ref, value)
        if unit.startswith("month") or unit.startswith("mo"):
            return _shift_months(ref, value)
        if unit.startswith("week") or unit.startswith("wk"):
            return ref + timedelta(weeks=value)
        return ref + timedelta(days=value)

    m = re.search(rf"\b(last|this|next)\s+(?P<month>{_MONTH_NAME_RE})\b", text, re.I)
    if m:
        direction = m.group(1).lower()
        month = _month_to_int(m.group("month"))
        if not month:
            return None
        year = ref.year
        if direction == "last":
            if ref.month <= month:
                year -= 1
        elif direction == "next":
            if ref.month >= month:
                year += 1
        return datetime(year, month, 1, tzinfo=timezone.utc)

    m = re.search(rf"\b(?P<month>{_MONTH_NAME_RE})\b", text, re.I)
    if m:
        month = _month_to_int(m.group("month"))
        if month:
            return datetime(ref.year, month, 1, tzinfo=timezone.utc)

    return None


def _parse_year_only(text: str) -> Optional[datetime]:
    text = (text or "").strip()
    if not text:
        return None
    m = _YEAR_RE.search(text)
    if not m:
        return None
    year = int(m.group(0))
    return datetime(year, 1, 1, tzinfo=timezone.utc)


def _parse_time_any(text: str, reference_time: Optional[datetime]) -> Optional[datetime]:
    return (
        _parse_iso_like(text)
        or _parse_numeric_date(text, reference_time)
        or _parse_month_name_date(text, reference_time)
        or _parse_quarter(text)
        or _parse_season(text)
        or _parse_year_only(text)
        or _parse_relative(text, reference_time)
    )


def parse_time_text(
    happen_time: str,
    obs_time: str,
    mode: str,
    reference_time: Optional[datetime] = None,
) -> Optional[datetime]:
    """
    Best-effort parser for time strings produced by OpenIE.

    This intentionally avoids heavyweight dependencies; it supports:
    - ISO-like timestamps (YYYY-MM-DD / full ISO)
    - numeric dates (YYYY/MM/DD, YYYY-MM, MM/DD/YYYY)
    - month-name dates (March 2020, Mar 5 2020)
    - quarters and seasons (Q1 2020, summer 2021)
    - relative expressions (last year, 2 weeks ago)

    For non-parseable values, returns None.
    """
    mode = (mode or "happen_else_obs").strip().lower()
    happen_time = (happen_time or "").strip()
    obs_time = (obs_time or "").strip()

    obs_dt = _parse_time_any(obs_time, reference_time)
    ref_dt = reference_time or obs_dt
    happen_dt = _parse_time_any(happen_time, ref_dt)

    if mode == "happen":
        return happen_dt
    if mode == "obs":
        return obs_dt
    # happen_else_obs
    return happen_dt or obs_dt


def time_to_scalar(dt: Optional[datetime]) -> float:
    if dt is None:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return float(dt.timestamp())


def normalize_time_text(
    happen_time: str,
    obs_time: str,
    reference_time: Optional[datetime] = None,
) -> str:
    text = (happen_time or "").strip()
    if not text:
        return ""
    dt = parse_time_text(
        happen_time=text,
        obs_time=obs_time,
        mode="happen_else_obs",
        reference_time=reference_time,
    )
    if dt is None:
        return ""
    return dt.date().isoformat()


@dataclass(frozen=True)
class QueryTime:
    """
    Time context used at scoring time.

    For now we use system time ("now"). Later we can plug in a query-time parser.
    """

    unix_seconds: float

    @classmethod
    def now(cls) -> "QueryTime":
        return cls(unix_seconds=float(datetime.now(tz=timezone.utc).timestamp()))
