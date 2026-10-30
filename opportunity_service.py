"""Discovery, validation, provenance, and persistence for student opportunities.

This module deliberately keeps web retrieval separate from SQLite operations.  Web
content is treated as untrusted data and only a configured search provider is used.
"""
from __future__ import annotations

import html
import json
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Optional, Protocol
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from urllib import robotparser

from database import get_db
from task_service import parse_and_validate_iso_date

ALLOWED_TYPES = {"hackathon", "student_competition", "coding_competition", "innovation_challenge"}
ALLOWED_TRUST = {"official", "secondary", "unverified"}
ALLOWED_PLATFORMS = {"Unstop", "Devpost"}
DATE_FIELDS = {
    "registration_deadline", "application_deadline", "submission_deadline",
    "event_start_date", "event_end_date", "results_announcement_date",
}
URL_FIELDS = {"source_url", "official_source_url", "registration_url"}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def validate_url(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    candidate = str(value).strip()
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("URLs must be public HTTP or HTTPS URLs.")
    return candidate


def platform_for_url(value: Optional[str], require_listing: bool = False) -> Optional[str]:
    """Return Unstop/Devpost only for exact platform hosts and listing paths."""
    url = validate_url(value)
    if not url:
        return None
    parsed = urlparse(url)
    host = parsed.hostname.lower() if parsed.hostname else ""
    if host not in {"unstop.com", "www.unstop.com"} and host not in {"devpost.com", "www.devpost.com"}:
        return None
    if host in {"unstop.com", "www.unstop.com"}:
        platform = "Unstop"
        path_parts = [part for part in parsed.path.lower().split("/") if part]
        valid_path = len(path_parts) >= 2 and path_parts[0] == "hackathons" and bool(path_parts[1])
    else:
        platform = "Devpost"
        path_parts = [part for part in parsed.path.lower().split("/") if part]
        valid_path = len(path_parts) >= 2 and path_parts[0] == "hackathons" and bool(path_parts[1])
    if require_listing and not valid_path:
        return None
    return platform


def _optional_text(value: Any) -> Optional[str]:
    text = str(value).strip() if value is not None else ""
    return text or None


def _normalise_dates(record: Dict[str, Any]) -> None:
    for field in DATE_FIELDS:
        if record.get(field):
            parsed = parse_opportunity_date(record[field], reference_date=_reference_date(record.get("source_retrieved_at")), context=field)
            if parsed is None:
                raise ValueError(f"Ambiguous or invalid date for {field}: {record[field]!r}")
            record[field] = parsed


def _reference_date(value: Optional[str] = None) -> date:
    """Use retrieval date for relative dates; fall back to configured UTC today."""
    if value:
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
        except ValueError:
            pass
    return datetime.now(timezone.utc).date()


def parse_opportunity_date(value: Any, reference_date: Optional[date] = None, context: str = "") -> Optional[str]:
    """Parse explicit common date formats without guessing ambiguous numeric dates.

    Month/day values without a year use the supplied retrieval date's year. Relative
    phrases are accepted only for a deadline-like context and a reliable reference.
    """
    if value is None or not str(value).strip():
        return None
    raw = " ".join(str(value).strip().split())
    try:
        if (("T" in raw and any(c.isdigit() for c in raw)) or (raw.endswith("Z") and any(c.isdigit() for c in raw))) and not re.search(r"^(?:in|within|tomorrow|next)", raw, re.I):
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None
    iso = re.fullmatch(r"(20\d{2})-(\d{2})-(\d{2})", raw)
    if iso:
        try:
            return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3))).isoformat()
        except ValueError:
            return None
    for fmt in ("%d %B %Y", "%B %d, %Y", "%d %b %Y", "%B %d %Y", "%d %b, %Y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            pass
    numeric = re.fullmatch(r"(\d{1,2})([/\-])(\d{1,2})(?:[/\-](\d{4}))?", raw)
    if numeric:
        first, second, year = int(numeric.group(1)), int(numeric.group(3)), numeric.group(4)
        # Ambiguous day/month ordering: if both numbers can be months (<=12),
        # do not guess whether it is MM/DD or DD/MM.
        if first <= 12 and second <= 12:
            return None
        if year is None:
            if reference_date is None:
                return None
            year = str(reference_date.year)
        try:
            return date(int(year), second, first).isoformat()
        except ValueError:
            return None
    ref = reference_date
    for fmt in ("%d %B", "%B %d", "%d %b", "%b %d"):
        try:
            if ref is None:
                return None
            parsed = datetime.strptime(raw, fmt).date().replace(year=ref.year)
            if parsed < ref and (ref - parsed).days > 30:
                parsed = parsed.replace(year=ref.year + 1)
            return parsed.isoformat()
        except ValueError:
            pass
    # Relative phrases: require an explicit reference date AND deadline context
    if ref is not None and re.search(r"deadline|close|submit|apply|application|registration|due", context, re.I):
        if re.fullmatch(r"tomorrow", raw, re.I):
            return (ref + timedelta(days=1)).isoformat()
        if re.fullmatch(r"next\s+week", raw, re.I):
            return (ref + timedelta(days=7)).isoformat()
        relative = re.fullmatch(r"(?:within|in)\s+(\d+)\s+days?", raw, re.I)
        if relative:
            return (ref + timedelta(days=int(relative.group(1)))).isoformat()
    return None


def validate_opportunity(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a validated copy; no missing value is fabricated."""
    value = dict(data)
    title = _optional_text(value.get("title"))
    if not title:
        raise ValueError("Opportunity title is required.")
    opportunity_type = _optional_text(value.get("opportunity_type"))
    if opportunity_type not in ALLOWED_TYPES:
        raise ValueError(f"Unsupported opportunity type: {opportunity_type!r}")
    value["title"] = title
    value["normalized_title"] = normalize_title(title)
    value["opportunity_type"] = opportunity_type
    for field in URL_FIELDS:
        value[field] = validate_url(value.get(field))
    _normalise_dates(value)
    if value.get("is_online") is not None:
        value["is_online"] = 1 if bool(value["is_online"]) else 0
    value["status"] = value.get("status") if value.get("status") in {"discovered", "open", "closed", "unknown"} else "discovered"
    trust = value.get("source_trust", "unverified")
    default_conf = "high" if trust == "official" else ("medium" if trust == "secondary" else "low")
    value["extraction_confidence"] = value.get("extraction_confidence") if value.get("extraction_confidence") in {"high", "medium", "low"} else default_conf
    value["source_retrieved_at"] = value.get("source_retrieved_at") or utc_now()
    value["source_retrieved_at"] = parse_and_validate_iso_date(value["source_retrieved_at"])
    provenance = value.get("field_provenance") or {}
    if isinstance(provenance, str):
        try:
            provenance = json.loads(provenance)
        except json.JSONDecodeError:
            provenance = {}
    value["field_provenance"] = provenance
    return value


def _confidence_from_provenance(provenance: Dict[str, Any]) -> str:
    levels = {entry.get("confidence") for entry in provenance.values() if isinstance(entry, dict)}
    levels.discard(None)
    if "official" in levels and len(levels) > 1:
        return "mixed"
    if "official" in levels:
        return "official"
    if "secondary" in levels:
        return "secondary"
    return "unverified"


def _duplicate(conn, item: Dict[str, Any]):
    urls = [u for u in (item.get("registration_url"), item.get("official_source_url"), item.get("source_url")) if u]
    if urls:
        marks = ",".join("?" for _ in urls)
        found = conn.execute(
            f"SELECT * FROM opportunities WHERE registration_url IN ({marks}) OR official_source_url IN ({marks}) OR source_url IN ({marks}) LIMIT 1",
            urls + urls + urls,
        ).fetchone()
        if found:
            return found
    return conn.execute(
        """SELECT * FROM opportunities WHERE normalized_title = ?
           AND COALESCE(organizer, '') = COALESCE(?, '')
           AND COALESCE(event_start_date, '') = COALESCE(?, '') LIMIT 1""",
        (item["normalized_title"], item.get("organizer"), item.get("event_start_date")),
    ).fetchone()


def save_opportunity(data: Dict[str, Any], db_path=None) -> Dict[str, Any]:
    """Insert an opportunity or carefully enrich its existing duplicate."""
    item = validate_opportunity(data)
    now = utc_now()
    trust = item.get("source_trust", "unverified")
    trust = trust if trust in ALLOWED_TRUST else "unverified"
    source_url = item.get("source_url") or item.get("official_source_url") or item.get("registration_url")
    source_summary = _optional_text(item.get("source_summary"))
    provenance = item["field_provenance"]
    for field, val in item.items():
        if field in {"field_provenance", "source_trust", "source_summary"} or val is None:
            continue
        provenance.setdefault(field, {"confidence": trust, "source_url": source_url})
    item["confidence_level"] = _confidence_from_provenance(provenance)

    columns = [
        "title", "normalized_title", "description", "opportunity_type", "platform", "organizer", "source_url", "official_source_url",
        "registration_url", "registration_deadline", "application_deadline", "submission_deadline", "event_start_date", "event_end_date", "results_announcement_date", "location", "is_online", "format",
        "eligibility_information", "eligible_education_levels", "eligible_regions", "required_skills", "team_size_rules", "themes", "participation_cost", "prize_information", "status",
        "source_name", "source_retrieved_at", "last_verified_at", "confidence_level", "extraction_confidence", "field_provenance", "created_at", "updated_at",
    ]
    with get_db(db_path) as conn:
        old = _duplicate(conn, item)
        if old:
            merged = dict(old)
            old_prov = json.loads(merged.get("field_provenance") or "{}")
            # Preserve existing values and record conflicting evidence without silent erasure
            for field in columns:
                incoming = item.get(field)
                if field in {"created_at", "updated_at", "field_provenance", "confidence_level"} or incoming is None:
                    continue
                prior = old_prov.get(field, {})
                old_val = merged.get(field)
                if old_val is not None and old_val != incoming:
                    conflicts = prior.setdefault("conflicts", [])
                    conflict_entry = {
                        "value": incoming,
                        "source_url": source_url,
                        "confidence": trust,
                        "retrieved_at": item["source_retrieved_at"],
                    }
                    if not any(c.get("value") == incoming and c.get("source_url") == source_url for c in conflicts):
                        conflicts.append(conflict_entry)
                if merged.get(field) is None or trust == "official" or prior.get("confidence") != "official":
                    merged[field] = incoming
                    new_entry = dict(provenance.get(field, {"confidence": trust, "source_url": source_url}))
                    if "conflicts" in prior:
                        new_entry["conflicts"] = prior["conflicts"]
                    old_prov[field] = new_entry
            merged["field_provenance"] = json.dumps(old_prov, sort_keys=True)
            merged["confidence_level"] = _confidence_from_provenance(old_prov)
            merged["source_retrieved_at"] = item["source_retrieved_at"]
            if trust == "official":
                merged["last_verified_at"] = item["source_retrieved_at"]
            merged["updated_at"] = now
            assignments = ", ".join(f"{c} = ?" for c in columns if c != "created_at")
            params = [merged.get(c) for c in columns if c != "created_at"] + [old["id"]]
            conn.execute(f"UPDATE opportunities SET {assignments} WHERE id = ?", params)
            opportunity_id, is_duplicate = old["id"], True
        else:
            item["last_verified_at"] = item["source_retrieved_at"] if trust == "official" else None
            item["field_provenance"] = json.dumps(provenance, sort_keys=True)
            item["created_at"] = now
            item["updated_at"] = now
            cur = conn.execute(
                f"INSERT INTO opportunities ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
                [item.get(c) for c in columns],
            )
            opportunity_id, is_duplicate = cur.lastrowid, False
        if source_url:
            conn.execute(
                """INSERT INTO opportunity_sources (opportunity_id, source_url, source_name, source_trust, retrieved_at, raw_summary, field_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(opportunity_id, source_url) DO UPDATE SET retrieved_at=excluded.retrieved_at,
                raw_summary=excluded.raw_summary, field_data=excluded.field_data, source_trust=excluded.source_trust""",
                (opportunity_id, source_url, item.get("source_name"), trust, item["source_retrieved_at"], source_summary, json.dumps(provenance, sort_keys=True)),
            )
        row = conn.execute("SELECT * FROM opportunities WHERE id = ?", (opportunity_id,)).fetchone()
    result = dict(row)
    result["field_provenance"] = json.loads(result["field_provenance"] or "{}")
    result["is_duplicate"] = is_duplicate
    return result


def list_opportunities(opportunity_type: Optional[str] = None, query: Optional[str] = None, platform: Optional[str] = None, limit: int = 100, db_path=None) -> List[Dict[str, Any]]:
    conditions, params = [], []
    if opportunity_type:
        if opportunity_type not in ALLOWED_TYPES:
            raise ValueError("Unsupported opportunity type filter.")
        conditions.append("opportunity_type = ?")
        params.append(opportunity_type)
    if query and query.strip():
        conditions.append("(LOWER(title) LIKE ? OR LOWER(COALESCE(organizer, '')) LIKE ? OR LOWER(COALESCE(description, '')) LIKE ?)")
        params.extend([f"%{query.lower().strip()}%"] * 3)
    if platform:
        if platform == "Both platforms":
            conditions.append("platform IN ('Unstop', 'Devpost')")
        elif platform in ALLOWED_PLATFORMS:
            conditions.append("platform = ?")
            params.append(platform)
        else:
            raise ValueError("Unsupported opportunity platform filter.")
    where = " WHERE " + " AND ".join(conditions) if conditions else ""
    with get_db(db_path) as conn:
        rows = conn.execute(f"SELECT * FROM opportunities{where} ORDER BY source_retrieved_at DESC, id DESC LIMIT ?", params + [limit]).fetchall()
    results = [dict(row) for row in rows]
    for row in results:
        row["field_provenance"] = json.loads(row["field_provenance"] or "{}")
    return results


class SearchProvider(Protocol):
    def search(self, query: str, count: int = 10) -> List[Dict[str, Any]]: ...


class BraveSearchProvider:
    """Official Brave Search API adapter. Configure BRAVE_SEARCH_API_KEY to enable."""
    endpoint = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("BRAVE_SEARCH_API_KEY")
        if not self.api_key:
            raise RuntimeError("Opportunity search is not configured. Set BRAVE_SEARCH_API_KEY to enable Brave Search.")

    def search(self, query: str, count: int = 10) -> List[Dict[str, Any]]:
        from urllib.parse import urlencode
        restricted_query = f"(site:unstop.com/hackathons OR site:devpost.com/hackathons) {query}"
        req = Request(f"{self.endpoint}?{urlencode({'q': restricted_query, 'count': min(count, 20)})}", headers={"Accept": "application/json", "X-Subscription-Token": self.api_key, "User-Agent": "RAMS-Opportunity-Discovery/1.0"})
        try:
            with urlopen(req, timeout=12) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Search provider request failed: {exc}") from exc
        return [{"title": r.get("title"), "url": r.get("url"), "description": r.get("description"), "source_name": platform_for_url(r.get("url"), require_listing=True) or "Brave Search", "source_trust": "secondary"} for r in payload.get("web", {}).get("results", [])]


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: List[str] = []
        self.title = ""
        self.links: List[Dict[str, str]] = []
        self._in_title = False
        self._ignore_depth = 0
        self._current_link: Optional[Dict[str, str]] = None

    def handle_starttag(self, tag, attrs):
        tag_lower = tag.lower()
        if tag_lower in ("script", "style", "noscript"):
            self._ignore_depth += 1
            return
        if self._ignore_depth > 0:
            return
        self._in_title = (tag_lower == "title")
        if tag_lower == "a":
            href = dict(attrs).get("href")
            if href:
                item = {"href": href.strip(), "text": ""}
                self.links.append(item)
                self._current_link = item

    def handle_endtag(self, tag):
        tag_lower = tag.lower()
        if tag_lower in ("script", "style", "noscript"):
            self._ignore_depth = max(0, self._ignore_depth - 1)
            return
        if tag_lower in ("p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "br", "section", "article"):
            if self.parts and self.parts[-1] != "\n":
                self.parts.append("\n")
        if tag_lower in ("td", "th"):
            if self.parts and self.parts[-1] != " ":
                self.parts.append(" ")
        if tag_lower == "a":
            self._current_link = None
        if tag_lower == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._ignore_depth > 0:
            return
        clean = " ".join(data.split())
        if clean:
            self.parts.append(clean)
            if self._in_title:
                self.title = (self.title + " " + clean).strip()
            if self._current_link is not None:
                self._current_link["text"] = (self._current_link["text"] + " " + clean).strip()


class RespectfulPageFetcher:
    """Small, opt-in HTML fetcher: robots-aware, no retries, and throttled per process."""
    _robots: Dict[str, robotparser.RobotFileParser] = {}
    _last_fetch = 0.0
    def fetch(self, url: str) -> Dict[str, str]:
        url = validate_url(url)
        parsed = urlparse(url)
        root = f"{parsed.scheme}://{parsed.netloc}"
        rp = self._robots.get(root)
        if not rp:
            rp = robotparser.RobotFileParser(root + "/robots.txt")
            try: rp.read()
            except Exception: return {"url": url, "text": "", "title": "", "links": [], "fetch_error": "robots policy could not be read"}
            self._robots[root] = rp
        if not rp.can_fetch("RAMS-Opportunity-Discovery", url):
            return {"url": url, "text": "", "title": "", "links": [], "fetch_error": "robots policy disallows retrieval"}
        time.sleep(max(0, 0.5 - (time.monotonic() - self._last_fetch)))
        req = Request(url, headers={"User-Agent": "RAMS-Opportunity-Discovery/1.0", "Accept": "text/html,application/xhtml+xml"})
        try:
            with urlopen(req, timeout=10) as response:
                if "html" not in response.headers.get("Content-Type", ""):
                    return {"url": url, "text": "", "title": "", "links": [], "fetch_error": "not an HTML page"}
                body = response.read(500_000).decode("utf-8", errors="replace")
            self._last_fetch = time.monotonic()
            parser = _TextExtractor(); parser.feed(body)
            return {"url": url, "text": "\n".join(parser.parts)[:30_000], "title": parser.title, "links": parser.links}
        except Exception as exc:
            return {"url": url, "text": "", "title": "", "links": [], "fetch_error": str(exc)}


_MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
_DATE_TOKEN = rf"(?:20\d{{2}}-\d{{2}}-\d{{2}}|\d{{1,2}}[/-]\d{{1,2}}(?:[/-]\d{{4}})?|\d{{1,2}}\s+{_MONTH}(?:,?\s+20\d{{2}})?|{_MONTH}\s+\d{{1,2}}(?:,?\s+20\d{{2}})?)"


def _extract_date_near(text: str, labels: Iterable[str], reference_date: Optional[date] = None) -> Optional[str]:
    for label in labels:
        match = re.search(label + rf"[^.]{{0,60}}?(?P<date>{_DATE_TOKEN}|(?:within|in)\s+\d+\s+days?|tomorrow|next\s+week)", text, re.I)
        if match:
            parsed = parse_opportunity_date(match.group("date"), reference_date=reference_date, context=label)
            if parsed:
                return parsed
    return None


def extract_dates(text: str, reference_date: Optional[date] = None) -> Dict[str, Optional[str]]:
    """Extract distinct date concepts from source text conservatively."""
    ref = reference_date or datetime.now(timezone.utc).date()
    result = {
        "registration_deadline": _extract_date_near(
            text,
            (r"registration\s+(?:closes?|deadline|ends?)", r"registration\s+by", r"register\s+by", r"apply\s+by"),
            ref,
        ),
        "application_deadline": _extract_date_near(
            text,
            (r"application\s+deadline", r"applications?\s+(?:close|due|deadline)", r"apply\s+before"),
            ref,
        ),
        "submission_deadline": _extract_date_near(
            text,
            (r"submission\s+deadline", r"submit\s+by", r"submissions?\s+(?:close|due)", r"applications?\s+must\s+be\s+submitted(?:\s+by)?"),
            ref,
        ),
        "event_start_date": _extract_date_near(
            text,
            (r"event\s+(?:starts?|begins?)", r"starts?\s+on", r"event\s+date", r"takes?\s+place\s+(?:on|starting)", r"begins?\s+on"),
            ref,
        ),
        "event_end_date": _extract_date_near(
            text,
            (r"event\s+ends?", r"ends?\s+on", r"concludes?\s+on", r"through"),
            ref,
        ),
        "results_announcement_date": _extract_date_near(
            text,
            (r"results?(?:\s+(?:will\s+be|are))?\s+(?:announced?|released?|out|on)", r"winners?(?:\s+(?:will\s+be|are))?\s+announced?", r"results?\s+date"),
            ref,
        ),
    }
    # Event date range: e.g. "takes place from November 20 to November 22, 2026",
    # "event dates: November 20-22, 2026", "held from 20 November to 22 November 2026"
    range_match = re.search(
        rf"(?:event\s+dates?|hackathon\s+dates?|takes?\s+place\s+from|runs?\s+from|held\s+from)\s*[:\-]?\s*"
        rf"(?P<start>(?:{_MONTH}\s+\d{{1,2}}|\d{{1,2}}\s+{_MONTH})),?\s*"
        rf"(?:-|to|through)\s*"
        rf"(?P<end>(?:(?:{_MONTH}\s+)?\d{{1,2}}|\d{{1,2}}\s+{_MONTH})),?\s*"
        rf"(?P<year>20\d{{2}})",
        text,
        re.I,
    )
    if range_match:
        year = range_match.group("year")
        start_str = range_match.group("start")
        end_str = range_match.group("end")
        start = parse_opportunity_date(f"{start_str}, {year}", ref, "event")
        if re.fullmatch(r"\d{1,2}", end_str.strip()):
            start_month_match = re.search(_MONTH, start_str, re.I)
            start_month = start_month_match.group(0) if start_month_match else ""
            end = parse_opportunity_date(f"{start_month} {end_str}, {year}", ref, "event")
        else:
            end = parse_opportunity_date(f"{end_str}, {year}", ref, "event")
        if start:
            result["event_start_date"] = start
        if end:
            result["event_end_date"] = end
    return result


def _extract_labeled_text(text: str, labels: Iterable[str], max_length: int = 500) -> Optional[str]:
    boundary = r"(?:\s+(?:prizes?|prize\s+pool|awards?|organized\s+by|organizer|eligibility|education\s+level|eligible\s+regions|required\s+skills?|team\s+size|themes?|participation\s+fee|entry\s+fee|location|venue|format)\s*[:\-])"
    for label in labels:
        pat = rf"{label}\s*[:\-]?\s*(?P<value>(?:(?!{boundary})[^.\n;]){{2,{max_length}}})"
        match = re.search(pat, text, re.I)
        if match:
            value = " ".join(match.group("value").split()).strip(" -:")
            if value:
                return value
    return None


def extract_candidate(result: Dict[str, Any], opportunity_type: str, fetched: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Conservative extraction: only captures explicit ISO date and URL evidence."""
    fetched = fetched or {}
    text = " ".join(filter(None, [result.get("description"), fetched.get("text")]))
    url = validate_url(result.get("url"))
    lower = text.lower()
    retrieved_at = utc_now()
    reference_date = _reference_date(retrieved_at)
    dates = extract_dates(text, reference_date)
    provenance = {}
    trust = result.get("source_trust", "secondary")
    source_url = url
    platform = platform_for_url(url, require_listing=True)

    # Extract registration URL: check both link path and anchor text, validating each candidate
    registration_url = None
    for link_entry in fetched.get("links", []):
        if isinstance(link_entry, dict):
            href = link_entry.get("href", "")
            anchor = link_entry.get("text", "")
        else:
            href = str(link_entry)
            anchor = ""
        if not href or re.search(r"^(?:javascript|mailto|data|tel):", href.strip(), re.I):
            continue
        if re.search(r"register|registration|apply|signup|sign-up|join|form", href, re.I) or \
           re.search(r"register|apply|sign\s*up|join|registration|submit\s+application", anchor, re.I):
            resolved = urljoin(url, href.strip()) if url else href.strip()
            try:
                registration_url = validate_url(resolved)
                if registration_url:
                    break
            except ValueError:
                continue
    organizer = _extract_labeled_text(text, (r"organized\s+by", r"organizer", r"hosted\s+by"), 160)
    eligibility = _extract_labeled_text(text, (r"eligibility", r"who\s+can\s+participate", r"eligible\s+for"), 600)
    education = _extract_labeled_text(text, (r"education\s+level", r"students?\s+from", r"open\s+to"), 200)
    regions = _extract_labeled_text(text, (r"eligible\s+(?:countries|regions)", r"available\s+in"), 200)
    skills = _extract_labeled_text(text, (r"required\s+skills?", r"skills?\s+required"), 250)
    team_rules = _extract_labeled_text(text, (r"team\s+size", r"teams?\s+of"), 180)
    themes = _extract_labeled_text(text, (r"themes?", r"domains?", r"tracks?"), 250)
    fees = _extract_labeled_text(text, (r"participation\s+fee", r"entry\s+fee", r"cost\s+to\s+participate"), 160)
    prizes = _extract_labeled_text(text, (r"prizes?", r"prize\s+pool", r"awards?"), 300)
    if "hybrid" in lower:
        event_format = "hybrid"
    elif "online" in lower or "virtual" in lower:
        event_format = "online"
    elif "offline" in lower or "in-person" in lower or "onsite" in lower:
        event_format = "offline"
    else:
        event_format = None
    extraction_confidence = "high" if trust == "official" else ("medium" if trust == "secondary" else "low")
    for field in ("title", "description"):
        provenance[field] = {"confidence": trust, "source_url": source_url}
    return {
        "title": result.get("title") or fetched.get("title"), "description": _optional_text(result.get("description")) or _optional_text(text[:1000]),
        "organizer": organizer, "opportunity_type": opportunity_type, "platform": platform, "source_url": url,
        "official_source_url": url if trust == "official" else None, "registration_url": registration_url,
        **dates, "format": event_format, "is_online": 1 if event_format == "online" else (0 if event_format == "offline" else None),
        "location": "Online" if event_format == "online" else _extract_labeled_text(text, (r"location", r"venue"), 160),
        "eligibility_information": eligibility, "eligible_education_levels": education, "eligible_regions": regions,
        "required_skills": skills, "team_size_rules": team_rules, "themes": themes,
        "participation_cost": fees, "prize_information": prizes,
        "source_name": result.get("source_name") or urlparse(url).netloc, "source_retrieved_at": retrieved_at,
        "source_trust": trust, "extraction_confidence": extraction_confidence,
        "source_summary": _optional_text(result.get("description")), "field_provenance": provenance,
    }


def discover_opportunities(query: str, opportunity_type: str, provider: Optional[SearchProvider] = None, fetcher: Optional[RespectfulPageFetcher] = None, db_path=None) -> Dict[str, Any]:
    if not query or not query.strip(): raise ValueError("A search query is required.")
    if opportunity_type not in ALLOWED_TYPES: raise ValueError("Unsupported opportunity type.")
    provider = provider or BraveSearchProvider()
    fetcher = fetcher or RespectfulPageFetcher()
    try:
        results = provider.search(query.strip())
    except Exception as exc:
        return {"success": False, "query": query, "error": str(exc), "saved": [], "candidates_found": 0}
    saved, skipped = [], []
    for result in results:
        try:
            url = validate_url(result.get("url"))
            platform = platform_for_url(url, require_listing=True)
            if not platform:
                skipped.append({"url": result.get("url"), "reason": "unsupported source or non-listing page"})
                continue
            fetched = fetcher.fetch(url)
            candidate = extract_candidate(result, opportunity_type, fetched)
            saved.append(save_opportunity(candidate, db_path=db_path))
        except (ValueError, TypeError) as exc:
            skipped.append({"url": result.get("url"), "reason": str(exc)})
    return {"success": True, "query": query, "candidates_found": len(results), "saved": saved, "skipped": skipped}
