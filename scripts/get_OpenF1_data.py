"""Download and validate the raw OpenF1 race data"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_ROOT = "https://api.openf1.org/v1"
YEARS = (2023, 2024, 2025, 2026)

MIN_REQUEST_INTERVAL_SECONDS = 2.1
MAX_ATTEMPTS = 4


class OpenF1Client:
    def __init__(self) -> None:
        self._last_request_started = 0.0

    def get(self, endpoint: str, params: list[tuple[str, str | int]]) -> tuple[bytes, Any, dict[str, Any]]:
        query = urlencode(params)
        url = f"{API_ROOT}/{endpoint}?{query}"

        for attempt in range(1, MAX_ATTEMPTS + 1):
            elapsed = time.monotonic() - self._last_request_started
            if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
                time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)

            self._last_request_started = time.monotonic()
            retrieved_at = datetime.now(timezone.utc).isoformat()
            request = Request(url, headers={"User-Agent": "f1-weekend-predictor/data-ingestion"})

            try:
                with urlopen(request, timeout=45) as response:
                    payload = response.read()
                    parsed = json.loads(payload.decode("utf-8"))
                    metadata = {
                        "endpoint": endpoint,
                        "url": url,
                        "retrieved_at": retrieved_at,
                        "status": response.status,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "bytes": len(payload),
                    }
                    return payload, parsed, metadata
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
                retryable = not isinstance(error, HTTPError) or error.code == 429 or error.code >= 500
                if attempt == MAX_ATTEMPTS or not retryable:
                    raise RuntimeError(f"OpenF1 request failed: {url}") from error
                time.sleep(2**attempt)

        raise RuntimeError(f"OpenF1 request failed: {url}")


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def save_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        file.write(payload)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write("\n")


def validate_year(
    year: int,
    scheduled_sessions: list[dict[str, Any]],
    candidate_sessions: list[dict[str, Any]],
    results: list[dict[str, Any]],
    drivers: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    candidate_keys = {row["session_key"] for row in candidate_sessions}
    result_keys = {row["session_key"] for row in results}
    completed_keys = sorted(result_keys)
    rows_per_session: dict[int, int] = {}

    for row in results:
        key = row["session_key"]
        rows_per_session[key] = rows_per_session.get(key, 0) + 1

    result_ids = [(row.get("session_key"), row.get("driver_number")) for row in results]
    duplicate_results = len(result_ids) - len(set(result_ids))
    if duplicate_results:
        errors.append(f"{duplicate_results} duplicate session/driver result rows")

    driver_ids = [(row.get("session_key"), row.get("driver_number")) for row in drivers]
    duplicate_drivers = len(driver_ids) - len(set(driver_ids))
    if duplicate_drivers:
        errors.append(f"{duplicate_drivers} duplicate session/driver metadata rows")

    driver_id_set = set(driver_ids)
    missing_driver_metadata = sorted(set(result_ids) - driver_id_set)
    if missing_driver_metadata:
        errors.append(f"{len(missing_driver_metadata)} results have no matching driver metadata")

    missing_position_rows = 0
    sessions_with_missing_positions: list[int] = []
    for session_key in completed_keys:
        session_rows = [row for row in results if row["session_key"] == session_key]
        positions = [row.get("position") for row in session_rows]
        missing_count = sum(position is None for position in positions)
        if missing_count:
            missing_position_rows += missing_count
            sessions_with_missing_positions.append(session_key)
        classified_positions = [position for position in positions if position is not None]
        if len(classified_positions) != len(set(classified_positions)):
            errors.append(f"session {session_key} has duplicate finishing positions")

    if missing_position_rows:
        warnings.append(
            f"{missing_position_rows} non-classified result rows have a null position across "
            f"{len(sessions_with_missing_positions)} sessions; normalization must define their final ordering"
        )

    unexpected_result_keys = result_keys - candidate_keys
    if unexpected_result_keys:
        errors.append(f"results returned for unexpected sessions: {sorted(unexpected_result_keys)}")

    past_without_results = sorted(candidate_keys - result_keys)
    if past_without_results:
        warnings.append(
            "Past scheduled race sessions with no published result were excluded: "
            + ", ".join(str(key) for key in past_without_results)
        )

    row_counts = list(rows_per_session.values())
    return {
        "year": year,
        "scheduled_race_sessions": len(scheduled_sessions),
        "past_scheduled_sessions": len(candidate_sessions),
        "completed_races_with_results": len(completed_keys),
        "completed_session_keys": completed_keys,
        "past_session_keys_without_results": past_without_results,
        "result_rows": len(results),
        "driver_rows": len(drivers),
        "minimum_field_size": min(row_counts) if row_counts else 0,
        "maximum_field_size": max(row_counts) if row_counts else 0,
        "duplicate_result_rows": duplicate_results,
        "duplicate_driver_rows": duplicate_drivers,
        "results_missing_driver_metadata": len(missing_driver_metadata),
        "result_rows_with_null_position": missing_position_rows,
        "sessions_with_null_positions": sessions_with_missing_positions,
        "errors": errors,
        "warnings": warnings,
    }


def download(output_root: Path) -> Path:
    started_at = datetime.now(timezone.utc)
    retrieval_id = started_at.strftime("%Y%m%dT%H%M%SZ")
    retrieval_dir = output_root / retrieval_id
    if retrieval_dir.exists():
        raise FileExistsError(f"Retrieval directory already exists: {retrieval_dir}")
    retrieval_dir.mkdir(parents=True)

    client = OpenF1Client()
    request_log: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    latest_2026_session: dict[str, Any] | None = None
    latest_2026_drivers: list[dict[str, Any]] = []

    for year in YEARS:
        year_dir = retrieval_dir / str(year)
        sessions_payload, sessions, metadata = client.get(
            "sessions", [("year", year), ("session_name", "Race")]
        )
        save_exact(year_dir / "race_sessions.json", sessions_payload)
        request_log.append(metadata)

        candidate_sessions = [
            session
            for session in sessions
            if session.get("date_end") and parse_timestamp(session["date_end"]) < started_at
        ]
        candidate_keys = [session["session_key"] for session in candidate_sessions]

        result_params = [("session_key", key) for key in candidate_keys]
        results_payload, results, metadata = client.get("session_result", result_params)
        save_exact(year_dir / "session_results.json", results_payload)
        request_log.append(metadata)

        completed_keys = sorted({row["session_key"] for row in results})
        driver_params = [("session_key", key) for key in completed_keys]
        drivers_payload, drivers, metadata = client.get("drivers", driver_params)
        save_exact(year_dir / "drivers.json", drivers_payload)
        request_log.append(metadata)

        validation = validate_year(year, sessions, candidate_sessions, results, drivers)
        validations.append(validation)

        if year == 2026 and completed_keys:
            session_by_key = {session["session_key"]: session for session in candidate_sessions}
            latest_key = max(completed_keys, key=lambda key: parse_timestamp(session_by_key[key]["date_end"]))
            latest_2026_session = session_by_key[latest_key]
            latest_2026_drivers = sorted(
                [row for row in drivers if row["session_key"] == latest_key],
                key=lambda row: row["driver_number"],
            )

    errors = [error for validation in validations for error in validation["errors"]]
    report = {
        "retrieval_id": retrieval_id,
        "started_at": started_at.isoformat(),
        "source": "OpenF1",
        "source_base_url": API_ROOT,
        "years": list(YEARS),
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "yearly_validation": validations,
    }
    save_json(retrieval_dir / "validation_report.json", report)

    current_grid = {
        "as_of_session": latest_2026_session,
        "selection_rule": "Drivers returned for the latest completed 2026 race session",
        "drivers": latest_2026_drivers,
    }
    save_json(retrieval_dir / "current_2026_grid.json", current_grid)

    manifest = {
        "retrieval_id": retrieval_id,
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source": "OpenF1",
        "raw_files_are_exact_http_response_bodies": True,
        "requests": request_log,
    }
    save_json(retrieval_dir / "manifest.json", manifest)

    if errors:
        raise RuntimeError(f"Validation failed; inspect {retrieval_dir / 'validation_report.json'}")
    return retrieval_dir


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "data" / "raw" / "openf1",
        help="Directory under which an immutable timestamped retrieval is created.",
    )
    args = parser.parse_args()
    retrieval_dir = download(args.output_root.resolve())
    print(retrieval_dir)


if __name__ == "__main__":
    main()
