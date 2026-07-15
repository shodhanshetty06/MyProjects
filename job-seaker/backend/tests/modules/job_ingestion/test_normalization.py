"""Unit tests for the pure normalization helpers.

No DB, network, or Docker services required.
"""

from __future__ import annotations

from app.modules.job_ingestion.models import EmploymentType, SalaryPeriod, WorkMode
from app.modules.job_ingestion.normalization import (
    compute_content_hash,
    parse_employment_type,
    parse_location,
    parse_salary,
    parse_work_mode,
)


def test_parse_salary_range_with_currency_and_period() -> None:
    result = parse_salary("$120,000 - $150,000 a year")
    assert result.salary_min == 120000
    assert result.salary_max == 150000
    assert result.currency == "USD"
    assert result.period == SalaryPeriod.YEARLY


def test_parse_salary_hourly_range() -> None:
    result = parse_salary("$45 - $60 / hr")
    assert result.salary_min == 45
    assert result.salary_max == 60
    assert result.period == SalaryPeriod.HOURLY


def test_parse_salary_k_suffix() -> None:
    result = parse_salary("£40k-£60k")
    assert result.salary_min == 40000
    assert result.salary_max == 60000
    assert result.currency == "GBP"


def test_parse_salary_defaults_to_hourly_for_small_unlabeled_numbers() -> None:
    result = parse_salary("$45 - $60")
    assert result.period == SalaryPeriod.HOURLY


def test_parse_salary_defaults_to_yearly_for_large_unlabeled_numbers() -> None:
    result = parse_salary("$120,000 - $150,000")
    assert result.period == SalaryPeriod.YEARLY


def test_parse_salary_handles_none_and_blank() -> None:
    for raw in (None, "", "   "):
        result = parse_salary(raw)
        assert result.salary_min is None
        assert result.salary_max is None
        assert result.period == SalaryPeriod.UNKNOWN


def test_parse_employment_type_matches_keywords() -> None:
    assert parse_employment_type("Full-time") == EmploymentType.FULL_TIME
    assert parse_employment_type("Part time") == EmploymentType.PART_TIME
    assert parse_employment_type("Contractor") == EmploymentType.CONTRACT
    assert parse_employment_type("Summer Internship") == EmploymentType.INTERNSHIP
    assert parse_employment_type("Temp position") == EmploymentType.TEMPORARY
    assert parse_employment_type("Freelance gig") == EmploymentType.FREELANCE
    assert parse_employment_type(None) == EmploymentType.UNKNOWN
    assert parse_employment_type("something else") == EmploymentType.UNKNOWN


def test_parse_work_mode_prefers_explicit_flag() -> None:
    assert parse_work_mode(location_raw="New York, NY", remote_flag=True) == WorkMode.REMOTE


def test_parse_work_mode_falls_back_to_keywords() -> None:
    assert parse_work_mode(location_raw="Remote - US", remote_flag=None) == WorkMode.REMOTE
    assert parse_work_mode(location_raw="Austin, TX (Hybrid)", remote_flag=None) == WorkMode.HYBRID
    assert (
        parse_work_mode(location_raw="On-site in Austin, TX", remote_flag=None) == WorkMode.ONSITE
    )
    assert parse_work_mode(location_raw="Austin, TX", remote_flag=None) == WorkMode.UNKNOWN


def test_parse_location_splits_city_region_country() -> None:
    result = parse_location("Austin, Texas, United States")
    assert result.city == "Austin"
    assert result.region == "Texas"
    assert result.country == "United States"
    assert result.country_code == "US"
    assert result.is_remote is False


def test_parse_location_handles_remote() -> None:
    result = parse_location("Remote")
    assert result.is_remote is True
    assert result.city is None


def test_parse_location_handles_none() -> None:
    result = parse_location(None)
    assert result.city is None
    assert result.is_remote is False


def test_compute_content_hash_stable_and_sensitive_to_change() -> None:
    kwargs = dict(
        title="Software Engineer",
        description="Build things.",
        location_raw="Remote",
        salary_raw="$100k",
    )
    first = compute_content_hash(**kwargs)
    second = compute_content_hash(**kwargs)
    assert first == second

    changed = compute_content_hash(**{**kwargs, "title": "Senior Software Engineer"})
    assert changed != first
