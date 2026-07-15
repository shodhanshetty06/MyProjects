"""Unit tests for `app.modules.ats_resume_generator.diffing` — pure functions,
no DB/HTTP/AI I-O required.
"""
from __future__ import annotations

from app.modules.ats_resume_generator import diffing


def test_diff_text_all_equal() -> None:
    segments = diffing.diff_text("Backend engineer", "Backend engineer")
    assert len(segments) == 1
    assert segments[0].op == "equal"
    assert segments[0].text == "Backend engineer"


def test_diff_text_replace_produces_delete_then_insert() -> None:
    segments = diffing.diff_text("Backend engineer", "Senior backend engineer")

    assert [(s.op, s.text) for s in segments] == [
        ("delete", "Backend"),
        ("insert", "Senior backend"),
        ("equal", "engineer"),
    ]
    # Reconstructing the "new" side from insert/equal segments should equal the input.
    rebuilt = " ".join(s.text for s in segments if s.op in ("equal", "insert") and s.text)
    assert rebuilt == "Senior backend engineer"


def test_diff_text_empty_old_is_all_insert() -> None:
    segments = diffing.diff_text("", "New summary text")
    assert all(s.op != "delete" for s in segments)
    assert "".join(s.text for s in segments if s.op == "insert") != ""


def test_diff_text_empty_new_is_all_delete() -> None:
    segments = diffing.diff_text("Old summary text", "")
    assert any(s.op == "delete" for s in segments)
    assert all(s.op != "insert" for s in segments)


def test_diff_string_list_added_removed_unchanged() -> None:
    result = diffing.diff_string_list(["Python", "SQL", "Docker"], ["python", "Kubernetes", "SQL"])
    assert result.added == ["Kubernetes"]
    assert result.removed == ["Docker"]
    # "Python"/"python" and "SQL" are both present on both sides (case-insensitively)
    assert result.unchanged == ["python", "SQL"]  # casing preserved from the "new" side


def test_diff_string_list_ignores_blank_entries() -> None:
    result = diffing.diff_string_list(["", " ", "Python"], ["Python", ""])
    assert result.added == []
    assert result.removed == []
    assert result.unchanged == ["Python"]


def test_diff_projects_detects_added_removed_changed_unchanged() -> None:
    old = [
        {"name": "Order Service", "description": "Built a service.", "highlights": ["Shipped v1"]},
        {"name": "Legacy Tool", "description": "Old internal tool.", "highlights": []},
    ]
    new = [
        {
            "name": "Order Service",
            "description": "Built and scaled a service.",
            "highlights": ["Shipped v1", "Cut latency 30%"],
        },
        {"name": "New Dashboard", "description": "Analytics dashboard.", "highlights": []},
    ]

    diffs = {d.name: d for d in diffing.diff_projects(old, new)}

    assert diffs["Order Service"].status == "changed"
    assert diffs["Legacy Tool"].status == "removed"
    assert diffs["New Dashboard"].status == "added"


def test_diff_projects_unchanged_when_identical() -> None:
    project = {"name": "Order Service", "description": "Built a service.", "highlights": ["Shipped v1"]}
    diffs = diffing.diff_projects([project], [dict(project)])
    assert len(diffs) == 1
    assert diffs[0].status == "unchanged"


def test_diff_projects_matches_by_case_insensitive_name() -> None:
    old = [{"name": "Order Service", "description": "A", "highlights": []}]
    new = [{"name": "order service", "description": "A", "highlights": []}]
    diffs = diffing.diff_projects(old, new)
    assert len(diffs) == 1
    assert diffs[0].status == "unchanged"
