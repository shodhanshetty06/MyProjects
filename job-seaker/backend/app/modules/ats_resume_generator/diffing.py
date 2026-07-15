"""Pure text/list diff utilities backing the ATS Resume Generator's
"Diff View" — comparing an optimized version against its own original
snapshot, or against another version. No DB/HTTP I-O so it's unit-testable
in isolation, mirroring `app.modules.matching.scoring`.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass


@dataclass(frozen=True)
class TextDiffSegment:
    op: str  # "equal" | "insert" | "delete"
    text: str


def diff_text(old: str, new: str) -> list[TextDiffSegment]:
    """Word-level diff between two strings, as tagged segments a UI can
    render with strikethrough/highlight styling.
    """
    old_tokens = old.split()
    new_tokens = new.split()
    matcher = difflib.SequenceMatcher(a=old_tokens, b=new_tokens, autojunk=False)

    segments: list[TextDiffSegment] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            segments.append(TextDiffSegment(op="equal", text=" ".join(old_tokens[i1:i2])))
        elif tag == "delete":
            segments.append(TextDiffSegment(op="delete", text=" ".join(old_tokens[i1:i2])))
        elif tag == "insert":
            segments.append(TextDiffSegment(op="insert", text=" ".join(new_tokens[j1:j2])))
        elif tag == "replace":
            segments.append(TextDiffSegment(op="delete", text=" ".join(old_tokens[i1:i2])))
            segments.append(TextDiffSegment(op="insert", text=" ".join(new_tokens[j1:j2])))
    return segments


@dataclass(frozen=True)
class ListDiff:
    added: list[str]
    removed: list[str]
    unchanged: list[str]


def diff_string_list(old: list[str], new: list[str]) -> ListDiff:
    """Order-insensitive, case-insensitive diff between two string lists
    (e.g. skills), preserving original casing for display.
    """
    old_normalized = {s.strip().lower(): s for s in old if s.strip()}
    new_normalized = {s.strip().lower(): s for s in new if s.strip()}

    added = [new_normalized[k] for k in new_normalized if k not in old_normalized]
    removed = [old_normalized[k] for k in old_normalized if k not in new_normalized]
    unchanged = [new_normalized[k] for k in new_normalized if k in old_normalized]
    return ListDiff(added=added, removed=removed, unchanged=unchanged)


@dataclass(frozen=True)
class ProjectDiff:
    name: str
    status: str  # "added" | "removed" | "changed" | "unchanged"
    description_diff: list[TextDiffSegment]
    highlights_diff: ListDiff


def diff_projects(old: list[dict], new: list[dict]) -> list[ProjectDiff]:
    """Project-by-project diff, matched by case-insensitive project name."""
    old_by_name = {(p.get("name") or "").strip().lower(): p for p in old}
    new_by_name = {(p.get("name") or "").strip().lower(): p for p in new}

    results: list[ProjectDiff] = []
    for key in dict.fromkeys(list(old_by_name) + list(new_by_name)):
        old_project = old_by_name.get(key)
        new_project = new_by_name.get(key)
        if old_project is None and new_project is not None:
            results.append(
                ProjectDiff(
                    name=new_project.get("name", ""),
                    status="added",
                    description_diff=diff_text("", new_project.get("description") or ""),
                    highlights_diff=diff_string_list([], new_project.get("highlights") or []),
                )
            )
        elif new_project is None and old_project is not None:
            results.append(
                ProjectDiff(
                    name=old_project.get("name", ""),
                    status="removed",
                    description_diff=diff_text(old_project.get("description") or "", ""),
                    highlights_diff=diff_string_list(old_project.get("highlights") or [], []),
                )
            )
        elif old_project is not None and new_project is not None:
            description_diff = diff_text(
                old_project.get("description") or "", new_project.get("description") or ""
            )
            highlights_diff = diff_string_list(
                old_project.get("highlights") or [], new_project.get("highlights") or []
            )
            changed = bool(highlights_diff.added or highlights_diff.removed) or any(
                seg.op != "equal" for seg in description_diff
            )
            results.append(
                ProjectDiff(
                    name=new_project.get("name", ""),
                    status="changed" if changed else "unchanged",
                    description_diff=description_diff,
                    highlights_diff=highlights_diff,
                )
            )
    return results


__all__ = [
    "TextDiffSegment",
    "diff_text",
    "ListDiff",
    "diff_string_list",
    "ProjectDiff",
    "diff_projects",
]
