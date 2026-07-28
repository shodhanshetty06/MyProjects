"""Plain string templates rendering a notification event into subject/body text.

A dict of `event_type -> (subject_fmt, body_fmt)` f-string-style templates —
no templating engine dependency (e.g. Jinja2) since the content here is
short and structured. `render` is intentionally forgiving of missing keys in
`context` (falls back to an empty string) so a template change never crashes
delivery of an otherwise-valid notification.
"""

from __future__ import annotations

from app.modules.notification.models import NotificationEventType

_TEMPLATES: dict[NotificationEventType, tuple[str, str]] = {
    NotificationEventType.APPLICATION_STATUS_CHANGED: (
        "Your application status was updated",
        "{body}",
    ),
    NotificationEventType.EMAIL_REVIEW_NEEDED: (
        "An email needs your review",
        "{body}",
    ),
    NotificationEventType.GMAIL_REAUTH_REQUIRED: (
        "Reconnect your Gmail account",
        "{body}",
    ),
    NotificationEventType.RESUME_READY: (
        "Your resume is ready",
        "{body}",
    ),
    NotificationEventType.NEW_MATCHING_JOB: (
        "A new job matches your profile",
        "{body}",
    ),
}

_DEFAULT_SUBJECT = "Notification from AI Job Search Platform"


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def render(*, event_type: NotificationEventType, title: str, body: str) -> tuple[str, str]:
    """Returns `(subject, body_text)` for an outbound email.

    `title`/`body` are the already-composed strings the caller (typically
    `NotificationService.notify`) generated for the in-app notification —
    reused here as the email content rather than duplicating copy per
    channel, with the template only supplying a channel-appropriate subject.
    """
    subject_fmt, body_fmt = _TEMPLATES.get(event_type, (title or _DEFAULT_SUBJECT, "{body}"))
    subject = subject_fmt.format_map(_SafeDict(title=title, body=body))
    rendered_body = body_fmt.format_map(_SafeDict(title=title, body=body))
    return subject, rendered_body


__all__ = ["render"]
