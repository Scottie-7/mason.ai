"""Notification configuration helpers."""

from __future__ import annotations

import os


class NotificationManager:
    """Expose simple feature flags for notification channels."""

    def is_email_configured(self) -> bool:
        return bool(os.environ.get("SENDGRID_API_KEY"))

    def is_sms_configured(self) -> bool:
        return bool(os.environ.get("TWILIO_ACCOUNT_SID") and os.environ.get("TWILIO_AUTH_TOKEN"))


__all__ = ["NotificationManager"]
