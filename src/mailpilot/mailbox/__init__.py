"""Mailbox adapters: Gmail for production, a local folder for demos and tests."""

from mailpilot.mailbox.base import Mailbox, label_for
from mailpilot.mailbox.folder import FolderMailbox
from mailpilot.mailbox.parsing import clean_body, parse_address, reply_subject

__all__ = ["FolderMailbox", "Mailbox", "clean_body", "label_for", "parse_address", "reply_subject"]
