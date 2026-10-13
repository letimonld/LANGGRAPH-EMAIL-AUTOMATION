from __future__ import annotations

import pytest

from mailpilot.guardrails import ReplyPolicy, check_reply

POLICY = ReplyPolicy.build(
    sender="emma@example.nl",
    support_address="support@parcelo.example",
    allowed_link_domains=["parcelo.example"],
    known_tracking_numbers=["PCL100000001NL"],
    max_chars=400,
)
OK = "Hi Emma, thanks for your message. Your parcel PCL100000001NL is on its way."


def test_clean_reply_passes():
    assert check_reply(OK, POLICY) == []


def test_company_links_and_addresses_are_allowed():
    text = (
        OK + " Track it at https://track.parcelo.example/x, or write to privacy@parcelo.example "
        "or support@parcelo.example. We'll reply to emma@example.nl."
    )
    assert check_reply(text, POLICY) == []


def test_public_addresses_from_help_center_are_allowed():
    policy = ReplyPolicy.build(
        sender="a@b.example",
        support_address="s@parcelo.example",
        allowed_link_domains=["parcelo.example"],
        public_addresses=["claims@carrier.example"],
    )
    assert check_reply(OK.replace("PCL100000001NL", "") + " claims@carrier.example", policy) == []


@pytest.mark.parametrize(
    ("text", "needle"),
    [
        ("", "too short"),
        (OK * 10, "characters"),
        (OK + " Also PCL123456789NL.", "PCL123456789NL"),
        (OK + " See http://evil.example/login.", "evil.example"),
        (OK + " See www.bit.ly/abc", "bit.ly"),
        (OK + " Other customer: bob@example.com", "bob@example.com"),
        (OK + " We will refund the full amount.", "commitment"),
        (OK + " Your refund has been approved.", "commitment"),
        (OK + " We guarantee delivery tomorrow.", "commitment"),
        (OK + " Wij zullen u het bedrag terugbetalen.", "commitment"),
        (OK + " Wir werden Ihnen den Betrag erstatten.", "commitment"),
        (OK + " Nous allons vous rembourser.", "commitment"),
        (OK + " Best, [Your Name]", "placeholder"),
        (OK + " Dear {{customer_name}}", "placeholder"),
        (OK + " TODO add date", "placeholder"),
        (OK + " As an AI I cannot say.", "AI"),
        (OK + " According to the knowledge base...", "AI"),
    ],
)
def test_each_rule_fires(text: str, needle: str):
    issues = check_reply(text, POLICY)
    assert issues, text
    assert any(needle.lower() in issue.lower() for issue in issues), issues


def test_cannot_guarantee_is_fine():
    assert check_reply(OK + " Unfortunately we cannot guarantee a delivery day.", POLICY) == []


def test_lookalike_domain_is_rejected():
    assert check_reply(OK + " https://parcelo.example.evil.com/", POLICY)
