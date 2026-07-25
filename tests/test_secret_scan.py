"""The secret scan must catch real credential shapes and not cry wolf.

This repository is PUBLIC, so a key in a tracked file is published on the next
push and stays in history after deletion. The scan is the last line of defence
against that, so its patterns are tested directly.
"""

import pytest

from selfcheck import SECRET_PATTERNS


def _match(text: str):
    return [name for name, pattern in SECRET_PATTERNS if pattern.search(text)]


# Every fixture is CONSTRUCTED rather than written as a literal, so this file
# does not itself contain a matchable credential shape. That matters: the scan
# runs over all tracked files including this one, and the alternative - an
# exclusion list - could later hide a real key. (The scanner caught exactly
# this on its first live run, which is the behaviour we want.)
@pytest.mark.parametrize("sample,expected", [
    ("pplx-" + "A" * 40, "Perplexity API key"),
    ("sk-" + "b" * 40, "OpenAI API key"),
    ("AKIA" + "A" * 16, "AWS access key"),
    ("ghp_" + "c" * 36, "GitHub token"),
    ("xoxb-" + "1234567890-abcdef", "Slack token"),
    ("-----BEGIN RSA " + "PRIVATE KEY-----", "private key block"),
])
def test_detects_credential_shapes(sample, expected):
    assert expected in _match(f"config = '{sample}'")


def test_detects_a_key_embedded_in_ordinary_code():
    src = 'HEADERS = {"Authorization": "Bearer pplx-' + "Z" * 44 + '"}'
    assert "Perplexity API key" in _match(src)


@pytest.mark.parametrize("benign", [
    "the pplx- prefix identifies Perplexity keys",   # prose, no key body
    "sk-",                                            # too short
    "AKIA",                                           # too short
    "PERPLEXITY_API_KEY=your-key-here",               # placeholder
    "load_key() reads PERPLEXITY_API_KEY from the environment",
    "https://api.perplexity.ai/search",
])
def test_does_not_flag_benign_text(benign):
    assert _match(benign) == [], f"false positive on: {benign}"


def test_research_web_source_contains_no_key():
    """The helper reads its key from the environment; if a literal ever gets
    pasted into it, this fails before the scan even runs."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "research_web.py").read_text(
        encoding="utf-8")
    assert _match(src) == []
    assert "os.environ.get" in src
