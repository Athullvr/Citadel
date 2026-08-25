import pytest
from features import FEATURE_NAMES, extract_features


def test_features_keys_and_types():
    feats = extract_features("Research 5 topics", ["web_search"])
    for name in FEATURE_NAMES:
        assert name in feats
        assert isinstance(feats[name], (int, float))


def test_empty_and_whitespace_input():
    feats_empty = extract_features("", [])
    assert feats_empty["text_char_len"] == 0
    assert feats_empty["text_word_len"] == 0
    assert feats_empty["num_tools"] == 0
    assert feats_empty["open_ended_keyword_hits"] == 0
    assert feats_empty["narrow_keyword_hits"] == 0
    assert feats_empty["max_explicit_count"] == 0
    assert feats_empty["sum_explicit_counts"] == 0

    feats_ws = extract_features("   \n\t  ", None)  # type: ignore[arg-type]
    assert feats_ws["text_char_len"] == 0
    assert feats_ws["num_tools"] == 0


def test_no_keyword_match_input():
    text = "Deploy the artifact to the production server cluster immediately."
    feats = extract_features(text, ["deploy_tool"])
    assert feats["open_ended_keyword_hits"] == 0
    assert feats["narrow_keyword_hits"] == 0
    assert feats["num_tools"] == 1
    assert feats["text_char_len"] == len(text)


def test_max_length_input():
    # Long text with multiple repeated clauses and numbers
    text = ("Research competitor features and summarize the following details. " * 50).strip()
    assert len(text) > 3000
    feats = extract_features(text, ["web_search", "draft_document"])
    assert feats["text_char_len"] == len(text)
    assert feats["open_ended_keyword_hits"] >= 50
    assert feats["narrow_keyword_hits"] >= 50
    assert feats["num_tools"] == 2


def test_unicode_and_special_characters():
    text = "🔍 Investigar 3 fuentes y redigir 2 correos electrónicos con café ☕! ¿Cuál es el resultado?"
    feats = extract_features(text, ["search", "email"])
    assert feats["text_char_len"] == len(text)
    assert feats["num_tools"] == 2
    assert isinstance(feats["is_question"], int)


def test_count_pattern_extraction():
    text = "Find 5 sources, review 10 competitors, and draft 3 emails to 4 attendees."
    feats = extract_features(text, ["web_search"])
    assert feats["max_explicit_count"] == 10
    assert feats["sum_explicit_counts"] == 5 + 10 + 3 + 4


def test_case_insensitivity():
    lower_text = "research and calculate what is this?"
    upper_text = "RESEARCH AND CALCULATE WHAT IS THIS?"
    lower_feats = extract_features(lower_text, ["calc"])
    upper_feats = extract_features(upper_text, ["calc"])
    assert lower_feats["open_ended_keyword_hits"] == upper_feats["open_ended_keyword_hits"]
    assert lower_feats["narrow_keyword_hits"] == upper_feats["narrow_keyword_hits"]
    assert lower_feats["is_question"] == upper_feats["is_question"] == 1
