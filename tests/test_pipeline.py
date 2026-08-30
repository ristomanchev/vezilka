"""
Unit tests for the pure functions in build_stt_dataset_v2_new.py.

Run:
    .venv/bin/python -m pytest tests/ -q
or without pytest:
    .venv/bin/python tests/test_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import build_stt_dataset_v2_new as b


# --------------------------------------------------------------------------- #
# normalize_text
# --------------------------------------------------------------------------- #

def test_normalize_strips_emoji_and_hashtags_and_urls():
    out = b.normalize_text("Здраво 👋 сите #мкд виж https://x.com/a  @user")
    assert "👋" not in out
    assert "#" not in out and "@" not in out
    assert "http" not in out
    assert out.startswith("Здраво")


def test_normalize_collapses_whitespace_and_quotes():
    assert b.normalize_text("  а   б\n в ") == "а б в"
    assert b.normalize_text("„тест“") == '"тест"'


def test_normalize_keeps_case_and_sentence_punctuation():
    assert b.normalize_text("Ова е тест.") == "Ова е тест."


# --------------------------------------------------------------------------- #
# label_is_sane
# --------------------------------------------------------------------------- #

def test_label_sane_accepts_normal_macedonian_sentence():
    ok, why = b.label_is_sane("Готвам директно на неа и веднаш можам да го допрам местото.")
    assert ok, why


def test_label_sane_rejects_too_few_words():
    ok, why = b.label_is_sane("Здраво")
    assert not ok and why == "too few words"


def test_label_sane_rejects_mostly_latin():
    ok, why = b.label_is_sane("This is an english sentence about nothing")
    assert not ok and why in ("not enough Cyrillic", "mostly Latin")


def test_label_sane_rejects_blacklisted_brand():
    ok, why = b.label_is_sane("купи сега во MILD HOME STORE веднаш")
    assert not ok and why.startswith("blacklist")


def test_label_sane_rejects_repetitive_hallucination():
    ok, why = b.label_is_sane("да да да да да да да да")
    assert not ok and why == "repetitive"


# --------------------------------------------------------------------------- #
# word_recall / agreement
# --------------------------------------------------------------------------- #

def test_word_recall_full_match():
    r = b.word_recall("работната површина останува безбедна",
                      "РАБОТНАТА ПОВРШИНА ОСТАНУВА БЕЗБЕДНА")
    assert r == 1.0


def test_word_recall_partial():
    r = b.word_recall("работната површина останува безбедна на допир",
                      "нешто сосема друго тука")
    assert r == 0.0


def test_agreement_gold_case():
    label = "Веројатно се прашувате како е тоа можно."
    ocr = "ВЕРОЈАТНО СЕ ПРАШУВАТЕ КАКО Е ТОА МОЖНО? ТОА Е УРЕД"
    score, recall = b.agreement(label, ocr)
    assert score >= b.AGREEMENT_THRESHOLD
    assert recall >= b.WORD_RECALL_MIN


def test_agreement_rejects_wide_ocr_blob_with_few_shared_words():
    # Whisper garble from a dialect clip vs a big unrelated OCR blob.
    label = "Мене на тебе китекен да ми свараш на каф."
    ocr = ("Ла и кафенце ни сварии тука не на тебе ке ти текне да ми свараш.. "
           "Ај арно се имме, да не се раскараме ајде... Еден избор MILD HOME STORE")
    score, recall = b.agreement(label, ocr)
    assert not (score >= b.AGREEMENT_THRESHOLD and recall >= b.WORD_RECALL_MIN)


def test_agreement_empty_ocr():
    assert b.agreement("нешто тука", "") == (0, 0.0)


# --------------------------------------------------------------------------- #
# grouping helpers
# --------------------------------------------------------------------------- #

def test_ocr_text_for_window_filters_by_time():
    ocr = [
        {"time": 0.0, "text": "прв"},
        {"time": 5.0, "text": "во прозорецот"},
        {"time": 5.5, "text": "во прозорецот"},   # near-dup, should collapse
        {"time": 20.0, "text": "далеку"},
    ]
    out = b.ocr_text_for_window(ocr, 4.8, 6.0)
    assert "во прозорецот" in out
    assert "прв" not in out and "далеку" not in out
    assert out.count("во прозорецот") == 1


def test_split_is_deterministic_and_per_video():
    a1 = b.split_for_video("VID_A.mp4")
    a2 = b.split_for_video("VID_A.mp4")
    assert a1 == a2 and a1 in ("train", "test")


def test_clean_filename():
    assert b.clean_filename("DVTZhGAjMMX") == "dvtzhgajmmx"
    assert b.clean_filename("baba ruza #2") == "baba_ruza_2"


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(funcs) - failed}/{len(funcs)} passed")
    sys.exit(1 if failed else 0)
