import pytest

from backend.core.risk_rules import (
    detect_risk_flags,
    has_high_risk_pattern,
    should_auto_accept,
    should_force_review,
    is_forbidden_change,
    edit_distance_ratio,
    extract_numbers,
    extract_urls,
    extract_emails,
    extract_currency,
)


class TestRiskFlags:
    def test_detect_number(self):
        flags = detect_risk_flags("价格99元")
        assert "number" in flags

    def test_detect_currency(self):
        flags = detect_risk_flags("￥99.00")
        assert "currency" in flags

    def test_detect_percent(self):
        flags = detect_risk_flags("折扣50%")
        assert "percent" in flags

    def test_detect_date(self):
        flags = detect_risk_flags("2024年3月15日")
        assert "date" in flags

    def test_detect_time(self):
        flags = detect_risk_flags("14:30")
        assert "time" in flags

    def test_detect_phone(self):
        flags = detect_risk_flags("13800138000")
        assert "phone_cn" in flags

    def test_detect_url(self):
        flags = detect_risk_flags("https://example.com")
        assert "url" in flags

    def test_detect_email(self):
        flags = detect_risk_flags("test@example.com")
        assert "email" in flags

    def test_detect_model(self):
        flags = detect_risk_flags("iPhone15Pro")
        assert "model" in flags

    def test_no_risk(self):
        flags = detect_risk_flags("你好世界")
        assert len(flags) == 0

    def test_empty_text(self):
        flags = detect_risk_flags("")
        assert len(flags) == 0

    def test_has_high_risk(self):
        assert has_high_risk_pattern("￥99") is True
        assert has_high_risk_pattern("你好") is False


class TestAutoAccept:
    def test_auto_accept_normal(self):
        result = should_auto_accept(
            "限时优惠", "限时优惠", 0.9, 0.95, False
        )
        assert result is True

    def test_auto_accept_low_ocr_conf(self):
        result = should_auto_accept(
            "限时优惠", "限时优惠", 0.5, 0.95, False
        )
        assert result is False

    def test_auto_accept_low_llm_conf(self):
        result = should_auto_accept(
            "限时优惠", "限时优惠", 0.9, 0.8, False
        )
        assert result is False

    def test_auto_accept_high_risk_original(self):
        result = should_auto_accept(
            "￥99", "￥99", 0.9, 0.95, False
        )
        assert result is False

    def test_auto_accept_high_risk_corrected(self):
        result = should_auto_accept(
            "优惠", "￥99", 0.9, 0.95, False
        )
        assert result is False

    def test_auto_accept_high_edit_distance(self):
        result = should_auto_accept(
            "你好世界", "再见宇宙", 0.9, 0.95, False
        )
        assert result is False

    def test_auto_accept_short_text(self):
        result = should_auto_accept(
            "a", "b", 0.9, 0.95, False
        )
        assert result is False

    def test_auto_accept_needs_human(self):
        result = should_auto_accept(
            "你好", "你好", 0.9, 0.95, True
        )
        assert result is False

    def test_auto_accept_tiny_region(self):
        result = should_auto_accept(
            "你好", "你好", 0.9, 0.95, False, [0, 0, 100, 10]
        )
        assert result is False


class TestForceReview:
    def test_force_review_high_risk(self):
        result = should_force_review("￥99", "￥99", 0.9, 0.95, False)
        assert result is True

    def test_force_review_low_ocr(self):
        result = should_force_review("你好", "你好", 0.5, 0.95, False)
        assert result is True

    def test_force_review_low_llm(self):
        result = should_force_review("你好", "你好", 0.9, 0.8, False)
        assert result is True

    def test_force_review_high_edit_distance(self):
        result = should_force_review("你好世界", "再见宇宙", 0.9, 0.95, False)
        assert result is True

    def test_force_review_short_text_changed(self):
        result = should_force_review("你", "好", 0.9, 0.95, False)
        assert result is True

    def test_force_review_tiny_region(self):
        result = should_force_review("你好", "你好", 0.9, 0.95, False, [0, 0, 100, 10])
        assert result is True

    def test_no_force_review(self):
        result = should_force_review("你好", "你好", 0.9, 0.95, False)
        assert result is False


class TestForbiddenChange:
    def test_forbidden_number_change(self):
        assert is_forbidden_change("价格99", "价格88") is True

    def test_forbidden_url_change(self):
        assert is_forbidden_change("https://a.com", "https://b.com") is True

    def test_forbidden_email_change(self):
        assert is_forbidden_change("a@b.com", "c@d.com") is True

    def test_forbidden_currency_change(self):
        assert is_forbidden_change("￥99", "￥88") is True

    def test_not_forbidden(self):
        assert is_forbidden_change("你好", "您好") is False

    def test_not_forbidden_same_numbers(self):
        assert is_forbidden_change("价格99", "价钱99") is False


class TestEditDistance:
    def test_same_string(self):
        assert edit_distance_ratio("hello", "hello") == 0.0

    def test_completely_different(self):
        assert edit_distance_ratio("abc", "xyz") == 1.0

    def test_one_char_diff(self):
        ratio = edit_distance_ratio("hello", "hallo")
        assert abs(ratio - 0.2) < 0.01

    def test_empty_strings(self):
        assert edit_distance_ratio("", "") == 0.0

    def test_one_empty(self):
        assert edit_distance_ratio("hello", "") == 1.0


class TestExtraction:
    def test_extract_numbers(self):
        assert extract_numbers("价格99元") == ["99"]

    def test_extract_urls(self):
        result = extract_urls("访问https://example.com")
        assert len(result) >= 1

    def test_extract_emails(self):
        result = extract_emails("联系test@example.com")
        assert len(result) >= 1

    def test_extract_currency(self):
        result = extract_currency("价格￥99.00")
        assert len(result) >= 1
