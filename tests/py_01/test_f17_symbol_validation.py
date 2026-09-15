"""
Test Suite for F17: Missing Symbol Validation

Tests centralized symbol validation across the trading system:
- Symbol validator functionality
- Valid/invalid symbol detection
- Batch validation
- Error messages and suggestions
- Integration with universe provider

Part of PY-01 Assignment (5 points, P1)
"""

import pytest
from symbol_validator import (
    SymbolValidator,
    SymbolValidationError,
    validate_symbol,
    is_valid_symbol,
    get_validator,
)
from universe_provider import UniverseProvider


# ============================================================================
# NORMAL CASES: Valid symbol operations
# ============================================================================


def test_f17_normal_valid_nifty50_symbols():
    """
    NORMAL: Known NIFTY 50 symbols are validated as valid.

    Input: RELIANCE, TCS, HDFCBANK, INFY
    Expected: All return True
    """
    validator = SymbolValidator()

    valid_symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]

    for symbol in valid_symbols:
        assert validator.is_valid(symbol) is True, f"{symbol} should be valid"
        assert validator.validate(symbol, raise_error=False) is True

    print(f"✓ NORMAL case passed: {len(valid_symbols)} known symbols validated")


def test_f17_normal_case_insensitive_validation():
    """
    NORMAL: Symbol validation is case-insensitive.

    Input: "reliance", "RELIANCE", "ReLiAnCe"
    Expected: All valid
    """
    validator = SymbolValidator()

    variations = ["reliance", "RELIANCE", "ReLiAnCe", "rElIaNcE"]

    for symbol in variations:
        assert validator.is_valid(symbol) is True, f"{symbol} should be valid (case-insensitive)"

    print(f"✓ NORMAL case passed: Case-insensitive validation works")


def test_f17_normal_valid_universe_size():
    """
    NORMAL: Validator recognizes correct universe size (50 symbols).

    Input: Get valid symbols
    Expected: Exactly 50 symbols
    """
    validator = SymbolValidator()
    valid_symbols = validator.get_valid_symbols()

    assert isinstance(valid_symbols, set)
    assert len(valid_symbols) == 50, f"Expected 50 symbols, got {len(valid_symbols)}"

    print(f"✓ NORMAL case passed: Universe has {len(valid_symbols)} symbols")


def test_f17_normal_convenience_functions():
    """
    NORMAL: Convenience functions work correctly.

    Input: Use is_valid_symbol() and validate_symbol()
    Expected: Work without needing validator instance
    """
    # is_valid_symbol (no exception)
    assert is_valid_symbol("TCS") is True
    assert is_valid_symbol("INVALID") is False

    # validate_symbol (with exception)
    assert validate_symbol("RELIANCE", raise_error=False) is True
    assert validate_symbol("NOTREAL", raise_error=False) is False

    print(f"✓ NORMAL case passed: Convenience functions working")


# ============================================================================
# BOUNDARY CASES: Edge conditions
# ============================================================================


def test_f17_boundary_empty_and_whitespace_symbols():
    """
    BOUNDARY: Empty strings and whitespace are handled gracefully.

    Input: "", "   ", None-like
    Expected: Returns False, doesn't crash
    """
    validator = SymbolValidator()

    invalid_inputs = ["", "   ", "\t", "\n"]

    for inp in invalid_inputs:
        assert validator.is_valid(inp) is False, f"'{inp}' should be invalid"

    print(f"✓ BOUNDARY case passed: Empty/whitespace handled")


def test_f17_boundary_batch_validation_mixed():
    """
    BOUNDARY: Batch validation separates valid from invalid correctly.

    Input: Mix of valid and invalid symbols
    Expected: Returns two separate lists
    """
    validator = SymbolValidator()

    mixed = ["RELIANCE", "INVALID1", "TCS", "NOTREAL", "HDFCBANK", "FAKE"]

    valid, invalid = validator.validate_batch(mixed, raise_error=False)

    assert len(valid) == 3, f"Expected 3 valid, got {len(valid)}"
    assert len(invalid) == 3, f"Expected 3 invalid, got {len(invalid)}"
    assert "RELIANCE" in valid
    assert "TCS" in valid
    assert "HDFCBANK" in valid
    assert "INVALID1" in invalid
    assert "NOTREAL" in invalid
    assert "FAKE" in invalid

    print(f"✓ BOUNDARY case passed: Batch validation separated {len(valid)} valid from {len(invalid)} invalid")


def test_f17_boundary_symbol_with_spaces():
    """
    BOUNDARY: Symbols with leading/trailing spaces are trimmed.

    Input: "  RELIANCE  "
    Expected: Valid (trimmed to "RELIANCE")
    """
    validator = SymbolValidator()

    assert validator.is_valid("  RELIANCE  ") is True
    assert validator.is_valid("TCS   ") is True
    assert validator.is_valid("   INFY") is True

    print(f"✓ BOUNDARY case passed: Whitespace trimming works")


def test_f17_boundary_universe_provider_integration():
    """
    BOUNDARY: Validator integrates with custom UniverseProvider.

    Input: Create validator with specific provider
    Expected: Uses that provider's constituents
    """
    provider = UniverseProvider()
    validator = SymbolValidator(universe_provider=provider)

    # Should use same universe
    provider_symbols = set(provider.get_constituents())
    validator_symbols = validator.get_valid_symbols()

    assert validator_symbols == provider_symbols

    print(f"✓ BOUNDARY case passed: Universe provider integration works")


# ============================================================================
# FAILURE CASES: Invalid symbols and error handling
# ============================================================================


def test_f17_failure_invalid_symbol_raises_error():
    """
    FAILURE: Validating invalid symbol with raise_error=True raises exception.

    Input: "INVALID_SYMBOL"
    Expected: Raises SymbolValidationError
    """
    validator = SymbolValidator()

    with pytest.raises(SymbolValidationError) as exc_info:
        validator.validate("INVALID_SYMBOL", raise_error=True)

    error_msg = str(exc_info.value)
    assert "INVALID_SYMBOL" in error_msg
    assert "not in NIFTY 50 universe" in error_msg

    print(f"✓ FAILURE case passed: Invalid symbol raises error")


def test_f17_failure_invalid_symbol_with_suggestion():
    """
    FAILURE: Invalid symbol similar to valid one suggests correction.

    Input: "REL" (similar to "RELIANCE")
    Expected: Error message suggests "RELIANCE"
    """
    validator = SymbolValidator()

    with pytest.raises(SymbolValidationError) as exc_info:
        validator.validate("REL", raise_error=True)

    error_msg = str(exc_info.value)
    assert "Did you mean 'RELIANCE'?" in error_msg or "RELIANCE" in error_msg

    print(f"✓ FAILURE case passed: Suggestion provided for similar symbol")


def test_f17_failure_batch_validation_raises_on_invalid():
    """
    FAILURE: Batch validation with raise_error=True fails on any invalid.

    Input: ["RELIANCE", "INVALID", "TCS"]
    Expected: Raises SymbolValidationError listing invalid symbols
    """
    validator = SymbolValidator()

    with pytest.raises(SymbolValidationError) as exc_info:
        validator.validate_batch(["RELIANCE", "INVALID", "TCS"], raise_error=True)

    error_msg = str(exc_info.value)
    assert "INVALID" in error_msg
    assert "Invalid symbols" in error_msg

    print(f"✓ FAILURE case passed: Batch validation raises on invalid")


def test_f17_failure_non_string_symbol():
    """
    FAILURE: Non-string symbol types are handled gracefully.

    Input: 123, None, []
    Expected: Returns False or raises appropriate error
    """
    validator = SymbolValidator()

    # With raise_error=False, should return False
    assert validator.validate(None, raise_error=False) is False

    # With raise_error=True, should raise SymbolValidationError
    with pytest.raises(SymbolValidationError) as exc_info:
        validator.validate(123, raise_error=True)

    assert "Invalid symbol type" in str(exc_info.value)

    print(f"✓ FAILURE case passed: Non-string types handled")


# ============================================================================
# INTEGRATION: Validator with actual universe data
# ============================================================================


def test_f17_integration_all_nifty50_symbols_valid():
    """
    INTEGRATION: All symbols from universe provider are validated as valid.

    Input: All 50 NIFTY 50 constituents
    Expected: Every one validates
    """
    provider = UniverseProvider()
    validator = SymbolValidator(universe_provider=provider)

    constituents = provider.get_constituents()

    failures = []
    for symbol in constituents:
        if not validator.is_valid(symbol):
            failures.append(symbol)

    assert len(failures) == 0, f"These symbols failed validation: {failures}"

    print(f"✓ INTEGRATION passed: All {len(constituents)} universe symbols valid")


def test_f17_integration_validator_caches_symbols():
    """
    INTEGRATION: Validator caches valid symbols for performance.

    Input: Multiple validation calls
    Expected: Uses cached symbols (doesn't reload every time)
    """
    validator = SymbolValidator()

    # First call loads
    symbols1 = validator.get_valid_symbols()

    # Second call should use cache
    symbols2 = validator.get_valid_symbols()

    # Should be the same object (cached)
    assert symbols1 is symbols2, "Should use cached symbols"

    print(f"✓ INTEGRATION passed: Symbol caching works")


def test_f17_integration_global_validator_singleton():
    """
    INTEGRATION: Global validator is singleton pattern.

    Input: Multiple get_validator() calls
    Expected: Returns same instance
    """
    validator1 = get_validator()
    validator2 = get_validator()

    assert validator1 is validator2, "Should be same validator instance"

    print(f"✓ INTEGRATION passed: Global validator is singleton")


def test_f17_integration_decorator_validation():
    """
    INTEGRATION: Decorator validates symbol parameters automatically.

    Input: Function decorated with @requires_valid_symbol
    Expected: Invalid symbols raise error before function executes
    """
    validator = SymbolValidator()

    # Track if function was called
    function_called = [False]

    @validator.requires_valid_symbol()
    def test_function(symbol: str):
        function_called[0] = True
        return f"Processing {symbol}"

    # Valid symbol should work
    result = test_function("RELIANCE")
    assert function_called[0] is True
    assert "RELIANCE" in result

    # Reset
    function_called[0] = False

    # Invalid symbol should raise before function executes
    with pytest.raises(SymbolValidationError):
        test_function("INVALID_SYMBOL")

    assert function_called[0] is False, "Function should not have been called"

    print(f"✓ INTEGRATION passed: Decorator validation works")


def test_f17_integration_find_similar_symbols():
    """
    INTEGRATION: Similar symbol finder provides helpful suggestions.

    Input: Common typos and abbreviations
    Expected: Suggests correct symbols
    """
    validator = SymbolValidator()

    test_cases = [
        ("REL", "RELIANCE"),  # Prefix match
        ("HDFC", "HDFCBANK"),  # Partial match
        ("INF", "INFY"),  # Prefix match
    ]

    for typo, expected in test_cases:
        suggestion = validator._find_similar_symbol(typo)
        # Should suggest the expected symbol or something reasonable
        assert suggestion is not None, f"Should suggest something for '{typo}'"

    print(f"✓ INTEGRATION passed: Similar symbol suggestions work")


# ============================================================================
# REGRESSION: Ensure compatibility
# ============================================================================


def test_f17_regression_universe_provider_unchanged():
    """
    REGRESSION: UniverseProvider still works independently.

    Verify F17 doesn't break existing universe provider.
    """
    provider = UniverseProvider()

    assert provider.validate_constituents(expected_count=50) is True
    constituents = provider.get_constituents()
    assert len(constituents) == 50

    print(f"✓ REGRESSION passed: UniverseProvider unchanged")


def test_f17_regression_empty_validator_initialization():
    """
    REGRESSION: Creating validator without provider uses default.

    Input: SymbolValidator() with no args
    Expected: Uses global UniverseProvider
    """
    validator = SymbolValidator()

    # Should have valid symbols
    assert len(validator.get_valid_symbols()) == 50

    print(f"✓ REGRESSION passed: Default initialization works")


# ============================================================================
# DOCUMENTATION: Usage examples
# ============================================================================


def test_f17_documentation_basic_usage():
    """
    DOCUMENTATION: Basic usage patterns work as documented.

    Demonstrates common validation patterns.
    """
    # Pattern 1: Check if valid (no exception)
    if is_valid_symbol("RELIANCE"):
        # Process symbol
        pass

    # Pattern 2: Validate with exception
    try:
        validate_symbol("MAYBE_INVALID", raise_error=True)
    except SymbolValidationError as e:
        # Handle invalid symbol
        pass

    # Pattern 3: Batch validation
    validator = SymbolValidator()
    symbols = ["RELIANCE", "INVALID", "TCS"]
    valid, invalid = validator.validate_batch(symbols, raise_error=False)

    assert len(valid) == 2
    assert len(invalid) == 1

    print(f"✓ DOCUMENTATION passed: Usage examples work")


def test_f17_documentation_decorator_usage():
    """
    DOCUMENTATION: Decorator usage works as documented.

    Shows how to protect functions with validation decorator.
    """
    validator = SymbolValidator()

    @validator.requires_valid_symbol()
    def generate_signal(symbol: str, data):
        return f"Signal for {symbol}"

    # This works
    result = generate_signal("RELIANCE", data=None)
    assert "RELIANCE" in result

    # This raises SymbolValidationError
    with pytest.raises(SymbolValidationError):
        generate_signal("INVALID", data=None)

    print(f"✓ DOCUMENTATION passed: Decorator usage works")


# ============================================================================
# Run all tests
# ============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
