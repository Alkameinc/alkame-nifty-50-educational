"""
Symbol Validator

Centralized symbol validation for the trading system.
Ensures all operations use valid NIFTY 50 universe symbols.

Part of F17 fix: Missing symbol validation
"""

import logging
from functools import wraps
from typing import Callable, Any

from universe_provider import UniverseProvider

logger = logging.getLogger(__name__)


class SymbolValidationError(ValueError):
    """Raised when an invalid symbol is provided to a trading operation."""

    pass


class SymbolValidator:
    """
    Validates that symbols belong to the configured trading universe.

    Provides:
    - Direct validation method
    - Decorator for automatic validation
    - Clear error messages with suggestions
    """

    def __init__(self, universe_provider: UniverseProvider | None = None):
        """
        Initialize validator with universe provider.

        Args:
            universe_provider: Provider for valid symbols (uses global default if None)
        """
        if universe_provider is None:
            self.universe_provider = UniverseProvider()
        else:
            self.universe_provider = universe_provider

        self._valid_symbols_cache: set[str] | None = None

    def get_valid_symbols(self) -> set[str]:
        """
        Get set of valid symbols from universe.

        Returns:
            Set of valid symbol strings (e.g., {"RELIANCE", "TCS", ...})
        """
        if self._valid_symbols_cache is None:
            constituents = self.universe_provider.get_constituents()
            self._valid_symbols_cache = set(constituents)

        return self._valid_symbols_cache

    def is_valid(self, symbol: str) -> bool:
        """
        Check if a symbol is valid.

        Args:
            symbol: Symbol to validate (case-insensitive)

        Returns:
            True if symbol is in valid universe, False otherwise
        """
        if not symbol:
            return False

        symbol_upper = symbol.strip().upper()
        valid_symbols = self.get_valid_symbols()

        return symbol_upper in valid_symbols

    def validate(self, symbol: str, *, raise_error: bool = True) -> bool:
        """
        Validate a symbol and optionally raise error if invalid.

        Args:
            symbol: Symbol to validate
            raise_error: If True, raises SymbolValidationError for invalid symbols

        Returns:
            True if valid

        Raises:
            SymbolValidationError: If symbol is invalid and raise_error=True
        """
        if not symbol or not isinstance(symbol, str):
            if raise_error:
                raise SymbolValidationError(
                    f"Invalid symbol type: expected string, got {type(symbol).__name__}"
                )
            return False

        symbol_upper = symbol.strip().upper()

        if not self.is_valid(symbol_upper):
            if raise_error:
                # Provide helpful error with similar symbols
                suggestion = self._find_similar_symbol(symbol_upper)
                error_msg = (
                    f"Invalid symbol '{symbol}': not in NIFTY 50 universe "
                    f"({len(self.get_valid_symbols())} valid symbols)."
                )
                if suggestion:
                    error_msg += f" Did you mean '{suggestion}'?"

                raise SymbolValidationError(error_msg)
            return False

        return True

    def validate_batch(self, symbols: list[str], *, raise_error: bool = True) -> tuple[list[str], list[str]]:
        """
        Validate a batch of symbols.

        Args:
            symbols: List of symbols to validate
            raise_error: If True, raises SymbolValidationError if any symbol is invalid

        Returns:
            (valid_symbols, invalid_symbols) tuple

        Raises:
            SymbolValidationError: If any symbol is invalid and raise_error=True
        """
        valid = []
        invalid = []

        for symbol in symbols:
            if self.is_valid(symbol):
                valid.append(symbol.strip().upper())
            else:
                invalid.append(symbol)

        if invalid and raise_error:
            raise SymbolValidationError(
                f"Invalid symbols: {', '.join(invalid)}. "
                f"Valid universe has {len(self.get_valid_symbols())} symbols."
            )

        return valid, invalid

    def _find_similar_symbol(self, symbol: str) -> str | None:
        """
        Find a similar valid symbol (for typo suggestions).

        Args:
            symbol: The invalid symbol

        Returns:
            Similar valid symbol or None
        """
        symbol_upper = symbol.upper()
        valid_symbols = self.get_valid_symbols()

        # Exact substring match
        for valid in valid_symbols:
            if symbol_upper in valid or valid in symbol_upper:
                return valid

        # Common prefixes (first 3 chars)
        if len(symbol) >= 3:
            prefix = symbol_upper[:3]
            for valid in valid_symbols:
                if valid.startswith(prefix):
                    return valid

        # Return None if no good match
        return None

    def requires_valid_symbol(self, param_name: str = "symbol") -> Callable:
        """
        Decorator that validates symbol parameter before function execution.

        Args:
            param_name: Name of the symbol parameter to validate (default: "symbol")

        Returns:
            Decorator function

        Example:
            @validator.requires_valid_symbol()
            def generate_signal(symbol: str, ...):
                # symbol is guaranteed to be valid here
                pass

            @validator.requires_valid_symbol("stock_symbol")
            def analyze(stock_symbol: str, ...):
                pass
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Get symbol from kwargs or args
                symbol = kwargs.get(param_name)

                # If not in kwargs, try to get from args based on function signature
                if symbol is None and args:
                    import inspect

                    sig = inspect.signature(func)
                    param_names = list(sig.parameters.keys())

                    if param_name in param_names:
                        param_index = param_names.index(param_name)
                        if param_index < len(args):
                            symbol = args[param_index]

                # Validate symbol
                if symbol is not None:
                    try:
                        self.validate(symbol, raise_error=True)
                    except SymbolValidationError as e:
                        logger.error(
                            f"Symbol validation failed in {func.__name__}('{symbol}'): {e}"
                        )
                        raise

                # Call original function
                return func(*args, **kwargs)

            return wrapper

        return decorator


# Global default instance
_default_validator: SymbolValidator | None = None


def get_validator() -> SymbolValidator:
    """Get or create the global default validator instance."""
    global _default_validator
    if _default_validator is None:
        _default_validator = SymbolValidator()
    return _default_validator


def validate_symbol(symbol: str, *, raise_error: bool = True) -> bool:
    """
    Convenience function for validating a single symbol using the global validator.

    Args:
        symbol: Symbol to validate
        raise_error: If True, raises SymbolValidationError for invalid symbols

    Returns:
        True if valid

    Raises:
        SymbolValidationError: If symbol is invalid and raise_error=True
    """
    return get_validator().validate(symbol, raise_error=raise_error)


def is_valid_symbol(symbol: str) -> bool:
    """
    Convenience function to check if a symbol is valid (no exception).

    Args:
        symbol: Symbol to check

    Returns:
        True if valid, False otherwise
    """
    return get_validator().is_valid(symbol)


# ============================================================================
# Self-test
# ============================================================================
if __name__ == "__main__":
    from config import configure_logging

    configure_logging(log_filename="symbol_validator_selftest.log")
    logger.info("Running symbol_validator.py self-test...")

    print("\n=== SYMBOL VALIDATOR SELF-TEST ===")

    validator = SymbolValidator()
    valid_symbols = validator.get_valid_symbols()
    print(f"Valid universe: {len(valid_symbols)} symbols")

    # Test 1: Valid symbols
    test_valid = ["RELIANCE", "TCS", "HDFCBANK", "INFY"]
    for sym in test_valid:
        is_valid = validator.is_valid(sym)
        print(f"  {sym}: {'✓ valid' if is_valid else '✗ invalid'}")
        assert is_valid is True, f"{sym} should be valid"

    # Test 2: Invalid symbols
    test_invalid = ["INVALID", "NOTREAL", ""]
    for sym in test_invalid:
        is_valid = validator.is_valid(sym)
        print(f"  {sym or '(empty)'}: {'✓ valid' if is_valid else '✗ invalid (expected)'}")
        assert is_valid is False, f"{sym} should be invalid"

    # Test 3: Validation with error
    try:
        validator.validate("INVALID_SYMBOL", raise_error=True)
        assert False, "Should have raised SymbolValidationError"
    except SymbolValidationError as e:
        print(f"\n✓ Caught expected error: {e}")

    # Test 4: Batch validation
    mixed_symbols = ["RELIANCE", "INVALID", "TCS", "NOTREAL"]
    valid, invalid = validator.validate_batch(mixed_symbols, raise_error=False)
    print(f"\nBatch validation:")
    print(f"  Valid: {valid}")
    print(f"  Invalid: {invalid}")
    assert len(valid) == 2
    assert len(invalid) == 2

    # Test 5: Similar symbol suggestion
    try:
        validator.validate("REL", raise_error=True)
    except SymbolValidationError as e:
        print(f"\n✓ Suggestion test: {e}")
        assert "RELIANCE" in str(e), "Should suggest RELIANCE"

    # Test 6: Case insensitivity
    assert validator.is_valid("reliance") is True
    assert validator.is_valid("RELIANCE") is True
    assert validator.is_valid("ReLiAnCe") is True

    # Test 7: Convenience functions
    assert is_valid_symbol("TCS") is True
    assert is_valid_symbol("INVALID") is False

    print("\n✓ All tests passed")
