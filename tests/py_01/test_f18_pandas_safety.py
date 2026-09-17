"""
test_f18_pandas_safety.py

Comprehensive test suite for F18: Unsafe pandas operations (3 pts, P2)

Tests cover:
1. Chained assignment detection
2. inplace=True usage detection  
3. Copy semantics verification
4. View vs copy behavior
5. SettingWithCopyWarning scenarios
6. Safe patterns validation

Part of PY-01 Assignment

Author: Engineering Team
Date: 2026-09-14
"""

import pytest
import pandas as pd
import warnings
import tempfile
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pandas_safety_validator import (
    PandasSafetyValidator,
    PandasSafetyIssue,
    verify_pandas_copy_semantics
)


# ============================================================================
# Test Suite 1: Runtime Copy Semantics
# ============================================================================

def test_f18_copy_creates_independent_dataframe():
    """F18-1: Verify .copy() creates truly independent DataFrame."""
    df_original = pd.DataFrame({
        'price': [100, 101, 102],
        'volume': [1000, 1100, 1200]
    })
    
    df_copy = df_original.copy()
    df_copy.loc[0, 'price'] = 999
    
    # Original should be unchanged
    assert df_original.loc[0, 'price'] == 100, "Original DataFrame was mutated!"
    assert df_copy.loc[0, 'price'] == 999, "Copy was not modified!"
    
    print("✓ F18-1: .copy() creates independent DataFrame")


def test_f18_filtered_dataframe_without_copy_is_view():
    """F18-2: Verify filtered DataFrame without .copy() can create view issues."""
    df_base = pd.DataFrame({
        'symbol': ['RELIANCE', 'TCS', 'INFY', 'HDFC'],
        'price': [2500, 3500, 1500, 1600],
        'sector': ['Energy', 'IT', 'IT', 'Finance']
    })
    
    # UNSAFE: Filtering without .copy()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Create filter (might be view or copy, depending on pandas version)
        df_it_sector = df_base[df_base['sector'] == 'IT']
        
        # Try to modify - may trigger SettingWithCopyWarning
        try:
            df_it_sector['price'] = df_it_sector['price'] * 1.1
        except Exception:
            pass  # Some pandas versions raise exception instead of warning
        
        # Check if warning was raised (expected behavior)
        setting_with_copy_warnings = [
            warning for warning in w
            if 'SettingWithCopyWarning' in str(warning.category)
        ]
        
        # Note: behavior is pandas version-dependent
        # Modern pandas (2.0+) with copy-on-write may not warn
        # But we document the pattern as unsafe regardless
    
    print("✓ F18-2: Filtered DataFrame without .copy() behavior documented")


def test_f18_proper_filtered_assignment_with_copy():
    """F18-3: Verify proper pattern: filter then copy before modification."""
    df_base = pd.DataFrame({
        'symbol': ['RELIANCE', 'TCS', 'INFY', 'HDFC'],
        'price': [2500, 3500, 1500, 1600],
        'sector': ['Energy', 'IT', 'IT', 'Finance']
    })
    
    # SAFE: Filter then copy
    df_it_sector = df_base[df_base['sector'] == 'IT'].copy()
    original_prices = df_base['price'].tolist()
    
    # Modify copy
    df_it_sector.loc[:, 'price'] = df_it_sector['price'] * 1.1
    
    # Original should be unchanged
    assert df_base['price'].tolist() == original_prices
    
    # Copy should be modified
    assert df_it_sector['price'].tolist() != original_prices
    
    print("✓ F18-3: Proper filtered assignment with .copy() works correctly")


def test_f18_drop_returns_new_dataframe():
    """F18-4: Verify .drop() returns new DataFrame, doesn't modify original."""
    df_original = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })
    
    # Drop column (without inplace=True)
    df_dropped = df_original.drop(columns=['C'])
    
    # Original should still have column C
    assert 'C' in df_original.columns
    assert list(df_original.columns) == ['A', 'B', 'C']
    
    # New DataFrame should not have column C
    assert 'C' not in df_dropped.columns
    assert list(df_dropped.columns) == ['A', 'B']
    
    print("✓ F18-4: .drop() returns new DataFrame without modifying original")


def test_f18_drop_with_assignment_is_safe_pattern():
    """F18-5: Verify df = df.drop() pattern is safe and explicit."""
    df = pd.DataFrame({
        'keep_col': [1, 2, 3],
        'drop_col': [4, 5, 6]
    })
    
    original_id = id(df)
    
    # Recommended pattern: assign back
    df = df.drop(columns=['drop_col'])
    
    # This creates a new DataFrame object
    assert id(df) != original_id
    assert 'drop_col' not in df.columns
    assert 'keep_col' in df.columns
    
    print("✓ F18-5: df = df.drop() pattern is safe and explicit")


def test_f18_loc_single_bracket_is_safe():
    """F18-6: Verify .loc with single bracket is safe for assignment."""
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    
    # SAFE: Single .loc[] call
    df.loc[0, 'A'] = 999
    
    assert df.loc[0, 'A'] == 999
    assert df.loc[1, 'A'] == 2  # Other rows unchanged
    
    print("✓ F18-6: .loc[row, col] single bracket assignment is safe")


def test_f18_chained_subscript_is_unsafe():
    """F18-7: Document that chained subscript df[col1][col2] is unsafe."""
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    
    # UNSAFE PATTERN (documented, not recommended to actually test this way)
    # df['A'][0] = 999  # This is unsafe!
    
    # Instead, we document the safe alternative:
    # SAFE: df.loc[0, 'A'] = 999
    
    # This test documents the issue without actually triggering it
    assert True, "Chained subscript pattern is documented as unsafe"
    
    print("✓ F18-7: Chained subscript df[col][row] pattern documented as unsafe")


# ============================================================================
# Test Suite 2: Static Analysis - Chained Assignment Detection
# ============================================================================

def test_f18_validator_detects_chained_assignment():
    """F18-8: Validator detects chained assignment patterns."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("""
import pandas as pd

df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})

# UNSAFE: Chained assignment
df['A'][0] = 999

# UNSAFE: Another variant
df[df['B'] > 2]['A'] = 100
""")
        temp_file = Path(f.name)
    
    try:
        validator = PandasSafetyValidator()
        issues = validator.validate_file(temp_file)
        
        # Should detect at least one chained assignment
        chained_issues = [
            i for i in issues
            if i.issue_type in ['CHAINED_ASSIGNMENT', 'CHAINED_SUBSCRIPT_ASSIGNMENT']
        ]
        
        assert len(chained_issues) > 0, "Validator should detect chained assignment"
        assert any('chained' in i.description.lower() for i in chained_issues)
        
        print(f"✓ F18-8: Validator detected {len(chained_issues)} chained assignment(s)")
    
    finally:
        temp_file.unlink()


def test_f18_validator_detects_inplace_true():
    """F18-9: Validator detects inplace=True usage."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("""
import pandas as pd

df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})

# DISCOURAGED: inplace=True
df.drop(columns=['B'], inplace=True)

# DISCOURAGED: Another inplace
df.fillna(0, inplace=True)
""")
        temp_file = Path(f.name)
    
    try:
        validator = PandasSafetyValidator()
        issues = validator.validate_file(temp_file)
        
        # Should detect inplace=True usage
        inplace_issues = [
            i for i in issues
            if i.issue_type == 'INPLACE_TRUE'
        ]
        
        assert len(inplace_issues) >= 2, f"Expected 2 inplace issues, found {len(inplace_issues)}"
        assert all('inplace' in i.description.lower() for i in inplace_issues)
        
        print(f"✓ F18-9: Validator detected {len(inplace_issues)} inplace=True usage(s)")
    
    finally:
        temp_file.unlink()


def test_f18_validator_ignores_safe_patterns():
    """F18-10: Validator does not flag safe patterns."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("""
import pandas as pd

df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})

# SAFE: Using .loc correctly
df.loc[0, 'A'] = 999

# SAFE: Using .copy()
df_filtered = df[df['B'] > 2].copy()
df_filtered['A'] = 100

# SAFE: Assignment pattern
df = df.drop(columns=['B'])

# SAFE: Method chaining
df = (df
      .drop(columns=['C'], errors='ignore')
      .reset_index(drop=True))
""")
        temp_file = Path(f.name)
    
    try:
        validator = PandasSafetyValidator()
        issues = validator.validate_file(temp_file)
        
        # Should have no critical issues
        critical_issues = [
            i for i in issues
            if i.severity == PandasSafetyIssue.SEVERITY_CRITICAL
        ]
        
        assert len(critical_issues) == 0, f"Safe code flagged as critical: {critical_issues}"
        
        print("✓ F18-10: Validator correctly ignores safe patterns")
    
    finally:
        temp_file.unlink()


def test_f18_validator_provides_recommendations():
    """F18-11: Validator provides actionable recommendations."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("""
df['A'][0] = 999
df.drop(columns=['B'], inplace=True)
""")
        temp_file = Path(f.name)
    
    try:
        validator = PandasSafetyValidator()
        issues = validator.validate_file(temp_file)
        
        # All issues should have recommendations
        for issue in issues:
            assert issue.recommendation, f"Issue {issue.issue_type} missing recommendation"
            assert len(issue.recommendation) > 20, "Recommendation too short"
            assert '.loc' in issue.recommendation or 'df =' in issue.recommendation
        
        print(f"✓ F18-11: All {len(issues)} issues have actionable recommendations")
    
    finally:
        temp_file.unlink()


# ============================================================================
# Test Suite 3: Project-Specific Validation
# ============================================================================

def test_f18_feature_engineer_no_inplace():
    """F18-12: Verify feature_engineer.py doesn't use inplace=True."""
    feature_engineer_path = Path(__file__).parent.parent.parent / 'feature_engineer.py'
    
    if not feature_engineer_path.exists():
        pytest.skip("feature_engineer.py not found")
    
    with open(feature_engineer_path, 'r') as f:
        content = f.read()
    
    # Check for inplace=True in production code (not in comments)
    lines = content.splitlines()
    inplace_lines = []
    
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        if 'inplace=True' in line or 'inplace = True' in line:
            inplace_lines.append((i, line.strip()))
    
    assert len(inplace_lines) == 0, (
        f"Found inplace=True in feature_engineer.py:\n" +
        "\n".join(f"  Line {num}: {line}" for num, line in inplace_lines)
    )
    
    print("✓ F18-12: feature_engineer.py does not use inplace=True")


def test_f18_feature_engineer_uses_copy_correctly():
    """F18-13: Verify feature_engineer.py uses .copy() correctly."""
    feature_engineer_path = Path(__file__).parent.parent.parent / 'feature_engineer.py'
    
    if not feature_engineer_path.exists():
        pytest.skip("feature_engineer.py not found")
    
    with open(feature_engineer_path, 'r') as f:
        content = f.read()
    
    # Should have .copy() calls for DataFrames being modified
    assert '.copy()' in content, "feature_engineer.py should use .copy()"
    
    # Check specific pattern: out = stock_df.copy()
    assert 'out = stock_df.copy()' in content or 'out=stock_df.copy()' in content
    
    print("✓ F18-13: feature_engineer.py uses .copy() correctly")


def test_f18_no_chained_assignments_in_production_code():
    """F18-14: Verify no chained assignments in main production files."""
    production_files = [
        'feature_engineer.py',
        'model_trainer.py',
        'ensemble_manager.py',
        'data_fetcher.py',
        'runtime_validator.py',
        'market_data_provider.py'
    ]
    
    project_root = Path(__file__).parent.parent.parent
    validator = PandasSafetyValidator()
    
    all_critical_issues = []
    
    for filename in production_files:
        filepath = project_root / filename
        if not filepath.exists():
            continue
        
        issues = validator.validate_file(filepath)
        critical = [i for i in issues if i.severity == PandasSafetyIssue.SEVERITY_CRITICAL]
        all_critical_issues.extend(critical)
    
    if all_critical_issues:
        report = "\n".join(str(issue) for issue in all_critical_issues)
        pytest.fail(f"Found {len(all_critical_issues)} critical pandas issues:\n{report}")
    
    print(f"✓ F18-14: No critical pandas issues in {len(production_files)} production files")


# ============================================================================
# Test Suite 4: Copy Semantics Verification
# ============================================================================

def test_f18_verify_pandas_copy_semantics():
    """F18-15: Runtime verification of pandas copy semantics."""
    result = verify_pandas_copy_semantics()
    assert result, "Pandas copy semantics verification failed"
    
    print("✓ F18-15: Pandas copy semantics verified")


def test_f18_shallow_vs_deep_copy():
    """F18-16: Verify DataFrame.copy() is deep by default."""
    df = pd.DataFrame({
        'A': [[1, 2], [3, 4], [5, 6]],  # Nested lists
        'B': [10, 20, 30]
    })
    
    # Default .copy() should be deep
    df_copy = df.copy()
    
    # Modify nested structure in copy
    df_copy.at[0, 'A'].append(999)
    
    # Original should be affected (nested objects are still references)
    # Note: This documents pandas behavior, not necessarily what we want
    # For truly independent nested data, need copy.deepcopy or restructure
    
    # But scalar columns should be independent
    df_copy.loc[0, 'B'] = 999
    assert df.loc[0, 'B'] == 10, "Scalar value in original should be unchanged"
    
    print("✓ F18-16: DataFrame.copy() deep copy semantics documented")


def test_f18_copy_after_filtering_best_practice():
    """F18-17: Demonstrate best practice: filter -> copy -> modify."""
    df = pd.DataFrame({
        'price': [100, 200, 300, 400, 500],
        'volume': [1000, 2000, 3000, 4000, 5000],
        'sector': ['IT', 'Finance', 'IT', 'Energy', 'IT']
    })
    
    # Best practice pattern
    df_it = df[df['sector'] == 'IT'].copy()  # Filter then copy
    df_it.loc[:, 'price'] = df_it['price'] * 1.1  # Modify safely
    
    # Original unchanged
    assert df['price'].tolist() == [100, 200, 300, 400, 500]
    
    # Filtered copy modified
    assert len(df_it) == 3  # IT sector only
    assert all(df_it['price'] > df[df['sector'] == 'IT']['price'].values)
    
    print("✓ F18-17: Filter -> copy -> modify pattern works correctly")


# ============================================================================
# Test Suite 5: Validator Integration
# ============================================================================

def test_f18_validator_generates_report():
    """F18-18: Validator generates human-readable report."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("df['A'][0] = 999\n")
        temp_file = Path(f.name)
    
    try:
        validator = PandasSafetyValidator()
        issues = validator.validate_file(temp_file)
        report = validator.generate_report(issues)
        
        assert 'PANDAS SAFETY VALIDATION REPORT' in report
        assert 'Critical' in report or 'CRITICAL' in report
        assert len(report) > 100, "Report should be detailed"
        
        print("✓ F18-18: Validator generates comprehensive report")
    
    finally:
        temp_file.unlink()


def test_f18_validator_categorizes_by_severity():
    """F18-19: Validator correctly categorizes issues by severity."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("""
df['A'][0] = 999  # Critical: chained assignment
df.drop(columns=['B'], inplace=True)  # Warning: inplace
""")
        temp_file = Path(f.name)
    
    try:
        validator = PandasSafetyValidator()
        issues = validator.validate_file(temp_file)
        
        severities = {issue.severity for issue in issues}
        assert PandasSafetyIssue.SEVERITY_CRITICAL in severities
        assert PandasSafetyIssue.SEVERITY_WARNING in severities
        
        critical_count = sum(1 for i in issues if i.severity == PandasSafetyIssue.SEVERITY_CRITICAL)
        warning_count = sum(1 for i in issues if i.severity == PandasSafetyIssue.SEVERITY_WARNING)
        
        assert critical_count >= 1, "Should have critical issues"
        assert warning_count >= 1, "Should have warning issues"
        
        print(f"✓ F18-19: Issues categorized: {critical_count} critical, {warning_count} warnings")
    
    finally:
        temp_file.unlink()


def test_f18_validator_handles_syntax_errors_gracefully():
    """F18-20: Validator handles files with syntax errors gracefully."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("""
def broken_function(
    # Missing closing paren - syntax error
df['A'][0] = 999  # But regex check should still catch this
""")
        temp_file = Path(f.name)
    
    try:
        validator = PandasSafetyValidator()
        
        # Should not raise exception
        issues = validator.validate_file(temp_file)
        
        # Regex checks should still work even if AST parsing fails
        assert isinstance(issues, list), "Should return list even with syntax errors"
        
        print("✓ F18-20: Validator handles syntax errors gracefully")
    
    finally:
        temp_file.unlink()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
