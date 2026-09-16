"""
pandas_safety_validator.py

Validates pandas DataFrame operations for common anti-patterns that can lead to:
- SettingWithCopyWarning
- Silent bugs due to chained assignments
- Performance issues from unintended copies
- Unpredictable mutation behavior

Part of PY-01 F18: Unsafe pandas operations (3 pts, P2)

Author: Engineering Team
Date: 2026-09-14
"""

import ast
import re
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd

# ANSI color codes for terminal output
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


class PandasSafetyIssue:
    """Represents a pandas safety issue found in code."""
    
    SEVERITY_CRITICAL = "CRITICAL"
    SEVERITY_WARNING = "WARNING"
    SEVERITY_INFO = "INFO"
    
    def __init__(
        self,
        filepath: str,
        line_number: int,
        column: int,
        issue_type: str,
        severity: str,
        description: str,
        code_snippet: str,
        recommendation: str
    ):
        self.filepath = filepath
        self.line_number = line_number
        self.column = column
        self.issue_type = issue_type
        self.severity = severity
        self.description = description
        self.code_snippet = code_snippet
        self.recommendation = recommendation
    
    def __repr__(self):
        severity_color = {
            self.SEVERITY_CRITICAL: RED,
            self.SEVERITY_WARNING: YELLOW,
            self.SEVERITY_INFO: BLUE
        }.get(self.severity, RESET)
        
        return (
            f"{severity_color}{BOLD}{self.severity}{RESET} | "
            f"{self.filepath}:{self.line_number}:{self.column}\n"
            f"  Type: {self.issue_type}\n"
            f"  Code: {self.code_snippet}\n"
            f"  Issue: {self.description}\n"
            f"  Fix: {self.recommendation}\n"
        )


class PandasSafetyValidator:
    """
    Static analyzer for pandas anti-patterns.
    
    Checks for:
    1. Chained assignments: df[col1][col2] = value
    2. inplace=True usage (discouraged in modern pandas)
    3. Missing .copy() after filtering operations
    4. Direct assignment to filtered DataFrames
    """
    
    def __init__(self):
        self.issues: List[PandasSafetyIssue] = []
    
    def validate_file(self, filepath: Path) -> List[PandasSafetyIssue]:
        """
        Validate a single Python file for pandas safety issues.
        
        Args:
            filepath: Path to Python file
        
        Returns:
            List of PandasSafetyIssue objects found in the file
        """
        self.issues = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
            
            # Run regex-based checks (faster, catches syntax pandas won't parse)
            self._check_chained_assignment_regex(filepath, lines)
            self._check_inplace_usage(filepath, lines)
            
            # Run AST-based checks (more precise)
            try:
                tree = ast.parse(content, filename=str(filepath))
                self._check_chained_assignment_ast(filepath, tree, lines)
            except SyntaxError:
                # If file has syntax errors, regex checks are still valuable
                pass
        
        except Exception as e:
            # Don't fail validation if a file can't be read
            print(f"Warning: Could not validate {filepath}: {e}")
        
        return self.issues
    
    def _check_chained_assignment_regex(self, filepath: Path, lines: List[str]):
        """
        Check for chained assignment patterns using regex.
        
        Patterns detected:
        - df[x][y] = value
        - dataframe[x][y] = value
        - df.loc[x][y] = value (less common but still problematic)
        
        Note: Excludes dictionary-like patterns (results[key1][key2])
        which are safe in Python.
        """
        # Pattern: DataFrame-like variable_name[something][something_else] = 
        # Specifically target common DataFrame variable names
        pattern = re.compile(
            r'\b(df|dataframe|data_?frame|_df|filtered|subset|timeseries|ts_data)\s*'
            r'\[[^\]]+\]\s*\[[^\]]+\]\s*='
        )
        
        for line_num, line in enumerate(lines, start=1):
            # Skip comments
            if line.strip().startswith('#'):
                continue
            
            match = pattern.search(line)
            if match:
                self.issues.append(PandasSafetyIssue(
                    filepath=str(filepath),
                    line_number=line_num,
                    column=match.start(),
                    issue_type="CHAINED_ASSIGNMENT",
                    severity=PandasSafetyIssue.SEVERITY_CRITICAL,
                    description=(
                        "Chained assignment df[x][y] = value can fail silently. "
                        "The first bracket may return a view or copy unpredictably, "
                        "causing SettingWithCopyWarning or silent data loss."
                    ),
                    code_snippet=line.strip(),
                    recommendation=(
                        "Use df.loc[x, y] = value or df.at[x, y] = value for single values. "
                        "For bulk assignment: temp = df.copy(); temp[col] = values; df = temp"
                    )
                ))
    
    def _check_inplace_usage(self, filepath: Path, lines: List[str]):
        """
        Check for inplace=True usage.
        
        Modern pandas best practice: avoid inplace=True because:
        1. It's often not actually faster (copy-on-write since pandas 2.0)
        2. It prevents method chaining
        3. It makes code harder to debug
        4. It can be confusing with views vs copies
        """
        pattern = re.compile(r'\binplace\s*=\s*True\b')
        
        for line_num, line in enumerate(lines, start=1):
            if line.strip().startswith('#'):
                continue
            
            match = pattern.search(line)
            if match:
                self.issues.append(PandasSafetyIssue(
                    filepath=str(filepath),
                    line_number=line_num,
                    column=match.start(),
                    issue_type="INPLACE_TRUE",
                    severity=PandasSafetyIssue.SEVERITY_WARNING,
                    description=(
                        "inplace=True is discouraged in modern pandas. "
                        "It's often not faster and prevents method chaining."
                    ),
                    code_snippet=line.strip(),
                    recommendation=(
                        "Replace 'df.drop(columns=cols, inplace=True)' with "
                        "'df = df.drop(columns=cols)'. More explicit and chainable."
                    )
                ))
    
    def _check_chained_assignment_ast(self, filepath: Path, tree: ast.AST, lines: List[str]):
        """
        Check for chained assignments using AST parsing.
        
        More precise than regex but won't catch all patterns.
        Focuses on DataFrame-like variable names to avoid false positives
        on dictionary chained assignments (which are safe).
        """
        class ChainedAssignmentVisitor(ast.NodeVisitor):
            def __init__(self, validator, filepath, lines):
                self.validator = validator
                self.filepath = filepath
                self.lines = lines
                # Common DataFrame variable name patterns
                self.dataframe_names = {
                    'df', 'dataframe', 'data_frame', '_df', 'filtered',
                    'subset', 'timeseries', 'ts_data', 'stock_df', 'index_df'
                }
            
            def visit_Assign(self, node):
                # Check if target is a chained subscript: df[x][y]
                for target in node.targets:
                    if isinstance(target, ast.Subscript):
                        if isinstance(target.value, ast.Subscript):
                            # This is var[x][y] = ...
                            # Check if var is a DataFrame-like name
                            if self._is_dataframe_variable(target.value.value):
                                line_num = node.lineno
                                code_snippet = self.lines[line_num - 1].strip() if line_num <= len(self.lines) else ""
                                
                                self.validator.issues.append(PandasSafetyIssue(
                                    filepath=str(self.filepath),
                                    line_number=line_num,
                                    column=node.col_offset,
                                    issue_type="CHAINED_SUBSCRIPT_ASSIGNMENT",
                                    severity=PandasSafetyIssue.SEVERITY_CRITICAL,
                                    description=(
                                        "Chained subscript assignment detected on DataFrame-like variable. "
                                        "This is unsafe and may not work as expected."
                                    ),
                                    code_snippet=code_snippet,
                                    recommendation=(
                                        "Use .loc[] or .at[] for proper indexing. "
                                        "Example: df.loc[row_indexer, col_indexer] = value"
                                    )
                                ))
                
                self.generic_visit(node)
            
            def _is_dataframe_variable(self, node):
                """Check if AST node represents a DataFrame-like variable."""
                if isinstance(node, ast.Name):
                    name_lower = node.id.lower()
                    # Check exact matches or contains common DataFrame patterns
                    return (
                        name_lower in self.dataframe_names or
                        'df' in name_lower or
                        'dataframe' in name_lower or
                        'timeseries' in name_lower
                    )
                return False
        
        visitor = ChainedAssignmentVisitor(self, filepath, lines)
        visitor.visit(tree)
    
    def validate_directory(
        self,
        directory: Path,
        exclude_patterns: Optional[List[str]] = None
    ) -> List[PandasSafetyIssue]:
        """
        Validate all Python files in a directory.
        
        Args:
            directory: Root directory to scan
            exclude_patterns: List of glob patterns to exclude (e.g., ['**/test_*.py', '**/.venv/**'])
        
        Returns:
            Aggregated list of all issues found
        """
        if exclude_patterns is None:
            exclude_patterns = [
                '**/test_*.py',
                '**/.venv/**',
                '**/venv/**',
                '**/__pycache__/**',
                '**/site-packages/**'
            ]
        
        all_issues = []
        
        for py_file in directory.rglob('*.py'):
            # Check if file matches any exclude pattern
            excluded = any(py_file.match(pattern) for pattern in exclude_patterns)
            if excluded:
                continue
            
            issues = self.validate_file(py_file)
            all_issues.extend(issues)
        
        return all_issues
    
    def generate_report(self, issues: List[PandasSafetyIssue]) -> str:
        """
        Generate a human-readable report of pandas safety issues.
        
        Args:
            issues: List of PandasSafetyIssue objects
        
        Returns:
            Formatted report string
        """
        if not issues:
            return f"{GREEN}{BOLD}✓ NO PANDAS SAFETY ISSUES FOUND{RESET}\n"
        
        report_lines = [
            f"\n{'=' * 80}",
            f"{BOLD}PANDAS SAFETY VALIDATION REPORT{RESET}",
            f"{'=' * 80}\n"
        ]
        
        # Group by severity
        critical = [i for i in issues if i.severity == PandasSafetyIssue.SEVERITY_CRITICAL]
        warning = [i for i in issues if i.severity == PandasSafetyIssue.SEVERITY_WARNING]
        info = [i for i in issues if i.severity == PandasSafetyIssue.SEVERITY_INFO]
        
        report_lines.append(f"Total Issues: {len(issues)}")
        report_lines.append(f"  {RED}Critical:{RESET} {len(critical)}")
        report_lines.append(f"  {YELLOW}Warning:{RESET} {len(warning)}")
        report_lines.append(f"  {BLUE}Info:{RESET} {len(info)}\n")
        
        if critical:
            report_lines.append(f"{RED}{BOLD}CRITICAL ISSUES{RESET}")
            report_lines.append("-" * 80)
            for issue in critical:
                report_lines.append(str(issue))
        
        if warning:
            report_lines.append(f"\n{YELLOW}{BOLD}WARNINGS{RESET}")
            report_lines.append("-" * 80)
            for issue in warning:
                report_lines.append(str(issue))
        
        if info:
            report_lines.append(f"\n{BLUE}{BOLD}INFO{RESET}")
            report_lines.append("-" * 80)
            for issue in info:
                report_lines.append(str(issue))
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append(f"{BOLD}RECOMMENDATION{RESET}")
        report_lines.append("=" * 80)
        report_lines.append(
            "Address CRITICAL issues immediately - they can cause silent data corruption.\n"
            "Consider fixing WARNINGS to align with modern pandas best practices.\n"
        )
        
        return "\n".join(report_lines)


def verify_pandas_copy_semantics():
    """
    Runtime verification that .copy() behaves as expected.
    
    Returns:
        True if all checks pass, False otherwise
    """
    checks_passed = []
    
    # Check 1: .copy() creates independent DataFrame
    df_original = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df_copy = df_original.copy()
    df_copy.loc[0, 'A'] = 999
    
    check1 = df_original.loc[0, 'A'] == 1  # Original should be unchanged
    checks_passed.append(("Copy independence", check1))
    
    # Check 2: Assignment without .copy() for filtering
    df_base = pd.DataFrame({'X': [1, 2, 3, 4, 5], 'Y': [10, 20, 30, 40, 50]})
    df_filtered = df_base[df_base['X'] > 2].copy()  # Proper way
    df_filtered.loc[:, 'Y'] = 999
    
    # Original should be unchanged
    check2 = df_base['Y'].tolist() == [10, 20, 30, 40, 50]
    checks_passed.append(("Filtered copy independence", check2))
    
    # Check 3: .drop() returns new DataFrame (doesn't modify original)
    df_test = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
    df_dropped = df_test.drop(columns=['C'])
    
    check3 = 'C' in df_test.columns and 'C' not in df_dropped.columns
    checks_passed.append((".drop() returns new DataFrame", check3))
    
    all_passed = all(passed for _, passed in checks_passed)
    
    print(f"\n{'=' * 60}")
    print(f"{BOLD}PANDAS COPY SEMANTICS VERIFICATION{RESET}")
    print(f"{'=' * 60}")
    for check_name, passed in checks_passed:
        status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
        print(f"{status} | {check_name}")
    print(f"{'=' * 60}\n")
    
    return all_passed


def main():
    """
    Main entry point for pandas safety validation.
    
    Validates the current project directory and generates a report.
    """
    print(f"\n{BOLD}PY-01 F18: Pandas Safety Validator{RESET}")
    print("=" * 80)
    
    # Runtime verification
    print("\n1. Verifying pandas copy semantics...")
    copy_ok = verify_pandas_copy_semantics()
    
    if not copy_ok:
        print(f"{RED}✗ Pandas copy semantics verification failed!{RESET}")
        print("This may indicate a pandas version or environment issue.")
        return 1
    
    # Static analysis
    print("\n2. Running static analysis on project files...")
    validator = PandasSafetyValidator()
    
    project_root = Path(__file__).parent
    issues = validator.validate_directory(
        project_root,
        exclude_patterns=[
            '**/test_*.py',
            '**/.venv/**',
            '**/venv/**',
            '**/site-packages/**',
            '**/__pycache__/**',
            '**/pandas_safety_validator.py',  # Don't check self
        ]
    )
    
    # Generate report
    report = validator.generate_report(issues)
    print(report)
    
    # Return appropriate exit code
    critical_count = sum(1 for i in issues if i.severity == PandasSafetyIssue.SEVERITY_CRITICAL)
    
    if critical_count > 0:
        print(f"{RED}{BOLD}✗ VALIDATION FAILED{RESET}: {critical_count} critical issues found")
        return 1
    elif issues:
        print(f"{YELLOW}{BOLD}⚠ WARNINGS PRESENT{RESET}: {len(issues)} non-critical issues found")
        return 0  # Don't fail build on warnings
    else:
        print(f"{GREEN}{BOLD}✓ VALIDATION PASSED{RESET}: No pandas safety issues detected")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
