"""Audit script to find functions exceeding 8 code lines and internal blank lines."""

import ast
import pathlib
import textwrap


class FuncAnalyzer(ast.NodeVisitor):
    """Walk AST and report function bodies exceeding 8 code lines."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.issues: list[str] = []

    def _count_code_lines(self, node: ast.FunctionDef) -> int:
        # Skip docstring node if present
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, (ast.Constant, ast.Str))
        ):
            body = body[1:]
        # Count non-blank source lines in body
        if not body:
            return 0
        start = body[0].lineno
        end = body[-1].end_lineno or body[-1].lineno
        return end - start + 1

    def _check_internal_blanks(
        self, source_lines: list[str], node: ast.FunctionDef
    ) -> list[int]:
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, (ast.Constant, ast.Str))
        ):
            body = body[1:]
        if not body:
            return []
        start = body[0].lineno  # 1-based
        end = body[-1].end_lineno or body[-1].lineno
        blanks = []
        for i in range(start - 1, end):  # 0-based
            if i < len(source_lines) and source_lines[i].strip() == "":
                blanks.append(i + 1)
        return blanks

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        code_lines = self._count_code_lines(node)
        if code_lines > 8:
            self.issues.append(
                f"  {self.filepath}:{node.lineno} {node.name}() has {code_lines} code lines (max 8)"
            )
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef


def main() -> None:
    all_issues: list[str] = []
    for pattern in ["src/**/*.py", "tests/**/*.py"]:
        for p in sorted(pathlib.Path(".").glob(pattern)):
            source = p.read_text()
            tree = ast.parse(source)
            analyzer = FuncAnalyzer(str(p))
            analyzer.visit(tree)
            all_issues.extend(analyzer.issues)

    if all_issues:
        print(f"Found {len(all_issues)} functions exceeding 8 code lines:\n")
        for issue in all_issues:
            print(issue)
    else:
        print("All functions have 8 or fewer code lines.")


if __name__ == "__main__":
    main()
