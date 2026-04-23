#!/usr/bin/env python3
"""
LaTeX Formula Checker for Markdown files.
检查 Markdown 文件中的 LaTeX 公式语法正确性。

用法:
    python3 check_latex_formula.py <file.md> [file2.md ...]
    python3 check_latex_formula.py --all  # 检查当前目录及子目录所有 .md 文件
"""

import re
import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional

class LaTeXChecker:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.errors = []

    def check_file(self, filepath: str) -> List[Tuple[int, str]]:
        """检查单个 Markdown 文件中的 LaTeX 公式"""
        errors = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            return [(0, f"无法读取文件: {e}")]

        # 提取所有 LaTeX 公式（行内和行间）
        formulas = []  # (line_num, formula, is_block)

        for i, line in enumerate(lines, 1):
            # 跳过代码块
            if '```' in line:
                continue

            # 匹配行内公式 $...$
            inline_pattern = r'\$([^\$]+)\$'
            for match in re.finditer(inline_pattern, line):
                formula = match.group(1)
                # 排除转义的 $ (通常不成对)
                if formula.count('$') == 0:
                    formulas.append((i, formula, False))

            # 匹配行间公式 $$...$$
            block_pattern = r'\$\$([^\$]+)\$\$'
            for match in re.finditer(block_pattern, line):
                formula = match.group(1)
                formulas.append((i, formula, True))

        # 检查每个公式
        for line_num, formula, is_block in formulas:
            # 检查 LaTeX 语法
            formula_errors = self._check_formula(formula, line_num, is_block)
            errors.extend(formula_errors)

        return errors

    def _check_formula(self, formula: str, line_num: int, is_block: bool) -> List[Tuple[int, str]]:
        """检查单个公式的语法"""
        errors = []
        prefix = f"行 {line_num}"

        # 1. 检查括号匹配
        errors.extend(self._check_brackets(formula, line_num, is_block))

        # 2. 检查大括号匹配
        errors.extend(self._check_braces(formula, line_num, is_block))

        # 3. 检查环境命令配对
        errors.extend(self._check_environments(formula, line_num, is_block))

        # 4. 检查特殊符号
        errors.extend(self._check_special_symbols(formula, line_num, is_block))

        return errors

    def _check_brackets(self, formula: str, line_num: int, is_block: bool) -> List[Tuple[int, str]]:
        """检查方括号 [] 和圆括号 () 匹配"""
        errors = []
        prefix = f"行 {line_num}"

        # 检查方括号
        open_count = formula.count('[')
        close_count = formula.count(']')
        if open_count != close_count:
            errors.append((line_num, f"{prefix}: 方括号不匹配 [{open_count} vs {close_count}]: {formula[:50]}..."))

        # 检查圆括号
        open_paren = formula.count('(')
        close_paren = formula.count(')')
        if open_paren != close_paren:
            errors.append((line_num, f"{prefix}: 圆括号不匹配 ({open_paren} vs {close_paren}): {formula[:50]}..."))

        # 检查 \left[ 但没有 \right] 或不匹配
        left_right_pattern = r'\\left\['
        right_left_pattern = r'\\right\]'
        left_count = len(re.findall(left_right_pattern, formula))
        right_count = len(re.findall(right_left_pattern, formula))
        if left_count != right_count:
            errors.append((line_num, f"{prefix}: \\left[ 和 \\right] 不匹配 [{left_count} vs {right_count}]"))

        return errors

    def _check_braces(self, formula: str, line_num: int, is_block: bool) -> List[Tuple[int, str]]:
        """检查大括号 {} 匹配"""
        errors = []

        # 简单计数法（排除转义的 \{ 和 \}）
        # 移除转义的括号
        clean = re.sub(r'\\\{|\\\}', '', formula)

        open_count = clean.count('{')
        close_count = clean.count('}')
        if open_count != close_count:
            errors.append((line_num, f"行 {line_num}: 大括号不匹配 {{{open_count} vs {close_count}}}: {formula[:50]}..."))

        return errors

    def _check_environments(self, formula: str, line_num: int, is_block: bool) -> List[Tuple[int, str]]:
        """检查环境命令配对（cases, aligned 等）"""
        errors = []

        # 检查 \begin 和 \end 配对
        begins = re.findall(r'\\begin\{(\w+)\}', formula)
        ends = re.findall(r'\\end\{(\w+)\}', formula)

        for env in begins:
            if ends.count(env) != begins.count(env):
                errors.append((line_num, f"行 {line_num}: 环境 \\begin{{{env}}} 和 \\end{{{env}}} 不匹配"))
                break

        return errors

    def _check_special_symbols(self, formula: str, line_num: int, is_block: bool) -> List[Tuple[int, str]]:
        """检查特殊符号"""
        errors = []

        # 检查常见的 cos sin 后面跟 [] (应该用 parentheses)
        # 匹配 \cos[ 或 \sin[ 等
        trig_with_bracket = re.findall(r'\\(?:cos|sin|tan|log|exp)\[', formula)
        if trig_with_bracket:
            errors.append((line_num, f"行 {line_num}: 发现 \\{trig_with_bracket[0][1:]}[ ，三角函数后应用 () 而非 []: {formula[:50]}..."))

        # 检查下标语法 _ 或 ^ 后紧跟多字符需要大括号
        # 例如 _12 应该检查是否应为 _{12}
        # 这个检查可能产生误报，所以默认关闭
        if self.verbose:
            # 检查下标后紧跟字母数字的情况
            pattern = r'[_\^]([a-zA-Z]{2,})'
            matches = re.findall(pattern, formula)
            for match in matches:
                errors.append((line_num, f"行 {line_num}: 下标 后跟多字符 '{match}' ，考虑使用 _{{{{match}}}} 或 ^{{{{match}}}}"))

        return errors


def main():
    import argparse
    parser = argparse.ArgumentParser(description='检查 Markdown 文件中的 LaTeX 公式')
    parser.add_argument('files', nargs='*', help='要检查的 .md 文件')
    parser.add_argument('--all', '-a', action='store_true', help='检查当前目录及子目录所有 .md 文件')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细信息')
    args = parser.parse_args()

    checker = LaTeXChecker(verbose=args.verbose)
    files_to_check = []

    if args.all:
        # 递归查找所有 .md 文件
        for md_file in Path('.').rglob('*.md'):
            # 排除 node_modules 和其他常见忽略目录
            if 'node_modules' not in str(md_file) and '.git' not in str(md_file):
                files_to_check.append(str(md_file))
    elif args.files:
        files_to_check = args.files
    else:
        print("用法: python3 check_latex_formula.py <file.md> [file2.md ...]")
        print("      python3 check_latex_formula.py --all")
        sys.exit(1)

    total_errors = 0

    for filepath in files_to_check:
        errors = checker.check_file(filepath)
        if errors:
            print(f"\n{'='*60}")
            print(f"文件: {filepath}")
            print('='*60)
            for line_num, error_msg in errors:
                print(f"  ❌ {error_msg}")
            total_errors += len(errors)
        else:
            if args.verbose:
                print(f"✓ {filepath} - 无问题")

    print(f"\n{'='*60}")
    print(f"检查完成: {len(files_to_check)} 个文件, {total_errors} 个错误")
    print('='*60)

    return 1 if total_errors > 0 else 0


if __name__ == '__main__':
    sys.exit(main())