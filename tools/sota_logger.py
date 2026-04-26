#!/usr/bin/env python3
"""
SotaLogger: SOTAFollow 项目日志工具
=======================================
记录项目更新日志，支持 commit、PR merge、论文精读报告更新等操作
日志风格：emoji + 开源学术风格
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# 日志轮转配置
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
MAX_ROTATED_FILES = 5


class SotaLogger:
    """SOTAFollow 项目日志记录器"""

    def __init__(self, log_path: Optional[str] = None):
        """
        初始化 Logger

        Args:
            log_path: 日志文件路径，默认为项目根目录的 sotafollow.log
        """
        if log_path is None:
            # 默认使用项目根目录的 sotafollow.log
            self.log_path = Path(__file__).parent.parent / "sotafollow.log"
        else:
            self.log_path = Path(log_path)

        self._ensure_log_directory()

    def _ensure_log_directory(self):
        """确保日志目录存在"""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _check_rotation(self):
        """检查是否需要日志轮转"""
        if not self.log_path.exists():
            return

        file_size = self.log_path.stat().st_size
        if file_size >= MAX_LOG_SIZE:
            self._rotate_logs()

    def _rotate_logs(self):
        """执行日志轮转"""
        # 删除最旧的日志
        oldest = self.log_path.parent / f"{self.log_path.name}.{MAX_ROTATED_FILES}"
        if oldest.exists():
            oldest.unlink()

        # 轮转现有日志
        for i in range(MAX_ROTATED_FILES - 1, 0, -1):
            current = self.log_path.parent / f"{self.log_path.name}.{i}"
            if current.exists():
                next_file = self.log_path.parent / f"{self.log_path.name}.{i + 1}"
                current.rename(next_file)

        # 当前日志变为 .1
        rotated = self.log_path.parent / f"{self.log_path.name}.1"
        if self.log_path.exists():
            self.log_path.rename(rotated)

    def _get_timestamp(self) -> str:
        """获取格式化的时间戳"""
        now = datetime.now()
        return f"📅 {now.strftime('%Y-%m-%d')} ⏰ {now.strftime('%H:%M:%S')}"

    def _get_git_user(self) -> str:
        """获取当前 Git 用户"""
        try:
            result = subprocess.run(
                ["git", "config", "user.name"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else "Unknown"
        except Exception:
            return "Unknown"

    def _write_log(self, entry: str):
        """写入日志条目"""
        self._check_rotation()
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(entry + "\n")

    def log_commit(self, commit_msg: str, files: List[str] = None):
        """
        记录 Git commit

        Args:
            commit_msg: commit 消息
            files: 变更的文件列表
        """
        files = files or []

        # 解析 commit 类型
        emoji = "🌳"
        type_label = "COMMIT"

        if "[VLA]" in commit_msg:
            emoji = "🤖"
            type_label = "VLA"
        elif "[RL]" in commit_msg:
            emoji = "🎯"
            type_label = "RL"
        elif "[WM]" in commit_msg:
            emoji = "🌍"
            type_label = "WM"
        elif "[优化]" in commit_msg:
            emoji = "⚡"
            type_label = "OPTIMIZE"
        elif "[修复]" in commit_msg:
            emoji = "🐛"
            type_label = "FIX"
        elif "[Docs]" in commit_msg or "文档" in commit_msg:
            emoji = "📚"
            type_label = "DOCS"

        timestamp = self._get_timestamp()
        git_user = self._get_git_user()

        entry = f"""
{timestamp} {emoji} [{type_label}]
   📝 {commit_msg}"""

        if files:
            # 分类文件
            vla_files = [f for f in files if "VLA/" in f or "π" in f]
            rl_files = [f for f in files if "RL/" in f]
            wm_files = [f for f in files if "WorldModel/" in f]

            if vla_files:
                entry += f"\n   🤖 VLA: {', '.join(vla_files[:3])}"
            if rl_files:
                entry += f"\n   🎯 RL: {', '.join(rl_files[:3])}"
            if wm_files:
                entry += f"\n   🌍 WM: {', '.join(wm_files[:3])}"
            if len(files) > 3:
                entry += f"\n   📂 及其他 {len(files) - 3} 个文件"

        entry += f"\n   └── 👤 {git_user}"

        self._write_log(entry)

    def log_pr_merge(self, pr_num: int, title: str, summary: str = ""):
        """
        记录 PR merge

        Args:
            pr_num: PR 编号
            title: PR 标题
            summary: PR 内容摘要
        """
        timestamp = self._get_timestamp()
        git_user = self._get_git_user()

        entry = f"""
{timestamp} 🔀 [PR MERGE] #{pr_num}
   📜 {title}"""

        if summary:
            entry += f"\n   💡 {summary}"

        entry += f"\n   👤 Merger: {git_user}"

        self._write_log(entry)

    def log_operation(self, operation: str, detail: str, category: str = "OPERATION"):
        """
        记录通用操作

        Args:
            operation: 操作名称
            detail: 操作详情
            category: 操作类别 (OPERATION, RESEARCH, etc.)
        """
        timestamp = self._get_timestamp()

        # 根据操作类型选择 emoji
        emoji_map = {
            "OPERATION": "⚙️",
            "RESEARCH": "🔬",
            "LEARNING": "📖",
            "ANALYSIS": "📊",
        }
        emoji = emoji_map.get(category, "⚙️")

        entry = f"""
{timestamp} {emoji} [{category}]
   📦 {operation}
   📝 {detail}"""

        self._write_log(entry)

    def log_reading_report(self, report_name: str, action: str, paper_info: str = ""):
        """
        记录精读报告更新

        Args:
            report_name: 报告名称 (如 "VISTA_精读报告.md")
            action: 操作类型 ("created" | "updated" | "reviewed" | "optimized")
            paper_info: 论文信息 (如 "arXiv 2602.10983")
        """
        timestamp = self._get_timestamp()

        # 操作类型映射
        action_map = {
            "created": ("✨", "CREATED"),
            "updated": ("📝", "UPDATED"),
            "reviewed": ("✅", "REVIEWED"),
            "optimized": ("⚡", "OPTIMIZED"),
        }
        emoji, label = action_map.get(action, ("📝", "UPDATED"))

        entry = f"""
{timestamp} {emoji} [READING REPORT] {label}
   📄 {report_name}"""

        if paper_info:
            entry += f"\n   🏛️ {paper_info}"

        entry += f"\n   └── 🤖 Auto-logged by SotaLogger"

        self._write_log(entry)

    def log_git_push(self, branch: str = "main", status: str = "success"):
        """
        记录 Git push 操作

        Args:
            branch: 推送的分支
            status: 推送状态 ("success" | "failed")
        """
        timestamp = self._get_timestamp()
        emoji = "✅" if status == "success" else "❌"
        status_label = "PUSH SUCCESS" if status == "success" else "PUSH FAILED"

        entry = f"""
{timestamp} {emoji} [{status_label}]
   🌿 Branch: {branch}
   └── 🤖 Auto-logged by SotaLogger"""

        self._write_log(entry)


def get_logger() -> SotaLogger:
    """获取全局 Logger 实例"""
    return SotaLogger()


# 命令行接口
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""
╔══════════════════════════════════════════════════════════════╗
║                    SotaLogger 使用指南                        ║
╠══════════════════════════════════════════════════════════════╣
║  用法: python3 sota_logger.py <command> [args]               ║
╠══════════════════════════════════════════════════════════════╣
║  命令:                                                        ║
║    commit <message>       记录 commit                         ║
║    pr <pr_num> <title>    记录 PR merge                       ║
║    operation <op> <detail> 记录通用操作                        ║
║    report <name> <action> 记录精读报告更新                    ║
║    push <branch> [status] 记录 git push                       ║
║    view [lines]           查看最近日志 (默认 20 行)             ║
╠══════════════════════════════════════════════════════════════╣
║  示例:                                                        ║
║    python3 sota_logger.py commit "更新 VISTA 精读报告"         ║
║    python3 sota_logger.py report "VISTA_精读报告.md" created   ║
║    python3 sota_logger.py operation "新增论文" "VISTA arXiv"   ║
║    python3 sota_logger.py view 50                             ║
╚══════════════════════════════════════════════════════════════╝
        """)
        sys.exit(0)

    logger = SotaLogger()
    command = sys.argv[1]

    if command == "commit" and len(sys.argv) >= 3:
        logger.log_commit(sys.argv[2])
        print("✅ Commit logged successfully!")

    elif command == "pr" and len(sys.argv) >= 4:
        pr_num = int(sys.argv[2])
        title = sys.argv[3]
        summary = sys.argv[4] if len(sys.argv) > 4 else ""
        logger.log_pr_merge(pr_num, title, summary)
        print(f"✅ PR #{pr_num} logged successfully!")

    elif command == "operation" and len(sys.argv) >= 4:
        operation = sys.argv[2]
        detail = sys.argv[3]
        logger.log_operation(operation, detail)
        print("✅ Operation logged successfully!")

    elif command == "report" and len(sys.argv) >= 4:
        name = sys.argv[2]
        action = sys.argv[3]
        info = sys.argv[4] if len(sys.argv) > 4 else ""
        logger.log_reading_report(name, action, info)
        print(f"✅ Report {action} logged successfully!")

    elif command == "push" and len(sys.argv) >= 3:
        branch = sys.argv[2]
        status = sys.argv[3] if len(sys.argv) > 3 else "success"
        logger.log_git_push(branch, status)
        print(f"✅ Push to {branch} logged!")

    elif command == "view":
        lines = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        if logger.log_path.exists():
            with open(logger.log_path, "r", encoding="utf-8") as f:
                content = f.readlines()
                print("".join(content[-lines:]))
        else:
            print("📭 日志文件尚未创建")

    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)
