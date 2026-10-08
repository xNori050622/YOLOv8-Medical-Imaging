"""检查 README 里那些「手写数字」是否仍然与仓库真实状态一致。

为什么需要
----------
README 的「What this fork adds」写着「新增 27 个文件 / 修改 7 个 / +7721 −328 行」，
「Regression tests」的表格写着每个测试文件有多少条用例、合计 104 条；workflow 里
又抄了一份依赖版本号。这些全是手写的，改一行代码就会失真，而失真时**没有任何东西
会报错**：读者只能自己去 `git diff` 数一遍。

这个脚本把它们变成可执行断言，在 CI 里跑：

  1. README 的自述统计  == `git diff <base> --shortstat -- . ':(exclude)runs'`；
  2. README 用例表的每个数字 == 对应文件里 `def test_` 的真实条数，且与正文的合计一致；
  3. workflow 里 `pip install` 的版本 == requirements.txt 里的 pin（防止两份漂移）。

只依赖标准库 + git，不需要 torch、不需要数据集。

基线为什么是一串 SHA 而不是 `origin/master`
-------------------------------------------
README 那句话讲的是「相对**上游**加了什么」，而 `origin/master` 到底指哪边取决于
你从哪克隆：上游的仓库里 `origin` 是上游，fork 的仓库里 `origin` 是 fork（CI 用
`actions/checkout` 检出的正是 fork，此时 `origin/master` == HEAD，diff 是空的）。
所以这里把基线钉成**上游 master 的 tip**——fork 正是从这个提交长出来的，它在
fork 的历史里一定是个祖先提交，任何完整克隆都能解析，也不必联网。上游之后若又
推了新提交，本仓库的统计口径也不受影响：重新 rebase 时再换这串 SHA 即可。

注意：第 1 项按 **git 已跟踪的文件** 计算（`git diff <base>` 的语义），所以你新增了
文件但还没 `git add` 时，本地可能先 PASS、提交后 CI 才 FAIL —— 这时按脚本给出的
那行数字改 README 即可。

用法::

    python tools/check_repo_consistency.py              # 本地：没有 origin/master 就跳过统计检查
    python tools/check_repo_consistency.py --strict     # CI：origin/master 必须存在，否则算失败
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
README = PROJECT_ROOT / "README.md"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "tests.yml"

# 基线 = 上游 sevdaimany/YOLOv8-Medical-Imaging 的 master tip（2023-09-07 "Add all files"）。
# fork 从它长出来（本仓库领先 21 个提交），所以它在 fork 的历史里是祖先提交，
# 完整克隆（CI 里 fetch-depth: 0）一定能解析；不依赖 remote 叫什么名字，也不联网。
UPSTREAM_TIP = "5d13edfdc0b08db1cc4f8b017308b240b1424b69"
DEFAULT_BASE = UPSTREAM_TIP
EXCLUDE_RUNS = ":(exclude)runs"

# "adds 28 files and modifies 7 (35 files, +8004 / −333 lines)"
# 注意 README 里的减号是 U+2212（数学减号），不是 ASCII 连字符，两种都收。
# 单词之间用 \s+ 而不是空格：markdown 里这句话可能被换行折成两行，不该因此判定「找不到」。
STAT_RE = re.compile(
    r"adds\s+(\d+)\s+files\s+and\s+modifies\s+(\d+)"
    r"\s+\((\d+)\s+files,\s+\+(\d+)\s+/\s+[\u2212-](\d+)\s+lines\)"
)
# "| `tests/test_metrics.py` | 29 | ... |"
# 必须带 re.MULTILINE：这张表在 README 中间，^ 要能匹配每一行的行首。
TABLE_ROW_RE = re.compile(r"^\|\s*`(tests/[^`]+\.py)`\s*\|\s*(\d+)\s*\|", re.MULTILINE)
# "`tests/` holds 104 plain-`assert` checks"
TOTAL_RE = re.compile(r"holds (\d+) plain-`assert` checks")
PIN_RE = re.compile(r"\b([A-Za-z0-9_.\-]+)==([0-9][^\s\"']*)")


def _git(*args: str) -> str | None:
    """跑一条只读 git 命令；失败返回 None。"""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def _base_available(base: str) -> bool:
    return _git("rev-parse", "--verify", "--quiet", base) is not None


def _is_ancestor_of_head(base: str) -> bool:
    """基线必须是 HEAD 的祖先，否则「相对上游加了什么」这句话就无从谈起。"""
    return _git("merge-base", "--is-ancestor", base, "HEAD") is not None


def _describe_base(base: str) -> str:
    """把基线描述成 `5d13edf 2023-09-07 Add all files`，让 CI 日志里看得见对不对。"""
    line = _git("log", "-1", "--format=%h %cs %s", base)
    return line.strip() if line else base


def _diff_counts(base: str) -> dict | None:
    """返回 {added, modified, files, insertions, deletions}（已排除 runs/）。"""
    shortstat = _git("diff", base, "--shortstat", "--", ".", EXCLUDE_RUNS)
    if shortstat is None:
        return None
    numbers = re.search(
        r"(\d+) files? changed"
        r"(?:, (\d+) insertions?\(\+\))?"
        r"(?:, (\d+) deletions?\(-\))?",
        shortstat,
    )
    if not numbers:
        return None
    names = {
        flag: _git("diff", base, "--diff-filter=" + flag, "--name-only", "--", ".", EXCLUDE_RUNS)
        for flag in ("A", "M")
    }
    if any(value is None for value in names.values()):
        return None
    return {
        "files": int(numbers.group(1)),
        "insertions": int(numbers.group(2) or 0),
        "deletions": int(numbers.group(3) or 0),
        "added": len([line for line in names["A"].splitlines() if line.strip()]),
        "modified": len([line for line in names["M"].splitlines() if line.strip()]),
    }


def check_readme_stats(base: str, strict: bool) -> tuple[list[str], str]:
    """README 的自述统计 vs git 真实差量。返回 (问题列表, 情况说明)。"""
    text = README.read_text(encoding="utf-8")
    match = STAT_RE.search(text)
    if not match:
        return ["README 里找不到「adds N files and modifies M (F files, +X / −Y lines)」那句话"], ""
    claimed = dict(
        zip(
            ("added", "modified", "files", "insertions", "deletions"),
            (int(value) for value in match.groups()),
        )
    )

    if not _base_available(base):
        note = f"本地找不到 {base}，跳过统计比对"
        if strict:
            return [
                f"{base} 不存在：需要完整克隆（CI 里由 actions/checkout 的 fetch-depth: 0 提供）"
            ], note
        return [], note

    label = _describe_base(base)
    if not _is_ancestor_of_head(base):
        return [
            f"基线 {label} 不是 HEAD 的祖先：README 的统计只对「fork 分叉自上游的那个提交」"
            "有意义；如果刚 rebase 到更新的上游，请把脚本里的 UPSTREAM_TIP 换成新 tip"
        ], f"对比基线 {label}"

    actual = _diff_counts(base)
    if actual is None:
        return [f"读不到 git diff {base} 的结果"], f"对比基线 {label}"

    problems = []
    for key, label in (
        ("added", "新增文件数"),
        ("modified", "修改文件数"),
        ("files", "文件总数"),
        ("insertions", "新增行数"),
        ("deletions", "删除行数"),
    ):
        if claimed[key] != actual[key]:
            problems.append(f"{label}：README 写 {claimed[key]}，实际 {actual[key]}")
    if problems:
        problems.append(
            "请把 README「What this fork adds」里的数字改成："
            f"adds {actual['added']} files and modifies {actual['modified']} "
            f"({actual['files']} files, +{actual['insertions']} / \u2212{actual['deletions']} lines)."
        )
    return problems, f"对比基线 {label}"


def check_test_table() -> tuple[list[str], str]:
    """README 用例表的数字 vs 文件里真实的 def test_ 条数。"""
    text = README.read_text(encoding="utf-8")
    rows = TABLE_ROW_RE.findall(text)
    if not rows:
        return ["README 的 Regression tests 表格没解析出任何用例行"], ""

    problems = []
    total_actual = 0
    for relative, claimed_text in rows:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            problems.append(f"表格里列了 {relative}，但文件不存在")
            continue
        actual = len(re.findall(r"(?m)^def test_", path.read_text(encoding="utf-8")))
        total_actual += actual
        if actual != int(claimed_text):
            problems.append(f"{relative}：表格写 {claimed_text} 条，实际 {actual} 条")

    total_match = TOTAL_RE.search(text)
    if not total_match:
        problems.append("README 正文里找不到「holds N plain-`assert` checks」那句合计")
    elif int(total_match.group(1)) != total_actual:
        problems.append(f"正文合计写 {total_match.group(1)} 条，表格实际相加是 {total_actual} 条")

    return problems, f"{len(rows)} 个测试文件，实际合计 {total_actual} 条"


def check_dependency_pins() -> tuple[list[str], str]:
    """workflow 里 pip install 的版本 vs requirements.txt 的 pin。"""
    if not WORKFLOW.is_file():
        return [f"找不到 {WORKFLOW.relative_to(PROJECT_ROOT)}"], ""

    pinned = {}
    for line in REQUIREMENTS.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        for name, version in re.findall(r"\b([A-Za-z0-9_.\-]+)==([^\s;]+)", line):
            pinned[name.lower()] = version

    installed = {}
    for line in WORKFLOW.read_text(encoding="utf-8", errors="replace").splitlines():
        if "pip install" not in line or line.strip().startswith("#"):
            continue
        for name, version in PIN_RE.findall(line):
            if name.lower() == "pip":
                continue
            installed[name.lower()] = version

    if not installed:
        return ["workflow 里没解析出任何 pip install 的版本 pin"], ""

    problems = []
    for name, version in sorted(installed.items()):
        if name not in pinned:
            problems.append(f"{name} 被 workflow 装了，但 requirements.txt 里没有它")
        elif pinned[name] != version:
            problems.append(
                f"{name}：workflow 固定 {version}，requirements.txt 固定 {pinned[name]}"
            )
    return problems, f"{len(installed)} 个包在两边一致"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查 README / workflow 里的手写数字是否失真")
    parser.add_argument("--base", default=DEFAULT_BASE, help=f"对比基线（默认 {DEFAULT_BASE}）")
    parser.add_argument("--strict", action="store_true", help="基线不存在时算失败（CI 用）")
    args = parser.parse_args(argv)

    checks = (
        ("README 自述统计", lambda: check_readme_stats(args.base, args.strict)),
        ("README 用例表", check_test_table),
        ("依赖版本 pin", check_dependency_pins),
    )

    failed = 0
    for title, run in checks:
        problems, note = run()
        if problems:
            failed += 1
            print(f"FAIL  {title}" + (f"（{note}）" if note else ""))
            for problem in problems:
                print(f"        · {problem}")
        else:
            print(f"PASS  {title}" + (f"（{note}）" if note else ""))

    print("-" * 68)
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
