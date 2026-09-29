"""make_classify_val.py 的回归测试。

用法（两种都行）：
    python tests/test_make_classify_val.py     # 零依赖，直接跑
    pytest tests/test_make_classify_val.py     # 装了 pytest 也可以

为什么要专门测这个
------------------
这个脚本会**真的移动**用户数据集里的文件，所以它必须满足几条硬性质：

  1. 移动而不是复制：train+val 的张数必须等于原 train 的张数，一张都不能丢；
  2. train 与 val 严格不相交：否则 val 上的精度是灌水的；
  3. test/ 一根手指都不能碰：它是对外报告用的唯一纯净集合；
  4. 可复现：同样的 seed 必须切出同一批文件，否则报告里的数字无法重现；
  5. 可撤销：--undo 之后目录必须和切分之前一模一样。

所有用例都跑在 tempfile 造的假数据集上，**不会碰到真实的
classification/Covid19-dataset**。
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # 支持 python tests/test_make_classify_val.py 直接运行
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import make_classify_val as splitter

# 假数据集的类别与样本数。故意让每类张数不同，这样"按类别分别算比例"才测得出来
CLASS_SIZES = {"Covid": 10, "Normal": 10, "Viral Pneumonia": 5}
TEST_SIZES = {"Covid": 3, "Normal": 2, "Viral Pneumonia": 2}
RATIO = 0.2
SEED = 0


# ------------------------------------------------------------------ 小工具
def _build_dataset(root: Path) -> Path:
    """造出 <root>/train/<类别>/*.png 与 <root>/test/<类别>/*.png。"""
    for split, sizes in (("train", CLASS_SIZES), ("test", TEST_SIZES)):
        for name, count in sizes.items():
            directory = root / split / name
            directory.mkdir(parents=True)
            stem = name.lower().replace(" ", "_")
            for index in range(count):
                (directory / f"{stem}_{index}.png").write_bytes(b"fake image")
    # 顺便放一个假的 ultralytics 缓存，验证切完会被清掉
    (root / "train.cache").write_text("stale", encoding="utf-8")
    return root


def _names(root: Path, split: str, name: str) -> set:
    directory = root / split / name
    return {path.name for path in directory.glob("*") if path.is_file()}


def _quiet(function, *args, **kwargs):
    """跑的时候把脚本的打印吞掉，顺便把输出返回给需要断言提示语的用例。"""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = function(*args, **kwargs)
    return code, buffer.getvalue()


# ========================================================== 切分的基本性质
def test_split_is_stratified_per_class():
    """每类单独按比例取：10 张取 2 张，5 张取 1 张。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = _build_dataset(Path(temporary))
        code, _ = _quiet(splitter.split_train, root, RATIO, SEED, False, False)
        assert code == 0
        assert len(_names(root, "val", "Covid")) == 2
        assert len(_names(root, "val", "Normal")) == 2
        assert len(_names(root, "val", "Viral Pneumonia")) == 1


def test_train_and_val_are_disjoint_and_lossless():
    """移动而非复制：张数守恒，且 train 与 val 没有交集。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = _build_dataset(Path(temporary))
        before = {name: len(_names(root, "train", name)) for name in CLASS_SIZES}
        _quiet(splitter.split_train, root, RATIO, SEED, False, False)
        for name in CLASS_SIZES:
            train, val = _names(root, "train", name), _names(root, "val", name)
            assert not (train & val), f"{name} 的 train 与 val 出现同一张图"
            assert len(train) + len(val) == before[name], f"{name} 张数不守恒"


def test_test_split_is_untouched():
    """test/ 是对外报告用的纯净集合，切 val 时不能动它。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = _build_dataset(Path(temporary))
        before = {name: _names(root, "test", name) for name in TEST_SIZES}
        _quiet(splitter.split_train, root, RATIO, SEED, False, False)
        for name in TEST_SIZES:
            assert _names(root, "test", name) == before[name]


# ========================================================= 可复现 / 可撤销
def test_split_is_deterministic():
    """同样的 seed 必须切出同一批文件，否则报告里的数字无法复现。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = _build_dataset(Path(temporary))
        _quiet(splitter.split_train, root, RATIO, SEED, False, False)
        first = {name: _names(root, "val", name) for name in CLASS_SIZES}
        _quiet(splitter.undo, root)
        _quiet(splitter.split_train, root, RATIO, SEED, False, False)
        second = {name: _names(root, "val", name) for name in CLASS_SIZES}
        assert first == second, f"两次切分结果不同：{first} != {second}"


def test_different_seed_gives_different_split():
    """反向确认上面的用例不是空转：换 seed 应该真换一批文件。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = _build_dataset(Path(temporary))
        _quiet(splitter.split_train, root, RATIO, SEED, False, False)
        first = {name: _names(root, "val", name) for name in CLASS_SIZES}
        _quiet(splitter.undo, root)
        _quiet(splitter.split_train, root, RATIO, SEED + 1, False, False)
        second = {name: _names(root, "val", name) for name in CLASS_SIZES}
        assert first != second


def test_undo_restores_the_original_tree():
    """--undo 之后必须和切分前一模一样，val/ 与清单都要消失。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = _build_dataset(Path(temporary))
        before = {name: _names(root, "train", name) for name in CLASS_SIZES}
        _quiet(splitter.split_train, root, RATIO, SEED, False, False)
        code, _ = _quiet(splitter.undo, root)
        assert code == 0
        for name in CLASS_SIZES:
            assert _names(root, "train", name) == before[name]
        assert not (root / "val").exists(), "val/ 没有被清理干净"
        assert not (root / splitter.MANIFEST_NAME).exists()


def test_manifest_records_seed_ratio_and_pairs():
    """清单是 --undo 的唯一依据，格式不能变。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = _build_dataset(Path(temporary))
        _quiet(splitter.split_train, root, RATIO, SEED, False, False)
        manifest = json.loads((root / splitter.MANIFEST_NAME).read_text(encoding="utf-8"))
        assert manifest["seed"] == SEED
        assert manifest["ratio"] == RATIO
        assert set(manifest["moved"]) == set(CLASS_SIZES)
        pairs = manifest["moved"]["Covid"]
        assert len(pairs) == 2
        for old_path, new_path in pairs:
            assert old_path.startswith("train/") and new_path.startswith("val/")


# ============================================================= 防御性行为
def test_second_split_is_refused():
    """val/ 里已经有文件时必须拒绝，并提示怎么撤销，而不是再切一次。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = _build_dataset(Path(temporary))
        _quiet(splitter.split_train, root, RATIO, SEED, False, False)
        snapshot = {name: _names(root, "val", name) for name in CLASS_SIZES}
        code, text = _quiet(splitter.split_train, root, RATIO, SEED, False, False)
        assert code == 1
        assert "--undo" in text, "拒绝的时候要告诉用户怎么撤销"
        assert {name: _names(root, "val", name) for name in CLASS_SIZES} == snapshot


def test_dry_run_does_not_touch_files():
    with tempfile.TemporaryDirectory() as temporary:
        root = _build_dataset(Path(temporary))
        before = {name: _names(root, "train", name) for name in CLASS_SIZES}
        code, text = _quiet(splitter.split_train, root, RATIO, SEED, False, True)
        assert code == 0
        assert "dry-run" in text
        for name in CLASS_SIZES:
            assert _names(root, "train", name) == before[name]
        assert not (root / "val").exists()
        assert not (root / splitter.MANIFEST_NAME).exists()


def test_stale_cache_is_removed():
    """图片列表变了还留着 train.cache，ultralytics 会按旧列表训练。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = _build_dataset(Path(temporary))
        assert (root / "train.cache").is_file()
        _quiet(splitter.split_train, root, RATIO, SEED, False, False)
        assert not (root / "train.cache").exists(), "过期缓存没清掉"


def test_tiny_class_keeps_at_least_one_image_in_train():
    """样本极少的类别不能把 train 抽空：1 张的整类跳过，2 张的只抽走 1 张。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for count, name in ((1, "One"), (2, "Two")):
            directory = root / "train" / name
            directory.mkdir(parents=True)
            for index in range(count):
                (directory / f"{index}.png").write_bytes(b"fake image")
        _quiet(splitter.split_train, root, 0.2, SEED, False, False)
        assert len(_names(root, "train", "One")) == 1
        assert not (root / "val" / "One").exists()
        assert len(_names(root, "train", "Two")) == 1
        assert len(_names(root, "val", "Two")) == 1


def test_ratio_out_of_range_is_rejected():
    with tempfile.TemporaryDirectory() as temporary:
        root = _build_dataset(Path(temporary))
        code, text = _quiet(splitter.main, ["--data", str(root), "--ratio", "0"])
        assert code == 1
        assert "ratio" in text


def test_cli_end_to_end():
    """走一遍 main()，确认参数解析与调用链是通的。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = _build_dataset(Path(temporary))
        code, text = _quiet(splitter.main,
                            ["--data", str(root), "--ratio", "0.2", "--seed", "0"])
        assert code == 0
        assert "互不相交" in text
        assert (root / "val" / "Covid").is_dir()
        assert (root / splitter.MANIFEST_NAME).is_file()


# ================================================================== 运行器
def main() -> int:
    """收集本模块所有 test_* 函数并执行，返回退出码（0 = 全部通过）。"""
    tests = [(name, function) for name, function in sorted(globals().items())
             if name.startswith("test_") and callable(function)]

    failures = 0
    for name, function in tests:
        try:
            function()
        except Exception as error:  # 测试失败要报出原因而不是中断整轮
            failures += 1
            print(f"FAIL  {name}: {type(error).__name__}: {error}")
        else:
            print(f"PASS  {name}")

    print("-" * 68)
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
