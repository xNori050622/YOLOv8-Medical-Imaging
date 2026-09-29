"""数据划分的回归测试：图片必须和它自己的标签落在同一个 split。

用法（两种都行）：
    python tests/test_split_alignment.py     # 零依赖，直接跑
    pytest tests/test_split_alignment.py     # 装了 pytest 也可以

为什么要专门测这个
------------------
原作者用 splitfolders.ratio 划分 segmentation/data_，它会把 images/ 与 labels/
当成两个子目录**分别**洗牌再切分。但两边文件数并不相等（BUSI 的 normal 类只有
图片、没有掩码），于是切完以后图片和标签对不上，例如：

    val/images/malignant (0).png  配上  val/labels/malignant (3).txt

训练时图片配到别人的掩码、评估时预测对不上真值，而且全程不报错。这里用
「13 张图片、10 张有标签」的合成目录把「同病例同 split」钉死，任何 seed 都必须成立。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # 支持 python tests/test_split_alignment.py 直接运行
    sys.path.insert(0, str(PROJECT_ROOT))

from segmentation import masks_to_polygons

# 模仿 BUSI：benign/malignant 有掩码，normal 只有图片
LABELLED_CASES = [
    "benign (0)", "benign (1)", "benign (2)", "benign (3)", "benign (4)", "benign (5)",
    "malignant (0)", "malignant (1)", "malignant (2)", "malignant (3)",
]
UNLABELLED_CASES = ["normal (0)", "normal (1)", "normal (2)"]

SPLITS = ("train", "val", "test")


# ------------------------------------------------------------------ 小工具
def _make_staging(root: Path) -> Path:
    """造出 data_/{images,labels}：13 张图片，其中 10 张各有一个标签文件。"""
    staging = root / "staging"
    (staging / "images").mkdir(parents=True)
    (staging / "labels").mkdir(parents=True)

    for case in LABELLED_CASES:
        (staging / "images" / f"{case}.png").write_bytes(b"fake image")
        (staging / "labels" / f"{case}.txt").write_text(
            "0 0.10 0.10 0.20 0.10 0.20 0.20\n", encoding="utf-8"
        )
    for case in UNLABELLED_CASES:
        (staging / "images" / f"{case}.png").write_bytes(b"fake image")
    return staging


def _stems(directory: Path) -> set:
    return {path.stem for path in directory.glob("*") if path.is_file()}


def _misaligned(output_dir: Path) -> dict:
    """{split: 有标签却没有对应图片的病例名}——非空就说明划分错配了。"""
    report = {}
    for split in SPLITS:
        labels = _stems(output_dir / split / "labels")
        images = _stems(output_dir / split / "images")
        if labels - images:
            report[split] = labels - images
    return report


def _layout(output_dir: Path) -> dict:
    return {split: sorted(_stems(output_dir / split / "images")) for split in SPLITS}


# ============================================================ 核心回归
def test_split_keeps_image_and_label_in_same_split():
    """图片与标签必须落在同一个 split（这正是原实现会错的地方）。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        staging = _make_staging(root)

        stats = masks_to_polygons.split_train_test_val(
            input_dir=staging, output_dir=root / "data", keep_staging=True
        )

        assert _misaligned(root / "data") == {}, "图片和它的标签被分到了不同的 split"
        assert sum(item["images"] for item in stats.values()) == 13, "图片总数不对"
        assert sum(item["labels"] for item in stats.values()) == 10, "标签总数不对"
        for split in SPLITS:
            assert (root / "data" / split / "images").is_dir(), f"{split}/images 不存在"


def test_every_case_is_used_exactly_once():
    """每个病例只能出现在一个 split 里，三个 split 合起来必须是全集。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        staging = _make_staging(root)

        masks_to_polygons.split_train_test_val(
            input_dir=staging, output_dir=root / "data", keep_staging=True
        )

        layout = _layout(root / "data")
        placed = [case for split in SPLITS for case in layout[split]]
        assert sorted(placed) == sorted(LABELLED_CASES + UNLABELLED_CASES)
        assert len(placed) == len(set(placed)), "同一个病例被放进了多个 split"


def test_alignment_holds_for_any_seed():
    """换 seed 只改变哪些病例进哪个 split，不允许改变「同病例同 split」。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        staging = _make_staging(root)

        for seed in (0, 7, 1337, 20260929):
            output = root / f"data_{seed}"
            stats = masks_to_polygons.split_train_test_val(
                input_dir=staging, output_dir=output, keep_staging=True, seed=seed
            )
            assert _misaligned(output) == {}, f"seed={seed} 时划分错配"
            assert sum(item["images"] for item in stats.values()) == 13


def test_split_is_reproducible_with_same_seed():
    """同一个 seed 必须给出完全相同的划分（沿用原实现的 seed=1337 可复现）。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        staging = _make_staging(root)

        first = masks_to_polygons.split_train_test_val(
            input_dir=staging, output_dir=root / "a", keep_staging=True, seed=1337
        )
        second = masks_to_polygons.split_train_test_val(
            input_dir=staging, output_dir=root / "b", keep_staging=True, seed=1337
        )

        assert first == second
        assert _layout(root / "a") == _layout(root / "b")


# ============================================================ 边界与清理
def test_case_without_any_label_is_still_placed():
    """normal 类没有标签文件，图片不能因此被丢掉。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        staging = _make_staging(root)

        masks_to_polygons.split_train_test_val(
            input_dir=staging, output_dir=root / "data", keep_staging=True
        )

        placed = set()
        for split in SPLITS:
            placed |= _stems(root / "data" / split / "images")
        assert set(UNLABELLED_CASES) <= placed, "没有掩码的病例被漏掉了"


def test_split_without_any_label_file_does_not_crash():
    """只有图片、一个标签都没有时也要能划分（labels/ 为空）。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        staging = root / "staging"
        (staging / "images").mkdir(parents=True)
        for case in LABELLED_CASES:
            (staging / "images" / f"{case}.png").write_bytes(b"fake image")

        stats = masks_to_polygons.split_train_test_val(
            input_dir=staging, output_dir=root / "data", keep_staging=True
        )

        assert sum(item["images"] for item in stats.values()) == len(LABELLED_CASES)
        assert sum(item["labels"] for item in stats.values()) == 0


def test_missing_images_directory_raises():
    """没有 images/ 就明确报错，而不是悄悄产出一个空数据集。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        try:
            masks_to_polygons.split_train_test_val(
                input_dir=root / "nothing", output_dir=root / "data"
            )
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("缺少 images/ 时没有报 FileNotFoundError")


def test_staging_directory_is_removed_unless_kept():
    """默认删掉中间目录 data_；keep_staging=True 时保留。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        staging = _make_staging(root)

        masks_to_polygons.split_train_test_val(
            input_dir=staging, output_dir=root / "data", keep_staging=True
        )
        assert staging.is_dir(), "keep_staging=True 不应该删除 data_"

        masks_to_polygons.split_train_test_val(
            input_dir=staging, output_dir=root / "data"
        )
        assert not staging.exists(), "默认应该删除 data_"


def test_alignment_check_is_not_vacuous():
    """确认上面的断言不是空转：故意造出错配目录必须被 _misaligned 抓到。"""
    with tempfile.TemporaryDirectory() as temporary:
        data = Path(temporary) / "data"
        for split in SPLITS:
            (data / split / "images").mkdir(parents=True)
            (data / split / "labels").mkdir(parents=True)

        # 复刻实测到的错配：val 的图片是 malignant (0)，标签却是 malignant (3)
        (data / "val" / "images" / "malignant (0).png").write_bytes(b"x")
        (data / "val" / "labels" / "malignant (3).txt").write_text("", encoding="utf-8")

        assert _misaligned(data) == {"val": {"malignant (3)"}}


# ================================================================ 运行器
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
