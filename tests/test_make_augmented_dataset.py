"""make_augmented_dataset.py 的回归测试。

用法（两种都行）：
    python tests/test_make_augmented_dataset.py     # 零依赖，直接跑
    pytest tests/test_make_augmented_dataset.py     # 装了 pytest 也可以

为什么要专门测这个
-----------------
这个脚本产出的数据集是**整张消融表的输入**。它一旦出错，错误不会当场暴露，
而是以「增强臂」的名义写进论文表格。所以几条硬性质必须钉死：

  1. val/test 逐字节不变：否则基线与增强臂的数字不可比，实验失去意义；
  2. 原图必须保留：增强是增补而非替换，否则没有「有/无增强」之外的对照；
  3. CLAHE 是光度变换、弹性形变是几何变换：前者标签必须原样，后者标签必须
     拓扑不变（点数不变、坐标仍在 [0,1]）；
  4. 同样的 seed 必须产出同样的字节，否则实验无法复现；
  5. 已存在的输出目录必须拒绝写入，除非 --force（残留的 _aug*.png 会静默
     把数据集规模改掉）；
  6. 比对逻辑本身不能是恒真的 —— 故意篡改 val/ 之后必须报错。

所有用例都跑在 tempfile 造的假数据集上，**不会碰到真实的 segmentation/data**。
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # 支持 python tests/test_make_augmented_dataset.py
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from tools import make_augmented_dataset as maker  # noqa: E402

# 假数据集的规模。train 比 val/test 多，且 train 里混一张背景图（空标签）
COUNTS = {"train": 4, "val": 2, "test": 2}
BACKGROUND_INDEX = 3          # train_3 是背景图
POLYGON = "0 0.25 0.25 0.75 0.25 0.75 0.75"   # 一个三角形，3 个顶点
HEIGHT, WIDTH = 96, 128


def _fake_image(seed: int) -> np.ndarray:
    """造一张「暗背景 + 亮斑」的图：CLAHE 一定会改变它（纯梯度图可能不变）。"""
    image = np.full((HEIGHT, WIDTH), 30, dtype=np.uint8)
    center = (WIDTH // 2 + seed % 7, HEIGHT // 2)
    cv2.circle(image, center, 18, 200, -1)
    cv2.circle(image, center, 9, 120, -1)
    return image


def _build_dataset(root: Path) -> Path:
    """造出 <root>/{train,val,test}/{images,labels}，并塞一个假的 ultralytics cache。"""
    for split, count in COUNTS.items():
        images = root / split / "images"
        labels = root / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for index in range(count):
            stem = f"{split}_{index}"
            cv2.imwrite(str(images / f"{stem}.png"), _fake_image(index))
            text = "" if (split == "train" and index == BACKGROUND_INDEX) else POLYGON + "\n"
            (labels / f"{stem}.txt").write_text(text, encoding="utf-8")
    # ultralytics 会在 split 目录下留 labels.cache（内容是绝对路径）：
    # 它不是数据，比对时必须忽略，否则每次都会误报「不一致」
    (root / "val" / "labels.cache").write_text("D:\\stale\\paths", encoding="utf-8")
    return root


def _quiet(function, *args, **kwargs):
    """跑的时候把打印吞掉，顺便把输出返回给需要断言提示语的用例。"""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = function(*args, **kwargs)
    return code, buffer.getvalue()


def _run(source: Path, output: Path, *extra):
    """等价于命令行跑一次，返回 (退出码, 输出文本)。"""
    return _quiet(maker.main, ["--source", str(source), "--out", str(output), *extra])


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree(directory: Path) -> dict:
    """{相对路径: sha256}，只列文件。"""
    return {
        item.relative_to(directory).as_posix(): _digest(item)
        for item in sorted(directory.rglob("*"))
        if item.is_file()
    }


def _images(directory: Path) -> set:
    return {item.name for item in directory.glob("*") if item.is_file()}


def _parse_label(path: Path) -> list:
    """把 YOLO 分割标签解析成 [(class_id, [(x, y), ...]), ...]。"""
    instances = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        tokens = line.split()
        coords = [float(value) for value in tokens[1:]]
        instances.append((int(tokens[0]), list(zip(coords[0::2], coords[1::2]))))
    return instances


# ==================================================== 增强的基本性质
def test_train_is_augmented_and_originals_kept():
    """train 变 (1+copies) 倍，且每张原图都还在、逐字节未变。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = _build_dataset(root / "src")
        output = root / "out"
        code, _ = _run(source, output, "--clahe")
        assert code == 0

        expected = _images(source / "train" / "images") | {
            f"train_{index}_aug1.png" for index in range(COUNTS["train"])
        }
        assert _images(output / "train" / "images") == expected
        for name in _images(source / "train" / "images"):
            assert _digest(output / "train" / "images" / name) == \
                _digest(source / "train" / "images" / name), f"{name} 原图被改动了"


def test_val_and_test_are_byte_identical():
    """val/test 必须逐字节复制 —— 这是基线与增强臂可比的前提。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = _build_dataset(root / "src")
        output = root / "out"
        code, text = _run(source, output, "--clahe")
        assert code == 0

        for split in ("val", "test"):
            # 源目录里的 *.cache 是绝对路径派生物，本来就不该被复制
            before = {
                name: value for name, value in _tree(source / split).items()
                if not name.endswith(".cache")
            }
            assert _tree(output / split) == before, f"{split} 与源目录不一致"
            assert not any("_aug" in name for name in _images(output / split / "images"))
        assert "逐字节一致" in text


def test_no_cache_files_leak_into_output():
    """带绝对路径的过期 cache 绝不能带进产出目录，否则 ultralytics 会去读源目录。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = _build_dataset(root / "src")
        output = root / "out"
        assert (source / "val" / "labels.cache").is_file()

        code, _ = _run(source, output, "--clahe")
        assert code == 0
        assert list(output.rglob("*.cache")) == []


def test_clahe_keeps_labels_untouched():
    """CLAHE 是光度变换：标签必须原样，背景图的空标签也保持为空。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = _build_dataset(root / "src")
        output = root / "out"
        code, _ = _run(source, output, "--clahe")
        assert code == 0

        for index in range(COUNTS["train"]):
            original = _parse_label(source / "train" / "labels" / f"train_{index}.txt")
            augmented = _parse_label(output / "train" / "labels" / f"train_{index}_aug1.txt")
            assert len(augmented) == len(original)
            for (class_a, points_a), (class_b, points_b) in zip(original, augmented):
                assert class_a == class_b
                assert np.allclose(points_a, points_b, atol=1e-6)

        background = output / "train" / "labels" / f"train_{BACKGROUND_INDEX}_aug1.txt"
        assert background.read_text(encoding="utf-8") == ""


def test_elastic_keeps_label_topology_and_moves_points():
    """弹性形变是几何变换：顶点数/类别不变、坐标仍在 [0,1]，而且必须真的动了。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = _build_dataset(root / "src")
        output = root / "out"
        code, _ = _run(source, output, "--elastic", "--alpha", "3", "--sigma", "6")
        assert code == 0

        for index in range(COUNTS["train"]):
            if index == BACKGROUND_INDEX:
                continue
            original = _parse_label(source / "train" / "labels" / f"train_{index}.txt")
            augmented = _parse_label(output / "train" / "labels" / f"train_{index}_aug1.txt")
            assert len(augmented) == len(original)
            for (class_a, points_a), (class_b, points_b) in zip(original, augmented):
                assert class_a == class_b
                assert len(points_a) == len(points_b)
                assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in points_b)
            assert any(
                not np.allclose(points_a, points_b, atol=1e-9)
                for (_, points_a), (_, points_b) in zip(original, augmented)
            ), f"train_{index} 的顶点没有跟着形变走"


def test_data_yaml_points_at_output():
    """data.yaml 必须指向产出目录本身，类别顺序与 config 一致。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = _build_dataset(root / "src")
        output = root / "out"
        code, _ = _run(source, output, "--clahe")
        assert code == 0

        payload = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
        assert Path(payload["path"]).resolve() == output.resolve()
        assert payload["train"] == "train/images"
        assert payload["val"] == "val/images"
        assert payload["test"] == "test/images"
        assert payload["nc"] == len(config.SEGMENT_CLASS_NAMES)
        assert payload["names"] == list(config.SEGMENT_CLASS_NAMES)


def test_manifest_records_hyperparameters():
    """manifest 是消融表里那一行的出处，超参必须原样记下来。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = _build_dataset(root / "src")
        output = root / "out"
        code, _ = _run(source, output, "--clahe",
                       "--clip-limit", "3.5", "--tile-grid", "4", "--seed", "7")
        assert code == 0

        manifest = json.loads((output / maker.MANIFEST_NAME).read_text(encoding="utf-8"))
        assert manifest["copies"] == 1
        assert manifest["seed"] == 7
        assert manifest["clahe"] == {"clip_limit": 3.5, "tile_grid_size": 4}
        assert manifest["elastic"] is None
        assert manifest["splits_augmented"] == ["train"]
        assert manifest["splits_untouched"] == ["test", "val"]
        assert Path(manifest["source"]).resolve() == source.resolve()
        assert manifest["stats"]["train"]["augmented"] == COUNTS["train"]
        assert manifest["stats"]["val"]["augmented"] == 0


# ==================================================== 拒绝不安全的行为
def test_refuses_without_any_augmentation():
    """既不开 CLAHE 也不开 elastic，等于白复制一份，应当直接拒绝。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = _build_dataset(root / "src")
        output = root / "out"
        code, text = _run(source, output)
        assert code == 1
        assert "--clahe" in text
        assert not output.exists()


def test_refuses_non_empty_output_without_force():
    """已存在的输出目录必须拒绝写入，否则会静默混入上一次的文件。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = _build_dataset(root / "src")
        output = root / "out"
        (output / "train" / "images").mkdir(parents=True)
        ghost = output / "train" / "images" / "train_0_aug1.png"
        ghost.write_bytes(b"stale")

        code, text = _run(source, output, "--clahe")
        assert code == 1
        assert "--force" in text
        assert ghost.read_bytes() == b"stale", "被拒绝时不能动已有文件"
        assert not (output / "data.yaml").exists()


def test_force_removes_stale_files():
    """--force 必须先把目录整个删掉，否则残留的 _aug*.png 会把数据集规模改掉。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = _build_dataset(root / "src")
        output = root / "out"
        (output / "train" / "images").mkdir(parents=True)
        (output / "train" / "images" / "ghost_aug1.png").write_bytes(b"stale")
        (output / "data.yaml").write_text("path: nope", encoding="utf-8")

        code, _ = _run(source, output, "--clahe", "--force")
        assert code == 0
        assert not (output / "train" / "images" / "ghost_aug1.png").exists()
        assert _images(output / "train" / "images") == (
            _images(source / "train" / "images")
            | {f"train_{index}_aug1.png" for index in range(COUNTS["train"])}
        )


def test_rejects_output_inside_source():
    """输出目录放在源目录内部会被当成一个 split 卷进去，必须拒绝。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = _build_dataset(root / "src")
        code, text = _run(source, source / "nested", "--clahe")
        assert code == 1
        assert "源目录内部" in text


def test_rejects_unknown_split():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = _build_dataset(root / "src")
        code, text = _run(source, root / "out", "--clahe", "--splits", "train,valid")
        assert code == 1
        assert "valid" in text


def test_dry_run_writes_nothing():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = _build_dataset(root / "src")
        output = root / "out"
        code, text = _run(source, output, "--clahe", "--dry-run")
        assert code == 0
        assert "dry-run" in text
        assert not output.exists()


# ==================================================== 可复现性 & 校验有效性
def test_same_seed_reproduces_same_bytes():
    """同样的 seed 必须产出同样的增强图，否则实验结果无法复现。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = _build_dataset(root / "src")
        first, second = root / "one", root / "two"
        for output in (first, second):
            code, _ = _run(source, output, "--elastic", "--alpha", "3", "--seed", "5")
            assert code == 0
        assert _tree(first / "train" / "images") == _tree(second / "train" / "images")


def test_different_seed_changes_augmentation():
    """换 seed 必须换出不同的增强图，否则 --seed 是个摆设。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = _build_dataset(root / "src")
        first, second = root / "one", root / "two"
        for output, seed in ((first, "1"), (second, "2")):
            code, _ = _run(source, output, "--elastic", "--alpha", "3", "--seed", seed)
            assert code == 0
        assert _tree(first / "train" / "images") != _tree(second / "train" / "images")


def test_verification_detects_tampered_val():
    """把 val 改一个字节，校验必须报错 —— 证明这段校验不是恒真的。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = _build_dataset(root / "src")
        output = root / "out"
        code, _ = _run(source, output, "--clahe")
        assert code == 0

        checked, text = _quiet(maker.verify_untouched, source, output, ("train",))
        assert checked == ["test", "val"]      # train 被增强，不参与比对
        assert "逐字节一致" in text

        (output / "val" / "labels" / "val_0.txt").write_text(
            "0 0.1 0.1 0.2 0.1 0.2 0.2\n", encoding="utf-8")
        try:
            _quiet(maker.verify_untouched, source, output, ("train",))
        except RuntimeError as error:
            assert "内容不同" in str(error)
        else:
            raise AssertionError("val 被改了却没有报错，校验形同虚设")


def test_verification_detects_missing_data_file():
    """少一个标签文件也必须报错，不能只看内容。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = _build_dataset(root / "src")
        output = root / "out"
        code, _ = _run(source, output, "--clahe")
        assert code == 0

        (output / "test" / "labels" / "test_0.txt").unlink()
        try:
            _quiet(maker.verify_untouched, source, output, ("train",))
        except RuntimeError as error:
            assert "缺失 1 个" in str(error)
        else:
            raise AssertionError("少了一个标签文件却没有报错")


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
