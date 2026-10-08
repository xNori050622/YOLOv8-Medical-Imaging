"""training.resolve_data 的回归测试：显式传 --data 时绝不静默回退。

用法（两种都行）：
    python tests/test_training.py     # 零依赖，直接跑
    pytest tests/test_training.py     # 装了 pytest 也可以

为什么要专门测这个
-----------------
resolve_data() 在「传了 --data 但那个文件不存在」时，原先是无条件下一个分支：
    yaml_path = Path(data or profile["data"])
    if not yaml_path.is_file():
        yaml_path = dataset.build_segment_data_yaml()   # 丢掉传进来的路径
而 build_segment_data_yaml() 的默认目标正是 segmentation/data.yaml，且 overwrite=True。
于是命令行拼错一个字母，结果就是「拿未增强的数据训练，实验记录上却写着增强」——
不报错、不提示，整张消融表都是错的，而且它会顺手把默认 data.yaml 重写一遍。

这类静默错配比崩溃危险得多，所以这里把三条路径都钉住：
  1. 显式路径不存在 -> 必须抛 FileNotFoundError；
  2. 显式路径存在   -> 必须原样返回该路径；
  3. 不传 --data    -> 必须仍然回退到 config 里的默认位置（原有行为不能破）。
所有用例都不需要 ultralytics，也不碰真实数据集。
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # 支持 python tests/test_training.py 直接运行
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
import dataset  # noqa: E402
import training  # noqa: E402


def _quiet(function, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = function(*args, **kwargs)
    return result, buffer.getvalue()


def _build_segment_root(root: Path) -> Path:
    """造一个最小可用的分割数据集目录树（只要有 train/images 就够生成 data.yaml）。"""
    for split in ("train", "val", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        (root / split / "images" / f"{split}_0.png").write_bytes(b"fake")
        (root / split / "labels" / f"{split}_0.txt").write_text("", encoding="utf-8")
    return root


def test_segment_missing_explicit_yaml_raises():
    """--data 指向不存在的 data.yaml：必须报错，而不是悄悄用默认数据集。"""
    with tempfile.TemporaryDirectory() as temporary:
        missing = Path(temporary) / "nope" / "data.yaml"
        assert not missing.exists()

        try:
            training.resolve_data("segment", str(missing))
        except FileNotFoundError as error:
            assert str(missing) in str(error), "报错信息里要带上出错的路径"
            assert "make_augmented_dataset" in str(error), "要告诉用户怎么生成增强数据集"
        else:
            raise AssertionError(
                "显式路径不存在却回退了 —— 这正是会静默污染消融实验的那个 bug"
            )


def test_segment_failed_resolve_does_not_touch_default_yaml():
    """回退分支不该被触发，所以默认 data.yaml 必须一个字节都没变。

    与上一条用例同样的毛病：原先靠 `if not default_yaml.is_file(): return`
    在没有数据集的机器上早退，于是 CI 上这条用例打印 PASS 却一句断言都没跑
    （README 承诺的「无需数据集」因此名不副实）。这里自造一个默认 data.yaml，
    填成与本函数无关的哨兵内容，再断言它字节级未变。
    """
    with tempfile.TemporaryDirectory() as temporary:
        sentinel = Path(temporary) / "data.yaml"
        sentinel_bytes = b"path: sentinel\nnc: 3\n"
        sentinel.write_bytes(sentinel_bytes)
        # 诱饵：默认数据目录指向一棵真实存在的最小数据树。万一有人把
        #「显式 --data 不存在就静默回退」改回来，回退会真的去重写上面这份
        # 哨兵文件，于是本用例在任何机器（包括 CI）都会红。
        decoy_root = _build_segment_root(Path(temporary) / "decoy_data")

        original_split_dir = config.SEGMENT_SPLIT_DIR
        original_data_yaml = config.SEGMENT_DATA_YAML
        original_profile_data = training.TASK_PROFILES["segment"]["data"]
        config.SEGMENT_SPLIT_DIR = decoy_root
        config.SEGMENT_DATA_YAML = sentinel
        training.TASK_PROFILES["segment"]["data"] = sentinel
        try:
            missing = Path(temporary) / "nope" / "data.yaml"
            try:
                training.resolve_data("segment", str(missing))
            except FileNotFoundError:
                pass
            else:
                raise AssertionError(f"{missing} 不存在，却没有报错")
        finally:
            config.SEGMENT_SPLIT_DIR = original_split_dir
            config.SEGMENT_DATA_YAML = original_data_yaml
            training.TASK_PROFILES["segment"]["data"] = original_profile_data

        assert sentinel.read_bytes() == sentinel_bytes, "默认 data.yaml 被改写了"


def test_segment_explicit_yaml_is_used():
    """显式路径存在时必须原样返回它，不能偷偷换成默认数据集。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = _build_segment_root(Path(temporary) / "data_aug")
        yaml_path = dataset.build_segment_data_yaml(
            root=root, yaml_path=root / "data.yaml", overwrite=True)

        resolved = training.resolve_data("segment", str(yaml_path))
        assert resolved == str(yaml_path)
        assert resolved != str(config.SEGMENT_DATA_YAML)

        payload = yaml.safe_load(Path(resolved).read_text(encoding="utf-8"))
        assert Path(payload["path"]).resolve() == root.resolve()


def test_segment_without_data_falls_back_to_default():
    """不传 --data 时仍走 config 的默认位置（原有行为不能被这次修复改坏）。

    这条用例原先直接调 resolve_data("segment", None)，于是隐含依赖「本机已经
    跑过 --prepare」：默认分支会去 config.SEGMENT_SPLIT_DIR 生成 data.yaml，
    而 config.SEGMENT_SPLIT_DIR 只在真实数据就位时存在。在干净克隆 / CI 上
    它必然抛 FileNotFoundError，还会顺手改写仓库里的 segmentation/data.yaml ——
    一个测试套件不该有这两种行为，所以这里把「默认位置」临时指到 tempfile
    造的假数据集上：分支被完整走到，结果与机器状态无关，也不再碰仓库文件。
    """
    with tempfile.TemporaryDirectory() as temporary:
        root = _build_segment_root(Path(temporary) / "data")
        default_yaml = Path(temporary) / "data.yaml"
        assert not default_yaml.exists()

        original_split_dir = config.SEGMENT_SPLIT_DIR
        original_data_yaml = config.SEGMENT_DATA_YAML
        original_profile_data = training.TASK_PROFILES["segment"]["data"]
        config.SEGMENT_SPLIT_DIR = root
        config.SEGMENT_DATA_YAML = default_yaml
        training.TASK_PROFILES["segment"]["data"] = default_yaml
        try:
            assert training.resolve_data("segment", None) == str(config.SEGMENT_DATA_YAML)
            assert training.resolve_data("segment", "") == str(config.SEGMENT_DATA_YAML)
        finally:
            config.SEGMENT_SPLIT_DIR = original_split_dir
            config.SEGMENT_DATA_YAML = original_data_yaml
            training.TASK_PROFILES["segment"]["data"] = original_profile_data

        # 回退分支确实把 data.yaml 生成在了默认位置，并指向默认的数据目录
        payload = yaml.safe_load(default_yaml.read_text(encoding="utf-8"))
        assert Path(payload["path"]) == root
        assert payload["train"] == "train/images"


def test_detect_missing_explicit_yaml_raises():
    """检测任务同理由：显式路径不存在时不能去猜。"""
    with tempfile.TemporaryDirectory() as temporary:
        missing = Path(temporary) / "nope" / "data.yaml"
        try:
            training.resolve_data("detect", str(missing))
        except FileNotFoundError:
            pass
        else:
            raise AssertionError(f"{missing} 不存在，却返回了一个别的路径")


def test_classify_missing_dir_raises():
    """分类任务传的目录不存在时必须报错（这条行为本来就是对的，钉住别退化）。"""
    with tempfile.TemporaryDirectory() as temporary:
        missing = Path(temporary) / "nope"
        try:
            training.resolve_data("classify", str(missing))
        except FileNotFoundError as error:
            assert str(missing) in str(error)
        else:
            raise AssertionError(f"{missing} 不存在，却没有报错")


def test_unknown_task_is_rejected_before_loading_torch():
    """未知任务必须在导入 ultralytics 之前就被拒 —— 否则报错会变成 ModuleNotFoundError。"""
    try:
        training.train_task("segmentt")
    except ValueError as error:
        assert "segmentt" in str(error)
    else:
        raise AssertionError("未知任务没有被拒绝")


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
