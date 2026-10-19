<!-- This file is part of G05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
See LICENSE.txt in the project root; upstream components retain their applicable licenses. -->

# 将自定义采集数据接入 G0.5 A1Z

本项目的采集程序是一个可选来源。可以使用其他相机、遥操设备与文件格式；进入训练前，必须把每条 episode 映射为相同的 LeRobot v2.1 数据契约。现有 `scripts/convert_hdf5_to_v21.py` 只读取本项目采集的 HDF5 字段；它不会自动识别其他 HDF5、MCAP、CSV 或独立视频。

## 1. 先列出来源字段

每个来源先写一份映射表，记录：原始字段名、单位、采样时间戳来源、左右臂/相机对应关系、缺失值处理方式。至少确认下列目标字段：

| 目标字段 | 单臂两相机 | 双臂三相机 | 单位与形状 |
| --- | --- | --- | --- |
| `observation.state.left_arm` | 必须 | 必须 | float32，6 维，弧度，固定关节顺序 |
| `action.left_arm` | 必须 | 必须 | float32，6 维，执行目标，弧度 |
| `observation.state.left_gripper` / `action.left_gripper` | 必须 | 必须 | float32，1 维，0–1 |
| `observation.state.right_arm` / `action.right_arm` | 无 | 必须 | float32，6 维，弧度 |
| `observation.state.right_gripper` / `action.right_gripper` | 无 | 必须 | float32，1 维，0–1 |
| `observation.images.head_rgb` | 必须 | 必须 | uint8 RGB 图像，高×宽×3 |
| `observation.images.left_wrist_rgb` | 必须 | 必须 | uint8 RGB 图像，高×宽×3 |
| `observation.images.right_wrist_rgb` | 无 | 必须 | uint8 RGB 图像，高×宽×3 |
| task | 必须 | 必须 | 非空任务文本，例如“抓取并放置物体” |

动作必须是对应时刻**实际发送给从臂的目标**。不要把 Leader 原始量、观测状态或提前计算的增量误当动作。关节顺序、正方向和夹爪开合方向必须与部署端一致。若夹爪是 0–100 的开合百分比，可换算为 0–1；角度、脉冲宽度或电机位置等其他单位须先弄清设备定义，不能直接除以 100。

## 2. 对齐时间与 episode

每条 episode 应有开始、结束和任务文本。选择真实采样频率（通常 15 或 30 Hz），以各流共同有效的时间区间建立递增时间轴。每个时刻取已经采集完成的状态、动作和相机帧，记录最大采样年龄和丢帧数；不要使用未来图像、补黑图或把缺失帧伪装成有效数据。状态与动作保持采集时的配对关系。检查抓取接触时图像与动作是否同步，确保同一相机在所有 episode 的图像尺寸一致。

## 3. 写入 LeRobot v2.1

在 [本项目 HDF5 转换指南](hdf5_to_v21.md#1-创建独立转换环境)创建 Python 3.11 转换环境，锁定 `lerobot==0.3.3`；该包写出的数据格式为 v2.1。自定义适配器可复用 `scripts/convert_hdf5_to_v21.py` 中的 `feature_schema`、`writer_class` 和保存/校验逻辑，替换它的 `inspect_episode`、`decode_rgb` 与来源文件枚举。关键写入形式如下（`read_and_align_episodes` 是按来源格式实现的读取器）：

```python
from convert_hdf5_to_v21 import feature_schema, writer_class

Writer = writer_class()
dataset = Writer.create(
    repo_id="local/a1z",
    root=output_stage,             # 新目录，写入时标记 INCOMPLETE
    fps=fps,
    robot_type="a1z",
    features=feature_schema(camera_shapes, parts),
    video_backend="pyav",
)
try:
    for episode in read_and_align_episodes(source):
        for frame in episode.frames:
            # frame 的键、形状与上表一致；图像是 RGB uint8
            dataset.add_frame(frame, task=episode.task)
        dataset.save_episode()
finally:
    dataset.stop_image_writer()
```

`parts` 用转换器里的 `SINGLE_LEFT_PARTS` 或 `PARTS`；`camera_shapes` 填实际图像形状。不要只手工修改 `meta/info.json` 的版本号。先转换少量 episode，再检查全部数据。正式批量转换应沿用参考转换器的做法：输出到新目录、标记 `INCOMPLETE`、保存源文件清单与哈希、校验通过后再发布完成目录。若已有标准 v2.1，只需检查；若只是夹爪 0–100 且其余布局匹配，可按 README 第 4.4 节生成训练副本。

## 4. 进入训练前的验收

在模型环境的仓库根目录设置真实输出目录：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate g05-model
export G05_PYTHON="$CONDA_PREFIX/bin/python"
export A1Z_DATASET=""   # 填转换后的 v2.1 根目录
"$G05_PYTHON" scripts/require_v21.py
"$G05_PYTHON" scripts/check_v21_dataset.py "$A1Z_DATASET" --g05-loader
```

检查器会核对任务索引、帧时间戳、数值范围、归一化统计量、全部视频帧，并抽查 G0.5 的动作窗口。还要人工抽查各相机视频、左右映射、关节曲线和夹爪方向。通过后回到 [README 第 5 节](../README.md#5-训练先试跑再正式运行)，先完成 10 步训练与 checkpoint 保存，再开始正式训练。最终真机执行前仍须按 README 第 8 节检查设备和动作单位。
