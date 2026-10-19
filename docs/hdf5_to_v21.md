<!-- This file is part of G05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
See LICENSE.txt in the project root; upstream components retain their applicable licenses. -->

# A1Z HDF5 → LeRobot v2.1

适用于本仓库固定版本的 `a1z-data-collect`：双臂 14 维、三路 JPEG 相机、`arm/state`、`arm/action` 和各路单调时间戳。在 Linux PC 离线运行，不连接机械臂，不需要 GPU，也不改写原始 HDF5。

单臂 7 维、头部＋腕部两路相机数据可在转换器的检查和转换命令中追加 `--single`；训练时按 LoRA／全量选择 `scripts/train_single_lora.sh`／`scripts/train_single_full.sh`。15 Hz 相机以历史帧对齐时，采样年龄可能超过双臂默认的 50 ms；先查看实际时间戳，再按实测上限设置 `--max-age`，不要仅为通过检查盲目放宽。

## 1. 创建独立转换环境

在已完成 bootstrap 的仓库根目录打开终端。转换使用 Python 3.11，模型训练仍使用原来的 Python 3.10 环境：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n g05-convert python=3.11 pip -y
conda activate g05-convert
# Linux PC 使用 CPU 版 PyTorch，转换不需要 CUDA
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-convert.txt
python -c 'from importlib.metadata import version; import av, cv2, h5py; print("LeRobot:", version("lerobot")); print("H.264:", av.codec.Codec("h264", "w").name)'
```

必须显示 LeRobot `0.3.3`。这个包版本写入的数据格式是 **v2.1**，两者不是同一个版本号。写入使用[固定版本官方 API](https://github.com/huggingface/lerobot/blob/v0.3.3/src/lerobot/datasets/lerobot_dataset.py)，视频使用 H.264；不要升级转换环境里的 LeRobot。

若已有同名环境且版本不符，先改名保留，再重新创建。不要把这些依赖安装到 `g05-model`。

## 2. 填写路径并检查 HDF5

先在转换终端运行 `conda activate g05-convert`，再停止当前 episode 的录制，等待 HDF5 保存完成。输入目录递归查找 `episode_N.hdf5`，其他名字的临时文件不参与转换。建议先把少量已确认有效的完整 episode 放入一个试转换目录；不同任务可以混合，每个文件使用自己的 `task` 属性。

```bash
export HDF5_INPUT=""    # 填入原始 episode_N.hdf5 所在目录
export V21_OUTPUT=""    # 填入 data/ 下一个尚不存在的新数据集目录
export DATA_FPS=30      # 可选 15 或 30，按部署计划选择
python scripts/convert_hdf5_to_v21.py \
  --input "$HDF5_INPUT" --fps "$DATA_FPS" --check-only
# 单臂时在上述命令末尾追加 --single
```

检查通过会显示各 episode 的输出帧数、各路最大时间偏差。默认至少保留 32 帧；双臂模式不接受单臂输入，单臂模式需追加 `--single`；两种模式均不接受缺相机、坏图像、空任务、非单调时间戳、非有限数值或超出 `[0,1]` 的 SDK 夹爪值。修改真实数据前先核对来源，不能裁剪数值或补黑图绕过检查。

默认相机映射是 `cam_high` → `head_rgb`、`cam_left_wrist` → `left_wrist_rgb`、`cam_right_wrist` → `right_wrist_rgb`。源文件相机名不同时，在检查和转换命令中都追加 `--head 实际头部名称 --left-wrist 实际左腕名称 --right-wrist 实际右腕名称`。图像保留原始分辨率，同一路相机各 episode 的分辨率须一致。

任务默认来自 HDF5 的 `task` 属性。只有确认整个输入目录属于同一任务时，才使用 `--task '真实任务描述'` 覆盖所有文件。

## 3. 转换并完整检查

沿用同一个终端里的路径和 FPS：

```bash
python scripts/convert_hdf5_to_v21.py \
  --input "$HDF5_INPUT" --output "$V21_OUTPUT" --fps "$DATA_FPS"
python scripts/check_v21_dataset.py "$V21_OUTPUT"
# 单臂时仅转换命令追加 --single；检查器按数据元数据识别布局
```

转换器会先检查全部输入，再写入 `<V21_OUTPUT>.incomplete`。完成 Parquet、统计信息和全部视频帧检查后，才将目录移为 `V21_OUTPUT` 并移除 `INCOMPLETE` 标记。输出或临时目录已存在时直接报错，不覆盖、不删除、不自动续传。失败后保留现场，检查报错原因，重试时使用新的输出目录。

转换会临时把一个 episode 的图像解码为 PNG，再编码为视频；磁盘需容纳原始 HDF5、正式输出和该 episode 的临时图像。完成编码后自动清理这些 PNG。转换输出建议放在 Git 已忽略的 `data/` 下；真实数据、视频、日志和测试环境不提交到仓库。

正式输出结构：

```text
YOUR_V21_DATASET/
├── meta/
│   ├── info.json                  # codebase_version=v2.1
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   └── tasks.jsonl
├── data/chunk-000/episode_000000.parquet
├── videos/chunk-000/observation.images.head_rgb/episode_000000.mp4
├── videos/chunk-000/observation.images.left_wrist_rgb/episode_000000.mp4
├── videos/chunk-000/observation.images.right_wrist_rgb/episode_000000.mp4   # 仅双臂
├── conversion_report.json
└── validation_report.json
```

`conversion_report.json` 记录源文件相对路径、SHA-256、FPS、相机映射、时间偏差、重复使用帧数和裁剪时长。`validation_report.json` 记录转换完成时的检查结果；数据改动后应重新运行检查命令。

## 4. 验证 G0.5 读取并训练

在安装好 G0.5 模型环境的机器上，先 `conda activate g05-model`，再执行；如果转换在另一台机器完成，先传输整个输出目录。在训练机器填写实际数据路径：

```bash
export G05_PYTHON="$CONDA_PREFIX/bin/python"
export A1Z_DATASET=""   # 填入转换后的真实数据集根目录，下面应直接有 meta/
"$G05_PYTHON" scripts/require_v21.py
"$G05_PYTHON" scripts/check_v21_dataset.py "$A1Z_DATASET" --g05-loader
```

第二条检查全部数据和视频，并调用当前固定版本的 G0.5 v2.1 reader，从每个 episode 的开头、中间、结尾抽取 32 步动作窗口，检查任务索引和边界补齐。它不加载模型、不启动训练。

人工查看单臂两路／双臂三路输出视频，核对左右臂、相机位置、任务文本、夹爪方向和关节曲线，尤其是抓取接触阶段的同步。通过后按 [README 第 5 节](../README.md) 进行真实数据的 10 步训练，再按第 7 节重新加载 checkpoint。仅转换成功不能证明模型训练或真机任务已经通过。

## 时间与动作约定

- 状态读取 `arm/state`，动作读取控制器输出 `arm/action`；不使用 `arm/action_raw`，不提前计算关节增量。
- 双臂每条 14 维向量固定为左臂 6 关节、左夹爪、右臂 6 关节、右夹爪；单臂为 6 关节＋夹爪共 7 维。关节用弧度，夹爪保持 SDK `[0,1]` 定义，方向须与部署一致。
- 以所有流的共同时间范围生成固定频率时间轴。每个时刻取已完成采样的最近历史帧，不取未来帧、不做插值。状态与动作始终使用同一采集行，以该行 `follower_timestamps` 为参考，保持记录时的配对关系。
- `leader_timestamps` 是主臂读取完成时间，并非精确命令下发时间；相机时间戳也是读取完成时间，并非硬件曝光同步。默认要求主从读取间隔不超过 20 ms、每路采样年龄不超过 50 ms，超出就报错。时间戳检查通过仍需人工核对物理同步。
- `--max-arm-skew`、`--max-age` 的单位为秒。若报错，先检查丢帧、设备延迟和录制完整性；只有分析报告后才调整阈值。输出时钟从 0 开始、严格按 FPS 递增，边缘未覆盖片段被裁掉，裁剪与重复采样均写入报告。
- 30 Hz 转 15 Hz 会减少采样点并改变动作窗口覆盖时长。PC 配置里的 `control_frequency` 要与最终训练数据/模型的频率匹配，不能只修改 `info.json` 的 FPS 标签。

## 转换后复核

转换器在落盘前会生成 `validation_report.json`，但正式训练前仍需在训练机器重新运行：

```bash
"$G05_PYTHON" scripts/check_v21_dataset.py "$A1Z_DATASET" --g05-loader
```

同时人工抽查各相机视频、左右臂字段、夹爪方向、任务文本与动作同步；单臂仅检查头部和腕部两路图像。不要用训练 loss 代替原始数据检查。
