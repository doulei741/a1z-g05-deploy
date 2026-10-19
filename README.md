<!-- This file is part of G05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
See LICENSE.txt in the project root; upstream components retain their applicable licenses. -->

# G05_A1Z

**中文** | [English](README.en.md)

**本仓库自有的 A1Z 集成内容采用 [Galaxea Dynamics A1Z LICENSE AGREEMENT](LICENSE.txt) 许可。G0.5 及第三方组件保留各自许可。**

Built with Galaxea Dynamics A1Z

A1Z 的采集 → LeRobot v2.1 转换 → LoRA / 全量微调 → PC 直连推理。

模式统一为 `single`（1 臂、2 相机）和 `dual`（2 臂、3 相机）。以下命令使用 **Linux Bash 终端**，在仓库根目录执行；新终端先进入仓库、激活对应环境。首次安装和下载需联网。设备、数据路径和任务描述由使用者填写。

## 1. 系统与克隆

推荐基准：**Ubuntu 22.04.5 LTS、Linux 6.8 HWE、x86_64**，设备使用 SocketCAN 和 USB/V4L2。6.8 是 Ubuntu 22.04.5 的官方 HWE 内核，见 [Ubuntu 内核说明](https://ubuntu.com/kernel/lifecycle)；其他系统/内核需自行验证，不能只凭版本号判断设备兼容性。

| 用途 | 准备条件 |
| --- | --- |
| 推理 / LoRA | NVIDIA GPU，显存建议至少 24 GB；LoRA 主机内存建议 32 GB |
| 全量微调 | 显存建议至少 80 GB，主机内存建议 64 GB |
| 模型环境 | Python ≥3.10.16、<3.11；固定依赖为 PyTorch 2.7.1 / cu128，驱动及 CUDA 编译工具链需匹配 |
| 采集 / 控制 / 转换 | Python 3.11，使用独立环境 |

内存是准备建议，先用小步训练、保存和重载确认峰值；磁盘应同时容纳原始数据、转换结果、基模和多个 checkpoint。同一卡上先退出训练再推理。

```bash
sudo apt-get update
sudo apt-get install -y git curl python3-venv build-essential ffmpeg tmux rsync can-utils iproute2 nano
uname -r
nvidia-smi
nvcc --version   # 模型依赖含 CUDA 扩展；需先准备匹配的 CUDA 编译工具链
df -h .
git clone --branch main https://github.com/userguide-galaxea/a1z-g05-deploy.git G05_A1Z
cd G05_A1Z
python3 scripts/bootstrap.py
python3 scripts/verify_workspace.py
```

bootstrap 会按 `versions.json` 初始化固定版本并应用补丁；Leader 和 SDK 统一使用 data-collect 内嵌引用：`a1z-data-collect/a1z-t/GALAXEA-A1Z`；版本由该清单统一固定。此时提示权重未下载是正常的；不要单独更新子模块。

## 2. 安装环境

没有 Conda 时安装 Miniforge；已有 Conda 则跳过安装块：

```bash
curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh" -o /tmp/Miniforge3.sh
bash /tmp/Miniforge3.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda init bash
```

**设备环境**：采集、预览和真机客户端共用。

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n g05-robot python=3.11 pip -y
conda activate g05-robot
export A1Z_PYTHON="$CONDA_PREFIX/bin/python"
bash scripts/setup_robot.sh
```

**模型环境**：训练与模型服务共用。

```bash
conda create -n g05-model python=3.10 pip -y
conda activate g05-model
python -m pip install uv==0.12.5
export G05_PYTHON="$CONDA_PREFIX/bin/python"
REQ_FILE="$(mktemp)"
(cd GalaxeaVLA && uv export --frozen --python "$G05_PYTHON" --index-strategy unsafe-best-match --format requirements.txt --no-hashes --no-dev --output-file "$REQ_FILE" && uv pip install --python "$G05_PYTHON" --index-strategy unsafe-best-match -r "$REQ_FILE")
rm -f "$REQ_FILE"
python --version
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

若缺少 `nvcc`，先安装与固定 cu128 依赖匹配的 CUDA Toolkit 再安装模型依赖。安装出现 CUDA 扩展编译错误时，先检查 `nvcc --version` 与驱动/工具链，参照 [模型环境说明](https://github.com/OpenGalaxea/GalaxeaVLA/blob/89f2322b4ad016e192437adc1a2c253b05bab246/README_zh.md) 修复后继续。

**转换环境**：固定 LeRobot 0.3.3，其输出格式为 v2.1；不要与模型环境混装。

```bash
conda create -n g05-convert python=3.11 pip -y
conda activate g05-convert
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-convert.txt
python -c 'from importlib.metadata import version; import av,cv2,h5py; assert version("lerobot")=="0.3.3"; print(av.codec.Codec("h264","w").name)'
```

## 3. 下载模型资源

先用自己的 Hugging Face 账号在 [G0.5 模型页面](https://huggingface.co/OpenGalaxea/G05) 申请并取得访问权限，完成页面要求的模型条款确认。再登录同一账号下载：

```bash
conda activate g05-model
export G05_PYTHON="$CONDA_PREFIX/bin/python"
hf auth login
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="OpenGalaxea/G05", local_dir="checkpoints",
    allow_patterns=["g05-base/**", "action_tokenizer.pt", "qwen3_5_2b_base_processor/**"])
PY
python3 scripts/verify_workspace.py --require-model-assets
```

应有 `checkpoints/g05-base/.hydra/config.yaml`、基模 checkpoint、`dataset_stats.json`、`action_tokenizer.pt` 和 processor 目录。检查通过再继续。

## 4. 采集、预览与转换

本项目提供的 **data-collect / a1z-t 是一种可选采集方法**，也可以使用自己的采集程序和已有数据。无论来源和原始文件格式，**训练输入统一为 LeRobot v2.1**。

| 数据来源 | 操作 |
| --- | --- |
| 使用本项目采集 | 按 4.1–4.4 预览、采集、检查并转换 |
| 已有 LeRobot v2.1 数据 | 用 `scripts/check_v21_dataset.py` 校验，通过后进入第 5 节 |
| 自有 HDF5、CSV、MCAP、视频或其他版本数据 | 按 [自定义数据接入](docs/custom_data_to_v21.md) 编写适配，将数据转为 v2.1，再校验和训练 |

转换需正确映射关节状态与动作、时间戳、相机和任务文本；关节用弧度，夹爪统一 `[0,1]`。仅改扩展名或版本标签不能完成转换。

### 4.1 填写设备并检查画面

相机型号自行选择；当前采集与推理入口读取 Linux V4L2/OpenCV RGB 图像。选择能稳定输出目标分辨率和帧率的设备，`single` 两路、`dual` 三路；模板参数是示例，填写实际设备路径与分辨率，并实测多路同时读取。其他相机接口需自行适配。

曝光、增益、白平衡按设备支持的厂商工具或 V4L2 控件调整，以目标清晰、无过曝和明显拖影为准；本项目不固定曝光数值，也不保证设备重插后保留设置。重插或重启后重新预览复查，采集与推理尽量保持相机位置、画面和光照一致。

**场地布置建议：** 下列为通用建议，不限定相机或灯具型号；截图不能确定灯具实际位置。

| 项目 | 摆放与检查 |
| --- | --- |
| 第三视角／全局相机 | 固定在桌面前上方或斜上方，朝下覆盖完整抓取区、放置区和运动路径；调整机位让工作台占主要画面，减少墙面和杂物。这里对应头部 `head`，不是额外增加一路模型输入。 |
| 腕部相机 | 朝向夹爪接触区，接近、夹紧、抬起和松爪时都能看到物体；避免夹指挡住目标或物体出画。双臂逐一核对左腕、右腕。 |
| 灯光 | 用稳定、柔和的侧前上方补光，减少手臂阴影、直射反光和逆光；录一段动作检查闪烁、条纹和拖影，灯具位置以预览结果调整。 |
| 桌面与背景 | 优先哑光、简洁、与物体有对比的背景；黑方块可用浅色桌垫，目标区域清晰且固定。人员、屏幕和无关物品尽量不入镜。 |
| 设置与复现 | 曝光、增益和白平衡以物体细节清楚为准；设备支持时固定已验证设置。保留布置照片及机位、分辨率／FPS、目标范围记录，采集与推理保持一致。 |
| 安装与通信 | 固定相机支架，线缆留足运动余量；用稳定设备路径绑定视角，按下面命令同时读两／三路至少 30 秒，确认无掉线。 |

先固定场景验证，再逐步扩大物体位置等变化范围，新增变化须有对应演示。开始前对照预览确认物体、夹爪和目标区域全程可见，可参考 [LeRobot 数据采集建议](https://huggingface.co/docs/lerobot/il_robots#tips-for-gathering-data)。

```bash
conda activate g05-robot
export A1Z_PYTHON="$CONDA_PREFIX/bin/python"
export ARM_MODE=single   # 双臂改为 dual
mkdir -p "$HOME/a1z-local"
cp -n a1z-data-collect/a1z-t/teleop_app/config.yaml "$HOME/a1z-local/a1z-t.yaml"
nano "$HOME/a1z-local/a1z-t.yaml"
cp -n "configs/collect_${ARM_MODE}.example.yaml" "$HOME/a1z-local/collect_${ARM_MODE}.yaml"
nano "$HOME/a1z-local/collect_${ARM_MODE}.yaml"
export G05_COLLECT_CONFIG="$HOME/a1z-local/collect_${ARM_MODE}.yaml"
ip -details link show type can
ls -l /dev/serial/by-id/ /dev/v4l/by-path/
bash scripts/collect.sh "$ARM_MODE" --check
python scripts/preview_cameras.py --config "$G05_COLLECT_CONFIG"
# 关闭预览后，可做无窗口连续读取检查：
python scripts/preview_cameras.py --config "$G05_COLLECT_CONFIG" --check-only --duration 30
```

填写真实 `task_description`、`hdf5_cfg.task_name`、保存目录、采集条数 `num_episodes`、Leader 串口、Follower CAN 和相机路径；保存目录建议填绝对路径（相对路径以 `a1z-data-collect/` 为基准）。`single` 预览显示 Head（头部）、Wrist（腕部）；`dual` 显示 Head（头部）、Left Wrist（左腕）、Right Wrist（右腕），逐一核对实际安装位置。CAN 必须 UP，串口/视频设备须有访问权限。若提示权限不足，Ubuntu 可执行 `sudo usermod -aG dialout,video "$USER"`，退出系统登录后重新登录；设备使用其他权限规则时按其 udev 配置处理。Leader 首次使用先按 [a1z-t 说明](https://github.com/userguide-galaxea/a1z-t/blob/93235404adbb5b753da6fd4d62eccc60805cfeaa/README.md) 确认固件、舵机 ID 和夹爪校准。a1z-t 由 data-collect 内部子模块管理；双臂需要两个独立串口。复制其 `teleop_app/config.yaml` 到本地，按硬件选择 `a1z_t_version: EVT` 或 `PVT`，并在采集 YAML 的 `arm_cfg.leader_config_path` 填写该文件绝对路径（null 使用上游默认配置）。相机稳定、方向对应正确后关闭预览，再启动采集。切换模式时另存对应配置，`cp -n` 不会覆盖旧文件。

若 CAN 接口尚未 UP，确认没有程序占用后设置；双臂对第二个接口重复执行。接口不存在时先按 [SDK 说明](https://github.com/userguide-galaxea/GALAXEA-A1Z/blob/156af498a947a830ca38efff4dd08fd2da96a200/README.md) 安装对应适配器驱动：

```bash
read -r -p '要配置的 CAN 接口（如 can0）: ' A1Z_CAN
sudo ip link set "$A1Z_CAN" down
sudo ip link set "$A1Z_CAN" type can bitrate 1000000
sudo ip link set "$A1Z_CAN" up
ip -details link show "$A1Z_CAN"
```

返回执行采集 `--check` 和相机预览，通过后继续。

### 4.2 开始采集

切换 single／dual 后，重新选择对应配置，不要沿用另一模式的环境变量：

```bash
export G05_COLLECT_CONFIG="$HOME/a1z-local/collect_${ARM_MODE}.yaml"
bash scripts/collect.sh "$ARM_MODE" --check
bash scripts/collect.sh "$ARM_MODE"
```

使用固定版本 data-collect / a1z-t。**S 开始、E 保存、R 丢弃重录、Q 退出**；停止不自动回零。先在现场确认运动范围和急停，再采少量数据回放；确认相机、动作、夹爪和任务正常后批量采集。不要同时运行其他遥操或推理客户端。

### 4.3 采集后检查数据

先保存并等待写入完成，抽查一条完整 episode（只读取数据、导出视频，不驱动机械臂）：

```bash
conda activate g05-robot
read -r -p '已保存的 episode_N.hdf5 完整路径: ' A1Z_EPISODE
python a1z-data-collect/tools/inspect_episode.py "$A1Z_EPISODE"
```

终端显示任务、时长、采样率、关节与夹爪范围，并在 HDF5 同目录生成各相机 MP4，用视频播放器打开检查：

- `single` 两路、`dual` 三路画面齐全，对应位置正确，无过曝、黑帧或明显卡顿。
- 视频包含完整操作；抓取、松爪与手臂动作在各画面中对应，夹爪方向正确。
- 任务文本为真实任务；状态与动作分别为 7／14 维，关节单位为弧度，夹爪在 `[0,1]`；时长与采样率符合实际录制。

首批检查通过后再批量采集；批量完成后抽查开头、中间、结尾的 episode。该工具提供人工检查信息，下一节 `--check-only` 才会对全部输入执行结构、时间戳和图像校验；失败或中断片段单独保留排查，不混入训练数据。自有格式按同样标准检查，再按适配说明转换。

### 4.4 本项目 HDF5 → LeRobot v2.1

以下转换器只支持本项目采集的 HDF5 字段结构；其他 HDF5 即使扩展名相同，也需先适配。录制保存完成后，在转换终端填写输入和全新的输出目录；FPS 须与计划部署频率一致：

```bash
conda activate g05-convert
export ARM_MODE=single   # 或 dual
read -r -p 'HDF5 数据目录: ' HDF5_INPUT
read -r -p '新的 v2.1 输出目录: ' V21_OUTPUT
SINGLE_ARGS=(); [[ "$ARM_MODE" != single ]] || SINGLE_ARGS=(--single)
python scripts/convert_hdf5_to_v21.py --input "$HDF5_INPUT" --fps 15 --max-age 0.10 "${SINGLE_ARGS[@]}" --check-only
python scripts/convert_hdf5_to_v21.py --input "$HDF5_INPUT" --output "$V21_OUTPUT" --fps 15 --max-age 0.10 "${SINGLE_ARGS[@]}"
python scripts/check_v21_dataset.py "$V21_OUTPUT"
```

先检查全部时间戳和图像，再写入、检查视频和元数据。失败目录带 `INCOMPLETE`，不能训练；输出目录已存在时拒绝覆盖。`--max-age 0.10` 是 15 Hz 对齐的时间预算，不是允许随意丢帧；同步报错时分析报告再调整，见 [转换说明](docs/hdf5_to_v21.md)。训练仅接受真正的 **LeRobot v2.1**。

### 4.5 入门 Demo：Pick Cube

**目标：抓起黑色方块，放入黑色框内。** 建议先用 `single`（头部＋腕部相机）完成整个流程；固定相机和框的位置，让方块及框边缘清晰可见。

下图取自抓放演示数据：双臂设备上的一侧机械臂将方块移至桌面另一位置，画面中没有黑框。它展示采集过程与相机视角，不代表模型推理效果；正文的黑框是入门任务的目标区域设置。配图仅裁取工作区、缩放和标注，原始数据未修改。

![方块抓放采集过程：初始场景、抓取、搬运、桌面放置](docs/images/pick-cube-sequence.jpg)

同一时刻的三路相机画面，从左到右为头部、左腕、右腕：

![Pick Cube 采集视角：头部、左腕、右腕](docs/images/pick-cube-camera-views.jpg)

在第 4.1 节创建的本地采集 YAML 中填写以下字段，保留其余设备配置：

```yaml
task_description: Pick up the black cube and place it inside the black box.
num_episodes: 100
hdf5_cfg:
  dataset_dir: ../data/raw
  task_name: pick_cube
  start_episode: null
```

`100` 是示例采集目标，不是保证成功的数据量。先采 3–5 条，按 4.3 节检查两路视频、动作和夹爪，再批量采集。每条从一致的起始姿态开始：接近 → 夹紧 → 抬起 → 移至框上方 → 放下并松爪 → 退开，完成后按 E 保存。方块位置要覆盖计划使用的区域，尤其要录清楚放置和松爪；失败片段按 R 丢弃重录。

按 4.4 节以 `ARM_MODE=single` 转换并校验，再按第 5 节选择 single LoRA 或全量训练，第 6 节看 loss，第 7–8 节加载和执行。推理 JSON 的 `task` 使用与上面 `task_description` 完全相同的文本；相机位置、物体、目标框及起始姿态保持一致。

验收时在采集覆盖范围内选 10 组未用于采集的方块位置，分别记录“抓起成功”和“最终放入框内成功”；方块完全进入框内并已松爪才计任务成功。比较 checkpoint 时使用相同的 10 组位置，结合验证 loss 选择模型。

**双臂扩展：** 使用 `dual` 和三路相机，在互不干涉的两侧区域各放一块方块和一个框，分别完成抓放；重新采集双臂数据，并把采集、推理任务文本统一改为 `Pick up both black cubes and place each one inside its corresponding black box.`。数据目录和 `task_name` 使用独立名称（如 `pick_cube_dual`），按 dual 流程转换、训练和推理。

## 5. 训练：先试跑，再正式运行

### 5.1 选择单／双臂和训练方式

两种训练都从第 3 节的原始 G0.5 基模开始，使用同一 v2.1 数据契约；各自独立运行，不需要先跑 LoRA 再跑全量。本节默认按完整 episode 划分验证集，至少需要 2 条有效数据；单条数据只用于录制与转换检查。

| 选择 | LoRA（`lora`） | 全量微调（`full`） |
| --- | --- | --- |
| 用途 | 通过低秩适配减少训练显存，适合先验证任务 | 更新完整模型，显存与 checkpoint 更大 |
| single 入口 | `scripts/train_single_lora.sh` | `scripts/train_single_full.sh` |
| dual 入口 | `scripts/train_lora.sh` | `scripts/train_full.sh` |
| 推理方式 | 加载该次训练的 checkpoint 和配套配置，保留 LoRA 结构 | 加载该次训练的 checkpoint 和配套配置 |

先创建训练会话，再在会话内执行本节后续命令；已存在则用 `tmux attach -t a1z-train` 返回：

```bash
tmux new -s a1z-train
```

会话内先 `cd` 到仓库根目录，激活模型环境并选择四种组合之一：

```bash
conda activate g05-model
export G05_PYTHON="$CONDA_PREFIX/bin/python"
export G05_OUTPUT_DIR="$PWD/outputs"
export ARM_MODE=single   # 或 dual
read -r -p '训练方式（lora 或 full）: ' TRAIN_MODE
export TRAIN_MODE
read -r -p '已校验的 v2.1 数据目录: ' A1Z_DATASET
export A1Z_DATASET CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1
unset G05_TRAIN_TASK
case "$ARM_MODE:$TRAIN_MODE" in
  single:lora|single:full) TRAIN_SCRIPT="scripts/train_single_${TRAIN_MODE}.sh" ;;
  dual:lora|dual:full) TRAIN_SCRIPT="scripts/train_${TRAIN_MODE}.sh" ;;
  *) echo 'ARM_MODE 只能是 single/dual，TRAIN_MODE 只能是 lora/full'; exit 2 ;;
esac
python scripts/full_preflight.py
python scripts/check_v21_dataset.py "$A1Z_DATASET" --g05-loader
bash "$TRAIN_SCRIPT" --dry-run
```

### 5.2 先跑 10 个优化步

继续使用同一会话的模式与路径：

```bash
export EXP_NAME="a1z_${ARM_MODE}_${TRAIN_MODE}_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "logs/$EXP_NAME"
set -o pipefail
bash "$TRAIN_SCRIPT" model.max_steps=10 model.batch_size=1 batch_size_val=1 model.num_workers=0 checkpointing_steps=10 eval_steps=10 data.val_set_proportion=0.1 logger.type=null logger.log_steps=1 2>&1 | tee "logs/$EXP_NAME/train.log"
printf '%s\n' "${PIPESTATUS[0]}" > "logs/$EXP_NAME/exit_code"
```

试跑须满足：退出码 0、到达 10 个优化步、loss 有限、checkpoint 保存成功；按第 7 节加载该产物后停止服务，再正式训练。仅配置解析或占用显存不算通过。

### 5.3 正式训练

| 正式训练推荐起点 | LoRA | 全量 |
| --- | --- | --- |
| 优化步数 | 20000 | 30000 |
| batch × 梯度累积（单卡） | 1 × 8 | 2 × 4 |
| 学习率 | 1e-4 | 4e-5 |
| rank / alpha | 8 / 16 | 关闭 LoRA |
| 权重 / 计算 | BF16 基础权重、FP32 适配器 / BF16 混合精度 | FP32 权重 / BF16 混合精度 |
| warmup / 调度 | 200 / cosine | 200 / cosine |
| 保存间隔 / 完整 episode 验证集 | 5000 步 / 约 10% | 同左 |

这些是显式覆盖的推荐起点，未改变底层默认配置。有效 batch = batch × 累积 × GPU 数；步数是优化器更新次数。先保持有效 batch，再按显存调整单次 batch。通过验证 loss、动作误差和实机成功率选择模型，不只看训练 loss 或最后一步。

试跑与重载检查通过后，在原训练会话运行以下命令（沿用 5.1 的变量）。关闭或更换终端后，先重新执行 5.1 的环境与选择命令。`--dry-run` 仅解析配置，不代替实际试跑：

**下面 A／B 二选一，再执行 C。** `ARM_MODE` 和 `A1Z_DATASET` 沿用 5.1；切换训练方式应先重做该方式的 10 步试跑。

**A. LoRA 参数与入口：**

```bash
export TRAIN_MODE=lora
STEPS=20000; BATCH=1; ACCUM=8; LR=1e-4
if [[ "$ARM_MODE" == single ]]; then TRAIN_SCRIPT=scripts/train_single_lora.sh; else TRAIN_SCRIPT=scripts/train_lora.sh; fi
```

**B. 全量微调参数与入口：**

```bash
export TRAIN_MODE=full
STEPS=30000; BATCH=2; ACCUM=4; LR=4e-5
if [[ "$ARM_MODE" == single ]]; then TRAIN_SCRIPT=scripts/train_single_full.sh; else TRAIN_SCRIPT=scripts/train_full.sh; fi
```

**C. 启动已选方式（A 或 B）的正式训练：**

```bash
export EXP_NAME="a1z_${ARM_MODE}_${TRAIN_MODE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "logs/$EXP_NAME"
set -o pipefail
bash "$TRAIN_SCRIPT" "model.max_steps=$STEPS" "model.batch_size=$BATCH" "model.grad_accumulation_steps=$ACCUM" "model.learning_rate=$LR" model.warmup_steps=200 model.lr_scheduler_type=cosine batch_size_val=1 checkpointing_steps=5000 data.val_set_proportion=0.1 eval_steps=1000 logger.type=wandb logger.mode=offline logger.workspace=null logger.log_steps=10 2>&1 | tee "logs/$EXP_NAME/train.log"
printf '%s\n' "${PIPESTATUS[0]}" > "logs/$EXP_NAME/exit_code"
```

记下实验名与 `logs/$EXP_NAME/train.log` 路径。退出码文件 `logs/$EXP_NAME/exit_code` 为 0 才表示命令正常结束。Ctrl+B 后按 D 脱离、训练继续；`tmux attach -t a1z-train` 返回。输出位于 `outputs/任务目录/实验名/`（具体路径见日志），包含 `.hydra/config.yaml`、`dataset_stats.json`、`checkpoints/step_N.pt` 和 `last.pt` 链接。验证集按完整 episode 划分；数据过少时先增加有效 episode，再判断泛化。

## 6. 查看实时 loss 图

另开终端，填写需要查看的日志：

```bash
python3 -m venv .venv-tools
.venv-tools/bin/python -m pip install flask
read -r -p '训练日志路径: ' TRAIN_LOG
.venv-tools/bin/python scripts/loss_server.py "$TRAIN_LOG" --host 127.0.0.1 --port 8888
```

本机访问 [http://127.0.0.1:8888](http://127.0.0.1:8888)，每 5 秒刷新 Loss / GradNorm（日志没有该指标时不显示数值）。远程机器在自己的电脑建立隧道后访问同一地址：

```bash
read -r -p '用户名@训练服务器: ' TRAIN_SSH
read -r -p 'SSH 端口: ' TRAIN_SSH_PORT
ssh -N -L 8888:127.0.0.1:8888 -p "${TRAIN_SSH_PORT:-22}" "$TRAIN_SSH"
```

图表需要浏览器能加载 CDN；离线可看 `/data` 的 JSON。看板只读日志，不控制训练。

## 7. 加载 checkpoint，启动模型服务

**终端 A：模型环境。** 先退出同卡训练。single 数据训练的模型配 single 客户端，dual 模型配 dual 客户端；LoRA 和全量的启动命令相同，区别由训练配置自动恢复。

从 `checkpoints/step_N.pt` 选择训练产物；`last.pt` 指向最新保存点。训练效果验证需选择本次微调产物。本项目 LoRA checkpoint 含基模与适配器，直接加载，无须手工合并或另传 PEFT adapter。

```bash
conda activate g05-model
export G05_PYTHON="$CONDA_PREFIX/bin/python"
find outputs -path '*/checkpoints/step_*.pt' -type f
read -r -p '要加载的 checkpoint 完整路径: ' MODEL_CKPT
export ACTION_STEPS=16 POLICY_HOST=127.0.0.1 POLICY_PORT=8765
export G05_TMUX_SESSION=a1z-policy
bash scripts/server_tmux.sh start "$MODEL_CKPT" model.model_weights_to_bf16=true
bash scripts/server_tmux.sh status
bash scripts/server_tmux.sh tail
```

`running` 只说明进程存在；持续查看 `tail`，出现 `Policy server listening` 后再进入第 8 节。服务只计算动作，不控制机械臂；关闭终端后仍在 tmux 运行。

迁移模型时保留整个运行目录：checkpoint、`.hydra/config.yaml`、`dataset_stats.json` 和 tokenizer sidecar；部署端也需第 3 节 processor/tokenizer 资源。不要只拷一个 `.pt`。大体积 checkpoint 可按 [部署说明](docs/deployment_modes.md) 导出推理权重。

切换 LoRA／全量或不同保存点，**先停止客户端**，在终端 A 执行以下命令，再重新运行上面的选权重与 `start` 命令：

```bash
export G05_TMUX_SESSION=a1z-policy
bash scripts/server_tmux.sh stop
```

仅停止模型服务不会替你停止正在运行的机械臂客户端。

## 8. 设备检查与推理执行

**终端 B：设备环境，仍在仓库根目录。** 先退出采集、预览和其他控制程序，保留终端 A 的模型服务。以下先配设备，再逐项测试：

```bash
conda activate g05-robot
export G05_CLIENT_PYTHON="$CONDA_PREFIX/bin/python"
export ARM_MODE=single   # 或 dual
mkdir -p "$HOME/a1z-local"
if [[ "$ARM_MODE" == single ]]; then TEMPLATE=pc-single; else TEMPLATE=pc-direct; fi
cp -n "configs/deployment/$TEMPLATE.example.json" "$HOME/a1z-local/client_${ARM_MODE}.json"
nano "$HOME/a1z-local/client_${ARM_MODE}.json"
bash scripts/client.sh --config "$HOME/a1z-local/client_${ARM_MODE}.json" --check
# 只打开相机：核对与采集时一致；Q/Esc 关闭预览后再继续。
python scripts/preview_cameras.py --config "$HOME/a1z-local/client_${ARM_MODE}.json"
bash scripts/client.sh --config "$HOME/a1z-local/client_${ARM_MODE}.json"
# 现场设备与急停就绪后，先真实观测，再执行模型动作：
bash scripts/client.sh --config "$HOME/a1z-local/client_${ARM_MODE}.json" --observe-only
bash scripts/client.sh --config "$HOME/a1z-local/client_${ARM_MODE}.json" --execute
```

推理前必须用该 JSON 预览：核对相机安装位置、朝向、头部／腕部顺序、分辨率和曝光与采集时一致；有变化先修正，不直接执行模型。预览只读取相机，不连接 CAN／Leader；关闭预览后再启动客户端。

JSON 填写 CAN、相机路径、与采集一致的任务和相机对应关系；`image_size` 为实际 `[宽,高]`（双臂模板须按实际采集分辨率修改），`control_frequency` 等于 v2.1 数据 FPS。服务和客户端在同一 PC 时用 `127.0.0.1:8765`，与终端 A 一致；`ACTION_STEPS=16` 是每次模型预测后执行的动作窗口，不是 16 Hz。`single` 使用 `can_channel`、`camera_head`、`camera_wrist`，不按机械安装侧命名；模型内部字段保持兼容，不要手工改权重或数据字段。`--check` 只查配置；默认模式用合成观测、不打开设备；`--observe-only` 会启动 SDK 和读取真实设备，不发送模型动作。

| 参数 | 模板值与含义 |
| --- | --- |
| `max_initial_joint_step` / `max_joint_step` | 都为 0.1 rad；分别限制首帧相对反馈、后续相邻模型目标的变化，超限报具体关节和数值 |
| `gripper_boundary_tolerance` | 0.02；容差内裁剪到 `[0,1]`，更大越界或非有限值拒绝，不能用于单位转换 |
| `max_cycles` | 3 条动作；现场短测通过后设 0 可持续运行，不是任务轮数 |
| `smoothing.enabled` | 默认 false；设 true 启用直接到 250 Hz 的平滑，保留速度、加速度和动作限制 |
| `smoothing.stale_timeout` | 1 秒；动作超时后保持，先查观测/传输/推理耗时，不靠放宽阈值掩盖掉线 |

短测完成后修改对应 JSON：`max_cycles=0` 持续运行，`smoothing.enabled=true` 启用平滑，然后重新执行 `--execute`；模式或模型改变时先停旧客户端。配置修改后重启客户端即可。Ctrl+C 停止，无自动回零；双臂查看两侧停止结果。保持状态可能仍有力矩，电机禁用不等于切断物理电源。相机报 `No such device` 时检查 USB 断开、供电、带宽和省电设置，设备重连后重开客户端。详细参数和错误处理见 [部署说明](docs/deployment_modes.md)。
