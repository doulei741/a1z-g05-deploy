<!-- This file is part of G05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
See LICENSE.txt in the project root; upstream components retain their applicable licenses. -->

# PC 本机推理与直连

模型服务与机械臂客户端在同一台 Linux PC。`single` 使用一臂和头部／腕部两路相机；`dual` 使用两臂和头部／左腕／右腕三路相机。完整安装与启动步骤见 [README 第 1～3、7～8 节](../README.md)。本文补充权重导出、参数和故障处理。

## 推理权重导出（可选）

训练中间 checkpoint 可能包含优化器和调度器状态。需要减少传输量时，可导出只供推理的权重；保留原始完整 checkpoint 用于续训。导出文件仍应放在同一运行目录的 `checkpoints/` 下，并保留 `.hydra/config.yaml`、`dataset_stats.json` 和 tokenizer sidecar。

```bash
conda activate g05-model
export G05_PYTHON="$CONDA_PREFIX/bin/python"
read -r -p '原始训练 checkpoint 完整路径: ' SOURCE_CKPT
read -r -p '新的推理 checkpoint 路径: ' INFER_CKPT
"$G05_PYTHON" scripts/export_infer_checkpoint.py "$SOURCE_CKPT" "$INFER_CKPT"
export ACTION_STEPS=16
bash scripts/server.sh "$INFER_CKPT" model.model_weights_to_bf16=true
```

导出会移除训练状态，将适用权重转换为 BF16；需保持 FP32 的参数仍保留 FP32。不能用导出文件恢复优化器状态。LoRA 与全量模型均需配套的训练配置，不能只迁移一个 `.pt`。

## 客户端配置与检查

先按 [README 第 2 节](../README.md#2-安装环境)安装 `g05-robot`。服务监听成功后，在仓库根目录另开终端，选择与模型一致的模式并填写设备：

```bash
conda activate g05-robot
export G05_CLIENT_PYTHON="$CONDA_PREFIX/bin/python"
export ARM_MODE=single   # 双臂改为 dual
mkdir -p "$HOME/a1z-local"
if [[ "$ARM_MODE" == single ]]; then TEMPLATE=pc-single; else TEMPLATE=pc-direct; fi
cp -n "configs/deployment/$TEMPLATE.example.json" "$HOME/a1z-local/client_${ARM_MODE}.json"
nano "$HOME/a1z-local/client_${ARM_MODE}.json"
bash scripts/client.sh --config "$HOME/a1z-local/client_${ARM_MODE}.json" --check
python scripts/preview_cameras.py --config "$HOME/a1z-local/client_${ARM_MODE}.json"
```

`single` 填写 `can_channel`、`camera_head` 和 `camera_wrist`；`dual` 填写 `left_can`、`right_can`、`camera_head`、`camera_left_wrist` 和 `camera_right_wrist`。核对实际安装位置，不按设备编号猜对应关系。相机预览只打开相机；确认画面顺序、视角、分辨率和曝光与采集一致，关闭预览后继续。

`image_size` 是实际 `[宽,高]`，`control_frequency` 与训练数据 FPS 一致，`task` 使用实际任务描述。CAN 接口须 UP，配置方法见 [SDK 说明](https://github.com/userguide-galaxea/GALAXEA-A1Z/blob/156af498a947a830ca38efff4dd08fd2da96a200/README.md)。本机模型服务和客户端默认使用 `127.0.0.1:8765`。

```bash
# 合成观测：不打开设备，检查模型响应和动作布局。
bash scripts/client.sh --config "$HOME/a1z-local/client_${ARM_MODE}.json"
# 设备和急停就绪后，再读取真实观测并执行短测。
bash scripts/client.sh --config "$HOME/a1z-local/client_${ARM_MODE}.json" --observe-only
bash scripts/client.sh --config "$HOME/a1z-local/client_${ARM_MODE}.json" --execute
```

`--check` 只检查配置，合成观测通过仅说明模型通信与动作格式可用。`--observe-only` 会启动 SDK、读取真实相机与关节，不发送模型动作，不能视为不接触设备的检查。`--execute` 才发送模型动作；首次执行须现场看护，不与采集或其他控制程序同时运行。

## 动作限制与持续运行

| 参数 | 模板值 | 含义 |
| --- | --- | --- |
| `max_initial_joint_step` | `0.1` rad | 首帧目标与启动反馈的最大差值 |
| `max_joint_step` | `0.1` rad | 后续相邻模型关节目标的最大差值，超限拒绝执行 |
| `gripper_boundary_tolerance` | `0.02` | 夹爪小幅越界可裁剪到 `[0,1]`，更大越界拒绝 |
| `max_cycles` | `3` | 动作条数上限；通过短测后设 `0` 可持续执行 |
| `control_frequency` | `15` Hz | 与训练数据 FPS 对应的动作发送频率 |

阈值不是关节速度设置。调整前检查单位、关节映射、训练数据分布和观测结果，不能靠放宽限制掩盖错误。夹爪的角度或 0～100 数据需先正确转换，边界容差不能用于单位换算。

## 可选的 250 Hz 平滑

单／双臂模板均带 `smoothing`，默认关闭。现场短测时可将 `smoothing.enabled` 设为 `true`，先保留 `max_cycles=3`；确认动作和停止行为后再持续运行。修改 JSON 后重启客户端，无需重训或重启模型服务。

```json
"smoothing": {
  "enabled": true,
  "frequency": 250,
  "max_velocity": 4.0,
  "max_acceleration": 20.0,
  "response_time": 0.04,
  "stale_timeout": 1.0
}
```

以上是 JSON 配置中的一个字段，放入完整配置对象并注意逗号。平滑线程直接向最多 250 Hz 的关节目标过渡，不包含额外的 30 Hz 预插值。速度单位为 rad/s，加速度单位为 rad/s²；夹爪随模型动作更新。已有步长与 SDK 限位检查继续生效。

平滑只处理已收到的目标，不预测未来动作。模型重算期间可保持最后目标；超过 `stale_timeout` 后停止平滑并尝试锁定最后指令，双臂同时处理两侧。`--observe-only` 不启动平滑发送线程。

`ACTION_STEPS` 是服务端每次预测后缓存的动作条数，改变它需重启模型服务；它与客户端的 `max_cycles` 和数据 FPS 不同。

## 停止与故障处理

- Ctrl+C 停止客户端，不自动回零；双臂核对两侧停止结果。保持可能仍有力矩，电机禁用不等于物理断电。
- 切换模型前先停客户端，再停止并重启模型服务；只停服务不能替你停止客户端。
- 相机 `No such device`：检查 USB 连接、供电、带宽与省电设置，重连后重新预览和打开客户端。
- SDK 故障、夹爪回零超时或 CAN 错误：先处理设备状态，不靠放宽模型动作阈值解决。
- 动作超时：检查观测、传输和模型推理耗时，不能用增大超时掩盖相机掉线。

统一操作顺序与参数定义见 [README 第 8 节](../README.md#8-设备检查与推理执行)。
