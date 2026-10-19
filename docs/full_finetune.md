<!-- This file is part of G05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
See LICENSE.txt in the project root; upstream components retain their applicable licenses. -->

# 全量微调：后台启动与检查

本页补充全量训练的后台入口。首次安装、模型下载、v2.1 数据检查和 10 步试跑先完成 [README 第 1～5 节](../README.md)。LoRA 与全量训练可独立选择；全量入口关闭 LoRA，显存需求按全量试跑结果评估。

## 1. 设置模式与数据

在仓库根目录激活模型环境，选择 `single` 或 `dual`，填写已校验数据的绝对路径，并使用新实验名：

```bash
conda activate g05-model
export G05_PYTHON="$CONDA_PREFIX/bin/python"
export G05_OUTPUT_DIR="$PWD/outputs"
read -r -p '已校验的 v2.1 数据目录: ' A1Z_DATASET
export A1Z_DATASET
export G05_ARM_MODE=single   # 双臂改为 dual
export EXP_NAME="a1z_full_$(date +%Y%m%d_%H%M%S)"
export NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0
"$G05_PYTHON" scripts/full_preflight.py
"$G05_PYTHON" scripts/check_v21_dataset.py "$A1Z_DATASET" --g05-loader
```

本项目 HDF5 先按 [转换指南](hdf5_to_v21.md) 转换；自有格式先按 [数据接入说明](custom_data_to_v21.md) 适配。预检与数据检查不会加载训练模型。

## 2. 创建后台训练会话

以下参数与 README 的全量推荐起点一致，先确认该方式的 10 步试跑与 checkpoint 重载通过：

```bash
bash scripts/train_full_tmux.sh start \
  model.max_steps=30000 model.batch_size=2 model.grad_accumulation_steps=4 \
  model.learning_rate=4e-5 model.warmup_steps=200 model.lr_scheduler_type=cosine \
  batch_size_val=1 model.num_workers=4 checkpointing_steps=5000 \
  data.val_set_proportion=0.1 eval_steps=1000 logger.log_steps=10
```

会话名为 `g05-full-$EXP_NAME`，日志为 `logs/$EXP_NAME/train.log`，模型输出目录见日志。`start` 只说明会话已创建，须确认实际优化步数增长。已有日志目录拒绝覆盖，不自动续训。

```bash
bash scripts/train_full_tmux.sh status
bash scripts/train_full_tmux.sh tail
```

Ctrl+C 退出日志查看，后台训练继续。重新登录后先设置相同的 `EXP_NAME` 再查询。正常退出后，`logs/$EXP_NAME/exit_code` 保存进程退出码；0 仍需结合目标步数和产物判断。关机或强制结束会话可能没有退出码。

仅停止自己的该次训练：

```bash
tmux send-keys -t "g05-full-$EXP_NAME" C-c
```

## 3. 实时 loss

另开终端进入仓库，填写同一个实验名：

```bash
python3 -m venv .venv-tools
.venv-tools/bin/python -m pip install flask
read -r -p '训练实验名: ' EXP_NAME
.venv-tools/bin/python scripts/loss_server.py "logs/$EXP_NAME/train.log" --host 127.0.0.1 --port 8888
```

远程访问见 [README 第 6 节](../README.md#6-查看实时-loss-图)。看板只读日志，不启动或停止训练。

## 4. 检查与部署产物

先确认退出码、实际更新步数与日志中记录的输出位置：

```bash
cat "logs/$EXP_NAME/exit_code"
read -r -p '本次训练输出目录: ' MODEL_RUN
ls -lh "$MODEL_RUN/last.pt"
cat "$MODEL_RUN/training_resources.json"
```

最终推理 checkpoint 不包含优化器与调度器状态。要完整续训，保留含训练状态的中间 checkpoint。迁移时保留 `.hydra/config.yaml`、`dataset_stats.json`、tokenizer sidecar 和 checkpoint。

退出同卡训练后，按 [README 第 7 节](../README.md#7-加载-checkpoint启动模型服务)加载权重，再按 [第 8 节](../README.md#8-设备检查与推理执行)预览相机、检查合成观测、检查真实观测并进行现场短测。大 checkpoint 可按 [部署说明](deployment_modes.md) 导出推理权重。

## 显存与数据要求

全量微调建议准备至少 80 GB GPU 显存和 64 GB 主机内存，实际以数据、配置和试跑峰值为准。普通 DDP 不减少单卡模型副本。单臂使用 7 维动作、两相机；双臂使用 14 维动作、三相机，数据格式均为 LeRobot v2.1。
