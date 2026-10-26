# 独立G0.5推理运行时

Built with Galaxea Dynamics A1Z。沿用仓库LICENSE.txt/NOTICE.txt与模型许可证。

## 入口与环境

本运行时不依赖GUI，CLI和GUI使用同一个 `RunConfig`、模型进程与动作执行器。
硬件端使用已安装A1Z SDK及相机依赖的Python（GUI项目推荐Python3.12）；模型使用
Python3.10及固定GalaxeaVLA依赖。运行时安装到各自虚拟环境，模型源码由当前目录的
固定GalaxeaVLA子模块加载，不能因共享Python环境误加载其他实验源码。

```bash
python -m pip install -e .
python -m g05_runtime.cli --capabilities
python -m g05_runtime.cli --config run.json --check
python -m g05_runtime.cli --config run.json --stub --duration 3
python -m g05_runtime.cli --config run.json --offline --duration 8
# 用户现场确认设备与环境后才执行：
python -m g05_runtime.cli --config run.json --execute
```

上述python是硬件/控制环境。`run.json`必须填写已登记推理bundle路径、模型Python、任务、
独立输出目录；执行硬件时还要填写三相机、左右CAN及safety_confirmed=true。
配置schema为1。示例见 `configs/runtime/baseline.example.json`。相对路径相对于配置文件。
`--check`不加载模型/设备；`--stub`CPU合成；`--offline`用bundle内录制观测实际加载GPU，
从不打开相机/CAN；`--execute`才打开硬件。模型预热在设备启动之前完成。

## 实验与复现

`runtime.json`声明该提交支持的模式；未支持的模式明确报错。三个方案仅在各自分支
加入ExperimentExecutor，公共基础保留历史节拍基线。GUI只管理进程和状态，不能消费动作。
事件通过G05_EVENT行传递；每次运行写入新的唯一目录，包含生效config、provenance、
server.log、events.jsonl与summary。停止先阻止新目标并hold/关闭硬件，再终止模型。
GPU/设备租约在各实验目录之间共享。同一时间只运行一个机器人方案。

```bash
python -m g05_runtime.report /path/to/run --output metrics.json
```

报告里的目标间隔不是CAN频率；回放没有实际反馈，task_success_verified始终false。
不会把文件存在/进程就绪/有限输出等同于实机成功。

## 分支与恢复

完整关系及上下文恢复入口 `PROJECT_MEMORY.md` / `AGENTS.md`。
先推送模型提交，再推送runtime子模块引用，最后更新GUI实验注册表。
不得自动选择实验胜出或自动训练。旧现场部署、模型权重和数据不覆盖。
