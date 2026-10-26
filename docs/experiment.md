# async 实验

方案②：完整块异步推理、按观测时刻对齐、quintic软过渡；夹爪不做加权平均。不是论文RTC。

独立CLI与GUI均调用同一个g05_runtime.cli；运行模式为`async`。本分支保留baseline作为对照。默认FM-only、FM10、预测32、30Hz、原有250Hz有界平滑及运动限制不变。配置示例`configs/runtime/async.example.json`，命令见`docs/runtime.md`。

每次运行独立目录，保留源码提交/dirty、权重SHA、生效参数、原始目标与反馈。CPU与录制观测GPU检查不是机械臂任务成功率。现场短测由用户启动，方案胜出尚待真实数据。

停止/缓冲耗尽/模型异常时不重复旧动作，不自动续训或启动下一次实机运行。其他电脑通过主GUI安装入口获取本分支的固定提交，不能复用原机器绝对路径。
