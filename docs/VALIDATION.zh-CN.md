# P0 验证记录

日期：2026-10-20。环境：Linux x64、Python 3.12.14，具体版本见 tested-environment.json。

- 最终科学核心与Qt线程回归：python -m pytest -q → 22 passed，7.50秒。
- E2E：真实PyTorch ANN → 整数NSGA-II → 项目保存/重载 → 预测完全一致 → CSV及来源JSON → 离屏Qt指标/图表/筛选视图，通过。
- 稳定性修复：工作线程结果显式排队到MainWindow的Qt Slot，再调用Python完成回调。回归断言完成回调运行于GUI线程；重复启动、取消和计算中关闭窗口通过。
- 独立科学QA：14项全部通过，涵盖传递分组/重复输入泄漏、仅训练集缩放、常量目标R²、解析目标方向和非支配关系、整数/固定输入、项目重开、非法数值、重复CSV头和保留列名。
- pymoo依赖存在108条弃用警告；本Linux安装使用纯Python回退路径，无编译加速，不影响本次测试结果。

未完成门槛：Windows原生构建/打包EXE实际运行结果由GitHub Actions决定；Windows人工点击和中文路径验收、无Python目标机验收、用户真实数据的模型质量和物理再仿真，不能由本Linux记录替代。
