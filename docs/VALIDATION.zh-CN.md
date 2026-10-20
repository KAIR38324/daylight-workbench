# v0.2 验证记录

日期：2026-10-20。环境：Linux x64、Python 3.12.14；本机依赖版本见 `tested-environment.json`。Windows 构建脚本使用 `requirements-dev.txt` 中更新的 PyInstaller 6.22.3，本机源码测试环境仍安装 6.12.0，不能把本机测试当作新版 Windows 打包验证。

- `python -m pytest -q`：66 passed，17.41 秒。本机出现 pymoo 依赖弃用警告；没有把警告当作测试失败。
- `python qa/v02_gui_smoke.py`：仅合成数据，真实离屏 Qt 窗口依次执行 CSV 导入、ANN 单次训练与训练/验证损失图、ANN/RF/XGBoost 小型组安全 CV、预算检查、NSGA-II、稳定 ID 表格/图联动、模拟 DeepSeek 计划本地应用、项目保存重开、仅目标列外部结果导入。1240×900 截图见 `qa/evidence/v02-*.png`。界面使用真实本地 Plotly JS，离屏环境无 WebGL 时测试 SVG 回退。
- `python qa/gui_smoke.py`：既有异步 GUI、取消、保存重载冒烟通过。
- `python -m daylight_workbench --self-test`：原有 ANN 流程与新增三模型合成 CV、优化、项目重开、模拟本地筛选、外部目标导入及离线图表资源检查通过。源码运行的自检**不是** Windows 打包 EXE 的自检。
- 关键独立回归：锁定测试标签极大值不能污染 ANN 训练张量；模型预测输出形状必须匹配目标数；XGBoost v2 项目拒绝有效但多输出的替换 Booster；交叉验证组/重复输入隔离、训练子集缩放、RF 排名不受测试标签变化、三模型保存重载一致、取消不采用半成品、图表稳定 ID 联动与 Pearson 非因果标签。
- DeepSeek 使用模拟传输与响应验证；没有真实密钥，也没有实际在线服务调用。真实 WebGL 3D 与 WebGL 平行坐标高亮未在离屏 Linux 上实测；其 JS 更新逻辑通过合同测试。

仍待完成：Windows 原生打包后 EXE 自检、目标设备人工点击及中文路径验收、无 Python 环境启动验收、用户真实数据的模型质量检查、物理约束和重新仿真验证。Windows CI 结果应单独记录，不能以源码测试或 EXE 文件生成替代。
