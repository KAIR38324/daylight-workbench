# 建筑采光多目标优化平台

Daylight Multi-objective Optimization Workbench · v0.3

一个以本地运行的 Python 桌面工作台：CSV/XLSX 数据 → 逐目标独立 ANN、随机森林或 XGBoost 代理模型 → NSGA-II 多目标优化 → 交互分析、偏好决策与导出。DeepSeek 建议是可选联网功能；本地建模、优化与审核后的偏好排序不依赖 API。

**科学边界：所有优化候选只是代理模型预测，不是 Radiance、EnergyPlus、Ladybug Tools 或实测验证结果。** 搜索边界内不等于物理可行，真实研究应复核误差、工程约束并重新仿真。

## 当前验证状态

- 源码已在 Linux / Python 3.12 上通过自动化测试与仅使用合成数据的离屏 Qt 流程。该流程覆盖逐目标独立建模与旧共享模式、三类模型交叉验证、优化、交互图表联动、项目保存重开、外部结果导入和本地模拟偏好计划；没有发送在线 API 请求。
- Windows x64 构建脚本包含源码测试、离屏 GUI 测试、PyInstaller 打包和打包程序自检。构建成功或 Linux 源码测试通过都不等于 Windows EXE 已验收。以 Windows CI 实际结果及目标设备人工测试为准。
- 用户提供的旧 CSV **仅用来了解文件结构**，其表头对应关系可能有误。本项目不以其训练、测算科研效果或上传 GitHub 仓库。演示及自动化测试数据均是明确标识的合成数据。

## 安装与运行

需要 64 位 Python 3.12。安装时需要联网获取依赖；建模、优化、绘图和项目操作可在断网后使用。

Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m daylight_workbench
```

Linux：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m daylight_workbench
```

Linux 图形桌面还需要 Qt 系统运行库。PyTorch、Qt WebEngine、Plotly 和三类模型会使安装包较大。

## 使用要点

1. 导入 CSV/XLSX，明确设置输入、目标、可选分组及每个目标的最小化/最大化方向。ID 和无关列应忽略；不自动推断数据列的物理意义。重复输入与相同分组绑定划分，避免跨集泄漏。
2. 默认对每个目标独立选择模型和参数，可分别得到 ANN、随机森林或 XGBoost；旧共享多输出/共同配置模式仍可手动选择。ANN 层卡片可逐层添加/删除隐藏层、设置宽度和激活函数；图示与模型使用相同规格，不提供自由连线。
3. 网格搜索预先显示候选数、折数与**所有目标合计**的拟合预算，超预算会拒绝启动，运行中可取消。默认锁定约 15% 测试集，剩余开发集做组安全 5 折交叉验证；另有明确标识的固定划分快速模式。ANN/XGBoost 只用外层训练折内部留出集早停，绝不以测试集选模型。各目标选优后用同一开发集重拟合，最后共同评估一次锁定测试集。
4. 查看各目标所选模型、MAE、MSE、RMSE、R² 原单位指标和测试实测/预测散点。ANN 显示真实训练/验证损失和早停信息，RF 显示真实树数进度，XGBoost 显示真实逐目标迭代指标，不伪造训练曲线。
5. 优化边界从最终模型实际拟合数据的输入列顺序和观测范围自动填入，可缩小范围、固定变量或指定整数类型。NSGA-II 显示实时代数、评估数量和前沿；本版不提供额外硬约束编辑器。所有候选需独立物理验证。
6. 结果页使用本地打包的交互图，提供目标空间散点/平行坐标、目标两两关系、逐目标分布、已选方案原单位对比，以及输入×目标 Pearson 相关热图。相关性不代表因果。选择按稳定候选 ID 在图表与表格间同步；表格和区间筛选默认折叠，可展开。支持选中/当前/全部候选导出。还可导入已有 CSV/XLSX 数值结果，手动映射目标和方向；仅有目标列也能分析，不会虚构输入。
7. 第五页“偏好决策”先由用户确认目标含义和方向，再审核可编辑权重、比较范围、归一化及可选硬阈值；点击“应用”后仅对现有候选本地排序，不宣称唯一物理最优。可选 DeepSeek 建议须先预览完整请求、主动发送并审核计划；只发送问题、列/目标映射、数值范围汇总及当前偏好，不发送训练行、候选明细或 ID。API 可能收费；密钥不保存到项目、配置或日志。尚未验证实际在线调用。
8. 保存 `.dwb` 项目并重开；导出 CSV 会附带来源 JSON。项目包含原始数据、模型及候选，**未加密**，分享前应先处理敏感数据。只打开可信来源的项目文件。

## 测试与 Windows 构建

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m pytest -q qa/v03_acceptance.py
python scripts/smoke.py --gui
python qa/gui_smoke.py
python qa/v02_gui_smoke.py
python -m daylight_workbench --self-test
```

离屏测试不是目标设备上的人工 GUI 验收。Windows x64 用 `powershell -ExecutionPolicy Bypass -File .\build.ps1` 构建；脚本会准备本地 Plotly JS、运行测试、打包并启动 EXE 自检。需分发完整的 `dist/DaylightWorkbench/` 文件夹，不能只复制 EXE。若 Windows 提示未知发布者，应按所在组织的安全策略审核，不应绕过系统安全警告。

旧版图文说明：[v0.2 入门指南](docs/建筑采光多目标优化平台%20入门指南.docx)。v0.3 完整流程说明将在最新手册同步完成后提供。

## 数据和实现限制

输入文件上限 128 MB；所选数值列不能有缺失或无穷值。小样本、常量目标、冲突标签或不均匀采样可能使代理模型不适合优化。.dwb 使用带哈希与结构校验的限定 ZIP 格式，旧 ANN 项目仍可读取；哈希只检测损坏，不认证作者。CSV 表头做常见电子表格公式前缀中和。无账号、遥测、云训练、Grasshopper 双向连接或自动 LBT/Radiance/EnergyPlus 调用。
