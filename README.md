# Daylight Workbench v0.2

一个以本地运行的 Python 桌面工作台：CSV/XLSX 数据 → ANN、随机森林或 XGBoost 代理模型 → NSGA-II 多目标优化 → 交互分析、筛选与导出。DeepSeek 自然语言筛选是可选联网功能，不影响离线建模和优化。

**科学边界：所有优化候选只是代理模型预测，不是 Radiance、EnergyPlus、Ladybug Tools 或实测验证结果。** 搜索边界内不等于物理可行，真实研究应复核误差、工程约束并重新仿真。

## 当前验证状态

- 源码已在 Linux / Python 3.12 上通过自动化测试与仅使用合成数据的离屏 Qt 流程。该流程覆盖三类模型交叉验证、优化、交互表格联动、项目保存重开、外部结果导入和模拟 LLM 计划；没有发送在线 API 请求。
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
2. 在 ANN 层卡片中逐层添加/删除隐藏层、设置宽度和激活函数。图示和模型使用相同规格，不提供自由连线。也可选择随机森林或 XGBoost。
3. 网格搜索预先显示候选数、折数与拟合预算，超预算会拒绝启动，运行中可取消。默认锁定约 15% 测试集，剩余开发集做组安全 5 折交叉验证；另有明确标识的固定划分快速模式。ANN/XGBoost 只用外层训练折内部留出集早停，绝不以测试集选模型。选择后用全部开发集重拟合，最后评估一次测试集。
4. 查看每个目标的 MAE、MSE、RMSE、R² 原单位指标和测试实测/预测散点。ANN 显示真实训练/验证损失和早停信息，RF 显示真实树数进度，XGBoost 显示真实逐目标迭代指标，不伪造训练曲线。
5. 优化边界从最终模型实际拟合数据的输入列顺序和观测范围自动填入，可缩小范围、固定变量或指定整数类型。NSGA-II 显示实时代数、评估数量和前沿；本版不提供额外硬约束编辑器。所有候选需独立物理验证。
6. 结果页使用本地打包的交互图，可与表格按稳定候选 ID 联动选择，并支持数值区间筛选、选中/当前/全部候选导出。还可导入已有 CSV/XLSX 数值结果，手动映射目标和方向；仅有目标列也能分析，不会虚构输入。
7. 可选 DeepSeek 筛选：在软件内填写密钥，先检查完整请求预览，再主动发送，审阅返回的筛选/排序计划后才应用。默认只发送问题、允许列名/类型和数值范围汇总，不发送训练行、候选明细或 ID。API 可能收费；密钥不保存到项目、配置或日志。没有实际密钥时，不声称在线调用通过。
8. 保存 `.dwb` 项目并重开；导出 CSV 会附带来源 JSON。项目包含原始数据、模型及候选，**未加密**，分享前应先处理敏感数据。只打开可信来源的项目文件。

## 测试与 Windows 构建

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/smoke.py --gui
python qa/gui_smoke.py
python qa/v02_gui_smoke.py
python -m daylight_workbench --self-test
```

离屏测试不是目标设备上的人工 GUI 验收。Windows x64 用 `powershell -ExecutionPolicy Bypass -File .\build.ps1` 构建；脚本会准备本地 Plotly JS、运行测试、打包并启动 EXE 自检。需分发完整的 `dist/DaylightWorkbench/` 文件夹，不能只复制 EXE。若 Windows 提示未知发布者，应按所在组织的安全策略审核，不应绕过系统安全警告。

## 数据和实现限制

输入文件上限 128 MB；所选数值列不能有缺失或无穷值。小样本、常量目标、冲突标签或不均匀采样可能使代理模型不适合优化。.dwb 使用带哈希与结构校验的限定 ZIP 格式，旧 ANN 项目仍可读取；哈希只检测损坏，不认证作者。CSV 表头做常见电子表格公式前缀中和。无账号、遥测、云训练、Grasshopper 双向连接或自动 LBT/Radiance/EnergyPlus 调用。
