# Daylight Workbench P0

离线的 Python 桌面工作台：CSV/XLSX → 多输出 ANN → NSGA-II → 数值区间筛选 → 候选 CSV 导出。

重要：所有优化输出均为“代理模型预测候选”，不是经 Radiance / EnergyPlus / Ladybug Tools 重新验证的结果。示例数据完全合成，只测试软件流程，不具有采光、眩光或能耗物理有效性。真实研究必须独立复核模型误差、几何约束、物理边界和再仿真结果。

## 当前交付状态

- 已实现可运行源码、中文 Qt 界面、科学核心、合成演示和自动化测试。
- Linux / Python 3.12 环境已通过 22 项核心/独立 QA/Qt线程回归测试，以及包含 Qt 视图的端到端冒烟测试。
- 已准备 Windows x64 CPU 版一文件夹构建脚本与 GitHub Actions；Linux 测试不能替代 Windows EXE 验收。Windows 构建结果以 CI 实际状态为准，不据此宣称已在 Windows 人工验收。
- 尚未收到真实数据；真实数据模型质量、中文路径下 Windows 完整人工操作、无 Python 环境启动验收仍是独立门槛。
- 不集成 LLM、LangChain、联网服务或 API 密钥。手动区间筛选为 P0 方案，不伪称 AI。

## 快速开始

要求：Python 3.12，64位。推荐新建虚拟环境。安装依赖需要联网，安装后核心流程离线运行。

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

Linux 需要图形会话及 Qt 系统运行库。无图形 CI 使用 QT_QPA_PLATFORM=offscreen 做测试，不等同真实桌面操作。CPU PyTorch 与 Qt 仍占较大磁盘空间，功能轻量不等于安装包很小。

## 操作流程

1. 数据与列定义：导入 .csv 或 .xlsx（首个工作表），显式设置自变量 X、目标 Y、忽略列和可选分组列。ID应忽略，不能把数值ID当作物理变量。类别列 P0 不支持，须预处理或忽略。目标优化方向必须明确选择最小化/最大化，演示已预设。
2. 校验划分：默认训练/验证/测试 70/15/15，种子42。相同输入向量即使目标冲突，也绑定到同一划分；分组关系与重复输入关系合并为传递分量，防止交叉泄漏。实际比例可能偏离，界面显示真实数量。默认保留完全重复行；只有用户勾选才删除，并显示数量。不会自动填缺失值。
3. ANN：默认64/64隐藏层、ReLU、线性输出、Adam学习率0.001、最多500轮、早停耐心35、批量64。X与Y的均值/标准差只由训练集计算。验证集早停并恢复最佳权重。测试集不参与缩放、梯度或早停。
4. 查看每个目标、每个划分的原单位 R²/MAE/RMSE，与训练均值基线比较；常量目标 R²明确标为未定义。图表显示独立测试集实测-预测散点和训练/验证损失。反复依据测试集调参会破坏测试独立性，应留新数据再验证。
5. NSGA-II：每个输入可设连续、整数、固定。默认上下界为训练子集观测范围，不允许超出。固定变量须上下界相等；整数边界须为整数，整数修复发生在真实遗传变量和模型评估之前。默认种群80、60代、种子42，最多20,000次预算。仅一个范围盒不代表相关变量组合有数据支持，更不等于物理可行。
6. 结果：可查看所有已评估唯一输入候选或全局预测Pareto候选；表格支持排序、多选。稳定候选ID在排序/筛选后不变；排序会清除选择以防误导出。输入/预测目标均可设数字区间。Pareto标记针对全部已评估候选，不随筛选重新定义。
7. 图表：选择任意目标对的散点图；平行坐标按本次全部候选各列最小/最大归一化。为保持UI响应，平行坐标仅抽样展示，表格与导出保留完整数据。
8. 导出当前筛选、选中行或所有唯一候选。CSV包含输入、预测目标、候选ID、代数及Pareto标记；旁边生成 .provenance.json，记录数据指纹、划分/训练/优化配置、种子、测试误差和CSV哈希。
9. 保存 .dwb 项目，可重新载入数据、明确列顺序、划分索引、模型、缩放器、历史和结果。仅训练后才能保存。改变影响模型的数据/列/训练配置会令旧模型和结果失效；改目标方向仅令优化结果失效。失败/取消不会采用任务半成品。

不建议把“至少12行”的软件门槛视作足够训练样本量。组数太少、目标常量、输入冲突、小样本或不均匀采样都可能让模型不适合优化。

## 数据与项目安全

- 数据文件上限128MB，拒绝重复/空列名、选择列中非数值、缺失或无穷值；至少需要12行有效数据和3个独立分量，每个划分至少2行。
- candidate_id、generation、pareto、predicted::前缀是保留结果名称，所选列不能占用。
- .dwb 是版本1 ZIP格式，仅含预期名称的JSON、CSV及PyTorch tensor-only state_dict。拒绝额外路径、重复成员、超大解压总量、哈希不匹配、列/架构不匹配或划分泄漏。
- 模型读取显式使用 torch.load(weights_only=True)，不执行任意Python对象反序列化；项目不解压至文件系统。哈希是损坏检测，不是作者签名。仍仅打开可信来源文件。
- 项目包含所选原始数据，未加密。分享项目即分享这些数据，请先脱敏；不会自动上传任何文件。
- CSV表头进行常见电子表格公式前缀中和。项目内部保留原始列名。
- 无账号系统、无遥测、无自动联网。不会保存任何 API 凭据。

## 测试

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/smoke.py --gui
python -m daylight_workbench --self-test
```

最后两项都用小型合成数据执行真实 ANN、整数 NSGA-II、项目重开、CSV/来源导出和离屏 Qt 视图。离屏测试不替代 Windows 上点击、中文路径、窗口缩放和打包启动的人工验收。

覆盖：组/重复输入传递防泄漏、训练专用缩放、常量目标指标、最小/最大方向与独立非支配关系核算、整数/固定变量、取消预检、非法数据、导入CSV/XLSX、来源校验、模型重开预测一致、坏项目/路径拒绝与筛选。pymoo 0.6.1.3 在本环境有依赖弃用警告，不影响已通过测试；若未编译扩展会使用较慢纯Python路径。

## Windows EXE 构建

Windows x64 安装Python3.12后执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\build.ps1
```

脚本依次安装CPU依赖、测试、离屏GUI冒烟、PyInstaller --onedir 打包，并运行打包EXE的 --self-test。产物是 dist/DaylightWorkbench/ 文件夹，入口 DaylightWorkbench.exe。必须完整解压并保留内部文件，不能只复制EXE。没有签名，Windows可能显示未知发布者；组织设备应按本单位策略审核，不要绕过安全警告。部分杀毒误报或依赖DLL问题需在Windows验证。

.github/workflows/windows-build.yml 在 push 或手动触发时构建同一流程，上传名为 DaylightWorkbench-Windows-x64 的14天构建产物。工作流执行与下载需要用户授权的仓库及GitHub账户；源码中不包含账号令牌。

## 结构与限制

- core.py：GUI独立的数值校验、无泄漏划分、训练、评估、NSGA-II、筛选与导出。
- project.py：受限版本化项目格式、哈希和结构校验、原子保存。
- ui.py：PySide6界面与单任务QThread，训练按小批次检查取消，优化按评估/代与归档检查取消。关闭计算中窗口时先取消并等任务结束，不强杀线程。
- demo.py：合成课堂形状数据，不是物理仿真。
- tests/ 与 qa/：实现测试及独立科学QA。

P0不含：物理耦合约束、类别/离散枚举、自适应主动学习、模型置信区间、自动超参搜索、LBT/Radiance/EnergyPlus调用、Grasshopper桥接、多用户、数据库、云端训练或LLM。研究质量验收需用户真实数据、输入/目标单位、期望方向、分组规则、允许误差和物理约束。
