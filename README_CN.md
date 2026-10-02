# CARE-RWCap
### 面向神经引导浮动随机游走电容提取的条件感知修正

[English](README.md) · [安装](docs/INSTALL.md) · [方法](docs/METHOD.md) · [评估](benchmarks/public10/README.md)

## 项目简介

CARE-RWCap 基于 [DeepRWCap](https://github.com/THU-numbda/deepRWCap)，将局部转移修正、有符号采样贡献和同次求解的条件自电容读出连接到浮动随机游走（FRW）流程中。本包提供核心实现、冻结模型、公开十案例输入/参考及三种传统 CPU 基线。

## 方法总览

[![CARE-RWCap 总览：物理输入、首次 Gradient 转移、后续 Poisson 转移、导体命中累积和条件终点读出](docs/figures/overview.png)](docs/figures/overview.pdf)

*[矢量 PDF](docs/figures/overview.pdf) · [原尺寸 PNG](docs/figures/overview.png)*

按图中的编号：

1. **CPGR（Conditional Parity Gradient Refinement）**：在满足严格输入反射条件的首次神经 Gradient 转移中，实施奇偶投影和贡献补偿。最终联合算子还会对符合条件的对置面概率求平均，梯度网络参数保持冻结。
2. **BPR（Baseline-Anchored Poisson Refinement）**：后续神经 Poisson 转移使用有界残差重加权修正面内条件分布，保留冻结锚点和原始 Poisson 选面器。
3. **CER（Conditional Endpoint Re-estimation）**：导体命中贡献累积完成后，从同一次求解中获得 raw 与有条件的耦合项自电容读出。参考电容只参与误差评估。

图示为整体框架；公开入口每次评估一个指定主导体，解析器要求上游已输出按逻辑导体汇总的唯一列。选面、输入有效性检查及回退规则见[论文与实现对应](docs/PAPER_MAPPING.md)，公式见[方法说明](docs/METHOD.md)。

## 环境与快速开始

冻结神经引擎对应 Ubuntu 24.04 / Linux x86_64、RTX 4090、Python 3.12、CUDA Toolkit 12.6、Torch/Torch-TensorRT 2.6.0+cu126 和 TensorRT 10.7，采用 FP16 部署。未验证其他 GPU/软件组合。

先克隆仓库：

```bash
git clone https://github.com/chmyo-Q/CARE-RWCap.git
cd CARE-RWCap
```

再按[安装说明](docs/INSTALL.md)配置系统库与 Python 依赖，完成后执行：

```bash
python scripts/check_environment.py --build-only
python scripts/build.py
python scripts/check_environment.py
python scripts/run.py --arm full --output runs/full_case8
```

输出目录必须是新目录。示例使用 case8、初始 seed 2029，保存 `result.out`、`metrics.json`、`run.json` 和日志，包含 raw/CER 电容、误差、walks、hops 和求解器耗时。CER 不额外运行一次求解。

```bash
python -m unittest discover -s tests -p 'test_*.py'
bash scripts/smoke_test.sh --output runs/smoke
```

第一条需要 NumPy，但不需要 GPU；第二条需要受支持的 GPU 环境。[验证说明](docs/VALIDATION.md)区分当前检查与此前 GPU 验收。

## 数据与模型

| 目录 | 内容 |
|---|---|
| `models/paper_p0/` | 本地训练的 DeepRWCap 基线的五个 FP16 引擎 |
| `models/bpr/` | BPR 泊松引擎，其余四个引擎与 P0 共享 |
| `models/checkpoints/` | FP32 权重与未编译 TorchScript |
| `benchmarks/public10/` | 十案例几何与参考电容 |
| `configs/paper_protocol.json` | 选定主导体、参考值、运行参数和 seed 协议 |

论文 P0 采用 DeepRWCap 架构，但并非官方发布的预训练权重。CPGR、CER 不新增训练检查点。模型来源见 [models/README.md](models/README.md)。

## 十案例评估

| 组名 | 泊松模型 | CPGR | 主比较读出 |
|---|---|---|---|
| `p0` | Paper P0 | 关闭 | raw |
| `bpr` | BPR | 关闭 | CER |
| `full` | BPR | 开启 | CER |

```bash
# 单 case8、seed 2029、三组快速检查
python scripts/benchmark.py --profile quick --output runs/quick

# 只生成完整计划，不启动求解
python scripts/benchmark.py --profile paper --plan-only --output runs/paper_plan

# 10 case × 10 seed × 3组，共300次正式求解，另有3次预热
python scripts/benchmark.py --profile paper --output runs/public10
```

各组同时保留 raw/CER，完整批次输出 `summary.json` 和 `summary.md`，包括逐case统计、等权宏平均、同口径差异、不确定性与采样/耗时指标。失败时保留输出；不完整批次不生成完整性能汇总。

协议采用初始 seeds **2029–2038**；训练和单案例默认 seed 是 **2029**。[历史参考](results/README.md)仅对应原始300-run。异步执行可能改变轨迹，已核验批次间 CPGR 增量收益的方向也曾变化，因此不承诺新批次必然获得相同均值或排序。更多定义见[评估协议](docs/REPRODUCIBILITY.md)。

## 参考结果

原始 public10 批次每个案例、每个求解臂各运行10次，宏平均对十个案例等权。

| 配置 | Macro SelfCapErr (%) |
|---|---:|
| DeepRWCap（paper P0），raw | 0.9278 |
| BPR + CER | 0.9001 |
| CARE-RWCap：BPR + CPGR + CER | 0.8110 |

[完整 raw/CER 对照](results/README.md)列出实际运行配置以及共用同次求解的读出结果。[论文与代码对应](docs/PAPER_MAPPING.md)说明各项实验的公开范围。

另外三个预先选定的 BPR checkpoint，在完整方法下得到0.8112%、0.8038%、0.8414%的宏平均误差，均值±样本标准差为 **0.8188±0.0199%**。这项实验复用历史 P0 对照。[补充数据与协议](results/training_seeds/README.md)包含全部300条 Full 和100条复用 P0 记录，可在 CPU 上重算表格：

```bash
python scripts/summarize_training_seeds.py --output outputs/training_seed_summary
```

该命令汇总已保存电容值，不执行新求解。推理入口仍使用包内的 training-seed2029 模型，紧凑补充材料不含另外两个模型文件。

## 传统 CPU 基线

附带上游 FRW-AGF、MicroWalk、FRW-FDM 二进制及其许可证。在 Linux x86_64 下，仅需 Python 3.10+ 标准库，无需神经网络环境：

```bash
chmod +x third_party/deeprwcap/baselines/rwcap_*
python3 scripts/baselines.py --method agf --output runs/agf_example
python3 scripts/baselines.py --method microwalk --output runs/microwalk_example
python3 scripts/baselines.py --method fdm --output runs/fdm_example
# 可选：所选CPU基线的完整100次评估
python3 scripts/baselines.py --method agf --profile paper --output runs/agf_public10
```

默认16线程，使用 raw SelfCapErr，不应用 CER；神经方法默认8个 worker。FRW-FDM 可能耗时较长。输出范围和实际验收状态见 [CPU基线说明](docs/CPU_BASELINES.md)。

## 目录与范围

```text
src/bpr/                         BPR网络与损失
src/readout/                     电容解析与CER
cpp/cpgr/                        投影、选择器与补偿
cpp/seed_control/                初始seed设置
models/                          冻结引擎与可检查权重
benchmarks/public10/             十案例输入与参考
configs/                         运行设置
scripts/                         编译、推理、评估
third_party/deeprwcap/           上游运行库、传统基线及许可证
results/                         原主表参考与训练种子补充结果
tests/                           CPU/GPU实现检查
docs/                            安装、方法和评估细节
docs/figures/                    正式overview PNG与矢量PDF
```

本项目修正模块和读出逻辑提供源码；上游求解核心与传统基线按二进制形式提供。完整训练数据/驱动、额外layout数据、专用显存实验、全部研发日志和论文绘图工程不在本包内。

[可选局部评估入口](docs/LOCAL_VALIDATION.md)需要用户已有原始参考数据，这些数据未随包提供，也不是运行public10的前提。详见[公开范围](docs/ARTIFACT_SCOPE.md)。

## 引用、致谢与反馈

论文题目为 **CARE-RWCap: Condition-Aware Refinement for Neural-Guided Floating Random Walk Capacitance Extraction**。[CITATION.cff](CITATION.cff)提供与当前终稿作者信息一致的软件引用；仓库未填写尚未获得的论文 DOI、卷期或录用信息。感谢 DeepRWCap 作者提供神经求解器、网络结构、测试案例和基线程序；使用这些上游组件时，也请引用原论文，BibTeX 见[英文首页](README.md#citation-and-acknowledgments)。

许可证见 [LICENSE](LICENSE) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。安装或实现问题可通过 GitHub Issues 提交，请附运行命令、环境及相关日志片段。
