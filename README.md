# memoRaxis_HippoRAG
这个项目致力于将HippoRAG的代码整合进memoRaxis
# memoRaxis_HippoRAG

`memoRaxis_HippoRAG` 是从 `memoRaxis` 中拆分出的 HippoRAG 单后端实验仓库。该仓库延续了 memoRaxis 中“推理适配器（R-axis）”与“记忆系统（M-axis）”解耦的总体设计思路，并将 HippoRAG 作为唯一的记忆后端实现，用于在 MemoryAgentBench 上完成 ingest、infer、evaluate、analyze 的完整实验流程。

从项目定位来看，`memoRaxis_HippoRAG` 不是一个通用型、面向生产环境交付的 RAG 产品仓库，而是一个面向研究验证、实验复现与结构化重构的轻量实验仓库。该仓库保留了原始框架中最重要的两个维度：

1. 以 HippoRAG 作为唯一 M 轴，实现单后端、单路径、可独立验证的知识存储与检索流程；
2. 在统一的 R1 / R2 / R3 推理范式下，观测 HippoRAG 对不同记忆类任务的支持能力、检索效果与最终问答表现。

本仓库的价值，主要不在于展示一个已经高度封装的成品系统，而在于为以下工作提供稳定起点：

- 将 HippoRAG 从原始多后端实验框架中独立出来；
- 让 M 轴后端的运行链条、依赖关系、索引目录和评测逻辑变得更清晰；
- 在不改变整体研究思路的前提下，为后续的模块清理、文档完善、参数固化、一键运行与长期维护提供基础。

---

## 1. 项目目标与设计思路

### 1.1 目标概述

该仓库聚焦于一个非常具体的目标：**在独立、单后端的仓库中验证 HippoRAG 的记忆组织与检索能力，并与统一的推理适配器相结合，形成可复现的完整实验链条。**

围绕这一目标，仓库当前关注以下几个方面：

- 将 HippoRAG 后端从多后端混合项目中剥离出来；
- 保持和 memoRaxis 主体框架一致的推理接口与实验形式；
- 在统一目录结构与脚本组织下，对四个典型记忆任务进行复现实验；
- 尽量降低“为了跑 HippoRAG 而被其他无关后端依赖链干扰”的概率；
- 为后续更系统的重构和结果分析提供稳定中间态。

### 1.2 R 轴与 M 轴的分离

本仓库延续了 memoRaxis 中的一个核心思想：**推理策略与记忆系统分离。**

这里的“分离”并不是简单的文件拆分，而是实验变量层面的分离：

- **R-axis（Reasoning Axis）**：回答问题时采用的推理范式；
- **M-axis（Memory Axis）**：证据如何存储、组织、检索。

在这个仓库中：

- M 轴固定为 HippoRAG；
- R 轴仍然保留三种适配器：
  - `R1`：SingleTurnAdaptor
  - `R2`：IterativeAdaptor
  - `R3`：PlanAndActAdaptor

这样的设计使得研究过程具有较清晰的控制变量意义。若希望考察 HippoRAG 的作用，可以在固定 R1 / R2 / R3 的前提下观察表现变化；若希望考察推理范式的影响，则可以在固定 HippoRAG 的前提下比较三种适配器的效果差异。

### 1.3 仓库当前的成熟度判断

从开发状态来看，当前仓库已经具备以下条件：

- 单后端结构明确；
- 支持分步脚本运行；
- 支持一键脚本串联四个任务；
- 支持基础数据预处理、索引构建、推理和评测；
- 可以作为研究型实验仓库进行小规模或中等规模验证。

但同时也需要客观指出：

- 当前仍然属于研究代码风格；
- 入口虽然比主仓库更清晰，但尚未完全统一为一个工业级 CLI 框架；
- 评测环节依然带有强实验属性；
- 运行成功仍依赖本地模型接口、依赖版本和数据集准备情况。

因此，更准确的表述是：**本仓库已经具备“可独立使用与复现”的条件，但仍然是为研究和实验服务的代码仓库，而非面向普通用户的产品形态。**

---

## 2. 仓库结构与目录说明

一个独立的单后端实验仓库，最重要的不是文件数量少，而是**目录含义清楚**。当前项目的核心目录结构如下：

```text
memoRaxis_HippoRAG/
├─ config/
│  ├─ config.yaml
│  └─ prompts.yaml
├─ docs/
│  └─ bluePrint.md
├─ external/
├─ MemoryAgentBench/
├─ scripts/
│  ├─ HippoRAG_MAB/
│  │  ├─ data/
│  │  ├─ ingest/
│  │  ├─ infer/
│  │  ├─ evaluate/
│  │  ├─ analyze/
│  │  └─ debug/
├─ src/
│  ├─ __init__.py
│  ├─ adaptors.py
│  ├─ benchmark_utils.py
│  ├─ config.py
│  ├─ hipporag_memory.py
│  ├─ llm_interface.py
│  ├─ logger.py
│  └─ memory_interface.py
├─ main.py
├─ run_all_tasks.py
└─ requirements.txt
```

下面对各部分作说明。

### 2.1 `config/`

该目录存放配置文件。

- `config.yaml`：模型接口、embedding 接口、其他运行配置；
- `prompts.yaml`：R1 / R2 / R3 三类 adaptor 使用的提示模板。

其中 `config.yaml` 是运行链路的关键文件。若模型接口或 embedding 接口不可用，后续 ingest / infer / evaluate 中的若干步骤都会失败。

### 2.2 `docs/`

该目录用于存放设计说明与架构文档，例如 `bluePrint.md`。  
其价值主要在于说明原始项目中各个组成部分之间的关系，而不是直接作为操作手册。

### 2.3 `MemoryAgentBench/`

该目录用于存放 MemoryAgentBench 的原始数据和由脚本生成的 preview 样本。  
当前仓库四个任务默认依赖的数据均从这一目录下读取。

### 2.4 `scripts/HippoRAG_MAB/`

这是整个仓库最直接的实验脚本区域。它按实验阶段划分为：

- `data/`：数据预处理
- `ingest/`：构建索引或写入后端
- `infer/`：使用 R1 / R2 / R3 进行回答
- `evaluate/`：进行机械评测或 Judge 评测
- `analyze/`：对结果做进一步汇总与分析
- `debug/`：用于接口检查、调试辅助等

这是当前仓库的主要执行入口来源。

### 2.5 `src/`

该目录承载项目的一方核心代码。  
与“只是把脚本堆在一起”的仓库不同，这一层表达的是仓库的设计抽象。

其中最重要的文件包括：

- `adaptors.py`：推理适配器实现
- `memory_interface.py`：记忆系统统一接口
- `hipporag_memory.py`：HippoRAG 后端接入实现
- `llm_interface.py`：模型调用抽象
- `config.py`：配置读取
- `benchmark_utils.py`：任务与数据相关辅助函数
- `logger.py`：日志系统

### 2.6 `run_all_tasks.py`

该文件是本仓库推荐的一键运行入口。  
它的目标不是替代全部脚本，而是在“初次复现”“标准跑通”“基础 smoke test”场景下，提供一条更稳定、更一致的统一路径。

### 2.7 `requirements.txt`

该文件列出仓库基础依赖。  
需要注意的是，HippoRAG 这类后端往往还涉及自身依赖链，因此仅安装 `requirements.txt` 不一定足以完成全部运行，通常还需要安装 HippoRAG 自身或其 vendored 目录下的依赖。

---

## 3. 运行环境与安装准备

### 3.1 推荐环境

当前仓库推荐使用：

- Python 3.11
- Windows / Linux / macOS 均可
- 推荐在 Anaconda 环境中运行
- 推荐统一在同一个终端体系中操作，不混用多个 shell

之所以推荐 Anaconda，有两个直接原因：

1. 该项目涉及较多实验型依赖，使用 Conda 可以更方便隔离环境；
2. 在 Windows 平台上，统一使用 Anaconda Prompt 可以减少路径、解释器和环境变量错乱的问题。

### 3.2 Anaconda 环境创建

推荐步骤如下：

```bash
conda create -n memoraxis_hipporag python=3.11 -y
conda activate memoraxis_hipporag
```

环境名可以自行调整，但建议与项目对应，以方便维护多个后端子项目。

### 3.3 进入项目根目录

Windows 下常见写法如下：

```bash
cd /d D:\memoRaxis_HippoRAG
```

Linux / macOS 下则直接：

```bash
cd /path/to/memoRaxis_HippoRAG
```

### 3.4 安装基础依赖

```bash
pip install -r requirements.txt
```

### 3.5 安装 HippoRAG 及其依赖

如果仓库中已经包含 vendored 的 HippoRAG 目录，并且该目录支持 editable install，则推荐：

```bash
pip install -e third_party/HippoRAG
```

如果采用的是外部安装方式，则应根据项目实际依赖说明安装对应版本的 HippoRAG 包。

需要强调的是：**本仓库的成功运行，不仅依赖一方代码本身，也依赖 HippoRAG 的正确安装、模型接口的可访问性以及数据集的完整性。**

---

## 4. 配置文件说明

### 4.1 配置文件位置

默认配置文件位于：

```text
config/config.yaml
```

### 4.2 配置的作用

该配置文件通常至少包含以下两类配置：

1. LLM 配置；
2. Embedding 配置。

在 HippoRAG 这类图式或结构化后端中，LLM 与 embedding 并不是只在 infer 阶段使用。很多时候：

- 数据抽取；
- 开放信息抽取（OpenIE）；
- 构图或关系组织；
- 检索阶段的某些增强逻辑；

都可能依赖模型接口。

### 4.3 配置示例

```yaml
llm:
  provider: openai_compat
  model: gpt-4o-mini
  base_url: "https://your-llm-base-url/v1"
  api_key: "YOUR_LLM_API_KEY"
  timeout: 120

embedding:
  provider: openai_compat
  model: text-embedding-3-small
  base_url: "https://your-embedding-base-url/v1"
  api_key: "YOUR_EMBEDDING_API_KEY"
  dim: 1536

database: {}
```

### 4.4 对外公开仓库时的建议形式

如果仓库需要公开，`config.yaml` 更适合作为占位版，真实 key 和接口地址由本地单独填写。  
这是因为：

- 真实 key 放在仓库中存在安全风险；
- 不同使用者的模型中转地址、模型名和 embedding 接口并不相同；
- 实验型项目在团队协作时更适合“模板配置 + 本地实际配置”的方式。

---

## 5. 数据准备与任务对应关系

### 5.1 四个任务

当前仓库围绕 MemoryAgentBench 中四个任务组织实验：

1. `Accurate_Retrieval`
2. `Conflict_Resolution`
3. `Long_Range_Understanding`
4. `Test_Time_Learning`

### 5.2 对应数据文件

当前脚本默认读取以下 parquet 文件：

```text
MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet
MemoryAgentBench/data/Conflict_Resolution-00000-of-00001.parquet
MemoryAgentBench/data/Long_Range_Understanding-00000-of-00001.parquet
MemoryAgentBench/data/Test_Time_Learning-00000-of-00001.parquet
```

如果其中任意一个文件缺失，则该任务对应链路无法执行。

### 5.3 preview samples 的作用

在评测阶段，部分脚本需要使用 preview JSON，而不是直接读取 parquet。  
因此，推荐在首次运行前先执行：

```bash
python scripts/HippoRAG_MAB/data/convert_all_data.py
```

或者单独执行：

```bash
python scripts/HippoRAG_MAB/data/convert_parquet_to_json.py
```

这样会在：

```text
MemoryAgentBench/preview_samples/
```

下生成对应任务的 preview 样本。

### 5.4 为什么预处理要单独存在

这一步的意义是把原始 parquet 数据转成后续评测脚本易于读取的 JSON 形式。  
它本身不属于 HippoRAG 的“记忆构建”，但属于整个实验链的一部分，因此通常被归入统一的一键运行流程中。

---

## 6. 运行方式总览

当前仓库提供两种主要使用方式：

1. **一键运行**
2. **CLI 分步运行**

这两种方式并不是互斥的，而是面向不同需求。

### 6.1 一键运行的适用场景

一键运行适合：

- 第一次验证仓库是否能跑通；
- 按文档快速复现实验；
- 小规模 smoke test；
- 对四个任务做统一批量运行；
- 给客户、合作者或评审一个清晰的入口。

### 6.2 CLI 分步运行的适用场景

分步 CLI 更适合：

- 调试某一阶段；
- 调参数后只重跑某一环；
- 局部检查 ingest / infer / evaluate 是否一致；
- 迭代做实验；
- 定位脚本级别问题。

从实际维护角度看，**推荐的工作流往往是：先用一键运行确认全链路可用，再用 CLI 分步运行做深入实验。**

---

## 7. 一键运行脚本：`run_all_tasks.py`

### 7.1 功能概述

`run_all_tasks.py` 用于将四个任务串成一条完整链路，依次执行：

1. 数据预处理
2. ingest
3. infer
4. evaluate

它本身不改变各个子脚本的核心逻辑，而是将它们按合理顺序统一组织起来。

### 7.2 最简完整命令

四个任务全跑：

```bash
python run_all_tasks.py --force_rebuild
```

这条命令表示：

- 重新构建需要的 index；
- 依默认参数执行四个任务；
- 按脚本默认设置完成 ingest、infer、evaluate。

### 7.3 只跑四个任务的 instance 0

```bash
python run_all_tasks.py --acc_instance_idx 0 --conflict_instance_idx 0 --long_instance_idx 0 --ttl_instance_idx 0 --force_rebuild
```

这条命令适合做“小规模完整验证”。  
它不会跑所有实例，而是每个任务只跑 `instance 0`。

### 7.4 更轻量的冒烟测试

```bash
python run_all_tasks.py --acc_instance_idx 0 --conflict_instance_idx 0 --long_instance_idx 0 --ttl_instance_idx 0 --acc_limit 3 --conflict_limit 3 --long_limit 1 --ttl_limit 3 --acc_max_chunks 50 --conflict_max_chunks 20 --long_max_chunks 20 --ttl_max_chunks 20 --force_rebuild
```

这类命令适合第一次验证环境是否正常。  
它通过减少：

- 实例数；
- 每个实例的问题数；
- ingest 时允许处理的最大 chunk 数；

来显著降低首轮运行成本。

---

## 8. 一键运行参数详细说明

本节对 `run_all_tasks.py` 的参数做完整说明，便于直接作为 README 操作手册使用。

### 8.1 通用参数

#### `--tasks`

作用：指定要执行的任务集合。

默认值等价于：

```bash
--tasks acc conflict long ttl
```

可选值：

- `acc`
- `conflict`
- `long`
- `ttl`

示例：

```bash
python run_all_tasks.py --tasks acc conflict --force_rebuild
```

表示只运行 Accurate Retrieval 和 Conflict Resolution。

#### `--adaptors`

作用：指定推理适配器。

默认值等价于：

```bash
--adaptors R1 R2 R3
```

可选值：

- `R1`
- `R2`
- `R3`
- `all`

示例：

```bash
python run_all_tasks.py --adaptors R1 R2 --force_rebuild
```

#### `--index_root`

作用：指定 HippoRAG 索引根目录。

默认值：

```bash
--index_root out/hipporag_indices
```

不同实验若希望并行保存不同索引，可为此参数指定不同目录。

#### `--output_suffix`

作用：给推理结果文件名增加后缀，便于区分不同实验设置。

默认值为空字符串。  
若需要保留不同实验结果，可以手动指定，例如：

```bash
python run_all_tasks.py --output_suffix exp1_r1r2 --adaptors R1 R2 --force_rebuild
```

需要注意的是，`Conflict_Resolution` 的评测脚本对文件名匹配较敏感，因此默认建议留空。

#### `--openie_mode`

作用：控制 HippoRAG 的 OpenIE 模式。

可选值：

- `online`
- `offline`

默认值为：

```bash
--openie_mode online
```

其含义取决于 HippoRAG 后端的实现方式。一般来说：

- `online`：在运行时调用接口进行抽取；
- `offline`：使用已有结果或离线流程。

#### `--force_rebuild`

作用：ingest 前删除已有索引目录并重建。

这与 LightRAG 路线中常见的 `--reset` 语义相近，但参数名不同。  
首次运行、索引结构发生变化、怀疑旧结果污染时，通常使用该参数。

#### `--skip_preprocess`

作用：跳过 preview 数据生成。

当前提是 `MemoryAgentBench/preview_samples/` 已经存在且完整时，可以节省时间。

#### `--skip_ingest`

作用：跳过 ingest，直接复用已有索引。

适合以下场景：

- 已经构建过索引；
- 当前只想修改 adaptor 或评测逻辑；
- 不希望重复消耗接口成本。

#### `--skip_infer`

作用：跳过推理阶段。  
适合只检查预处理与索引构建是否正常。

#### `--skip_eval`

作用：跳过评测阶段。  
适合先只看推理结果文件是否正常生成。

---

### 8.2 Accurate Retrieval 专属参数

#### `--acc_instance_idx`

用于指定 Accurate Retrieval 的实例编号。默认值为：

```bash
0
```

支持形式包括：

- `0`
- `0-3`
- `0,2,5`
- `0-2,5,7`

#### `--acc_limit`

用于限制每个实例回答的问题数。默认值为：

```bash
5
```

#### `--acc_chunk_size`

用于控制 ingest 时的 chunk 大小。默认值为：

```bash
850
```

#### `--acc_max_chunks`

用于控制 ingest 时允许处理的最大 chunk 数。默认值为：

```bash
-1
```

其中 `-1` 表示不限制。

---

### 8.3 Conflict Resolution 专属参数

#### `--conflict_instance_idx`

默认值：

```bash
0-7
```

#### `--conflict_limit`

默认值：

```bash
-1
```

表示不限制问题数。

#### `--conflict_min_chars`

默认值：

```bash
800
```

用于控制 ingest 时的最小文本块长度。

#### `--conflict_max_chunks`

默认值：

```bash
-1
```

用于限制最大 chunk 数。

---

### 8.4 Long Range Understanding 专属参数

#### `--long_instance_idx`

默认值：

```bash
0-39
```

#### `--long_limit`

默认值：

```bash
-1
```

#### `--long_chunk_size`

默认值：

```bash
1200
```

#### `--long_overlap`

默认值：

```bash
100
```

#### `--long_max_chunks`

默认值：

```bash
-1
```

Long Range 任务对文本切分比较敏感，因此 `chunk_size` 与 `overlap` 对效果影响通常较明显。

---

### 8.5 Test Time Learning 专属参数

#### `--ttl_instance_idx`

默认值：

```bash
0-5
```

#### `--ttl_limit`

默认值：

```bash
-1
```

#### `--ttl_max_chunks`

默认值：

```bash
-1
```

---

## 9. 一键运行常用样例

### 9.1 四个任务全跑

```bash
python run_all_tasks.py --force_rebuild
```

### 9.2 四个任务都只跑 instance 0

```bash
python run_all_tasks.py --acc_instance_idx 0 --conflict_instance_idx 0 --long_instance_idx 0 --ttl_instance_idx 0 --force_rebuild
```

### 9.3 只跑两个任务

```bash
python run_all_tasks.py --tasks acc conflict --force_rebuild
```

### 9.4 只跑 R1 和 R2

```bash
python run_all_tasks.py --adaptors R1 R2 --force_rebuild
```

### 9.5 跳过 ingest，复用已有索引

```bash
python run_all_tasks.py --skip_ingest
```

### 9.6 先做轻量 smoke test

```bash
python run_all_tasks.py --acc_instance_idx 0 --conflict_instance_idx 0 --long_instance_idx 0 --ttl_instance_idx 0 --acc_limit 3 --conflict_limit 3 --long_limit 1 --ttl_limit 3 --acc_max_chunks 50 --conflict_max_chunks 20 --long_max_chunks 20 --ttl_max_chunks 20 --force_rebuild
```

---

## 10. CLI 分步运行方式

虽然一键脚本是更统一的入口，但当前项目仍保留分步 CLI，这是研究型项目非常重要的使用方式。

### 10.1 数据预处理

```bash
python scripts/HippoRAG_MAB/data/convert_all_data.py
```

或：

```bash
python scripts/HippoRAG_MAB/data/convert_parquet_to_json.py
```

### 10.2 Ingest

#### Accurate Retrieval

```bash
python scripts/HippoRAG_MAB/ingest/ingest_accurate_retrieval.py --instance_idx 0 --chunk_size 850 --save_root out/hipporag_indices --openie_mode online --max_chunks -1 --force_rebuild
```

#### Conflict Resolution

```bash
python scripts/HippoRAG_MAB/ingest/ingest_conflict_resolution.py --instance_idx 0-7 --min_chars 800 --save_root out/hipporag_indices --openie_mode online --max_chunks -1 --force_rebuild
```

#### Long Range Understanding

```bash
python scripts/HippoRAG_MAB/ingest/ingest_long_range.py --instance_idx 0-39 --chunk_size 1200 --overlap 100 --save_root out/hipporag_indices --openie_mode online --max_chunks -1 --force_rebuild
```

#### Test Time Learning

```bash
python scripts/HippoRAG_MAB/ingest/ingest_test_time.py --instance_idx 0-5 --save_root out/hipporag_indices --openie_mode online --max_chunks -1 --force_rebuild
```

### 10.3 Infer

#### Accurate Retrieval

```bash
python scripts/HippoRAG_MAB/infer/infer_accurate_retrieval.py --instance_idx 0 --adaptor all --limit 5 --index_root out/hipporag_indices --output_suffix "" --openie_mode online
```

#### Conflict Resolution

```bash
python scripts/HippoRAG_MAB/infer/infer_conflict_resolution.py --instance_idx 0-7 --adaptor all --limit -1 --index_root out/hipporag_indices --output_suffix "" --openie_mode online
```

#### Long Range Understanding

```bash
python scripts/HippoRAG_MAB/infer/infer_long_range.py --instance_idx 0-39 --adaptor all --limit -1 --index_root out/hipporag_indices --output_suffix "" --openie_mode online
```

#### Test Time Learning

```bash
python scripts/HippoRAG_MAB/infer/infer_test_time.py --instance_idx 0-5 --adaptor all --limit -1 --index_root out/hipporag_indices --output_suffix "" --openie_mode online
```

### 10.4 Evaluate

#### Accurate Retrieval

```bash
python scripts/HippoRAG_MAB/evaluate/evaluate_mechanical.py --results out/acc_ret_results_0.json --instance MemoryAgentBench/preview_samples/Accurate_Retrieval/instance_0.json
```

#### Conflict Resolution

```bash
python scripts/HippoRAG_MAB/evaluate/evaluate_conflict_official.py
```

#### Long Range Understanding

```bash
python scripts/HippoRAG_MAB/evaluate/evaluate_long_range_A.py --results out/long_range_results_0.json --instance_folder MemoryAgentBench/preview_samples/Long_Range_Understanding
```

#### Test Time Learning

```bash
python scripts/HippoRAG_MAB/evaluate/evaluate_ttl_mechanical.py --results_pattern out/ttl_results_*.json
```

---

## 11. 推荐使用流程

从开发者视角来看，比较稳妥的实际使用顺序如下。

### 第一步：最小 smoke test

先只跑四个任务的 `instance 0`，并限制问题数与 chunk 数：

```bash
python run_all_tasks.py --acc_instance_idx 0 --conflict_instance_idx 0 --long_instance_idx 0 --ttl_instance_idx 0 --acc_limit 3 --conflict_limit 3 --long_limit 1 --ttl_limit 3 --acc_max_chunks 50 --conflict_max_chunks 20 --long_max_chunks 20 --ttl_max_chunks 20 --force_rebuild
```

这样做的好处是：

- 可以快速发现配置错误；
- 可以快速发现依赖问题；
- 可以避免首次运行就投入过高的 token / 时间成本；
- 可以判断链路是否真正打通。

### 第二步：标准运行

在 smoke test 通过后，再运行更接近默认规模的命令：

```bash
python run_all_tasks.py --force_rebuild
```

### 第三步：局部调参

当链路稳定后，推荐改用分步 CLI，按需重跑：

- 修改 chunk 参数后只重跑 ingest；
- 修改 adaptor 后只重跑 infer；
- 修改评测脚本后只重跑 evaluate。

这比每次都从头全跑更适合科研实验。

---

## 12. 输出结果与目录说明

### 12.1 索引目录

默认索引目录为：

```text
out/hipporag_indices/
```

其中会按任务与实例组织子目录。  
若不同实验需要隔离索引，可通过 `--index_root` 指定不同路径。

### 12.2 推理结果目录

推理结果通常输出到：

```text
out/
```

常见文件名包括：

```text
out/acc_ret_results_0.json
out/conflict_res_results_0.json
out/long_range_results_0.json
out/ttl_results_0.json
```

若指定了 `--output_suffix`，则文件名会增加相应后缀。

### 12.3 评测输出

评测结果通常直接打印在终端中，不同任务评测方式不同：

- Accurate Retrieval：机械评测
- Conflict Resolution：官方评测
- Long Range Understanding：LLM / Judge 型评测
- Test Time Learning：机械评测

其中 `Long_Range_Understanding` 的评测往往更慢，调用链也更重，这是任务性质与评测脚本设计共同决定的。

---

## 13. 常见问题与排查思路

### 13.1 `ModuleNotFoundError`

如果一进入脚本就报缺包，优先检查：

- 是否激活了正确的 Conda 环境；
- 是否已安装 `requirements.txt`；
- 是否已安装 HippoRAG 本体；
- 当前 Python 解释器是否确实来自目标环境。

### 13.2 配置读取失败

优先检查：

- `config/config.yaml` 是否存在；
- YAML 格式是否正确；
- `base_url` / `api_key` / `model` 是否写对；
- 模型中转服务是否可访问。

### 13.3 数据文件缺失

如果报 parquet 文件不存在，则说明：

- `MemoryAgentBench/data/` 下的数据未准备完整；
- 文件名与脚本默认预期不一致；
- 当前工作目录不是项目根目录。

### 13.4 结果文件找不到

若评测脚本找不到结果文件，优先检查：

- infer 是否成功结束；
- `output_suffix` 是否导致文件名变化；
- 任务名和实例号是否与预期一致；
- 当前 `out/` 目录中是否已有旧结果混杂。

### 13.5 运行过慢或成本过高

如果 ingest 或 infer 成本过高，可以优先尝试：

- 只跑 `instance 0`；
- 减小 `limit`；
- 设置 `max_chunks`；
- 先做 smoke test；
- 仅保留 `R1` 或 `R1 R2`；
- 尽量复用已有索引，避免重复 `--force_rebuild`。

---

## 14. 当前版本的边界与后续方向

### 14.1 当前边界

当前版本虽然已经可独立运行，但仍然具有以下边界：

- 入口仍以实验脚本为主，而非单一统一 CLI 框架；
- 参数体系虽已比原始多后端仓库清晰，但仍存在进一步统一的空间；
- 某些评测脚本仍然较依赖上游任务产物与固定文件命名；
- 依赖模型服务与本地实验环境的可用性。

### 14.2 后续方向

该仓库后续较有价值的改进方向包括：

- 进一步清理与 HippoRAG 无关的依赖；
- 统一结果 schema 与输出目录规范；
- 为 ingest / infer / evaluate 增加更明确的 CLI 帮助信息；
- 增加更强的 smoke test 与环境检查脚本；
- 沉淀更系统的实验记录与分析文档。

---

## 15. 项目总结

`memoRaxis_HippoRAG` 的主要意义，不在于提供一个封装完整、面向最终用户的 RAG 成品，而在于：

- 将 HippoRAG 从原始多后端实验框架中独立出来；
- 收敛实验变量；
- 保留统一的 R1 / R2 / R3 推理适配器；
- 为后续更稳健的重构、文档化、参数固化和实验复现提供一个清晰的中间态仓库。

从开发者视角看，这个仓库最重要的不是“已经足够完美”，而是它已经形成了一个明确、可操作、可验证的结构：  
**HippoRAG 作为单独记忆后端，已经能够在统一的推理框架下被独立运行、独立评测、独立改造。**

这正是它作为一个单后端实验仓库的价值所在。
