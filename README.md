# memoRaxis_HippoRAG

`memoRaxis_HippoRAG` 是从 `memoRaxis` 中拆分出的 HippoRAG 单后端实验仓库。
本仓库用于在 MemoryAgentBench 上完成四个任务的 `ingest -> infer -> evaluate` 实验，并支持统一的 token 轨迹记录。

## 1. 项目内容

本仓库固定：

- M-axis：HippoRAG
- R-axis：R1 / R2 / R3 三种 adaptor

支持四个任务：

- Accurate_Retrieval
- Conflict_Resolution
- Long_Range_Understanding
- Test_Time_Learning

## 2. 环境准备

建议使用 Python 3.11。

```bash
conda create -n memoraxis_hipporag python=3.11 -y
conda activate memoraxis_hipporag
pip install -r requirements.txt
```

将真实配置写入：

```text
config/config.yaml
```

如仓库仅提供模板文件，则复制一份：

```bash
cp config/config.example.yaml config/config.yaml
```

## 3. 数据准备

将四个 parquet 文件放入：

```text
MemoryAgentBench/data/
```

然后执行：

```bash
python scripts/HippoRAG_MAB/data/convert_all_data.py
```

生成：

```text
MemoryAgentBench/preview_samples/
```

## 4. 一键运行

四个任务都跑 instance 0：

```bash
python run_all_tasks.py --acc_instance_idx 0 --conflict_instance_idx 0 --long_instance_idx 0 --ttl_instance_idx 0 --run_id hippo_all_inst0 --force_rebuild
```

只做最小测试：

```bash
python run_all_tasks.py --acc_instance_idx 0 --conflict_instance_idx 0 --long_instance_idx 0 --ttl_instance_idx 0 --acc_limit 1 --conflict_limit 1 --long_limit 1 --ttl_limit 1 --run_id hippo_smoke_inst0 --skip_eval --force_rebuild
```

## 5. 四个任务单独运行

Accurate Retrieval：

```bash
python run_all_tasks.py --tasks acc --acc_instance_idx 0 --acc_limit 5 --acc_chunk_size 850 --run_id acc_inst0 --force_rebuild
```

Conflict Resolution：

```bash
python run_all_tasks.py --tasks conflict --conflict_instance_idx 0 --conflict_limit -1 --conflict_min_chars 800 --run_id conflict_inst0 --force_rebuild
```

Long Range Understanding：

```bash
python run_all_tasks.py --tasks long --long_instance_idx 0 --long_limit 1 --long_chunk_size 1200 --long_overlap 100 --run_id long_inst0 --force_rebuild
```

Test Time Learning：

```bash
python run_all_tasks.py --tasks ttl --ttl_instance_idx 0 --ttl_limit -1 --run_id ttl_inst0 --force_rebuild
```

## 6. Token 统计

统一 token 轨迹输出位置：

```text
out/token_traces/<run_id>.jsonl
```

你需要重点检查：

1. `Conflict_Resolution`、`Long_Range_Understanding`、`Test_Time_Learning` 是否存在 `stage=infer` 记录。
2. `Long_Range_Understanding` 是否存在 `stage=evaluate` 记录。
3. 结果 JSON 顶层是否包含 `run_id` 与 `token_trace`。

## 7. 常见问题

- 索引不存在：先运行 ingest，或取消 `--skip_ingest`。
- 配置读取失败：检查 `config/config.yaml`。
- 数据文件缺失：检查 `MemoryAgentBench/data/` 与 `preview_samples/`。
- 结果文件缺失：先确认 infer 是否成功完成。

## 8. 输出目录

索引目录：

```text
out/hipporag_indices/
```

推理结果：

```text
out/
```

评测结果：

```text
out/eval/
```

日志：

```text
log/
```

token 轨迹：

```text
out/token_traces/
```
