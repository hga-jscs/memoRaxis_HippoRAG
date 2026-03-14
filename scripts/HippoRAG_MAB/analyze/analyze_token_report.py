import argparse
import json
from pathlib import Path

import pandas as pd


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main(jsonl_path: str):
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"token trace not found: {path}")

    df = load_jsonl(path)
    if df.empty:
        raise ValueError("token trace is empty")

    out_dir = Path("out/token_reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = (
        df.groupby(["dataset", "stage", "substage", "adaptor", "api_kind", "model"], dropna=False)
        .agg(
            calls=("total_tokens", "count"),
            prompt_tokens=("prompt_tokens", "sum"),
            completion_tokens=("completion_tokens", "sum"),
            total_tokens=("total_tokens", "sum"),
            latency_ms=("latency_ms", "sum"),
        )
        .reset_index()
        .sort_values(["dataset", "stage", "total_tokens"], ascending=[True, True, False])
    )
    summary.to_csv(out_dir / "token_summary.csv", index=False, encoding="utf-8-sig")

    dataset_total = df.groupby(["dataset"], dropna=False)["total_tokens"].sum().reset_index()
    stage_total = df.groupby(["dataset", "stage"], dropna=False)["total_tokens"].sum().reset_index()
    substage_total = df.groupby(["dataset", "stage", "substage"], dropna=False)["total_tokens"].sum().reset_index()
    question_total = (
        df.groupby(["dataset", "instance_idx", "question_idx", "adaptor"], dropna=False)
        .agg(total_tokens=("total_tokens", "sum"), calls=("total_tokens", "count"))
        .reset_index()
        .sort_values(["dataset", "instance_idx", "question_idx"])
    )

    dataset_total.to_csv(out_dir / "dataset_total.csv", index=False, encoding="utf-8-sig")
    stage_total.to_csv(out_dir / "stage_total.csv", index=False, encoding="utf-8-sig")
    substage_total.to_csv(out_dir / "substage_total.csv", index=False, encoding="utf-8-sig")
    question_total.to_csv(out_dir / "question_total.csv", index=False, encoding="utf-8-sig")

    md = [
        "# Token Summary",
        "",
        "## Dataset Total",
        dataset_total.to_markdown(index=False),
        "",
        "## Stage Total",
        stage_total.to_markdown(index=False),
        "",
        "## Substage Total",
        substage_total.to_markdown(index=False),
        "",
        "## Detailed Summary",
        summary.to_markdown(index=False),
    ]
    (out_dir / "token_summary.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Saved report files to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze token tracing JSONL report")
    parser.add_argument("--input", type=str, required=True, help="Path to token JSONL file")
    args = parser.parse_args()
    main(args.input)
