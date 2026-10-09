from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge_files(inputs: list[Path], output: Path) -> dict:
    seen: set[str] = set()
    records: list[dict] = []
    duplicates = 0
    for input_path in inputs:
        if not input_path.exists():
            raise FileNotFoundError(f"Dataset file was not found: {input_path}")
        for line in input_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            work_id = str(record.get("id") or "")
            if work_id and work_id in seen:
                duplicates += 1
                continue
            if work_id:
                seen.add(work_id)
            records.append(record)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + chr(10))
    return {
        "input_files": [str(path) for path in inputs],
        "records_written": len(records),
        "duplicates_removed": duplicates,
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge yearly OpenAlex JSONL files without duplicate works.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", default="data/openalex_temporal_dataset.jsonl")
    args = parser.parse_args()
    print(json.dumps(merge_files([Path(path) for path in args.inputs], Path(args.output)), indent=2))


if __name__ == "__main__":
    main()
