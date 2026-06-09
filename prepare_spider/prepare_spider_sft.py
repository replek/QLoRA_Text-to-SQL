import json
import re
from pathlib import Path
from typing import Dict, List, Any


SPIDER_DIR = Path("spider")
OUTPUT_DIR = Path("processed_spider_sft")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_sql(sql: str) -> str:
    sql = sql.strip()
    sql = re.sub(r"\s+", " ", sql)
    if sql.endswith(";"):
        sql = sql[:-1].strip()
    return sql


def get_column_type(schema: Dict[str, Any], table_id: int, column_name: str) -> str:
    column_names = schema["column_names_original"]
    column_types = schema.get("column_types", [])

    for idx, (current_table_id, current_column_name) in enumerate(column_names):
        if current_table_id == table_id and current_column_name == column_name:
            if idx < len(column_types) and column_types[idx]:
                return column_types[idx].upper()

    return "TEXT"


def build_primary_key_map(schema: Dict[str, Any]) -> Dict[int, List[str]]:
    table_names = schema["table_names_original"]
    column_names = schema["column_names_original"]
    primary_keys = schema.get("primary_keys", [])

    primary_key_map = {table_id: [] for table_id in range(len(table_names))}

    for column_id in primary_keys:
        if column_id < len(column_names):
            table_id, column_name = column_names[column_id]
            if table_id != -1:
                primary_key_map[table_id].append(column_name)

    return primary_key_map


def build_foreign_key_map(schema: Dict[str, Any]) -> Dict[int, List[str]]:
    table_names = schema["table_names_original"]
    column_names = schema["column_names_original"]
    foreign_keys = schema.get("foreign_keys", [])

    foreign_key_map = {table_id: [] for table_id in range(len(table_names))}

    for source_column_id, target_column_id in foreign_keys:
        if source_column_id >= len(column_names) or target_column_id >= len(column_names):
            continue

        source_table_id, source_column_name = column_names[source_column_id]
        target_table_id, target_column_name = column_names[target_column_id]

        if source_table_id == -1 or target_table_id == -1:
            continue

        target_table_name = table_names[target_table_id]

        fk_line = (
            f"FOREIGN KEY ({source_column_name}) "
            f"REFERENCES {target_table_name}({target_column_name})"
        )

        foreign_key_map[source_table_id].append(fk_line)

    return foreign_key_map


def format_schema_as_create_tables(schema: Dict[str, Any]) -> str:
    table_names = schema["table_names_original"]
    column_names = schema["column_names_original"]

    tables = {table_id: [] for table_id in range(len(table_names))}

    for table_id, column_name in column_names:
        if table_id == -1:
            continue

        column_type = get_column_type(schema, table_id, column_name)
        tables[table_id].append((column_name, column_type))

    primary_key_map = build_primary_key_map(schema)
    foreign_key_map = build_foreign_key_map(schema)

    create_statements = []

    for table_id, table_name in enumerate(table_names):
        lines = []

        for column_name, column_type in tables[table_id]:
            lines.append(f"    {column_name} {column_type}")

        primary_keys = primary_key_map.get(table_id, [])
        if primary_keys:
            lines.append(f"    PRIMARY KEY ({', '.join(primary_keys)})")

        foreign_keys = foreign_key_map.get(table_id, [])
        for fk in foreign_keys:
            lines.append(f"    {fk}")

        table_body = ",\n".join(lines)
        create_statement = f"CREATE TABLE {table_name} (\n{table_body}\n);"
        create_statements.append(create_statement)

    return "\n\n".join(create_statements)


def build_prompt(question: str, schema_text: str, sql: str | None = None) -> str:
    prompt = f"""### Instruction:
You are a Text-to-SQL assistant. Generate a valid SQLite SQL query for the given question using only the provided database schema. Do not use tables or columns that are not present in the schema. Return only the SQL query.

### Database schema:
{schema_text}

### Question:
{question}

### SQL:
"""

    if sql is not None:
        prompt += normalize_sql(sql)

    return prompt


def build_sft_examples(
    examples: List[Dict[str, Any]],
    schemas_by_db_id: Dict[str, Dict[str, Any]],
    split_name: str,
    include_sql_in_text: bool = True,
) -> List[Dict[str, Any]]:
    processed = []

    for idx, example in enumerate(examples):
        db_id = example["db_id"]
        question = example["question"].strip()
        sql = normalize_sql(example["query"])

        if db_id not in schemas_by_db_id:
            raise ValueError(f"Schema not found for db_id={db_id}")

        schema = schemas_by_db_id[db_id]
        schema_text = format_schema_as_create_tables(schema)

        text = build_prompt(
            question=question,
            schema_text=schema_text,
            sql=sql if include_sql_in_text else None,
        )

        processed.append(
            {
                "id": f"{split_name}_{idx:06d}",
                "split": split_name,
                "db_id": db_id,
                "question": question,
                "schema": schema_text,
                "sql": sql,
                "text": text,
            }
        )

    return processed


def save_jsonl(items: List[Dict[str, Any]], path: Path):
    with open(path, "w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_json(items: List[Dict[str, Any]], path: Path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(items, file, ensure_ascii=False, indent=2)


def main():
    tables = load_json(SPIDER_DIR / "tables.json")
    schemas_by_db_id = {schema["db_id"]: schema for schema in tables}

    train_spider = load_json(SPIDER_DIR / "train_spider.json")

    train_others_path = SPIDER_DIR / "train_others.json"
    if train_others_path.exists():
        train_others = load_json(train_others_path)
    else:
        train_others = []

    dev = load_json(SPIDER_DIR / "dev.json")

    train_all = train_spider + train_others

    train_sft = build_sft_examples(
        examples=train_all,
        schemas_by_db_id=schemas_by_db_id,
        split_name="train",
        include_sql_in_text=True,
    )

    dev_sft = build_sft_examples(
        examples=dev,
        schemas_by_db_id=schemas_by_db_id,
        split_name="dev",
        include_sql_in_text=True,
    )

    dev_eval = build_sft_examples(
        examples=dev,
        schemas_by_db_id=schemas_by_db_id,
        split_name="dev",
        include_sql_in_text=False,
    )

    save_jsonl(train_sft, OUTPUT_DIR / "train_sft.jsonl")
    save_jsonl(dev_sft, OUTPUT_DIR / "dev_sft.jsonl")
    save_jsonl(dev_eval, OUTPUT_DIR / "dev_eval_prompts.jsonl")

    save_json(train_sft[:100], OUTPUT_DIR / "train_preview_100.json")
    save_json(dev_sft[:100], OUTPUT_DIR / "dev_preview_100.json")

    stats = {
        "train_examples": len(train_sft),
        "dev_examples": len(dev_sft),
        "unique_train_databases": len(set(item["db_id"] for item in train_sft)),
        "unique_dev_databases": len(set(item["db_id"] for item in dev_sft)),
        "train_output": str(OUTPUT_DIR / "train_sft.jsonl"),
        "dev_output": str(OUTPUT_DIR / "dev_sft.jsonl"),
        "dev_eval_prompts_output": str(OUTPUT_DIR / "dev_eval_prompts.jsonl"),
    }

    with open(OUTPUT_DIR / "stats.json", "w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()