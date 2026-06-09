#!/usr/bin/env python3
"""
analyze_sql_predictions.py
--------------------------
Detailed error analysis for Text-to-SQL QLoRA fine-tuning results.

Input file format (test_pred_gold.txt):
    <predicted SQL>
    <gold SQL>
    <empty line>
    <predicted SQL>
    ...

Usage:
    python analyze_sql_predictions.py test_pred_gold.txt
    python analyze_sql_predictions.py test_pred_gold.txt --output-dir ./results
"""

import re
import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import sqlparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from tabulate import tabulate

# ══════════════════════════════════════════════════════════════════════════════
# FILE PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_two_files(pred_path: str, gold_path: str) -> list[dict]:
    """
    Read predictions and gold answers from two separate files.
    Each file has one query per line.
    Gold lines may have a DB name appended after 2+ spaces — it is stripped.
    """
    with open(pred_path, "r", encoding="utf-8") as f:
        preds = [l.rstrip("\n") for l in f if l.strip()]

    with open(gold_path, "r", encoding="utf-8") as f:
        # strip trailing DB name: "SELECT ...   db_name"  →  "SELECT ..."
        golds = [re.split(r"\t|\s{2,}", l.rstrip("\n"))[0].strip()
                 for l in f if l.strip()]

    if len(preds) != len(golds):
        print(f"  ⚠  Length mismatch: {len(preds)} preds vs {len(golds)} golds")

    return [
        {"id": i, "pred": p, "gold": g}
        for i, (p, g) in enumerate(zip(preds, golds))
    ]

# ══════════════════════════════════════════════════════════════════════════════
# SQL NORMALISATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def normalize(sql: str) -> str:
    sql = sql.strip().lower()
    sql = re.sub(r"\s+", " ", sql)
    sql = sql.rstrip(";").strip()
    sql = re.sub(r'"([^"]*)"', r"'\1'", sql)
    sql = re.sub(r"`([^`]*)`", r"\1", sql)
    sql = re.sub(r"'(\d+(?:\.\d+)?)'", r"\1", sql)   # '2014' → 2014  ← ADD THIS
    return sql
def _resolve_aliases(sql: str) -> str:
    """Replace T1/T2-style aliases with the real table name they refer to."""
    alias_map = {}

    # catch: FROM table AS t1, JOIN table AS t1, FROM table t1, JOIN table t1
    for m in re.finditer(
        r"\b(?:from|join)\s+(\w+)\s+(?:as\s+)?(\w+)\b", sql, re.IGNORECASE
    ):
        table = m.group(1).lower()
        alias = m.group(2).lower()
        if alias not in SQL_KEYWORDS and table not in SQL_KEYWORDS:
            alias_map[alias] = table

    if not alias_map:
        return sql

    # replace alias.column → table.column
    def replace_ref(m):
        a = m.group(1).lower()
        return f"{alias_map.get(a, a)}.{m.group(2)}"

    sql = re.sub(r"\b(\w+)\.(\w+)\b", replace_ref, sql)

    # remove alias declarations (concert as t1 → concert, concert t1 → concert)
    for alias, table in alias_map.items():
        sql = re.sub(
            rf"\b({re.escape(table)})\s+(?:as\s+)?{re.escape(alias)}\b",
            r"\1", sql, flags=re.IGNORECASE,
        )

    return re.sub(r"\s+", " ", sql).strip()
def canonicalize(sql: str) -> str:
    sql = normalize(sql)
    sql = _resolve_aliases(sql)          # ← ADD: resolve T1/T2 before comparing
    m = re.match(r"^(select\s+)(.*?)(\s+from\b.*)", sql, re.IGNORECASE | re.DOTALL)
    if m:
        cols = [c.strip() for c in m.group(2).split(",")]
        sql  = m.group(1) + " , ".join(sorted(cols)) + m.group(3)
    return sql
def strip_string_literals(sql: str) -> str:
    """Replace quoted string content with placeholder (keeps structure)."""
    sql = re.sub(r"'[^']*'", "'?'", sql)
    sql = re.sub(r'"[^"]*"', '"?"', sql)
    return sql


# ══════════════════════════════════════════════════════════════════════════════
# SQL FEATURE EXTRACTORS
# ══════════════════════════════════════════════════════════════════════════════

# ------------ TABLES ----------------------------------------------------------

def extract_tables(sql: str) -> set[str]:
    """Extract table names from FROM and JOIN clauses."""
    sql = normalize(sql)
    tables = set()

    # FROM clause: FROM t1, t2, ...  (stop before WHERE/JOIN/GROUP/ORDER/etc.)
    from_re = re.compile(
        r"\bfrom\s+((?:[\w]+(?:\s+(?:as\s+)?\w+)?\s*,\s*)*[\w]+(?:\s+(?:as\s+)?\w+)?)"
        r"(?=\s+(?:join|where|group|order|having|limit|union|intersect|except|$)|\s*$)",
        re.IGNORECASE,
    )
    m = from_re.search(sql)
    if m:
        for part in m.group(1).split(","):
            # strip alias
            tbl = re.split(r"\s+", part.strip())[0].strip()
            if re.match(r"^\w+$", tbl) and tbl not in SQL_KEYWORDS:
                tables.add(tbl)

    # JOIN clauses
    for m in re.finditer(r"\bjoin\s+(\w+)", sql, re.IGNORECASE):
        tbl = m.group(1).lower()
        if tbl not in SQL_KEYWORDS:
            tables.add(tbl)

    return tables


# ------------ JOINS -----------------------------------------------------------

def extract_joins(sql: str) -> list[str]:
    """Return list of normalised join types (e.g. 'inner join', 'left join')."""
    sql = normalize(sql)
    joins = []
    for m in re.finditer(
        r"\b((?:inner|left(?:\s+outer)?|right(?:\s+outer)?|full(?:\s+outer)?|cross)\s+)?join\b",
        sql, re.IGNORECASE,
    ):
        qualifier = (m.group(1) or "").strip().lower()
        qualifier = re.sub(r"\s+outer", "", qualifier).strip()
        joins.append(f"{qualifier} join".strip() if qualifier else "join")
    return joins


# ------------ AGGREGATIONS ---------------------------------------------------

def extract_aggregations(sql: str) -> list[str]:
    """Return list of aggregate function names used."""
    return [m.group(1).lower()
            for m in re.finditer(r"\b(count|sum|avg|min|max)\s*\(", sql, re.IGNORECASE)]


# ------------ COLUMNS --------------------------------------------------------

def extract_all_column_refs(sql: str) -> set[str]:
    """Extract all explicit table.column references."""
    sql = normalize(strip_string_literals(sql))
    return {m.group(1).lower() for m in re.finditer(r"\b\w+\.(\w+)\b", sql)}


def extract_bare_columns(sql: str) -> set[str]:
    """
    Extract bare column names (no table prefix) that are not SQL keywords,
    numeric literals, or aggregate functions.
    """
    sql = normalize(strip_string_literals(sql))
    candidates = set()
    for m in re.finditer(r"\b([a-z_][a-z0-9_]*)\b", sql):
        w = m.group(1)
        if w not in SQL_KEYWORDS and not w.isdigit():
            candidates.add(w)
    return candidates


def extract_select_cols(sql: str) -> set[str]:
    """Extract column tokens from the SELECT clause."""
    sql = normalize(sql)
    m = re.search(r"\bselect\s+(.+?)\s+\bfrom\b", sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return set()

    sel = m.group(1)
    # remove aggregate wrappers
    sel = re.sub(r"\b(count|sum|avg|min|max|distinct)\s*\(", "", sel, flags=re.IGNORECASE)
    sel = sel.replace(")", "")
    cols = set()
    for part in sel.split(","):
        part = re.sub(r"\bas\s+\w+$", "", part.strip(), flags=re.IGNORECASE).strip()
        if "." in part:
            col = part.split(".")[-1].strip()
        else:
            col = part.strip()
        col = re.sub(r"[^a-z0-9_*]", "", col)
        if col and col not in SQL_KEYWORDS:
            cols.add(col)
    return cols


# ------------ CLAUSES --------------------------------------------------------

CLAUSE_RE = {
    "where":    re.compile(r"\bwhere\b",        re.IGNORECASE),
    "group_by": re.compile(r"\bgroup\s+by\b",   re.IGNORECASE),
    "order_by": re.compile(r"\border\s+by\b",   re.IGNORECASE),
    "having":   re.compile(r"\bhaving\b",       re.IGNORECASE),
    "limit":    re.compile(r"\blimit\b",        re.IGNORECASE),
    "distinct": re.compile(r"\bselect\s+distinct\b", re.IGNORECASE),
}


def has_clause(sql: str, clause: str) -> bool:
    return bool(CLAUSE_RE[clause].search(sql))


def extract_limit_value(sql: str) -> int | None:
    m = re.search(r"\blimit\s+(\d+)", sql, re.IGNORECASE)
    return int(m.group(1)) if m else None


def extract_order_directions(sql: str) -> list[str]:
    m = re.search(r"\border\s+by\s+(.+?)(?:\s+limit\b|\s*$)", sql,
                  re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    dirs = []
    for part in m.group(1).split(","):
        dirs.append("desc" if re.search(r"\bdesc\b", part, re.IGNORECASE) else "asc")
    return dirs


def extract_order_by_cols(sql: str) -> set[str]:
    sql = normalize(sql)
    m = re.search(r"\border\s+by\s+(.+?)(?:\s+limit\b|\s*$)", sql, re.IGNORECASE)
    if not m:
        return set()
    cols = set()
    for part in m.group(1).split(","):
        part = re.sub(r"\b(asc|desc)\b", "", part, flags=re.IGNORECASE).strip()
        col = part.split(".")[-1].strip() if "." in part else part.strip()
        col = re.sub(r"[^a-z0-9_]", "", col)
        if col:
            cols.add(col)
    return cols


def extract_group_by_cols(sql: str) -> set[str]:
    sql = normalize(sql)
    m = re.search(r"\bgroup\s+by\s+(.+?)(?:\s+having\b|\s+order\b|\s+limit\b|\s*$)",
                  sql, re.IGNORECASE)
    if not m:
        return set()
    cols = set()
    for part in m.group(1).split(","):
        col = part.strip().split(".")[-1].strip() if "." in part else part.strip()
        col = re.sub(r"[^a-z0-9_]", "", col)
        if col:
            cols.add(col)
    return cols


def extract_where_operators(sql: str) -> list[str]:
    m = re.search(
        r"\bwhere\s+(.+?)(?:\s+group\s+by|\s+order\s+by|\s+having|\s+limit\b|\s*$)",
        sql, re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return []
    chunk = m.group(1)
    ops = re.findall(
        r"\b(like|not\s+like|not\s+in|in|between|is\s+not\s+null|is\s+null|<>|!=|>=|<=|>|<|=)\b",
        chunk, re.IGNORECASE,
    )
    return [re.sub(r"\s+", " ", op.lower()) for op in ops]


# ------------ SET OPERATIONS & SUBQUERIES ------------------------------------
def extract_set_ops(sql: str) -> list[str]:
    """Extract set operations in order. UNION ALL is distinct from UNION."""
    sql = normalize(sql)
    ops = []
    for m in re.finditer(r"\b(union\s+all|union|intersect|except)\b", sql, re.IGNORECASE):
        ops.append(re.sub(r"\s+", " ", m.group(1).lower()))
    return ops
def split_set_op_branches(sql: str) -> list[str]:
    """Split a SQL query into its individual SELECT branches."""
    parts = re.split(r"\b(?:union\s+all|union|intersect|except)\b", sql, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def count_selects(sql: str) -> int:
    return len(re.findall(r"\bselect\b", sql, re.IGNORECASE))


def has_nested_select(sql: str) -> bool:
    return count_selects(sql) > 1


# ══════════════════════════════════════════════════════════════════════════════
# KEYWORDS BLACKLIST  (prevent keyword tokens being flagged as column names)
# ══════════════════════════════════════════════════════════════════════════════

SQL_KEYWORDS = {
    "select", "from", "where", "join", "on", "and", "or", "not", "in",
    "like", "is", "null", "as", "by", "group", "order", "having", "limit",
    "distinct", "inner", "left", "right", "full", "outer", "cross",
    "union", "intersect", "except", "all", "between", "case", "when",
    "then", "else", "end", "asc", "desc", "count", "sum", "avg", "min",
    "max", "exists", "any", "some", "into", "values", "set", "update",
    "delete", "insert", "create", "drop", "alter", "table", "index",
    "with", "recursive", "over", "partition", "row", "rows", "range",
    "unbounded", "preceding", "following", "current", "offset", "fetch",
    "first", "next", "only", "ties", "rollup", "cube", "grouping",
    "sets", "filter", "within", "collate", "cast", "convert", "trim",
    "leading", "trailing", "both", "substring", "extract", "position",
    "overlay", "normalize", "translate", "char", "varchar", "int",
    "integer", "float", "real", "double", "precision", "boolean", "date",
    "time", "timestamp", "interval", "true", "false", "unknown",
    "primary", "foreign", "key", "references", "unique", "default",
    "check", "constraint", "natural", "using", "lateral", "apply",
    "pivot", "unpivot", "for", "of", "at", "zone", "without",
}


# ══════════════════════════════════════════════════════════════════════════════
# CORE ERROR COMPARATOR
# ══════════════════════════════════════════════════════════════════════════════

ERROR_CATEGORIES = {
    # ── Structure ──────────────────────────────────────────────────────
    "missing_join":          "Structure",
    "extra_join":            "Structure",
    "wrong_join_type":       "Structure",
    "wrong_join_count":      "Structure",
    "missing_subquery":      "Structure",
    "extra_subquery":        "Structure",
    "missing_set_operation": "Structure",
    "extra_set_operation":   "Structure",
    "wrong_set_operation":   "Structure",
    # ── Columns ────────────────────────────────────────────────────────
    "hallucinated_column":   "Columns",
    "missing_column":        "Columns",
    "wrong_select_column":   "Columns",
    "missing_select_column": "Columns",
    # ── Tables ─────────────────────────────────────────────────────────
    "hallucinated_table":    "Tables",
    "missing_table":         "Tables",
    # ── Aggregation ────────────────────────────────────────────────────
    "missing_aggregation":   "Aggregation",
    "extra_aggregation":     "Aggregation",
    "wrong_aggregation":     "Aggregation",
    # ── Clauses ────────────────────────────────────────────────────────
    "missing_where":         "Clauses",
    "extra_where":           "Clauses",
    "missing_group_by":      "Clauses",
    "extra_group_by":        "Clauses",
    "missing_order_by":      "Clauses",
    "extra_order_by":        "Clauses",
    "missing_having":        "Clauses",
    "extra_having":          "Clauses",
    "missing_limit":         "Clauses",
    "extra_limit":           "Clauses",
    "missing_distinct":      "Clauses",
    "extra_distinct":        "Clauses",
    # ── Conditions ─────────────────────────────────────────────────────
    "wrong_order_direction": "Conditions",
    "wrong_order_by_column": "Conditions",
    "wrong_group_by_column": "Conditions",
    "wrong_where_operator":  "Conditions",
    "wrong_limit_value":     "Conditions",
    # ── Other ──────────────────────────────────────────────────────────
    "other_mismatch":        "Other",
}


def compare_pair(pred: str, gold: str) -> tuple[list[str], dict]:
    """
    Compare a predicted SQL against the gold SQL.
    Returns (list_of_error_tags, details_dict).
    Empty error list means exact match.
    """
    errors: list[str] = []
    details: dict = {}

    p = normalize(pred)
    g = normalize(gold)

    if canonicalize(p) == canonicalize(g):
        return [], {"exact_match": True}

    details["exact_match"] = False

    # ── Tables ──────────────────────────────────────────────────────────────
    p_tbls, g_tbls = extract_tables(p), extract_tables(g)
    missing_tbls = g_tbls - p_tbls
    halluc_tbls  = p_tbls - g_tbls
    if missing_tbls:
        errors.append("missing_table")
        details["missing_tables"] = sorted(missing_tbls)
    if halluc_tbls:
        errors.append("hallucinated_table")
        details["hallucinated_tables"] = sorted(halluc_tbls)

    # ── Columns ─────────────────────────────────────────────────────────────
    p_cols = extract_all_column_refs(p)
    g_cols = extract_all_column_refs(g)
    halluc_cols = p_cols - g_cols
    missing_cols = g_cols - p_cols

    if halluc_cols:
        errors.append("hallucinated_column")
        details["hallucinated_columns"] = sorted(halluc_cols)
    if missing_cols:
        errors.append("missing_column")
        details["missing_columns"] = sorted(missing_cols)

    # SELECT clause columns
    p_sel = extract_select_cols(p)
    g_sel = extract_select_cols(g)
    if p_sel != g_sel:
        extra_sel   = p_sel - g_sel
        missing_sel = g_sel - p_sel
        if extra_sel:
            errors.append("wrong_select_column")
            details["extra_select_cols"] = sorted(extra_sel)
        if missing_sel:
            errors.append("missing_select_column")
            details["missing_select_cols"] = sorted(missing_sel)

    # ── Joins ───────────────────────────────────────────────────────────────
    p_joins = extract_joins(p)
    g_joins = extract_joins(g)
    if p_joins != g_joins:
        details["pred_joins"] = p_joins
        details["gold_joins"] = g_joins
        if not p_joins and g_joins:
            errors.append("missing_join")
        elif p_joins and not g_joins:
            errors.append("extra_join")
        elif len(p_joins) != len(g_joins):
            errors.append("wrong_join_count")
        else:
            errors.append("wrong_join_type")

    # ── Aggregations ────────────────────────────────────────────────────────
    p_aggs = Counter(extract_aggregations(p))
    g_aggs = Counter(extract_aggregations(g))
    if p_aggs != g_aggs:
        details["pred_aggs"] = dict(p_aggs)
        details["gold_aggs"] = dict(g_aggs)
        if not p_aggs and g_aggs:
            errors.append("missing_aggregation")
        elif p_aggs and not g_aggs:
            errors.append("extra_aggregation")
        else:
            errors.append("wrong_aggregation")

    # ── Clause presence ─────────────────────────────────────────────────────
    for clause in ["where", "group_by", "order_by", "having", "limit", "distinct"]:
        ph = has_clause(p, clause)
        gh = has_clause(g, clause)
        if ph and not gh:
            errors.append(f"extra_{clause}")
        elif not ph and gh:
            errors.append(f"missing_{clause}")

    # ── ORDER BY details ────────────────────────────────────────────────────
    if has_clause(p, "order_by") and has_clause(g, "order_by"):
        p_dirs = extract_order_directions(p)
        g_dirs = extract_order_directions(g)
        if p_dirs != g_dirs:
            errors.append("wrong_order_direction")
            details["pred_order_dir"] = p_dirs
            details["gold_order_dir"] = g_dirs

        p_ob_cols = extract_order_by_cols(p)
        g_ob_cols = extract_order_by_cols(g)
        if p_ob_cols != g_ob_cols:
            errors.append("wrong_order_by_column")
            details["pred_order_cols"] = sorted(p_ob_cols)
            details["gold_order_cols"] = sorted(g_ob_cols)

    # ── GROUP BY columns ────────────────────────────────────────────────────
    if has_clause(p, "group_by") and has_clause(g, "group_by"):
        if extract_group_by_cols(p) != extract_group_by_cols(g):
            errors.append("wrong_group_by_column")

    # ── LIMIT value ─────────────────────────────────────────────────────────
    if has_clause(p, "limit") and has_clause(g, "limit"):
        pv, gv = extract_limit_value(p), extract_limit_value(g)
        if pv != gv:
            errors.append("wrong_limit_value")
            details["pred_limit"] = pv
            details["gold_limit"] = gv

    # ── WHERE operators ─────────────────────────────────────────────────────
    if has_clause(p, "where") and has_clause(g, "where"):
        if Counter(extract_where_operators(p)) != Counter(extract_where_operators(g)):
            errors.append("wrong_where_operator")

    # ── Set operations ──────────────────────────────────────────────────────
    p_sops = extract_set_ops(p)
    g_sops = extract_set_ops(g)
    if p_sops != g_sops:
        details["pred_set_ops"] = p_sops
        details["gold_set_ops"] = g_sops
        if not p_sops and g_sops:
            errors.append("missing_set_operation")
        elif p_sops and not g_sops:
            errors.append("extra_set_operation")
        elif len(p_sops) != len(g_sops):
            errors.append("wrong_set_op_count")
        else:
            # same count but different types — flag each mismatch specifically
            for ps, gs in zip(p_sops, g_sops):
                if ps != gs:
                    errors.append(f"set_op_{gs.replace(' ','_')}_as_{ps.replace(' ','_')}")
            errors.append("wrong_set_op_type")

    # Branch-level: even if set op type is right, are the SELECT branches correct?
    if g_sops:
        g_branches = split_set_op_branches(g)
        p_branches = split_set_op_branches(p)
        details["gold_branch_count"] = len(g_branches)
        details["pred_branch_count"] = len(p_branches)
        matched = sum(1 for gb in g_branches if any(normalize(pb) == gb for pb in p_branches))
        details["matching_branches"] = matched
        if matched < len(g_branches):
            errors.append("wrong_set_op_branch")
    # ── Subqueries ──────────────────────────────────────────────────────────
    p_sq = count_selects(p)
    g_sq = count_selects(g)
    if p_sq != g_sq:
        if p_sq < g_sq:
            errors.append("missing_subquery")
        else:
            errors.append("extra_subquery")
        details["pred_selects"] = p_sq
        details["gold_selects"] = g_sq
    if not errors:
        return [], {"exact_match": True}

    return errors, details


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def analyze_all(pairs: list[dict]) -> tuple[list[dict], Counter]:
    results = []
    error_counts: Counter = Counter()

    for pair in pairs:
        errors, details = compare_pair(pair["pred"], pair["gold"])
        is_correct = (errors == [])
        results.append({
            "id":         pair["id"],
            "pred":       pair["pred"],
            "gold":       pair["gold"],
            "errors":     errors,
            "details":    details,
            "is_correct": is_correct,
        })
        if is_correct:
            error_counts["__exact_match__"] += 1
        else:
            for e in errors:
                error_counts[e] += 1

    return results, error_counts

def analyze_set_operations(results: list[dict]) -> dict:
    """Deep-dive stats for queries that involve set operations."""
    set_op_results = [r for r in results if extract_set_ops(r["gold"])]

    per_op   = defaultdict(lambda: {"total": 0, "correct": 0, "wrong_as": Counter()})
    confusion = defaultdict(Counter)   # gold_op → pred_op → count

    for r in set_op_results:
        gold_ops = extract_set_ops(r["gold"])
        pred_ops = extract_set_ops(r["pred"])
        primary_gold = gold_ops[0] if gold_ops else "none"
        primary_pred = pred_ops[0] if pred_ops else "none"

        per_op[primary_gold]["total"] += 1
        confusion[primary_gold][primary_pred] += 1

        if gold_ops == pred_ops:
            per_op[primary_gold]["correct"] += 1
        else:
            per_op[primary_gold]["wrong_as"][primary_pred] += 1

    # branch-level accuracy across all set-op queries
    total_branches   = 0
    matched_branches = 0
    for r in set_op_results:
        g_branches = split_set_op_branches(normalize(r["gold"]))
        p_branches = split_set_op_branches(normalize(r["pred"]))
        total_branches += len(g_branches)
        matched_branches += sum(
            1 for gb in g_branches if any(normalize(pb) == gb for pb in p_branches)
        )

    return {
        "total":          len(set_op_results),
        "per_op":         dict(per_op),
        "confusion":      {k: dict(v) for k, v in confusion.items()},
        "total_branches": total_branches,
        "matched_branches": matched_branches,
    }

# ══════════════════════════════════════════════════════════════════════════════
# TEXT REPORT
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(results: list[dict], error_counts: Counter) -> str:
    total     = len(results)
    correct   = sum(1 for r in results if r["is_correct"])
    incorrect = total - correct

    lines = []
    SEP  = "═" * 72
    SEP2 = "─" * 72

    lines += [SEP, "  TEXT-TO-SQL PREDICTION ANALYSIS REPORT", SEP, ""]

    # ── Overview ────────────────────────────────────────────────────────────
    lines += [
        f"  Total examples  : {total:,}",
        f"  Exact matches   : {correct:,}  ({100*correct/total:.1f}%)",
        f"  Wrong           : {incorrect:,}  ({100*incorrect/total:.1f}%)",
        "",
    ]

    # ── Per-error stats table ────────────────────────────────────────────────
    lines += [SEP2, "  ERROR TYPE BREAKDOWN", SEP2]

    rows = []
    for err, cnt in error_counts.most_common():
        if err == "__exact_match__":
            continue
        pct_err   = 100 * cnt / incorrect if incorrect else 0
        pct_total = 100 * cnt / total
        cat       = ERROR_CATEGORIES.get(err, "Other")
        rows.append([err, cat, cnt, f"{pct_err:.1f}%", f"{pct_total:.1f}%"])

    lines.append(
        tabulate(
            rows,
            headers=["Error Type", "Category", "Count", "% of Wrong", "% of Total"],
            tablefmt="simple",
            colalign=("left", "left", "right", "right", "right"),
        )
    )
    lines.append("")

    # ── Category-level summary ───────────────────────────────────────────────
    lines += [SEP2, "  CATEGORY SUMMARY", SEP2]
    cat_totals: Counter = Counter()
    for err, cnt in error_counts.items():
        if err == "__exact_match__":
            continue
        cat_totals[ERROR_CATEGORIES.get(err, "Other")] += cnt

    cat_rows = []
    for cat, cnt in cat_totals.most_common():
        cat_rows.append([cat, cnt, f"{100*cnt/max(incorrect,1):.1f}%"])
    lines.append(
        tabulate(cat_rows, headers=["Category", "Count", "% of Wrong"],
                 tablefmt="simple", colalign=("left", "right", "right"))
    )
    lines.append("")

    # ── Errors-per-example distribution ─────────────────────────────────────
    lines += [SEP2, "  ERRORS PER INCORRECT EXAMPLE", SEP2]
    dist = Counter(len(r["errors"]) for r in results if not r["is_correct"])
    dist_rows = [
        [n, cnt, f"{100*cnt/max(incorrect,1):.1f}%"]
        for n, cnt in sorted(dist.items())
    ]
    lines.append(
        tabulate(dist_rows, headers=["# Errors", "Examples", "% of Wrong"],
                 tablefmt="simple", colalign=("right", "right", "right"))
    )
    lines.append("")

    # ── Top co-occurring error pairs ─────────────────────────────────────────
    lines += [SEP2, "  TOP CO-OCCURRING ERROR PAIRS", SEP2]
    co: Counter = Counter()
    for r in results:
        if not r["is_correct"]:
            errs = sorted(r["errors"])
            for i in range(len(errs)):
                for j in range(i + 1, len(errs)):
                    co[(errs[i], errs[j])] += 1
    if co:
        co_rows = [[f"{a} + {b}", cnt] for (a, b), cnt in co.most_common(10)]
        lines.append(
            tabulate(co_rows, headers=["Error Pair", "Count"],
                     tablefmt="simple", colalign=("left", "right"))
        )
    else:
        lines.append("  (no co-occurrences found)")
    lines.append("")

    # ── Sample errors ────────────────────────────────────────────────────────
    lines += [SEP2, "  SAMPLE ERRORS — TOP 8 ERROR TYPES", SEP2]
    top8 = [e for e, _ in error_counts.most_common() if e != "__exact_match__"][:8]
    for err in top8:
        lines.append(f"\n  [{err.upper()}]")
        samples = [r for r in results if err in r["errors"]][:3]
        for k, s in enumerate(samples, 1):
            lines.append(f"    Example {k}:")
            lines.append(f"      PRED : {s['pred']}")
            lines.append(f"      GOLD : {s['gold']}")
            rel = {k: v for k, v in s["details"].items() if k != "exact_match"}
            if rel:
                for dk, dv in rel.items():
                    lines.append(f"      {dk:<22}: {dv}")
# ── Set operation deep-dive ──────────────────────────────────────────────
    soa = analyze_set_operations(results)
    lines += [SEP2, "  SET OPERATION ANALYSIS", SEP2]

    if soa["total"] == 0:
        lines.append("  No set-operation queries found in gold.")
    else:
        lines.append(f"  Queries with set operations : {soa['total']}  "
                     f"({100*soa['total']/total:.1f}% of total)\n")

        # per-type accuracy
        acc_rows = []
        for op, stats in sorted(soa["per_op"].items()):
            acc = 100 * stats["correct"] / stats["total"] if stats["total"] else 0
            wrong_as = ", ".join(
                f"{k}×{v}" for k, v in stats["wrong_as"].most_common(3)
            ) or "—"
            acc_rows.append([op.upper(), stats["total"], stats["correct"],
                             f"{acc:.1f}%", wrong_as])
        lines.append(
            tabulate(acc_rows,
                     headers=["Set Op", "Total", "Correct", "Accuracy", "Wrong as (top 3)"],
                     tablefmt="simple", colalign=("left","right","right","right","left"))
        )
        lines.append("")

        # confusion matrix
        all_ops = sorted({op for ops in soa["confusion"].values() for op in ops}
                         | set(soa["confusion"].keys()))
        if len(all_ops) > 1:
            lines.append("  Confusion matrix  (rows = gold, cols = predicted):\n")
            header = ["gold \\ pred"] + [o.upper() for o in all_ops]
            conf_rows = []
            for gold_op in sorted(soa["confusion"]):
                row = [gold_op.upper()]
                for pred_op in all_ops:
                    row.append(soa["confusion"][gold_op].get(pred_op, 0))
                conf_rows.append(row)
            lines.append(tabulate(conf_rows, headers=header,
                                  tablefmt="simple", colalign=("left",) + ("right",)*len(all_ops)))
            lines.append("")

        # branch accuracy
        if soa["total_branches"] > 0:
            ba = 100 * soa["matched_branches"] / soa["total_branches"]
            lines.append(f"  Branch-level accuracy : {soa['matched_branches']} / "
                         f"{soa['total_branches']} branches correct  ({ba:.1f}%)")
            lines.append("  (each SELECT branch within a set op scored independently)\n")

        # samples for worst-performing op
        worst_op = min(
            soa["per_op"], key=lambda k: soa["per_op"][k]["correct"] / soa["per_op"][k]["total"]
        )
        lines.append(f"  Sample errors — {worst_op.upper()} (lowest accuracy):")
        samples = [r for r in results
                   if not r["is_correct"] and worst_op in extract_set_ops(r["gold"])][:3]
        for k, s in enumerate(samples, 1):
            lines.append(f"    {k}. PRED: {s['pred']}")
            lines.append(f"       GOLD: {s['gold']}")
    lines.append("")
    lines.append(SEP)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = {
    "Structure":   "#4e79a7",
    "Columns":     "#f28e2b",
    "Tables":      "#e15759",
    "Aggregation": "#76b7b2",
    "Clauses":     "#59a14f",
    "Conditions":  "#edc948",
    "Other":       "#b07aa1",
}

def _color_for(err: str) -> str:
    cat = ERROR_CATEGORIES.get(err, "Other")
    return PALETTE.get(cat, "#aaaaaa")


def generate_visualizations(results: list[dict], error_counts: Counter, output_dir: Path):
    total     = len(results)
    correct   = sum(1 for r in results if r["is_correct"])
    incorrect = total - correct

    top_errors = [(e, c) for e, c in error_counts.most_common(20)
                  if e != "__exact_match__"]

    # ── Figure 1: overview + top errors ─────────────────────────────────────
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle("Text-to-SQL Prediction Analysis", fontsize=15, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # Pie
    ax0 = fig.add_subplot(gs[0])
    ax0.pie(
        [correct, incorrect],
        labels=[f"Correct\n{correct:,} ({100*correct/total:.1f}%)",
                f"Incorrect\n{incorrect:,} ({100*incorrect/total:.1f}%)"],
        colors=["#59a14f", "#e15759"],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 11},
        autopct="",
    )
    ax0.set_title("Overall Accuracy", fontsize=12, pad=10)

    # Horizontal bar: top errors
    ax1 = fig.add_subplot(gs[1])
    if top_errors:
        labels_raw  = [e for e, _ in top_errors]
        values      = [c for _, c in top_errors]
        colors      = [_color_for(e) for e in labels_raw]
        labels_disp = [e.replace("_", " ") for e in labels_raw]
        y_pos = np.arange(len(labels_disp))

        bars = ax1.barh(y_pos, values, color=colors, edgecolor="white", height=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels_disp, fontsize=8.5)
        ax1.invert_yaxis()
        ax1.set_xlabel("Count", fontsize=10)
        ax1.set_title("Error Type Distribution (Top 20)", fontsize=12, pad=10)

        for bar, val in zip(bars, values):
            ax1.text(bar.get_width() + max(values) * 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     str(val), va="center", fontsize=8)

        # Legend for categories
        legend_patches = [
            plt.Rectangle((0, 0), 1, 1, color=c, label=cat)
            for cat, c in PALETTE.items()
        ]
        ax1.legend(handles=legend_patches, fontsize=7.5, loc="lower right",
                   title="Category", title_fontsize=8, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_dir / "fig1_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: category breakdown ────────────────────────────────────────
    cat_totals: Counter = Counter()
    for e, c in error_counts.items():
        if e != "__exact_match__":
            cat_totals[ERROR_CATEGORIES.get(e, "Other")] += c

    fig2, ax = plt.subplots(figsize=(8, 5))
    cats   = [k for k, _ in cat_totals.most_common()]
    values = [cat_totals[c] for c in cats]
    colors = [PALETTE.get(c, "#aaaaaa") for c in cats]
    bars   = ax.bar(cats, values, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values)*0.01,
                str(val), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Total error count (multi-counted per example)")
    ax.set_title("Errors by Category", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", labelsize=10)
    plt.tight_layout()
    fig2.savefig(output_dir / "fig2_categories.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ── Figure 3: clause errors (missing vs extra) ───────────────────────────
    clauses = ["where", "group_by", "order_by", "having", "limit", "distinct"]
    miss_v  = [error_counts.get(f"missing_{c}", 0) for c in clauses]
    extra_v = [error_counts.get(f"extra_{c}", 0)   for c in clauses]

    x   = np.arange(len(clauses))
    w   = 0.35
    fig3, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, miss_v,  w, label="Missing in pred", color="#e15759", edgecolor="white")
    b2 = ax.bar(x + w/2, extra_v, w, label="Extra in pred",   color="#f28e2b", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ").upper() for c in clauses], fontsize=10)
    ax.set_ylabel("Count")
    ax.set_title("Clause Errors: Missing vs Extra", fontsize=12, fontweight="bold")
    ax.legend()
    for bar in list(b1) + list(b2):
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig3.savefig(output_dir / "fig3_clauses.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # ── Figure 4: errors-per-example histogram ───────────────────────────────
    err_counts_per_ex = [len(r["errors"]) for r in results if not r["is_correct"]]
    if err_counts_per_ex:
        fig4, ax = plt.subplots(figsize=(7, 4))
        max_err = max(err_counts_per_ex)
        bins    = np.arange(0.5, max_err + 1.5, 1)
        ax.hist(err_counts_per_ex, bins=bins, color="#4e79a7", edgecolor="white", rwidth=0.8)
        ax.set_xlabel("# Distinct error types per example")
        ax.set_ylabel("# Examples")
        ax.set_xticks(range(1, max_err + 1))
        ax.set_title("Error Density per Incorrect Example", fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig4.savefig(output_dir / "fig4_error_density.png", dpi=150, bbox_inches="tight")
        plt.close(fig4)
    
    # ── Figure 5: set operation accuracy + confusion ─────────────────────────
    soa = analyze_set_operations(results)
    if soa["total"] > 0 and soa["per_op"]:
        ops      = sorted(soa["per_op"].keys())
        totals   = [soa["per_op"][o]["total"]   for o in ops]
        corrects = [soa["per_op"][o]["correct"]  for o in ops]
        wrongs   = [t - c for t, c in zip(totals, corrects)]

        fig5, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig5.suptitle("Set Operation Analysis", fontsize=13, fontweight="bold")

        # stacked bar: correct vs wrong per op type
        x = np.arange(len(ops))
        axes[0].bar(x, corrects, color="#59a14f", label="Correct",  edgecolor="white")
        axes[0].bar(x, wrongs, bottom=corrects, color="#e15759", label="Wrong", edgecolor="white")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([o.upper() for o in ops], fontsize=10)
        axes[0].set_ylabel("Query count")
        axes[0].set_title("Accuracy per Set Operation Type")
        axes[0].legend()
        for i, (c, t) in enumerate(zip(corrects, totals)):
            pct = 100 * c / t if t else 0
            axes[0].text(i, t + max(totals)*0.01, f"{pct:.0f}%", ha="center", fontsize=9)

        # confusion heatmap
        all_ops = sorted({o for v in soa["confusion"].values() for o in v} | set(ops))
        matrix  = np.array([
            [soa["confusion"].get(g, {}).get(p, 0) for p in all_ops]
            for g in all_ops
        ])
        im = axes[1].imshow(matrix, cmap="Blues")
        axes[1].set_xticks(range(len(all_ops)))
        axes[1].set_yticks(range(len(all_ops)))
        axes[1].set_xticklabels([o.upper() for o in all_ops], fontsize=9)
        axes[1].set_yticklabels([o.upper() for o in all_ops], fontsize=9)
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Gold")
        axes[1].set_title("Confusion Matrix")
        for i in range(len(all_ops)):
            for j in range(len(all_ops)):
                if matrix[i, j] > 0:
                    axes[1].text(j, i, str(int(matrix[i, j])),
                                 ha="center", va="center", fontsize=9,
                                 color="white" if matrix[i, j] > matrix.max()*0.6 else "black")
        plt.colorbar(im, ax=axes[1], shrink=0.8)

        plt.tight_layout()
        fig5.savefig(output_dir / "fig5_set_operations.png", dpi=150, bbox_inches="tight")
        plt.close(fig5)
    print(f"  Figures saved → {output_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# CSV / JSON EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_csv(results: list[dict], path: Path):
    rows = []
    for r in results:
        rows.append({
            "id":         r["id"],
            "is_correct": r["is_correct"],
            "errors":     "|".join(r["errors"]),
            "n_errors":   len(r["errors"]),
            "pred":       r["pred"],
            "gold":       r["gold"],
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def export_json(results: list[dict], error_counts: Counter, path: Path):
    summary = {k: v for k, v in error_counts.items() if k != "__exact_match__"}
    correct = sum(1 for r in results if r["is_correct"])
    out = {
        "total":         len(results),
        "correct":       correct,
        "incorrect":     len(results) - correct,
        "error_summary": summary,
        "results":       [
            {k: v for k, v in r.items()} for r in results
        ],
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Analyse Text-to-SQL predictions vs gold answers."
    )
    parser.add_argument("pred_file", nargs="?", help="C:/Users/user/Downloads/analysis_of_mistakes/gold.txt")
    parser.add_argument("gold_file", nargs="?", help="C:/Users/user/Downloads/analysis_of_mistakes/pred_1.txt")
    parser.add_argument(
        "--output-dir", "-o", default="./sql_analysis",
        help="Directory for output files (default: ./sql_analysis)"
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Skip matplotlib figures"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Input ──────────────────────────────────────────────────────────────
    if args.pred_file and args.gold_file:
        print(f"\n[1/4] Parsing  →  {args.pred_file}  +  {args.gold_file}")
        pairs = parse_two_files(args.pred_file, args.gold_file)
    else:
        print("\n[1/4] No input files given — generating demo data …")
        demo_pred = out_dir / "demo_pred.txt"
        demo_gold = out_dir / "demo_gold.txt"
        with open(demo_pred, "w") as fp, open(demo_gold, "w") as fg:
            for pred, gold in SYNTHETIC_PAIRS * 5:
                fp.write(pred + "\n")
                fg.write(gold + "     demo_db\n")   # simulates the DB-name suffix
        print(f"  Demo files written → {demo_pred}, {demo_gold}")
        pairs = parse_two_files(str(demo_pred), str(demo_gold))

    # ── Analysis ───────────────────────────────────────────────────────────
    print("[2/4] Analysing predictions …")
    results, error_counts = analyze_all(pairs)

    # ── Report ─────────────────────────────────────────────────────────────
    print("[3/4] Generating report …")
    report = generate_report(results, error_counts)
    report_path = out_dir / "analysis_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\n  Report saved → {report_path}")

    # ── Exports ────────────────────────────────────────────────────────────
    export_csv(results,  out_dir / "results.csv")
    export_json(results, error_counts, out_dir / "results.json")
    print(f"  CSV    saved → {out_dir / 'results.csv'}")
    print(f"  JSON   saved → {out_dir / 'results.json'}")

    # ── Figures ────────────────────────────────────────────────────────────
    if not args.no_viz:
        print("[4/4] Generating visualisations …")
        generate_visualizations(results, error_counts, out_dir)
    else:
        print("[4/4] Skipping visualisations (--no-viz)")

    print(f"\n✓ All outputs written to  {out_dir}/\n")


if __name__ == "__main__":
    main()