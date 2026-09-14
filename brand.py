"""
Choose a brand from the customer-support Twitter dataset.

Input:
    sample.csv (or the full Kaggle CSV)

Output:
    brand_comparison.csv
    thread_level_results.csv

What it does:
1. Reconstructs reply graphs from in_response_to_tweet_id + response_tweet_id.
2. Handles comma-separated response_tweet_id values (branching).
3. Sorts turns by created_at.
4. Identifies brands from outbound rows (inbound == False).
5. For each brand, measures:
      - thread count
      - average reconstructed thread length
      - in-thread assistance / likely resolution
      - DM deflection
      - customer acknowledgement
6. Runs an exploratory TF-IDF clustering signal for intent diversity.
7. Produces a ranked shortlist.

IMPORTANT:
This is a dataset-selection diagnostic, not the final evaluation methodology.
The outcome classifier is intentionally heuristic so its rules are inspectable.
"""

import argparse
import re
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


DM_RE = re.compile(
    r"\b(dm|direct message|private message|send us a dm|"
    r"message us privately|join us in dm)\b", re.I
)

POSITIVE_RE = re.compile(
    r"\b(thanks|thank you|cheers|working (now|ok|okay)|works (now|ok|okay)|"
    r"problem solved|solved|fixed|sorted|all good|got it|that worked|"
    r"appreciate (it|that)|brilliant thanks)\b", re.I
)

ACTION_RE = re.compile(
    r"\b(try|check|restart|reinstall|clear|delete|update|reset|follow|"
    r"click|open|select|use|log out|login|log in|change|turn off|"
    r"turn on|here'?s|here is|the answer|you (can|will)|"
    r"we (have|can|will))\b", re.I
)


def split_ids(value):
    """Turn NaN / '119249,119251' / '119249' into a list of string IDs."""
    if pd.isna(value):
        return []
    return [
        x.strip()
        for x in str(value).split(",")
        if x.strip() and x.strip().lower() != "nan"
    ]


def load_data(path):
    df = pd.read_csv(path)

    required = {
        "tweet_id", "author_id", "inbound", "created_at", "text",
        "response_tweet_id", "in_response_to_tweet_id"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for col in [
        "tweet_id", "author_id",
        "in_response_to_tweet_id", "response_tweet_id"
    ]:
        df[col] = df[col].astype("string").str.strip()

    df["inbound_bool"] = (
        df["inbound"].astype("string").str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )

    # Twitter export timestamps have a stable format; errors become NaT.
    df["created_dt"] = pd.to_datetime(
        df["created_at"], errors="coerce", utc=True
    )

    if df["tweet_id"].duplicated().any():
        dupes = df.loc[df["tweet_id"].duplicated(), "tweet_id"].tolist()
        raise ValueError(f"Duplicate tweet_id values found, e.g. {dupes[:10]}")

    return df


def build_graph(df):
    rows = df.set_index("tweet_id").to_dict("index")

    children = defaultdict(list)
    parents = {}

    # response_tweet_id can contain multiple IDs.
    for _, r in df.iterrows():
        tid = r["tweet_id"]
        for child in split_ids(r["response_tweet_id"]):
            children[tid].append(child)

    # Also use the explicit parent field.
    for _, r in df.iterrows():
        tid = r["tweet_id"]
        pids = split_ids(r["in_response_to_tweet_id"])
        if pids:
            parents[tid] = pids[0]

    # Make the graph consistent even if one of the two fields is incomplete.
    for child, parent in parents.items():
        if child in rows and parent in rows:
            children[parent].append(child)

    for parent in children:
        children[parent] = list(dict.fromkeys(children[parent]))

    return rows, children, parents


def is_customer(rows, tid):
    return tid in rows and rows[tid]["inbound_bool"] is True


def is_brand(rows, tid, brand):
    return (
        tid in rows
        and rows[tid]["inbound_bool"] is False
        and rows[tid]["author_id"] == brand
    )


def find_customer_roots(rows, parents):
    """Observed inbound tweets without an observed parent."""
    return [
        tid for tid, r in rows.items()
        if r["inbound_bool"] is True
        and (tid not in parents or parents.get(tid) not in rows)
    ]


def path_score(path, rows, brand):
    roles = []
    for tid in path:
        if is_customer(rows, tid):
            roles.append("C")
        elif is_brand(rows, tid, brand):
            roles.append("B")

    if not roles:
        return (-1, -1)

    alternations = sum(a != b for a, b in zip(roles, roles[1:]))
    return (alternations, len(roles))


def best_alternating_path(root, brand, rows, children):
    """
    For a branching tree, retain the branch with the strongest
    customer<->brand alternation, then the greatest usable depth.
    """
    best = [root]
    stack = [(root, [root])]

    while stack:
        tid, path = stack.pop()

        if path_score(path, rows, brand) > path_score(best, rows, brand):
            best = path

        for child in children.get(tid, []):
            if child in rows and child not in path:
                stack.append((child, path + [child]))

    return best


def sort_path(path, rows):
    return sorted(
        [rows[tid] for tid in path if tid in rows],
        key=lambda r: (
            r["created_dt"] if pd.notna(r["created_dt"])
            else pd.Timestamp.min.tz_localize("UTC")
        )
    )


def classify_thread(turns):
    brand_turns = [t for t in turns if t["inbound_bool"] is False]
    customer_turns = [t for t in turns if t["inbound_bool"] is True]

    if not brand_turns:
        return "no_brand_reply"

    last_brand = brand_turns[-1]
    last_customer = customer_turns[-1] if customer_turns else None

    last_brand_text = str(last_brand.get("text") or "")

    if DM_RE.search(last_brand_text):
        return "dm_deflection"

    if (
        last_customer is not None
        and pd.notna(last_customer["created_dt"])
        and pd.notna(last_brand["created_dt"])
        and last_customer["created_dt"] > last_brand["created_dt"]
    ):
        customer_text = str(last_customer.get("text") or "")
        if POSITIVE_RE.search(customer_text):
            return "resolved_ack"
        return "customer_replied"

    if ACTION_RE.search(last_brand_text) or len(last_brand_text.split()) >= 12:
        return "in_thread_assistance"

    return "brand_reply_no_resolution_signal"


def intent_diversity(texts):
    """Exploratory only: TF-IDF + KMeans, not a final intent taxonomy."""
    texts = [str(t).strip() for t in texts if str(t).strip()]

    if not SKLEARN_OK or len(texts) < 3:
        return np.nan, np.nan, np.nan

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1
        )
        X = vectorizer.fit_transform(texts)

        max_k = min(5, len(texts) - 1)
        best = None

        for k in range(2, max_k + 1):
            model = KMeans(n_clusters=k, random_state=42, n_init=20)
            labels = model.fit_predict(X)

            if len(set(labels)) < 2:
                continue

            score = silhouette_score(X, labels)

            if best is None or score > best[0]:
                best = (score, k, labels)

        if best is None:
            return np.nan, np.nan, np.nan

        score, k, labels = best
        return int(k), round(float(score), 3), str(
            sorted(Counter(labels).values(), reverse=True)
        )

    except Exception:
        return np.nan, np.nan, np.nan


def analyse(path):
    df = load_data(path)
    rows, children, parents = build_graph(df)

    brands = (
        df.loc[df["inbound_bool"] == False, "author_id"]
        .dropna()
        .value_counts()
        .index
        .tolist()
    )

    roots = find_customer_roots(rows, parents)

    results = []

    for brand in brands:
        seen = set()

        for root in roots:
            path_ids = best_alternating_path(
                root, brand, rows, children
            )

            if not any(is_brand(rows, tid, brand) for tid in path_ids):
                continue

            # Avoid counting the same root/brand thread more than once.
            key = (brand, root)
            if key in seen:
                continue
            seen.add(key)

            turns = sort_path(path_ids, rows)
            outcome = classify_thread(turns)

            customer_text = " ".join(
                str(t.get("text") or "")
                for t in turns
                if t["inbound_bool"] is True
            )

            results.append({
                "brand": brand,
                "root_tweet_id": root,
                "path_len": len(turns),
                "customer_turns": sum(
                    t["inbound_bool"] is True for t in turns
                ),
                "brand_turns": sum(
                    t["inbound_bool"] is False
                    for t in turns
                    if t["author_id"] == brand
                ),
                "outcome": outcome,
                "customer_text": customer_text,
            })

    thread_df = pd.DataFrame(results)

    if thread_df.empty:
        raise ValueError("No customer->brand threads could be reconstructed.")

    summary = (
        thread_df.groupby("brand")
        .agg(
            threads=("root_tweet_id", "nunique"),
            avg_path_len=("path_len", "mean"),
            resolved_ack=(
                "outcome", lambda x: (x == "resolved_ack").sum()
            ),
            in_thread_assistance=(
                "outcome", lambda x: (x == "in_thread_assistance").sum()
            ),
            dm_deflection=(
                "outcome", lambda x: (x == "dm_deflection").sum()
            ),
            no_brand_reply=(
                "outcome", lambda x: (x == "no_brand_reply").sum()
            ),
            other=(
                "outcome",
                lambda x: x.isin([
                    "customer_replied",
                    "brand_reply_no_resolution_signal"
                ]).sum()
            ),
        )
        .reset_index()
    )

    summary["resolved_or_assisted_pct"] = (
        (
            summary["resolved_ack"]
            + summary["in_thread_assistance"]
        )
        / summary["threads"]
        * 100
    )

    summary["dm_pct"] = (
        summary["dm_deflection"]
        / summary["threads"]
        * 100
    )

    # Intent diversity from customer issue text.
    intent_rows = []
    for brand in summary["brand"]:
        texts = thread_df.loc[
            thread_df["brand"] == brand, "customer_text"
        ].tolist()

        k, silhouette, sizes = intent_diversity(texts)

        intent_rows.append({
            "brand": brand,
            "intent_clusters": k,
            "intent_silhouette": silhouette,
            "intent_cluster_sizes": sizes,
        })

    intent_df = pd.DataFrame(intent_rows)
    summary = summary.merge(intent_df, on="brand", how="left")

    # Ranking is deliberately weighted toward enough data + useful support
    # behavior, not just a tiny brand with 1 perfect thread.
    summary["sample_confidence"] = (
        summary["threads"] / summary["threads"].max()
    )

    summary["selection_score"] = (
        0.50 * (summary["resolved_or_assisted_pct"] / 100)
        + 0.20 * (1 - summary["dm_pct"] / 100)
        + 0.20 * summary["sample_confidence"]
        + 0.10 * (
            summary["intent_silhouette"].fillna(0).clip(lower=0)
        )
    )

    summary = summary.sort_values(
        ["selection_score", "threads"],
        ascending=[False, False]
    )

    return df, thread_df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv",
        nargs="?",
        default="sample.csv",
        help="Path to the Kaggle CSV"
    )
    args = parser.parse_args()

    input_path = Path(args.csv)
    df, thread_df, summary = analyse(input_path)

    out_dir = input_path.parent
    summary_path = out_dir / "brand_comparison.csv"
    thread_path = out_dir / "thread_level_results.csv"

    summary.to_csv(summary_path, index=False)
    thread_df.to_csv(thread_path, index=False)

    print("\n=== BRAND SELECTION REPORT ===")
    print(f"Rows: {len(df)}")
    print(f"Brands: {summary['brand'].nunique()}")
    print(f"Reconstructed brand threads: {len(thread_df)}\n")

    cols = [
        "brand", "threads", "avg_path_len",
        "resolved_or_assisted_pct", "dm_pct",
        "intent_clusters", "intent_silhouette",
        "selection_score"
    ]

    print(summary[cols].round(3).to_string(index=False))

    print("\nTop candidates:")
    for _, r in summary.head(5).iterrows():
        print(
            f"  {r['brand']}: "
            f"{int(r['threads'])} threads, "
            f"{r['resolved_or_assisted_pct']:.1f}% "
            f"in-thread assistance/resolution, "
            f"{r['dm_pct']:.1f}% DM deflection"
        )

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {thread_path}")


if __name__ == "__main__":
    main()
