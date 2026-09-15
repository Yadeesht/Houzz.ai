import pandas as pd

# ============================================================
# CONFIG
# ============================================================

THREADS_FILE = "Data/gwr_threads.csv"
GOLDEN_FILE = "Data/gwr_golden_review.csv"
OUTPUT_FILE = "Data/data_retrieval.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading files...")

threads = pd.read_csv(THREADS_FILE)
golden = pd.read_csv(GOLDEN_FILE)

print(f"GWR threads: {threads['thread_id'].nunique():,}")
print(f"Golden threads: {golden['thread_id'].nunique():,}")


# ============================================================
# HOLD OUT GOLDEN THREADS
# ============================================================

golden_thread_ids = set(
    golden["thread_id"]
    .dropna()
    .astype(str)
)

threads["thread_id"] = threads["thread_id"].astype(str)

retrieval_threads = threads[
    ~threads["thread_id"].isin(golden_thread_ids)
].copy()

print(
    f"Threads remaining for retrieval: "
    f"{retrieval_threads['thread_id'].nunique():,}"
)


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

retrieval_threads["created_at"] = pd.to_datetime(
    retrieval_threads["created_at"],
    errors="coerce",
    utc=True
)

retrieval_threads = retrieval_threads.sort_values(
    ["thread_id", "created_at", "tweet_id"],
    kind="stable"
)


# ============================================================
# CREATE ONE RETRIEVAL DOCUMENT PER THREAD
# ============================================================

rows = []

for thread_id, thread in retrieval_threads.groupby(
    "thread_id",
    sort=False
):

    # Build:
    #
    # Customer: ...
    # GWRHelp: ...
    # Customer: ...
    # GWRHelp: ...
    #
    conversation_lines = []

    for row in thread.itertuples(index=False):

        if str(row.speaker).lower() == "customer":
            speaker = "Customer"
        else:
            speaker = "GWRHelp"

        text = str(row.text).strip()

        if not text:
            continue

        conversation_lines.append(
            f"{speaker}: {text}"
        )

    if not conversation_lines:
        continue

    rows.append({
        "thread_id": thread_id,
        "root_tweet_id": thread["root_tweet_id"].iloc[0],
        "data": "\n".join(conversation_lines)
    })


# ============================================================
# CREATE DATAFRAME
# ============================================================

retrieval_df = pd.DataFrame(rows)


# ============================================================
# SAVE
# ============================================================

retrieval_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 60)
print("DATA RETRIEVAL FILE CREATED")
print("=" * 60)

print(f"Output: {OUTPUT_FILE}")
print(f"Threads: {len(retrieval_df):,}")
print()
print("Columns:")
print(list(retrieval_df.columns))

