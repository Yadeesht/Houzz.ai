import pandas as pd
import numpy as np
from collections import defaultdict, deque


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "Data/twcs.csv"
OUTPUT_FILE = "Data/gwr_threads.csv"

GWR_AUTHOR = "GWRHelp"


# ============================================================
# HELPER: NORMALIZE IDS
# ============================================================

def normalize_id(value):
    """
    Convert a value into a list of clean tweet IDs.

    Handles:
        NaN
        123
        123.0
        "123"
        "123,456"
    """

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    result = []

    for item in value.split(","):
        item = item.strip()

        if not item:
            continue

        try:
            number = float(item)

            if number.is_integer():
                item = str(int(number))
        except ValueError:
            pass

        result.append(item)

    return result


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(
    INPUT_FILE,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ],
)

print(f"Loaded {len(df):,} tweets")


# ============================================================
# NORMALIZE tweet_id
# ============================================================

print("Normalizing tweet IDs...")

df["tweet_id"] = (
    pd.to_numeric(df["tweet_id"], errors="coerce")
    .astype("Int64")
    .astype("string")
    .fillna("")
)


# ============================================================
# BUILD REPLY GRAPH
# ============================================================

print("Building reply graph...")

# parent -> children
children = defaultdict(set)

# child -> parents
parents = defaultdict(set)

for tweet_id, response_ids, parent_ids in df[
    ["tweet_id", "response_tweet_id", "in_response_to_tweet_id"]
].itertuples(index=False):

    if not tweet_id:
        continue

    # --------------------------------------------------------
    # response_tweet_id
    #
    # tweet A has response B
    # => A -> B
    # --------------------------------------------------------

    for response_id in normalize_id(response_ids):

        children[tweet_id].add(response_id)
        parents[response_id].add(tweet_id)

    # --------------------------------------------------------
    # in_response_to_tweet_id
    #
    # tweet B is replying to A
    # => A -> B
    # --------------------------------------------------------

    for parent_id in normalize_id(parent_ids):

        children[parent_id].add(tweet_id)
        parents[tweet_id].add(parent_id)


print(f"Graph nodes with children: {len(children):,}")


# ============================================================
# IDENTIFY ROOT TWEETS
# ============================================================

# A root is a tweet for which we have no known parent.
#
# IMPORTANT:
# This means "root of the observed dataset/thread", not
# necessarily the absolute beginning of the real-world conversation.

all_tweet_ids = set(
    df.loc[df["tweet_id"] != "", "tweet_id"]
)

tweets_with_parents = set(parents.keys())

root_tweets = all_tweet_ids - tweets_with_parents

print(f"Root tweets: {len(root_tweets):,}")


# ============================================================
# BUILD BIDIRECTIONAL GRAPH
# ============================================================

print("Building bidirectional graph...")

adjacency = defaultdict(set)

for parent_id, child_ids in children.items():

    for child_id in child_ids:

        adjacency[parent_id].add(child_id)
        adjacency[child_id].add(parent_id)


# ============================================================
# FIND GWRHelp TWEETS
# ============================================================

gwr_mask = (
    df["author_id"].astype(str).str.strip() == GWR_AUTHOR
)

gwr_tweets = set(
    df.loc[gwr_mask, "tweet_id"]
    .dropna()
    .tolist()
)

print(f"GWRHelp tweets: {len(gwr_tweets):,}")


# ============================================================
# RECONSTRUCT THREADS
# ============================================================

print("Reconstructing GWRHelp threads...")

visited = set()

thread_rows = []

thread_number = 0


for gwr_tweet in gwr_tweets:

    if gwr_tweet in visited:
        continue

    # --------------------------------------------------------
    # 1. Find the root by walking backwards
    # --------------------------------------------------------

    current = gwr_tweet
    root = gwr_tweet

    backward_seen = set()

    while current in parents and current not in backward_seen:

        backward_seen.add(current)

        parent_list = list(parents[current])

        if not parent_list:
            break

        # Normally there is one parent.
        # If branching/inconsistent data gives multiple parents,
        # choose the earliest parent by timestamp later through
        # the observed component logic.
        #
        # Here we simply take one to locate the root.
        current = parent_list[0]

        root = current

    # --------------------------------------------------------
    # 2. Walk the entire connected component
    # --------------------------------------------------------

    thread_number += 1

    thread_id = f"GWR_{thread_number:06d}"

    queue = deque([root])

    component = set()

    while queue:

        tweet_id = queue.popleft()

        if tweet_id in component:
            continue

        component.add(tweet_id)

        for neighbor in adjacency.get(tweet_id, []):

            if neighbor not in component:
                queue.append(neighbor)

    # --------------------------------------------------------
    # 3. Mark all tweets in this thread as visited
    # --------------------------------------------------------

    visited.update(component)

    # --------------------------------------------------------
    # 4. Store thread information
    # --------------------------------------------------------

    for tweet_id in component:

        thread_rows.append(
            {
                "thread_id": thread_id,
                "root_tweet_id": root,
                "tweet_id": tweet_id,
            }
        )


print(f"Threads found: {thread_number:,}")
print(f"Tweets in GWRHelp threads: {len(thread_rows):,}")


# ============================================================
# JOIN THREAD INFO BACK TO ORIGINAL DATA
# ============================================================

print("Joining tweet information...")

threads = pd.DataFrame(thread_rows)

result = threads.merge(
    df,
    on="tweet_id",
    how="left",
)


# ============================================================
# PARSE CREATED_AT
# ============================================================

result["created_at"] = pd.to_datetime(
    result["created_at"],
    errors="coerce",
    utc=True,
)


# ============================================================
# SORT CHRONOLOGICALLY WITHIN EACH THREAD
# ============================================================

print("Sorting threads chronologically...")

result = result.sort_values(
    [
        "thread_id",
        "created_at",
        "tweet_id",
    ],
    kind="stable",
)


# ============================================================
# TURN INDEX
# ============================================================

result["turn_index"] = (
    result.groupby("thread_id")
    .cumcount()
    + 1
)


# ============================================================
# SPEAKER
# ============================================================

# TRUE  -> customer
# FALSE -> company/support

result["speaker"] = np.where(
    result["inbound"].astype(str).str.upper() == "TRUE",
    "customer",
    "company",
)


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

result = result[
    [
        "thread_id",
        "root_tweet_id",
        "turn_index",
        "tweet_id",
        "author_id",
        "speaker",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ]
]


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False,
)

print()
print("=" * 60)
print("DONE")
print("=" * 60)

print(f"Output file: {OUTPUT_FILE}")
print(f"Tweets: {len(result):,}")
print(f"Threads: {result['thread_id'].nunique():,}")


# ============================================================
# SHOW ONE EXAMPLE
# ============================================================

print()
print("=" * 60)
print("EXAMPLE THREAD")
print("=" * 60)

if len(result) > 0:

    sample_thread = result["thread_id"].iloc[0]

    sample = result[
        result["thread_id"] == sample_thread
    ]

    print(f"Thread ID: {sample_thread}")
    print(f"Root tweet: {sample['root_tweet_id'].iloc[0]}")
    print()

    for row in sample.itertuples(index=False):

        text = str(row.text).replace("\n", " ")

        if len(text) > 200:
            text = text[:200] + "..."

        print(
            f"{row.turn_index:>2}. "
            f"[{row.speaker}] "
            f"{row.created_at} | "
            f"{text}"
        )