import pandas as pd
import numpy as np
import re

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "Data/gwr_threads.csv"

PROFILE_OUTPUT = "Data/gwr_thread_profile.csv"
CANDIDATE_OUTPUT = "Data/gwr_candidate_pool.csv"

RANDOM_SEED = 42

# Number of candidate threads we want to inspect
TARGET_CANDIDATES = 600


# ============================================================
# LOAD
# ============================================================

print("Loading GWR threads...")

df = pd.read_csv(INPUT_FILE)

df["created_at"] = pd.to_datetime(
    df["created_at"],
    errors="coerce",
    utc=True
)

df["text"] = df["text"].fillna("")
df["speaker"] = df["speaker"].astype(str).str.lower()

print(f"Tweets loaded: {len(df):,}")
print(f"Threads loaded: {df['thread_id'].nunique():,}")


# ============================================================
# HELPER REGEX
# ============================================================

QUESTION_RE = re.compile(
    r"\?|"
    r"\bhow\b|"
    r"\bwhat\b|"
    r"\bwhen\b|"
    r"\bwhere\b|"
    r"\bwhy\b|"
    r"\bcan\b|"
    r"\bcould\b|"
    r"\bwould\b|"
    r"\bwill\b|"
    r"\bis\b|"
    r"\bdo\b|"
    r"\bdoes\b",
    re.IGNORECASE
)

COMPLAINT_RE = re.compile(
    r"\bproblem\b|"
    r"\bissue\b|"
    r"\bcomplaint\b|"
    r"\bterrible\b|"
    r"\bawful\b|"
    r"\brubbish\b|"
    r"\bdisappoint(ed|ing)?\b|"
    r"\bunhappy\b|"
    r"\bfrustrat(ed|ing|ion)?\b|"
    r"\bangry\b|"
    r"\bworst\b|"
    r"\bnot working\b|"
    r"\bdoesn't work\b|"
    r"\bdoes not work\b|"
    r"\bbroken\b|"
    r"\bfailed\b|"
    r"\bcan't\b|"
    r"\bcannot\b|"
    r"\bunable\b|"
    r"\bnever\b|"
    r"\bstill\b",
    re.IGNORECASE
)

DM_RE = re.compile(
    r"\bdm\b|"
    r"\bdirect message\b|"
    r"\bprivate message\b|"
    r"\bpm\b|"
    r"\binbox\b|"
    r"\bmessage us privately\b|"
    r"\bsend us a message\b",
    re.IGNORECASE
)

ESCALATION_RE = re.compile(
    r"\bcall us\b|"
    r"\bcall me\b|"
    r"\bphone\b|"
    r"\btelephone\b|"
    r"\bcontact us\b|"
    r"\bemail us\b|"
    r"\bemail\b|"
    r"\bcontact center\b|"
    r"\bcustomer service\b|"
    r"\bspeak to\b|"
    r"\bteam will\b|"
    r"\bescalat(e|ed|ion)\b",
    re.IGNORECASE
)


# ============================================================
# THREAD-LEVEL PROFILING
# ============================================================

profiles = []

for thread_id, thread in df.groupby("thread_id", sort=False):

    thread = thread.sort_values(
        ["created_at", "tweet_id"],
        kind="stable"
    )

    # --------------------------------------------------------
    # Basic counts
    # --------------------------------------------------------

    total_turns = len(thread)

    customer = thread[
        thread["speaker"] == "customer"
    ]

    company = thread[
        thread["speaker"] == "company"
    ]

    customer_turns = len(customer)
    company_turns = len(company)

    # --------------------------------------------------------
    # Single vs multi
    # --------------------------------------------------------

    if total_turns == 1:
        interaction_type = "single_turn"
    else:
        interaction_type = "multi_turn"

    # --------------------------------------------------------
    # Short vs long
    #
    # Don't choose arbitrary values yet.
    # We calculate message-count quantiles first.
    # --------------------------------------------------------

    text_lengths = thread["text"].str.len()

    total_text_chars = int(text_lengths.sum())
    avg_text_chars = float(text_lengths.mean())

    # --------------------------------------------------------
    # Customer text for question / complaint signals
    # --------------------------------------------------------

    customer_text = " ".join(
        customer["text"].astype(str).tolist()
    )

    question_signal = bool(
        QUESTION_RE.search(customer_text)
    )

    complaint_signal = bool(
        COMPLAINT_RE.search(customer_text)
    )

    # --------------------------------------------------------
    # DM / escalation signals
    # --------------------------------------------------------

    all_text = " ".join(
        thread["text"].astype(str).tolist()
    )

    dm_signal = bool(
        DM_RE.search(all_text)
    )

    escalation_signal = bool(
        ESCALATION_RE.search(all_text)
    )

    # --------------------------------------------------------
    # Where did DM/escalation appear?
    # --------------------------------------------------------

    company_text = " ".join(
        company["text"].astype(str).tolist()
    )

    company_dm_signal = bool(
        DM_RE.search(company_text)
    )

    company_escalation_signal = bool(
        ESCALATION_RE.search(company_text)
    )

    # --------------------------------------------------------
    # First / last speaker
    # --------------------------------------------------------

    first_speaker = (
        thread.iloc[0]["speaker"]
        if len(thread)
        else None
    )

    last_speaker = (
        thread.iloc[-1]["speaker"]
        if len(thread)
        else None
    )

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    if len(thread) >= 2:

        first_time = thread["created_at"].min()
        last_time = thread["created_at"].max()

        duration_minutes = (
            last_time - first_time
        ).total_seconds() / 60

    else:

        duration_minutes = 0

    profiles.append(
        {
            "thread_id": thread_id,

            "root_tweet_id": thread[
                "root_tweet_id"
            ].iloc[0],

            "total_turns": total_turns,

            "customer_turns": customer_turns,
            "company_turns": company_turns,

            "interaction_type": interaction_type,

            "total_text_chars": total_text_chars,
            "avg_text_chars": avg_text_chars,

            "question_signal": question_signal,
            "complaint_signal": complaint_signal,

            "dm_signal": dm_signal,
            "escalation_signal": escalation_signal,

            "company_dm_signal": company_dm_signal,
            "company_escalation_signal": company_escalation_signal,

            "first_speaker": first_speaker,
            "last_speaker": last_speaker,

            "duration_minutes": duration_minutes,
        }
    )


profile = pd.DataFrame(profiles)


# ============================================================
# SHORT / MEDIUM / LONG USING DISTRIBUTION
# ============================================================

# We use quantiles instead of inventing thresholds.
#
# roughly:
# bottom 33% -> short
# middle 34% -> medium
# top 33% -> long

q1 = profile["total_turns"].quantile(0.33)
q2 = profile["total_turns"].quantile(0.66)

def length_bucket(turns):

    if turns <= q1:
        return "short"

    elif turns <= q2:
        return "medium"

    return "long"


profile["length_bucket"] = (
    profile["total_turns"]
    .apply(length_bucket)
)


# ============================================================
# SIMPLE SUPPORT SHAPE
# ============================================================

def classify_support_shape(row):

    if (
        row["question_signal"]
        and row["complaint_signal"]
    ):
        return "question_and_complaint"

    if row["complaint_signal"]:
        return "complaint"

    if row["question_signal"]:
        return "question"

    return "other"


profile["customer_type"] = profile.apply(
    classify_support_shape,
    axis=1
)


# ============================================================
# SAVE FULL PROFILE
# ============================================================

profile.to_csv(
    PROFILE_OUTPUT,
    index=False
)

print()
print("=" * 60)
print("THREAD PROFILE CREATED")
print("=" * 60)

print(f"Threads: {len(profile):,}")
print(f"Output:  {PROFILE_OUTPUT}")


# ============================================================
# SHOW DISTRIBUTION
# ============================================================

print()
print("Interaction type:")
print(
    profile["interaction_type"]
    .value_counts()
)

print()
print("Length bucket:")
print(
    profile["length_bucket"]
    .value_counts()
)

print()
print("Customer type:")
print(
    profile["customer_type"]
    .value_counts()
)

print()
print("DM signal:")
print(
    profile["dm_signal"]
    .value_counts()
)

print()
print("Escalation signal:")
print(
    profile["escalation_signal"]
    .value_counts()
)


# ============================================================
# STRATIFIED CANDIDATE SAMPLING
# ============================================================

# We want the candidate pool to contain different
# conversation shapes.
#
# We create strata from:
#   interaction length
#   customer type
#   DM/escalation status

profile["stratum"] = (
    profile["interaction_type"]
    + "__"
    + profile["length_bucket"]
    + "__"
    + profile["customer_type"]
    + "__dm_"
    + profile["dm_signal"].astype(str)
)


# Allocate approximately evenly across strata.
strata = profile["stratum"].value_counts()

n_strata = len(strata)

base_per_stratum = max(
    1,
    TARGET_CANDIDATES // n_strata
)

candidate_parts = []

rng = np.random.RandomState(RANDOM_SEED)

for stratum, count in strata.items():

    subset = profile[
        profile["stratum"] == stratum
    ]

    n_take = min(
        len(subset),
        base_per_stratum
    )

    sampled = subset.sample(
        n=n_take,
        random_state=rng
    )

    candidate_parts.append(sampled)


candidates = pd.concat(
    candidate_parts,
    ignore_index=True
)


# If we are below target because some strata are small,
# fill the remaining slots randomly from unused threads.

remaining = TARGET_CANDIDATES - len(candidates)

if remaining > 0:

    unused = profile[
        ~profile["thread_id"].isin(
            candidates["thread_id"]
        )
    ]

    if len(unused) > 0:

        extra = unused.sample(
            n=min(remaining, len(unused)),
            random_state=rng
        )

        candidates = pd.concat(
            [candidates, extra],
            ignore_index=True
        )


# ============================================================
# SAVE CANDIDATES
# ============================================================

candidates = candidates.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

candidates.to_csv(
    CANDIDATE_OUTPUT,
    index=False
)


print()
print("=" * 60)
print("CANDIDATE POOL CREATED")
print("=" * 60)

print(f"Candidates: {len(candidates):,}")
print(f"Output:     {CANDIDATE_OUTPUT}")