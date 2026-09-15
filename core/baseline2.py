import pandas as pd
import re
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. LOAD GOLDEN SET
# ============================================================

FILE = "Data/gwr_golden_review.csv"

df = pd.read_csv(FILE)

# Keep only usable labels
df = df[df["intent"].notna()].copy()


# ============================================================
# 2. EXTRACT ONLY CUSTOMER MESSAGES
# ============================================================

def extract_customer_text(conversation):
    """
    Extract all customer turns from the reconstructed conversation.
    Ignores GWRHelp responses.
    """
    if pd.isna(conversation):
        return ""

    lines = str(conversation).splitlines()

    customer_parts = []
    collecting = False

    for line in lines:

        if line.startswith("Customer:"):
            collecting = True
            customer_parts.append(
                line.replace("Customer:", "", 1).strip()
            )

        elif re.match(r"^[A-Za-z0-9_]+:", line):
            # Another speaker started
            collecting = False

        elif collecting:
            customer_parts.append(line.strip())

    return " ".join(customer_parts)


df["customer_text"] = df["full_conversation"].apply(
    extract_customer_text
)


# ============================================================
# 3. SIMPLE KEYWORD RULES
# ============================================================
#
# These are intentionally small and high-signal.
# We are NOT trying to cover every possible wording.
#
# The idea:
#
# customer message
#        ↓
# keyword matching
#        ↓
# intent
#
# OTHER is the fallback when nothing matches.
# ============================================================

INTENT_RULES = {

    "TRAIN DELAY / CANCELLATION": [
        "delay",
        "delayed",
        "late",
        "lateness",
        "cancel",
        "cancelled",
        "cancellation",
        "disruption",
        "disrupted",
        "not running",
        "service suspended",
        "replacement bus"
    ],

    "REFUND / COMPENSATION": [
        "refund",
        "money back",
        "reimburse",
        "reimbursement",
        "compensation",
        "compensate"
    ],

    "BOOKING / TICKETS": [
        "ticket",
        "tickets",
        "booking",
        "book",
        "reservation",
        "reserve"
    ],

    "PAYMENT / CHARGES": [
        "charged",
        "charge",
        "payment",
        "paid",
        "card",
        "overcharged",
        "extra charge"
    ],

    "BAGGAGE / LOST PROPERTY": [
        "bag",
        "bags",
        "baggage",
        "luggage",
        "suitcase",
        "lost property",
        "lost item"
    ],

    "TECHNICAL / WEBSITE / APP": [
        "app",
        "website",
        "web site",
        "online",
        "login",
        "log in",
        "error",
        "bug",
        "not working",
        "can't access",
        "cannot access"
    ],

    "ACCESSIBILITY / SPECIAL ASSISTANCE": [
        "wheelchair",
        "accessible",
        "accessibility",
        "disabled",
        "mobility",
        "assistance",
        "assisted travel",
        "step free",
        "step-free"
    ],

    "STAFF / SERVICE COMPLAINT": [
        "staff",
        "driver",
        "conductor",
        "guard",
        "employee",
        "rude",
        "impolite",
        "attitude"
    ],

    "ROUTE / TIMETABLE / STATION": [
        "timetable",
        "schedule",
        "platform",
        "station",
        "departure",
        "depart",
        "arrival",
        "arrive",
        "route",
        "connection",
        "connecting"
    ],

    "TRAIN / SERVICE INFORMATION": [
        "carriage",
        "coach",
        "formation",
        "first class",
        "standard class",
        "wifi",
        "wi-fi",
        "toilet",
        "catering",
        "on board",
        "onboard",
        "seat",
        "seating",
        "capacity",
        "bike space"
    ],

    "GENERAL CUSTOMER SERVICE": [
        "help",
        "please advise",
        "enquiry",
        "question",
        "contact",
        "customer service"
    ]
}


# Compile regex once
COMPILED_RULES = {
    intent: [
        re.compile(
            rf"(?<!\w){re.escape(keyword)}(?!\w)",
            re.IGNORECASE
        )
        for keyword in keywords
    ]
    for intent, keywords in INTENT_RULES.items()
}


# ============================================================
# 4. KEYWORD CLASSIFIER
# ============================================================

def predict_intent(text):
    """
    Count matching keywords for each intent.
    Highest score wins.
    If nothing matches -> OTHER.
    """

    if not text:
        return "OTHER"

    scores = defaultdict(int)

    for intent, patterns in COMPILED_RULES.items():
        for pattern in patterns:
            if pattern.search(text):
                scores[intent] += 1

    # No keyword matched
    if not scores:
        return "OTHER"

    # Highest number of matched keywords
    return max(scores, key=scores.get)


df["predicted_intent"] = df["customer_text"].apply(
    predict_intent
)

ESCALATION_KEYWORDS = [
    # Money / refunds
    "refund",
    "money back",
    "reimburse",
    "reimbursement",
    "compensation",
    "compensate",

    # Payment / charges
    "charged",
    "charge",
    "payment",
    "paid",
    "overcharged",
    "extra charge",

    # Cancellation
    "cancel",
    "cancelled",
    "cancellation",

    # Financial loss / fees
    "fee",
    "cost",
    "cost me",
    "lost money",
    "financial",
    "pay back"
]

COMPILED_ESCALATION_RULES = [
    re.compile(
        rf"(?<!\w){re.escape(keyword)}(?!\w)",
        re.IGNORECASE
    )
    for keyword in ESCALATION_KEYWORDS
]


def should_escalate(text):
    """
    Simple high-stakes keyword flag.

    Returns:
        True  -> escalate to human
        False -> no escalation flag
    """

    if not text:
        return False

    for pattern in COMPILED_ESCALATION_RULES:
        if pattern.search(text):
            return True

    return False


df["escalate_to_human"] = df["customer_text"].apply(
    should_escalate
)

# ============================================================
# 5. EVALUATION FUNCTION
# ============================================================

def evaluate(data, name):

    y_true = data["intent"]
    y_pred = data["predicted_intent"]

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Samples   : {len(data)}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Macro P   : {precision:.4f}")
    print(f"Macro R   : {recall:.4f}")
    print(f"Macro F1  : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    labels = sorted(y_true.unique())

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )

    print(cm_df)


# ============================================================
# 6. EVALUATE FULL GOLDEN SET
# ============================================================

evaluate(
    df,
    "Baseline 2 — Full Golden Set"
)


# ============================================================
# 7. EVALUATE ONLY GOOD CANDIDATES
# ============================================================

good_candidates = df[
    df["evaluation_quality"] == "good_candidate"
].copy()

evaluate(
    good_candidates,
    "Baseline 2 — Good Candidates Only"
)


# ============================================================
# 8. SHOW INDIVIDUAL PREDICTIONS
# ============================================================

results = good_candidates[
    [
        "thread_id",
        "customer_text",
        "intent",
        "predicted_intent"
    ]
].copy()

results["correct"] = (
    results["intent"] == results["predicted_intent"]
)

print("\nSample predictions:")
print(
    results.head(3).to_string(index=False)
)


# ============================================================
# 9. SAVE RESULTS
# ============================================================

results.to_csv(
    "Data/baseline2_keyword_results.csv",
    index=False
)

print(
    "\nSaved predictions to baseline2_keyword_results.csv"
)