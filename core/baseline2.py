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
# 5. RESPONSE TEMPLATES & KEYWORD-BASED SELECTOR
# ============================================================
#
# Each intent has:
#   - sub_rules: A list of dicts with 'name', 'keywords', and 'response'.
#     If the customer text matches keywords in a sub-rule, that
#     specific response is selected.
#   - default: Fallback response for that intent if no sub-rule keywords match.
#

RESPONSE_TEMPLATES = {

    "TRAIN DELAY / CANCELLATION": {
        "sub_rules": [
            {
                "name": "replacement_bus",
                "keywords": ["replacement bus", "bus", "coach", "road transport", "rail replacement"],
                "response": "Rail replacement buses are operating on this route due to service disruption. Replacement buses usually pick up from the station front or designated bus bays. Please check station signage or speak to platform staff for exact pickup points."
            },
            {
                "name": "cancellation",
                "keywords": ["cancel", "cancelled", "cancellation", "suspended", "not running", "called off"],
                "response": "I am very sorry that your train has been cancelled. Your ticket is valid on the next available GWR service on the same route. If you choose not to travel due to the cancellation, you are entitled to a full fee-free refund from your point of purchase."
            },
            {
                "name": "delay",
                "keywords": ["delay", "delayed", "late", "lateness", "behind schedule", "held up", "waiting"],
                "response": "I'm sorry for the delay to your journey today. Please check real-time departures on www.gwr.com or the GWR app for live updates. If your journey arrives 15 minutes or more late, you can claim compensation via our Delay Repay scheme at www.gwr.com/delayrepay."
            }
        ],
        "default": "We're sorry for the disruption to your travel. Please check live departure boards at www.gwr.com for the latest service status, and visit www.gwr.com/delayrepay if your journey is delayed by 15+ minutes."
    },

    "REFUND / COMPENSATION": {
        "sub_rules": [
            {
                "name": "delay_repay",
                "keywords": ["delay repay", "delayed", "delay", "late", "lateness", "30 mins", "15 mins", "compensation", "compensate"],
                "response": "If your journey was delayed by 15 minutes or more, you can claim compensation under our Delay Repay scheme. Please submit a claim with a photo of your ticket within 28 days at www.gwr.com/delayrepay."
            },
            {
                "name": "ticket_refund",
                "keywords": ["ticket", "booking", "refund", "money back", "unused", "reimburse", "reimbursement", "return ticket"],
                "response": "For ticket refunds on unused journeys, please apply through your original retailer. If booked directly with GWR, you can submit a refund request via your online GWR account or at any staffed station ticket office."
            }
        ],
        "default": "To apply for a refund or compensation, please visit www.gwr.com/refunds. If your claim is regarding a delayed journey, please claim directly through www.gwr.com/delayrepay."
    },

    "BOOKING / TICKETS": {
        "sub_rules": [
            {
                "name": "collection_reference",
                "keywords": ["collection", "collect", "reference", "code", "e-ticket", "eticket", "smartcard", "email", "received", "haven't received"],
                "response": "If you have not received your booking confirmation or collection reference, please check your spam/junk folder. You can also view your active bookings in your online GWR account. If you still need help, please DM us your registered email."
            },
            {
                "name": "seat_reservation",
                "keywords": ["seat", "seats", "reservation", "reserved", "table", "window", "aisle"],
                "response": "Seat reservations are available on most long-distance GWR services. If you hold a flexible ticket, reservations are optional. You can reserve or amend a seat reservation at any station ticket office or by contacting us via DM."
            },
            {
                "name": "amend_change",
                "keywords": ["change", "amend", "amendment", "exchange", "upgrade", "first class"],
                "response": "Advance tickets can be amended prior to departure for a £10 admin fee plus any difference in fare via your online account or ticket office. Flexible tickets (Off-Peak/Anytime) can be changed or upgraded directly."
            }
        ],
        "default": "For ticket purchases, amendments, or collection queries, please visit www.gwr.com or manage your booking via your GWR account. Feel free to DM us if you need specific booking assistance."
    },

    "PAYMENT / CHARGES": {
        "sub_rules": [
            {
                "name": "overcharge",
                "keywords": ["overcharged", "extra charge", "double charged", "charged twice", "incorrect amount", "too much", "wrong charge"],
                "response": "I'm sorry to hear about an unexpected or duplicate charge. Please send us a direct message (DM) with your booking reference, travel date, and the last 4 digits of your payment card so our accounts team can investigate."
            },
            {
                "name": "payment_failed",
                "keywords": ["failed", "declined", "card", "payment", "contactless", "pending", "paid"],
                "response": "If your payment was interrupted or declined, pending charges on your bank statement are usually temporary authorisations that will release within a few days. For contactless queries, please DM us with the journey date and card details."
            }
        ],
        "default": "For billing or payment enquiries, please send us a DM with your booking reference or transaction details so we can look into this for you."
    },

    "BAGGAGE / LOST PROPERTY": {
        "sub_rules": [
            {
                "name": "lost_item",
                "keywords": ["lost", "left on", "forgot", "missing", "lost property", "lost item", "left behind", "left my"],
                "response": "I'm sorry to hear you left an item behind. Please report your lost property as soon as possible at www.gwr.com/lostproperty with a full description and your train/route details so our station teams can match it."
            },
            {
                "name": "bike_bicycle",
                "keywords": ["bike", "bicycle", "cycle", "bike space", "tandem"],
                "response": "Bike spaces are free but reservations are compulsory on many high-speed GWR services due to limited space. You can reserve a bicycle space when booking online or at any staffed ticket office."
            },
            {
                "name": "luggage_allowance",
                "keywords": ["luggage", "baggage", "suitcase", "bags", "bag", "allowance", "space"],
                "response": "Passengers can bring up to 3 items of luggage free of charge (two larger items and one small bag). Overhead racks and luggage stacks at coach ends are available on all our trains."
            }
        ],
        "default": "For lost property and baggage enquiries, please visit www.gwr.com/lostproperty or speak to a customer service representative at any staffed station."
    },

    "TECHNICAL / WEBSITE / APP": {
        "sub_rules": [
            {
                "name": "login_account",
                "keywords": ["login", "log in", "password", "account", "sign in", "reset", "cannot access", "can't access"],
                "response": "If you're having difficulty logging into your account, please try resetting your password using the 'Forgot Password' link on the login page. Clearing your browser cache and cookies can also help resolve login issues."
            },
            {
                "name": "app_website_error",
                "keywords": ["app", "website", "web site", "error", "bug", "crash", "glitch", "not working", "screen", "down"],
                "response": "We apologize for the technical issue with our app/website. Please check if you have the latest app update installed, or try completing your transaction via a web browser. If it persists, please DM us a screenshot of the error."
            }
        ],
        "default": "We're sorry for the technical difficulty. Please try refreshing the page or restarting the app. If the problem persists, please DM us details of the error and the device you're using."
    },

    "ACCESSIBILITY / SPECIAL ASSISTANCE": {
        "sub_rules": [
            {
                "name": "passenger_assist",
                "keywords": ["wheelchair", "ramp", "mobility", "disabled", "assistance", "assisted travel", "help off", "help on"],
                "response": "Passenger Assist is available across our network for anyone needing travel support or wheelchair ramps. You can book assistance at www.gwr.com/assisted-travel, via the Passenger Assistance app, or by calling 0800 197 1329 (open 24/7). You can also turn up and request help at any staffed station."
            },
            {
                "name": "step_free",
                "keywords": ["step free", "step-free", "accessible", "accessibility", "lift", "elevator", "stairs"],
                "response": "Station accessibility information, including step-free access and lift status, is available on the station pages of www.gwr.com and on National Rail Enquiries. If a lift is out of service, our team can arrange accessible alternative transport."
            }
        ],
        "default": "We are committed to making rail travel accessible. For assisted travel bookings or enquiries, please call our Assisted Travel team free on 0800 197 1329 or visit www.gwr.com/assisted-travel."
    },

    "STAFF / SERVICE COMPLAINT": {
        "sub_rules": [
            {
                "name": "staff_conduct",
                "keywords": ["rude", "impolite", "attitude", "staff", "driver", "conductor", "guard", "employee", "unhelpful"],
                "response": "We are very sorry to hear about your experience with our staff. We hold our teams to high professional standards. Please send us a DM with the date, time, station or service, and any staff details so we can refer this to station management."
            },
            {
                "name": "conditions_overcrowding",
                "keywords": ["terrible", "horrible", "appalling", "disgrace", "crowded", "filthy", "disgusting", "cold", "dirty"],
                "response": "We apologize for the poor conditions you experienced today. This is not the level of service we aim to deliver. Please share your train details with us via DM or submit formal feedback at www.gwr.com/contact so we can investigate."
            }
        ],
        "default": "We're sorry that your journey did not meet expectations. Please share details of your journey with us via DM or submit feedback directly at www.gwr.com/contact so we can address your concerns."
    },

    "ROUTE / TIMETABLE / STATION": {
        "sub_rules": [
            {
                "name": "missed_connection",
                "keywords": ["connection", "connecting", "missed", "miss", "change trains", "next train"],
                "response": "If your train was delayed causing you to miss a connecting service, your ticket is valid on the next available train to your destination. Station staff will also be able to advise on alternative routes."
            },
            {
                "name": "platform_info",
                "keywords": ["platform", "departure", "depart", "arrival", "arrive", "which platform", "track"],
                "response": "Live platform allocations and departure times are displayed on station screens and in the GWR app. Platforms are confirmed shortly before arrival, so please check the departure boards at the station."
            },
            {
                "name": "timetable_schedule",
                "keywords": ["timetable", "schedule", "frequency", "times", "first train", "last train", "route"],
                "response": "Timetables and route maps are available to download at www.gwr.com/timetables. You can also plan journeys and check live departures via the GWR app or National Rail Enquiries."
            }
        ],
        "default": "For timetable, platform, or route information, please check live departures at www.gwr.com or speak to station staff on the concourse."
    },

    "TRAIN / SERVICE INFORMATION": {
        "sub_rules": [
            {
                "name": "wifi_facilities",
                "keywords": ["wifi", "wi-fi", "toilet", "toilets", "catering", "cafe", "trolley", "plug", "socket", "power", "air con", "heating"],
                "response": "We apologise for the issues with onboard facilities. Please send us a DM with your train time, origin/destination, and coach letter so our on-train and maintenance crews can be notified to resolve this."
            },
            {
                "name": "formation_carriages",
                "keywords": ["carriage", "carriages", "coach", "coaches", "formation", "short formed", "busy", "rammed", "seat", "seating", "capacity"],
                "response": "We try to run all trains with their scheduled number of carriages, but short formations can occur when rolling stock undergoes urgent maintenance. We are sorry for the crowding and inconvenience this caused."
            },
            {
                "name": "first_class",
                "keywords": ["first class", "standard class", "upgrade", "complimentary"],
                "response": "First Class offers wider seats and complimentary refreshments on selected high-speed routes. Weekend First upgrades are also available on board selected services subject to availability."
            }
        ],
        "default": "For information regarding onboard amenities, carriage formations, and services, please visit www.gwr.com/onboard or ask your train manager during your journey."
    },

    "GENERAL CUSTOMER SERVICE": {
        "sub_rules": [
            {
                "name": "contact_dm",
                "keywords": ["dm", "private message", "call", "phone", "contact", "speak", "agent", "human"],
                "response": "Our team is here to help! Please send us a direct message (DM) with your journey details or booking reference, or call our customer service team on 0345 7000 125."
            },
            {
                "name": "greeting_enquiry",
                "keywords": ["help", "please advise", "enquiry", "question", "info", "information"],
                "response": "Hello! How can we assist you with your journey today? Please reply with your specific travel details or enquiry and we'll be happy to help."
            }
        ],
        "default": "Thanks for contacting GWR. Please let us know how we can assist you today, or send us a DM with your journey details."
    },

    "OTHER": {
        "sub_rules": [],
        "default": "Thank you for getting in touch with GWR. Please send us a direct message (DM) or reply with details about your journey, and our customer support team will be happy to assist you."
    }
}


# Compile sub-rule keyword regexes once for fast matching
COMPILED_RESPONSE_TEMPLATES = {}

for intent, config in RESPONSE_TEMPLATES.items():
    compiled_sub_rules = []
    for rule in config.get("sub_rules", []):
        compiled_patterns = [
            re.compile(
                rf"(?<!\w){re.escape(keyword)}(?!\w)",
                re.IGNORECASE
            )
            for keyword in rule["keywords"]
        ]
        compiled_sub_rules.append({
            "name": rule.get("name", ""),
            "patterns": compiled_patterns,
            "response": rule["response"]
        })
    COMPILED_RESPONSE_TEMPLATES[intent] = {
        "sub_rules": compiled_sub_rules,
        "default": config.get("default", "")
    }


def select_response(intent, text):
    """
    Select the best response template for a given intent based on sub-keywords.
    If no sub-rules match, fall back to the default response for that intent.
    """
    if not text:
        text = ""

    config = COMPILED_RESPONSE_TEMPLATES.get(intent)
    if not config:
        config = COMPILED_RESPONSE_TEMPLATES.get("OTHER", {})

    best_response = None
    max_matches = 0

    for rule in config.get("sub_rules", []):
        matches = sum(1 for pattern in rule["patterns"] if pattern.search(text))
        if matches > max_matches:
            max_matches = matches
            best_response = rule["response"]

    if best_response and max_matches > 0:
        return best_response

    return config.get("default", COMPILED_RESPONSE_TEMPLATES["OTHER"]["default"])


def generate_response(text, intent=None):
    """
    Generate a response directly from customer text:
    1. If intent is not provided, predict it using predict_intent(text).
    2. Select the right template using select_response(intent, text).
    """
    if intent is None:
        intent = predict_intent(text)
    return select_response(intent, text)


# Apply response generation to golden set dataframe
df["predicted_response"] = [
    select_response(intent, text)
    for intent, text in zip(df["predicted_intent"], df["customer_text"])
]


# ============================================================
# 6. EVALUATION FUNCTION
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


# ============================================================
# 7. EVALUATE FULL GOLDEN SET
# ============================================================

evaluate(
    df,
    "Baseline 2 — Full Golden Set"
)


# ============================================================
# 9. SHOW INDIVIDUAL PREDICTIONS & RESPONSES
# ============================================================

results = df[
    [
        "thread_id",
        "customer_text",
        "intent",
        "predicted_intent",
        "predicted_response",
        "escalate_to_human"
    ]
].copy()

results["correct"] = (
    results["intent"] == results["predicted_intent"]
)

print("\nSample predictions with responses:")
for _, row in results.head(5).iterrows():
    print("-" * 60)
    print(f"Thread ID        : {row['thread_id']}")
    print(f"Customer Text    : {row['customer_text'][:120]}...")
    print(f"Golden Intent    : {row['intent']}")
    print(f"Predicted Intent : {row['predicted_intent']} (Correct: {row['correct']})")
    print(f"Escalate to Human: {row['escalate_to_human']}")
    print(f"Generated Response:\n{row['predicted_response']}")


# ============================================================
# 10. SAVE RESULTS
# ============================================================

results.to_csv(
    "Data/baseline2_keyword_results.csv",
    index=False
)

print(
    "\nSaved predictions to Data/baseline2_keyword_results.csv"
)