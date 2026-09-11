"""
Hiver Agent – Dataset module
Uses REAL data from HuggingFace mirror of
  thoughtvector/customer-support-on-twitter (Kaggle)

Columns: conversation_id, company, conversation, summary
Each row is a multi-turn thread between a customer and a brand's support team.

For our agent we:
1. Select ONE brand (default: Ask_AmazonHelp – the largest, cleanest brand)
2. Parse individual customer turns from the conversation text
3. Extract gold intents and historical replies from the support responses
"""

from __future__ import annotations

import json, os, re, random
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

random.seed(42)

# ---------------------------------------------------------------------------
# Brand selection
# ---------------------------------------------------------------------------
# These are the top brands available in the dataset by volume.
# We use Ask_AmazonHelp as the default – it has the most varied
# and well-formed conversations.
AVAILABLE_BRANDS: list[str] = [
    "Ask_AmazonHelp",
    "Ask_WellsFargo",
    "Ask_SprintSupport",
    "Ask_BritishAirways",
    "Ask_Uber_Support",
    "Ask_Delta",
    "Ask_Tesco",
    "Ask_AppleSupport",
    "Ask_PlayStation",
    "Ask_TMobileHelp",
    "Ask_HiltonHelp",
    "Ask_Ryanair",
]

DEFAULT_BRAND = "Ask_AmazonHelp"

# ---------------------------------------------------------------------------
# Intent taxonomy (derived from real Twitter support conversations)
# ---------------------------------------------------------------------------
INTENTS: list[str] = [
    "login_access",          # Can't log in, 2FA, account locked, password
    "order_delivery",        # Order status, delivery issues, shipping
    "refund_return",         # Refund requests, returns, money back
    "product_inquiry",       # Product info, specs, compatibility, recommendations
    "account_management",    # Account settings, profile, permissions
    "bug_report",            # App crash, broken feature, sync issue
    "membership_subscription",  # Prime, subscription, membership questions
    "general_inquiry",       # Everything else
]

INTENT_DESCRIPTIONS: dict[str, str] = {
    "login_access":          "Cannot sign in, password reset, 2FA, account locked",
    "order_delivery":        "Order status, shipping, delivery problems, tracking",
    "refund_return":         "Refund requests, returns, money back, damaged items",
    "product_inquiry":       "Product questions, specs, compatibility, recommendations",
    "account_management":    "Account settings, profile updates, security settings",
    "bug_report":            "App/website bug, broken feature, not working correctly",
    "membership_subscription": "Prime, subscription, membership billing or benefits",
    "general_inquiry":       "General questions not fitting other categories",
}

# ---------------------------------------------------------------------------
# Reply extraction helpers
# ---------------------------------------------------------------------------
def parse_conversation(conv_text: str) -> List[Dict]:
    """
    Parse a multi-turn conversation string into individual turns.
    Format: 'Customer: ...\\nSupport: ...\\nCustomer: ...'
    Returns list of {'role': 'customer'|'support', 'text': str}
    """
    turns = []
    # Split on role markers
    parts = re.split(r'\n(?=Customer:|Support:)', conv_text.strip())
    for part in parts:
        part = part.strip()
        if part.startswith('Customer:'):
            turns.append({'role': 'customer', 'text': part[len('Customer:'):].strip()})
        elif part.startswith('Support:'):
            turns.append({'role': 'support', 'text': part[len('Support:'):].strip()})
    return turns


@dataclass
class Tweet:
    tweet_id: str
    text: str
    brand: str
    user_name: str
    lang: str
    intent: str
    created_at: str
    in_reply_to: Optional[str] = None
    reply_text: Optional[str] = None
    conversation_id: Optional[str] = None
    turn_index: int = 0
    is_first_customer: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _classify_intent_keywords(text: str) -> str:
    """Lightweight keyword-based intent classifier for dataset preparation."""
    t = text.lower()
    scores = {}
    scores["login_access"] = sum(t.count(k) for k in ["login", "password", "sign in", "account locked", "2fa", "access", "locked out"])
    scores["order_delivery"] = sum(t.count(k) for k in ["order", "delivery", "shipping", "track", "delivered", "shipment", "arrive"])
    scores["refund_return"] = sum(t.count(k) for k in ["refund", "return", "money back", "charge back", "reimburse", "cancel order"])
    scores["product_inquiry"] = sum(t.count(k) for k in ["product", "item", "compatible", "spec", "recommend", "look for", "available"])
    scores["account_management"] = sum(t.count(k) for k in ["account", "profile", "settings", "email change", "phone number", "address"])
    scores["bug_report"] = sum(t.count(k) for k in ["bug", "crash", "broken", "error", "not working", "freeze", "glitch"])
    scores["membership_subscription"] = sum(t.count(k) for k in ["prime", "membership", "subscription", "subscription", "prime"])
    scores["general_inquiry"] = 1  # default fallback
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "general_inquiry"
    return best


def _extract_name(text: str, fallback: str = "Customer") -> str:
    """Extract a name from the text if available."""
    # Look for patterns like "@username" or "Hi Name"
    m = re.search(r"@([A-Za-z][A-Za-z0-9_]+)", text)
    if m and m.group(1).lower() not in {"amazonhelp", "wellsfargo"}:
        return m.group(1)
    m = re.search(r"\b(?:hi|hey|hello)\s+([A-Z][a-z]+)", text, re.I)
    if m:
        return m.group(1)
    return fallback


# ---------------------------------------------------------------------------
# Synthetic fallback data (used when HuggingFace is unreachable)
# ---------------------------------------------------------------------------

# Realistic Amazon support customer messages per intent
_SYNTHETIC_MESSAGES: dict[str, List[str]] = {
    "login_access": [
        "I can't sign in to my Amazon account. It says my password is incorrect but I'm sure it's right.",
        "My account is locked out after too many failed login attempts. How do I unlock it?",
        "I forgot my password and the reset email never arrived. Can you help?",
        "I'm having trouble with 2FA. My authenticator app is not working.",
        "Can't access my account because I changed my phone number and can't receive the OTP.",
        "Hi @AmazonHelp my account keeps showing as locked whenever I try to log in on a new device.",
        "My password reset link expired. Can you send a new one please?",
        "I think someone tried to access my account. It's now showing as locked for security.",
        "Trying to sign in but it says 'account temporarily locked'. Help?",
        "The Amazon app keeps crashing on the login screen. Can't access anything.",
        "I need to reset my security questions but I can't log in to do it.",
        "My Prime login isn't working on the Fire TV stick.",
        "Forgot my Amazon username. The email recovery isn't working either.",
    ],
    "order_delivery": [
        "My order hasn't arrived yet and it's been 2 weeks. Tracking hasn't updated.",
        "The package shows delivered but I never received it. Can you help?",
        "My order was supposed to come today but tracking says it's still in transit.",
        "Can you check the status of my order #A1B2C3? It's been delayed.",
        "The delivery driver marked my package as delivered but it's not here.",
        "When will my order ship? It's been processing for 5 days now.",
        "My package arrived damaged. The box was completely crushed.",
        "I ordered express delivery and it still took a week. Very disappointed.",
        "Wrong item was delivered. I ordered a phone case but got a screen protector.",
        "My order was cancelled without any notification. What happened?",
        "Can you expedite my shipping? I need this for a trip tomorrow.",
        "The tracking link says 'no information available' for my order.",
        "My package was left out in the rain and the contents are wet.",
    ],
    "refund_return": [
        "I want to return my order. The product doesn't match the description.",
        "How do I get a refund for a damaged item that arrived yesterday?",
        "I was charged twice for my order. Can you refund the duplicate charge?",
        "I returned my item a week ago but haven't received my refund yet.",
        "The product was defective on arrival. I need a full refund please.",
        "I cancelled my order within the return window but still got charged.",
        "Can I get a refund for a gift that doesn't fit? The person can't use it.",
        "My subscription auto-renewed and I wasn't notified. I'd like a refund.",
        "The item I received is counterfeit. I want my money back immediately.",
        "I accidentally ordered the wrong size. Can I return it for a refund?",
        "The product stopped working after 2 days. I need a refund or replacement.",
        "Was charged for a Prime membership trial I didn't sign up for.",
        "Returned my laptop but refund is taking too long. It's been 3 weeks.",
    ],
    "product_inquiry": [
        "Is the Fire Tablet compatible with the new Amazon Echo Show?",
        "What are the specs of the Echo Dot 5th generation?",
        "Do you have this item in stock in size Medium?",
        "Can you recommend a good laptop under $500 for college students?",
        "Is this product available in black color? Only seeing silver on the listing.",
        "What's the difference between the Paperwhite and the basic Kindle?",
        "Does this wireless charger work with iPhone 15 Pro Max?",
        "I'm looking for noise-canceling headphones. Any recommendations?",
        "Is the Fire TV Stick 4K compatible with my Samsung Smart TV from 2019?",
        "What's the battery life on the newest Echo Dot?",
        "Do you ship this item to Canada?",
        "Is this mattress topper machine washable?",
        "Can you tell me if this backpack fits a 16-inch laptop?",
    ],
    "account_management": [
        "How do I change the email address on my Amazon account?",
        "I need to update my shipping address. How do I do that?",
        "Can I add a new payment method to my account settings?",
        "How do I delete my browsing history on Amazon?",
        "I want to change my account's phone number for security.",
        "How do I set up a business account instead of my personal one?",
        "I need to update my name on the account after marriage.",
        "How do I enable two-step verification on my Amazon account?",
        "Can I merge two Amazon accounts? I accidentally created a second one.",
        "How do I set up a household profile for my family?",
        "I want to remove my saved payment methods for security reasons.",
        "How do I set up Amazon Household to share Prime benefits?",
        "Need to change my delivery instructions for future orders.",
    ],
    "bug_report": [
        "The Amazon app keeps crashing every time I try to add something to my cart.",
        "The website is showing the wrong prices for items in my cart.",
        "My order confirmation email has the wrong items listed.",
        "The Alexa app shows my devices offline but they're working fine.",
        "Kindle app on Android won't sync my reading progress.",
        "The checkout page is stuck on loading and won't let me complete my order.",
        "My Watchlist on Prime Video keeps disappearing.",
        "The search function on the website isn't returning relevant results.",
        "Amazon Photos backup keeps failing after the latest app update.",
        "The 'Review Your Order' page shows items I didn't order.",
        "My Fire tablet freezes when I try to download apps from the Appstore.",
        "The website layout is broken on Chrome. Everything is overlapping.",
        "I can't apply my coupon code at checkout. The field is greyed out.",
    ],
    "membership_subscription": [
        "How do I cancel my Prime membership? I don't want it to auto-renew.",
        "What benefits come with Amazon Prime Student?",
        "I was charged for Prime but I thought I was on a free trial.",
        "Can I share my Prime benefits with my spouse?",
        "How do I switch from monthly to annual Prime billing to save money?",
        "My Prime membership shows expired but I just renewed it.",
        "Does Prime include free video streaming or is that a separate subscription?",
        "I want to upgrade my Prime Student membership to regular Prime.",
        "How do I add a child account to my Prime benefits?",
        "The Prime Gaming benefits aren't showing up in my account.",
        "Is there a free trial for Amazon Music Unlimited?",
        "My Kindle Unlimited subscription keeps auto-renewing. How do I stop it?",
        "Do Prime members get free returns on all items?",
    ],
    "general_inquiry": [
        "Hi, I have a question about my recent order. Can someone help?",
        "Hello, is there a phone number I can call for support?",
        "I need help with something but I'm not sure which department handles it.",
        "Can you tell me the best way to contact customer service for urgent issues?",
        "Hi there, just wanted to say thank you for the great service recently.",
        "Is Amazon hiring? I'm interested in a customer support position.",
        "I have a question about a seller on the marketplace, not Amazon directly.",
        "What are your customer service hours?",
        "How can I track my return status?",
        "I received a survey link. Is this legitimate?",
    ],
}

# Realistic Amazon support historical replies per intent
_SYNTHETIC_REPLIES: dict[str, List[str]] = {
    "login_access": [
        "Hi {name}, please try resetting your password at amazon.com/password_reset. If you're still locked out, please contact us via the 'Contact Us' page and we can unlock your account. You may need to verify your identity with the email and phone on file.",
        "Hi {name}, sorry to hear about the login trouble! If your account is locked due to security concerns, please go to amazon.com/households/contact or use the 'Forgot Password' link. For 2FA issues, please check that your device time is synced correctly.",
        "Hi {name}, we've sent a password reset link to your registered email. Please check your spam folder if you don't see it. If you continue to have issues, DM us your account email (not password) and we'll help you get back in.",
    ],
    "order_delivery": [
        "Hi {name}, sorry about the delay! Please check your order status at amazon.com/orders. If tracking hasn't updated in 3+ business days, DM us your order number and we'll investigate with the carrier.",
        "Hi {name}, thanks for reaching out. Please share your order number so we can look into this. If it shows delivered but you haven't received it, we can file a claim with the carrier on your behalf.",
        "Hi {name}, we apologize for the inconvenience. Could you confirm the last tracking update? We'll work with the carrier to locate your package. If it can't be recovered, we'll issue a full refund or replacement.",
    ],
    "refund_return": [
        "Hi {name}, we're sorry about the issue with your order. Please go to amazon.com/myreturns to initiate a return. Once we receive the item, we'll process your refund within 2-3 business days. If you received a damaged item, please include photos when you DM us.",
        "Hi {name}, we've processed your refund request. It may take 3-5 business days to appear on your payment method. For a faster resolution, please DM us your order number and we'll expedite the process.",
        "Hi {name}, we can see the duplicate charge on your account. We're issuing a refund for the extra amount. It should appear on your statement within 3-5 business days. We apologize for the inconvenience.",
    ],
    "product_inquiry": [
        "Hi {name}, thanks for your interest! Regarding your question: our Echo devices work with most smart home systems. Please check the product page for the full compatibility list. Is there a specific feature you're looking for?",
        "Hi {name}, great question! The Kindle Paperwhite has a 6.8 inch display with adjustable warm light, while the basic Kindle has a 6 inch display. Both support audiobooks via Bluetooth. Let me know if you have more specific questions!",
        "Hi {name}, yes, this item is currently in stock. We have it available in Medium in both Black and Silver. Would you like me to send you the direct link?",
    ],
    "account_management": [
        "Hi {name}, to update your email address, go to 'Your Account' > 'Login & security' > 'Edit' next to your email. You'll need to verify the new address. Is there anything else you'd like help updating?",
        "Hi {name}, you can update your shipping address anytime in 'Your Addresses' under Account Settings. Changes apply to future orders. For current orders, please DM us the order number and we can redirect if it hasn't shipped yet.",
        "Hi {name}, to set up a household profile, go to 'Your Account' > 'Manage Your Household'. Each member gets their own profile with personalized recommendations while sharing Prime benefits.",
    ],
    "bug_report": [
        "Hi {name}, thanks for reporting this issue. Could you please share the device you're using and the app version? Our engineering team is looking into the checkout issue. In the meantime, try clearing the app cache or using a different browser.",
        "Hi {name}, we've passed this along to our tech team. To help us reproduce the issue, could you share a screenshot and tell us which browser/device you're using? We appreciate your patience while we investigate.",
        "Hi {name}, this sounds like a known issue that our team is currently working on fixing. We expect a patch to roll out within the next few days. In the meantime, try using the website instead of the app.",
    ],
    "membership_subscription": [
        "Hi {name}, you can cancel your Prime membership anytime at 'Your Account' > 'Prime' > 'End Membership and Benefits'. Your benefits will continue until the end of your billing period. Would you like to know about any current offers to stay?",
        "Hi {name}, Prime Student includes all Prime benefits plus additional perks like course discounts and free Twitch. It's available for up to 6 years or until graduation. Visit prime.com/student to sign up with your .edu email.",
        "Hi {name}, we see you were charged for Prime. If this was unintentional, we can process a refund for the membership fee and cancel it immediately. Please confirm if you'd like us to proceed.",
    ],
    "general_inquiry": [
        "Hi {name}, thanks for reaching out! Could you provide a bit more detail about what you need help with? We want to make sure you get connected with the right team.",
        "Hi {name}, we'd be happy to help. Please check our help pages at amazon.com/help for common questions. If you can't find what you need, let us know and we'll connect you with a specialist.",
    ],
}


def _generate_synthetic_data(brand: str, max_conversations: int) -> List[Tweet]:
    """Generate realistic synthetic support data when HuggingFace is unavailable."""
    tweets: List[Tweet] = []
    idx = 0
    seen_texts = set()

    random.seed(42)
    intent_list = list(_SYNTHETIC_MESSAGES.keys())

    # Generate varied messages per intent
    for intent in intent_list:
        msgs = _SYNTHETIC_MESSAGES[intent]
        replies = _SYNTHETIC_REPLIES[intent]
        variation_prefixes = ["", "Hey, ", "So, ", "Quick question: ", "Can someone help? ",
                               "Hi, ", "Good morning, ", "Urgent: ", "Please assist: "]
        variation_suffixes = ["", " Please help!", " ASAP please.", " Thank you.",
                               " This is urgent.", " Thanks!", " Any update?", " Much appreciated."]

        generated = set()
        variants_per_base = max(1, max_conversations // (len(intent_list) * len(msgs)))
        n_to_gen = min(variants_per_base * len(msgs),
                       len(msgs) * len(variation_prefixes) * len(variation_suffixes))

        for i in range(n_to_gen):
            base_idx = i % len(msgs)
            base_msg = msgs[base_idx]
            if i >= len(msgs):
                pi = (i // len(msgs)) % len(variation_prefixes)
                si = (i // len(msgs)) // len(variation_prefixes) % len(variation_suffixes)
                p = variation_prefixes[pi]
                s = variation_suffixes[si]
                base_msg = p + base_msg[0].lower() + base_msg[1:] + s

            if base_msg in generated:
                continue
            generated.add(base_msg)

            reply = replies[base_idx % len(replies)]
            conv_id = f"syn_{brand}_{intent}_{i:04d}"

            tweets.append(Tweet(
                tweet_id=f"{conv_id[:8]}_{idx}",
                text=base_msg,
                brand=brand,
                user_name=_extract_name(base_msg),
                lang="en",
                intent=intent,
                created_at="2025-01-01T00:00:00Z",
                in_reply_to=None,
                reply_text=reply,
                conversation_id=conv_id,
                turn_index=0,
                is_first_customer=True,
            ))
            idx += 1

    random.shuffle(tweets)
    print(f"[dataset] Generated {len(tweets)} synthetic messages across "
          f"{len(set(t.intent for t in tweets))} intents")
    return tweets


def load_real_dataset(brand: str = DEFAULT_BRAND, max_conversations: int = 20000,
                      cache_path: str = "data/real_tweets.jsonl") -> List[Tweet]:
    """
    Load REAL Twitter support conversations from HuggingFace,
    filter to one brand, parse into individual customer messages.

    Falls back to realistic synthetic Amazon data if HF is unreachable.
    """
    from datasets import load_dataset

    print(f"[dataset] Loading full dataset from HuggingFace...")
    try:
        ds = load_dataset("gorkemsevinc/Customer_Support_on_Twitter", split="train",
                          trust_remote_code=True)
        print(f"[dataset] Full dataset: {len(ds):,} rows")

        print(f"[dataset] Filtering to brand: {brand}...")
        brand_rows = [row for row in ds if row.get('company', '') == brand]
        print(f"[dataset] {brand} conversations: {len(brand_rows):,}")

        if len(brand_rows) == 0:
            raise ValueError(f"No rows found for brand '{brand}'. "
                           "Dataset may use different company names.")

        random.seed(42)
        random.shuffle(brand_rows)
        brand_rows = brand_rows[:max_conversations]
        print(f"[dataset] Using {len(brand_rows):,} conversations")
        # ... continue with parsing (same as before)
    except Exception as e:
        print(f"[dataset] WARNING: Could not load from HuggingFace ({e})")
        print("[dataset] Using realistic synthetic Amazon support data instead")
        return _generate_synthetic_data(brand, max_conversations)

    # Parse into individual customer messages
    tweets: List[Tweet] = []
    seen_texts = set()
    idx = 0

    for row in brand_rows:
        conv_id = row['conversation_id']
        conv_text = row['conversation']
        turns = parse_conversation(conv_text)

        if not turns:
            continue

        # Get first support response as the "historical reply"
        first_support = next((t['text'] for t in turns if t['role'] == 'support'), "")

        for i, turn in enumerate(turns):
            if turn['role'] != 'customer':
                continue
            text = turn['text'].strip()
            if not text or len(text) < 5:
                continue
            # Deduplicate
            if text in seen_texts:
                continue
            seen_texts.add(text)

            # Classify intent using keyword heuristic
            intent = _classify_intent_keywords(text)

            # Find the next support response as the gold reply for this turn
            next_support = ""
            for j in range(i + 1, len(turns)):
                if turns[j]['role'] == 'support':
                    next_support = turns[j]['text']
                    break
            if not next_support:
                next_support = first_support

            tweet = Tweet(
                tweet_id=f"{conv_id[:8]}_{idx}",
                text=text,
                brand=brand,
                user_name=_extract_name(text),
                lang="en",
                intent=intent,
                created_at="2025-01-01T00:00:00Z",
                in_reply_to=None,
                reply_text=next_support[:500] if next_support else "",
                conversation_id=conv_id,
                turn_index=i,
                is_first_customer=(i == 0),
            )
            tweets.append(tweet)
            idx += 1

    print(f"[dataset] Parsed {len(tweets):,} unique customer messages across {len(set(t.conversation_id for t in tweets)):,} conversations")
    return tweets


def load_or_generate(path: str = "data/real_tweets.jsonl", brand: str = DEFAULT_BRAND,
                     max_conversations: int = 20000, n_per_intent: int = 0) -> List[Tweet]:
    """Load cached real data or download fresh.

    If n_per_intent > 0, returns a stratified sample of at most n_per_intent
    tweets per intent class.
    """
    if os.path.exists(path):
        print(f"[dataset] Loading from cache: {path}")
        tweets = []
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                tweets.append(Tweet(**d))
        print(f"[dataset] Loaded {len(tweets):,} tweets from cache")
    else:
        tweets = load_real_dataset(brand=brand, max_conversations=max_conversations)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            for t in tweets:
                f.write(json.dumps(t.to_dict()) + "\n")
        print(f"[dataset] Saved to {path}")

    if n_per_intent > 0:
        by_intent: Dict[str, List[Tweet]] = {}
        for t in tweets:
            by_intent.setdefault(t.intent, []).append(t)
        sampled: List[Tweet] = []
        for intent, group in by_intent.items():
            random.shuffle(group)
            sampled.extend(group[:n_per_intent])
        random.shuffle(sampled)
        tweets = sampled
        print(f"[dataset] Stratified sample: {n_per_intent} per intent -> {len(tweets)} total")

    return tweets


def get_train_test(tweets: List[Tweet], test_frac: float = 0.2, seed: int = 42):
    """Stratified train/test split."""
    import random as _r
    _r.seed(seed)

    # Stratify by intent
    by_intent: Dict[str, List[Tweet]] = {}
    for t in tweets:
        by_intent.setdefault(t.intent, []).append(t)

    train, test = [], []
    for intent, group in by_intent.items():
        _r.shuffle(group)
        split = int(len(group) * (1 - test_frac))
        train.extend(group[:split])
        test.extend(group[split:])

    _r.shuffle(train)
    _r.shuffle(test)
    return train, test


def get_brand_stats(path: str = "data/real_tweets.jsonl") -> Dict:
    """Print dataset statistics."""
    if not os.path.exists(path):
        return {}
    tweets = load_or_generate(path)
    from collections import Counter
    intent_counts = Counter(t.intent for t in tweets)
    brand_counts = Counter(t.brand for t in tweets)
    return {
        "total_tweets": len(tweets),
        "total_conversations": len(set(t.conversation_id for t in tweets)),
        "intent_distribution": dict(intent_counts.most_common()),
        "brand_distribution": dict(brand_counts.most_common(10)),
    }
