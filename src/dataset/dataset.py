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
AVAILABLE_BRANDS: list[str] = [
    "Ask_AmazonHelp", "Ask_WellsFargo", "Ask_SprintSupport",
    "Ask_BritishAirways", "Ask_Uber_Support", "Ask_Delta",
    "Ask_Tesco", "Ask_AppleSupport", "Ask_PlayStation",
    "Ask_TMobileHelp", "Ask_HiltonHelp", "Ask_Ryanair",
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
    """Parse a multi-turn conversation string into individual turns."""
    turns = []
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
    scores["general_inquiry"] = 1
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "general_inquiry"
    return best


def _extract_name(text: str, fallback: str = "Customer") -> str:
    m = re.search(r"@([A-Za-z][A-Za-z0-9_]+)", text)
    if m and m.group(1).lower() not in {"amazonhelp", "wellsfargo"}:
        return m.group(1)
    m = re.search(r"\b(?:hi|hey|hello)\s+([A-Z][a-z]+)", text, re.I)
    if m:
        return m.group(1)
    return fallback


# ---------------------------------------------------------------------------
# Synthetic fallback data (diverse base messages, no mechanical variations)
# ---------------------------------------------------------------------------
# ~50 genuinely diverse messages per intent covering paraphrases, edge cases,
# ambiguous cases, short/noisy inputs, and misspellings.
# This pool is used for both training and golden set construction.
# The golden set samples 29 per intent from this pool, ensuring diversity.

_SYNTHETIC_MESSAGES: dict[str, List[str]] = {
    "login_access": [
        # Direct problems
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
        # Paraphrases and edge cases
        "username n password both forgotten fml help pls @AmazonHelp",
        "keeps saying wrong password but ik it's right smh",
        "locked out of my account again. This keeps happening every few weeks.",
        "the verification code never comes thru. tried like 5 times already",
        "need to change my recovery email but i'm locked out of the account. ironic.",
        "can't sign in on my new phone. old phone broke and i didn't transfer auth app.",
        "got locked out after changing my password. now can't log in with old OR new.",
        "the password reset link says expired the second i click it. broken?",
        "2fa code taking 10+ minutes to arrive. by then it's expired.",
        "account says locked for suspicious activity but it was just me logging in from work.",
        "every time i try to reset, the email goes to an old address i dont use anymore.",
        "my son locked my account trying to guess my password. now im locked out too.",
        "cant log in from my kindle. works fine on phone though.",
        "the captcha on the login page wont load. stuck forever.",
        "hey my account was hacked and now the email is changed. need urgent help.",
        "forgot everything - username, password, recovery email. what do i do?",
        "new password wont work keeps saying invalid. the requirements say 8 chars min and mine is 10.",
        "ok so i changed my password yesterday and now nothing will log in. browser, app, all broken",
        "the login page keeps refreshing on its own. cant even type my password.",
        "the forgot username page says 'try again later' every time i click it",
        "my wife added her phone as a recovery option and now i cant verify its me",
        "receiving someone elses verification codes. thinks their account is linked to mine somehow",
        "can you manually reset my account? i live overseas and cant receive calls or texts",
        "account locked but the unlock page isnt loading. help?",
        "hi i locked myself out intentionally to reset settings but now i need back in lol",
        "still waiting on the unlock email. its been 2 hours. come on guys",
        "my work network blocks the authentication page. cant verify from office.",
        "forgot pw but the recovery phone is old. need to update recovery info first but i cant log in",
        "asked for a password reset 3 times today. still no emails. checked spam too.",
        "totally locked out. cant verify, cant reset, cant do anything. human help?",
        "the 2fa app says my amazon credentials are invalid but i can log in on the website fine",
        "login works on desktop but NOT on the iphone app. tried reinstalling twice.",
        "my account got locked during a sale and i missed the deal. frustrating.",
        "clicked 'forgot password' 10 mins ago and nothing. no email, no text, nothing.",
        "security email says new device logged in but it WAS me. why am i locked then?",
        "password requirements changed and my old password format no longer works.",
        "the account recovery form says to check my email for instructions but i never get them",
        "i think theres a bug. password reset says successful but login still fails.",
        "can you remove the lock? i can verify identity with credit card on file.",
    ],
    "order_delivery": [
        # Direct problems
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
        # Paraphrases and edge cases
        "where my order tho? been 10 days and still says processing. ridiculous.",
        "delivery guy just left the box on my porch in the rain. contents soaked.",
        "tracking hasn't moved in 4 days. stuck at 'departed facility'. whats going on?",
        "ordered a gift that needs to arrive by friday. its wednesday and still in another state.",
        "package marked delivered but i was home all day. never saw the driver.",
        "the delivery window said 9-5 but they came at 7am while i was sleeping.",
        "my neighbor said they signed for it but i never received anything.",
        "order status just changed to cancelled out of nowhere. i was charged too!",
        "express shipping was supposed to be 1-2 days. its been 8. still waiting.",
        "box came completely smashed. the item inside is cracked. need a replacement asap.",
        "ordered black, got white. not even the right color.",
        "i paid for same-day delivery and it came the next day. not cool.",
        "the tracking number doesn't work at all. just says 'invalid'.",
        "my order from last month still says 'pending'. is this normal?",
        "delivery guy threw the package over my fence. broke something fragile inside.",
        "can i change my delivery address? i moved and the package is going to my old place.",
        "ordered a set of 3 and only received 1. the rest says 'backordered' but that wasnt mentioned",
        "the package was left at the wrong house entirely. my neighbor 2 doors down got it.",
        "how do i know if my order was actually shipped or just prepared? its been 6 days.",
        "i need this laptop charger before my flight tomorrow. anyway to speed it up?",
        "delivery attempted but i got no door tag or anything. how do i reschedule?",
        "my frozen items melted during delivery. box was sitting in the sun for hours.",
        "just got a delivery notification but nothing was left at my door.",
        "the order confirmation says estimated delivery yesterday but still no package.",
        "3 packages all missing. 2 marked delivered, 1 stuck in transit for 2 weeks.",
        "i returned something last month and the refund still hasn't appeared on my card.",
        "package was stolen off my porch. can you help with a claim?",
        "ordered books for my kid's school project. due tomorrow. delivery says 'lost in transit'.",
        "my package is literally at the local facility 10 mins away but hasn't moved in 3 days.",
        "they delivered to the wrong apartment number. theres a 3 and a 5 that look similar.",
        "express delivery option was grayed out at checkout. why? i need fast shipping.",
        "the order confirmation email had the wrong items listed. worried about what im actually getting",
        "can you ship my pending order via overnight? i'll pay the difference.",
        "package came today but wrong size. ordered medium got large. annoying but not a dealbreaker",
        "my international order has been in customs for 2 weeks. how long does that usually take?",
        "ordered as a gift but they put the price sticker on the box. really?",
        "the delivery guy said he left it with my doorman but the doorman says nope. now what?",
        "i can see the delivery truck on my security cam driving past my house without stopping",
        "marked as 'out for delivery' at 8am its now 7pm and nothing. where is it?",
        "got the shipping notification but no tracking number. how do i track it?",
        "ordered 2 of the same item, one came and was wrong item, the other never shipped",
    ],
    "refund_return": [
        # Direct problems
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
        # Paraphrases and edge cases
        "double charged on my card. same order shows up twice. fix?",
        "this broke after wearing it once. $80 for one use is crazy.",
        "the refund was supposed to hit my account 3-5 business days ago. still waiting.",
        "ordered a medium, got listed as large but the tag says medium. confused.",
        "the product page said 'premium quality' and what i got feels like dollar store plastic.",
        "wrapped the gift and gave it away. turns out it was the wrong color. can i still return?",
        "my return was accepted but i still got charged. been a month now.",
        "the item arrived and the seal was already broken. looks used.",
        "i accidentally bought 2 subscriptions. want to cancel one and get refund for it.",
        "the replacement they sent was ALSO defective. third try now.",
        "was supposed to get a refund for a cancelled order but charge still pending on my bank.",
        "ordered a set of 4, received only 2. the rest say discontinued. but you charged me for 4.",
        "my return label expired before i could use it. the post office was closed.",
        "gift receipt didnt work for return. the store said it was a regular receipt.",
        "charged me for the free trial i never signed up for. cancel and refund please.",
        "the refund shows as 'processed' but my bank says they never received it.",
        "item arrived 2 weeks late so it was useless. event already passed. refund?",
        "the product is nothing like the photos. totally different design.",
        "opened the box and there was nothing inside. empty box. refund?",
        "returned within 30 days but still charged. the return tracking says delivered.",
        "my wife ordered the wrong size and i need to return it but lost the original packaging.",
        "was promised a refund for a late delivery but got a $5 coupon instead. not the same.",
        "the item was damaged in shipping and the return shipping is gonna cost me $15. seriously?",
        "i canceled the same-day subscription but got charged for a full month.",
        "the return window closed 2 days ago and i just realized the product is broken.",
        "bought the extended warranty and it wont cover the repair. feels like a scam.",
        "ordered shoes in my usual size and they fit like 2 sizes too small. different sizing entirely",
        "the refund went to a card i closed last month. where does the money go now?",
        "marked as return received 5 days ago but refund still pending. getting frustrated.",
        "the item showed up with water damage. box was soaked through. the product might be fine but im not risking it",
        "ordered the wrong color by accident. opened it and realized. 2 minutes after delivery. return ok?",
        "my refund was approved but it was less than what i paid. tax and shipping not included?",
        "gift was the wrong item entirely. ordered a book, got a board game. not even close",
        "the return portal keeps giving me an error. cant even start the return process.",
        "3 of my 5 items arrived damaged. do i return all 3 at once or separately?",
        "cancelled the order before it shipped but still seeing a charge on my statement.",
        "i returned it in perfect condition within the window but still got charged a restocking fee. why?",
        "the replacement item was also wrong. this is the second time. at this point just refund me",
        "wants me to pay return shipping for a defective item. that should be free right?",
        "the refund appeared but then disappeared from my account the next day. what happened?",
        "was charged a 'convenience fee' for returning in store. since when?",
        "returned the item 2 weeks ago. delivery confirmation shows received. still no refund.",
    ],
    "product_inquiry": [
        # Direct questions
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
        # Paraphrases and edge cases
        "whats the diff between the 11 and 12 inch ipad pro? which one for drawing?",
        "does this coffee maker work with ground coffee or only pods?",
        "need a standing desk that fits in a small apartment. any suggestions under $300?",
        "will this tent fit 4 adults comfortably or is it more of a 2-person tent?",
        "the product description says waterproof but the reviews say otherwise. which is it?",
        "can this bluetooth speaker pair with another one for stereo sound?",
        "looking for a gift for my mom who has arthritis. easy to use, large buttons?",
        "does the kindle paperwhite come in 32gb? only seeing 16gb on the listing.",
        "is the fire hd 10 good for reading comics or is the screen too small?",
        "can i use this laptop for photo editing? i need at least 16gb ram.",
        "the color options listed dont match whats in the photos. which ones are actually available?",
        "do you sell replacement parts for this? specifically the charging port.",
        "what age range is this toy recommended for? listing says 3+ but reviews mention small parts",
        "is this yoga mat eco-friendly? listing says 'non-toxic' but not specific.",
        "can the echo dot connect to my existing bose sound system?",
        "does this camera need a separate memory card or does it come with one?",
        "looking for a portable charger that works with both android and iphone. recommendations?",
        "the product page says 'bestseller' but has only 23 reviews. best in what category?",
        "will this vacuum work on hardwood floors or only carpet?",
        "does the fire tv stick support 4k hdr on my vizio tv?",
        "is this cookware oven safe up to what temperature? listing says nothing about it.",
        "can the echo show make video calls to facetime? or only alexa devices?",
        "this headphone listing says noise cancelling but its $30. is that real or marketing?",
        "looking for an e-reader with a physical page-turn button. does kindle have that?",
        "will this standing desk frame support a 35 inch monitor? max weight says 180 lbs.",
        "does this camping tent need separate poles or are they included?",
        "can i read library books on the kindle or only books bought from amazon?",
        "is this protein powder suitable for vegans? the ingredients list is confusing.",
        "the listing says 'set of 6' but the product title says 'set of 4'. which is correct?",
        "does this camera come with a case or is that sold separately?",
        "my kids have android tablets. will this fire tablet work with their apps?",
        "looking for a portable battery pack for camping. needs to charge phone and headlamp.",
        "is this watch face compatible with the new apple watch ultra?",
        "can you compare this to the previous version? is it worth upgrading?",
        "the description mentions 'premium leather' but the details say 'vegan leather'. which one?",
        "does this printer work with chromebook or only windows/mac?",
        "i have a samsung phone. will these wireless earbuds work? any compatibility issues?",
        "the listing says this fits queen mattress but the dimensions seem small for queen. can you confirm?",
        "is this air purifier loud on high setting? i need it for the bedroom.",
        "the product has 2.5 stars but is marked 'amazons choice'. how does that work?",
        "can i use this telescope for astrophotography or just casual viewing?",
        "does this keyboard work with xbox or only pc?",
        "theres a cheaper version of the same product. whats the difference between them?",
        "i need a gift box for a 10 year old girl. any ideas under $25?",
        "will this screen protector fit the iphone 15 plus or only the regular 15?",
        "the listing shows two different images for the same ASIN. which version do i get?",
        "does this lunch bag keep food cold for 8+ hours? i need it for a full work day.",
        "is the echo pop available in blue? only seeing white and black in stock.",
    ],
    "account_management": [
        # Direct actions
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
        # Paraphrases and edge cases
        "how do i remove a credit card from my account? old card expired.",
        "can i change the name on my account? recently got married and my last name changed.",
        "my old email was hacked so i need to change the email on my amazon account too.",
        "how do i set up a profile for my kids so they dont have access to my payment methods?",
        "the address on my account is wrong. how do i update it for future orders only?",
        "i want to close my account permanently. how do i do that?",
        "can i add a nickname to my account so my family knows which one is mine?",
        "my account shows a different name than what i set. how do i fix it?",
        "how do i make my account private? i dont want my browsing history visible to family members.",
        "i set up two-factor but lost my backup codes. what if i get locked out?",
        "the phone number on my account is old. how do i change it without logging in?",
        "can i create a separate profile for my business purchases under the same login?",
        "how do i turn off the 'save my payment info' setting? its too convenient and dangerous.",
        "my wife and i want to share prime but keep our accounts separate. how?",
        "i accidentally typed my address wrong on 3 recent orders. can i bulk update it?",
        "how do i change my country/region setting? i moved internationally.",
        "the saved addresses are all wrong. how do i clear them and start fresh?",
        "i want to link my amazon account to my audible account. how?",
        "can i set up a business account for tax purposes without losing my personal account?",
        "how do i update my emergency contact info on the account?",
        "my account has my old apartment address. ive moved 2 times since then.",
        "how do i deactivate my account without deleting my order history?",
        "the profile picture on my account wont update. says 'upload failed' every time.",
        "i need to add my partner to my account so they can manage subscriptions too.",
        "can i export my purchase history? doing taxes and need everything from last year.",
        "how do i disable one-click ordering? i keep accidentally buying things.",
        "the default address keeps reverting to my old one. is this a bug?",
        "i want to change my login email but the verification code goes to the old email. stuck.",
        "can i use a po box as my default shipping address?",
        "how do i set up voice purchasing restrictions so my echo dot doesnt order things by accident?",
        "my account shows someone elses name in the greeting. how did that happen?",
        "want to set up a business profile for vendor purchases. separate from my personal account.",
        "how do i change my preferred language? currently getting emails in a language i dont speak.",
        "i need to update my tax info for seller central. where is that in account settings?",
        "the delivery instructions field only lets me type 100 characters. need more.",
        "can i make an account for my elderly parent and set it up so i can manage it remotely?",
        "how do i opt out of all marketing emails? getting like 10 a day.",
        "my saved addresses keep disappearing. had to re-enter them 3 times this month.",
        "can i set up a joint account with my spouse so we both see orders and can manage returns?",
        "how do i change my default payment method? it keeps using an old card.",
        "the address autocomplete keeps suggesting my old neighborhood. how do i clear that?",
        "i want to update my password but the current one says its wrong and i cant reset it.",
        "my account has a bunch of addresses from hotels i stayed at. how do i clean those up?",
        "how do i set up multiple addresses under one account for different family members?",
    ],
    "bug_report": [
        # Direct bugs
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
        # Paraphrases and edge cases
        "app crashes the second i open it. tried reinstalling 3 times. iphone 12 if it matters.",
        "the price in my cart is $20 higher than what i saw on the product page. bug?",
        "my kindle just restarted mid-chapter and lost my place. plus the book now shows as 'not purchased'.",
        "alexa says no devices found but i have 3 echo dots that are clearly online.",
        "the checkout button is grayed out for no reason. cart has items. address is valid. card works.",
        "prime video keeps buffering every 2 minutes even on fast wifi. netflix works fine.",
        "my reading progress syncs from phone to tablet but NOT from tablet to phone. one-way only.",
        "the search bar autocomplete is showing products i searched for 6 months ago. creepy and unhelpful.",
        "photos backup says 'upload failed' for every single photo. works on wifi and cellular.",
        "order summary shows $0 total but says 'payment required' on the next screen.",
        "the app froze during checkout and now it says my order is confirmed but i got no confirmation email.",
        "fire tablet stuck on the amazon logo. boot loop. tried hard reset, still broken.",
        "on desktop chrome, the whole page is zoomed in and cut off. works fine on firefox.",
        "the coupon code i copied from your website says invalid when i paste it. 'already used' but i never used it.",
        "the 'continue browsing' recommendations are all products i already bought last week.",
        "kindle app on samsung tablet crashes specifically when i turn pages. scrolling works fine.",
        "my account shows a $50 charge for something i never ordered. the order details page is blank.",
        "the alexa app update broke routines. they worked fine yesterday.",
        "prime video subtitles are way out of sync. like 3 seconds behind the audio.",
        "the 'saved for later' list randomly deletes items. lost 8 things today.",
        "app keeps logging me out after 5 minutes. have to sign in every single time.",
        "the cart icon shows 3 items but when i open the cart its empty. clearing cache didnt help.",
        "order confirmation says i bought 5 of something. i only bought 1. trying to cancel the other 4.",
        "the website mobile view has buttons stacked on top of each other. cant click anything.",
        "my whisper sync stopped working. audiobook position not saving between devices.",
        "the 'deals of the day' page shows yesterdays deals. the timer says 'expired' for everything.",
        "fire tablet camera app crashes immediately. front and back camera both.",
        "prime membership shows as expired even though i just renewed. says 'upgrade to prime' everywhere.",
        "the account dropdown menu goes behind the search bar. cant see all options.",
        "app freezes on the loading screen after the splash screen. happens every single launch.",
        "the 'buy now' button charges a different price than what was displayed. $18 vs $14.",
        "prime video says 'not available in your region' but ive been watching it here for months.",
        "checkout page forgets my saved address every time. have to re-type it.",
        "the app icon on my home screen shows a notification badge that wont go away. 99+ for weeks.",
        "kindle book shows 'downloaded' but when i open it, its just a blank page.",
        "the 'arriving today' notification came at 11pm. the package wasnt there.",
        "my cart shows items at the old price but checkout wants the new higher price.",
        "the support chat window closes randomly in the middle of typing.",
        "alexa keeps setting timers on the wrong device. i told the living room echo but bedroom goes off.",
        "the order tracking page shows a different carrier than what the confirmation email said.",
        "amazon music keeps skipping songs on my echo dot. works fine on my phone though.",
        "the 'review your order' page shows items from a previous order, not the current one.",
        "app notification says package delivered but tracking says 'in transit'. which is it?",
        "the search results page shows the wrong product images. listing says 'wireless charger' but shows headphones.",
        "my prime benefit 'free same-day delivery' is showing as unavailable for items that say same-day delivery available.",
        "the website font changed overnight and now everything is blurry. zoom and clear cache didnt help.",
        "the return label generation page is stuck loading. tried 3 different browsers.",
    ],
    "membership_subscription": [
        # Direct questions
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
        # Paraphrases and edge cases
        "got charged $139 today and i thought my trial was still active. when does it end?",
        "can i pause my prime membership? going traveling for 3 months and wont need it.",
        "prime student verification says my .edu email is invalid. the school is definitely accredited.",
        "how much is prime annually vs monthly? want to switch but the website wont show me the annual option.",
        "my prime benefits just stopped working. free shipping now shows as $5.99. what gives?",
        "can i share my prime video with my parents in another state? they keep getting 'too many devices'.",
        "the prime free trial says im not eligible but ive never had prime before. how?",
        "canceled prime last month but got charged again this month. auto-renewal was supposed to be off.",
        "does prime gaming include the monthly free games or just the base subscription?",
        "i accidentally signed up for a 30-day trial of audible. how do i cancel before they charge me?",
        "prime shows as active but im not getting free two-day shipping. says 'delivery fees apply'.",
        "student prime says it expires in 2027. but i graduate this year. does it auto-convert?",
        "can i have prime AND prime student at the same time? switching from student to regular.",
        "prime video keeps asking me to subscribe separately for certain shows. isnt that supposed to be included?",
        "my prime membership benefits page shows nothing. no video, no music, no gaming. blank page.",
        "the kindle unlimited subscription renewed automatically. how do i get a refund for this month?",
        "was promised a free month of prime for signing up but got charged day 1.",
        "prime exclusive deals are showing regular prices for me. not getting the discount at checkout.",
        "how do i gift a prime membership to someone? want to get it for my parents.",
        "the prime trial page says 'try 30 days free' but the next screen charges me immediately.",
        "my amazon music unlimited plan just went up in price. is there a way to lock in the old rate?",
        "do i need separate prime memberships for my household or does one cover everyone?",
        "prime says 'your benefits end in 3 days' but i just paid for the year. what end?",
        "the prime video app on my tv says 'sign in with your prime account' but it IS my prime account.",
        "kindle unlimited shows as 'included with prime' during signup but then charges separately. misleading.",
        "how do i downgrade from annual prime to monthly? want to cancel but not lose everything immediately.",
        "prime reading books keep disappearing from my library after 30 days. what am i missing?",
        "the free trial banner wont go away even though im already paying for prime.",
        "does amazon music unlimited include hd and spatial audio or is that a higher tier?",
        "my prime membership isnt showing in my account. renewed 3 days ago but says 'not a prime member'.",
        "got an email saying my prime benefits are changing. what exactly is changing?",
        "the prime day deal said 'prime member exclusive' but it charged me the same price as non-prime.",
        "i renewed my prime but the free same-day delivery still shows as unavailable in my area.",
        "prime student keeps asking for re-verification every few months. is this normal?",
        "how do i make sure my prime auto-renews with a different card than my default?",
        "the audible free trial says 1 credit but then charges for the membership after 30 days. wanted just the book.",
        "prime says my free trial ends tomorrow but i want to cancel now before they charge $139.",
        "the prime benefits comparison page says 'photo storage unlimited' but my photos keep getting compressed.",
        "my amazon music family plan only works for 1 device at a time. thought it was 6.",
        "can i use my prime benefits in a different country? traveling and prime video says 'not available'.",
        "prime says im enrolled in share but my household members cant access prime video.",
        "the kindle unlimited subscription auto-renewed. how do i switch to buying books individually again?",
        "prime free trial showed $0.00 at checkout but i was charged $13 the next day. help?",
    ],
    "general_inquiry": [
        # Direct general questions
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
        # Paraphrases and edge cases
        "hey quick question - do you guys have black friday deals this year?",
        "the gift wrapping option was missing at checkout. can i request it after ordering?",
        "does the 'subscribe and save' program work for pet supplies too or just household items?",
        "i got an email saying my package is delayed but the tracking page says its on time. confused.",
        "can i schedule a delivery for a specific date? need it on a saturday.",
        "my question is about the warehouse location. is this item coming from the us or overseas?",
        "the 'ask a question' button on a product page is grayed out. why?",
        "do you price match? saw the same item cheaper at target.",
        "whats the difference between 'frequently bought together' and 'customers who viewed this also viewed'?",
        "can i change the delivery instructions after placing the order?",
        "i want to know if a product is actually worth buying based on your policies. 4 stars avg but 2000 reviews",
        "does this product come with a warranty? listing says nothing about it.",
        "the chat support said one thing and email said another. who do i trust?",
        "is there a way to save items to a wishlist without getting price change notifications?",
        "can i use my amazon balance to buy a gift card? seems circular but wanna know.",
        "do you offer student discounts on everything or just prime?",
        "the product page says 'in stock' but when i go to checkout it says 'unavailable'. why?",
        "is there a physical store i can return to or is it mail only?",
        "how do i know if a seller is official vs third party? there are like 20 listings for the same thing",
        "can i order something now and pick it up same day at a locker?",
        "what happens if the item i ordered goes on sale after i buy it? can i get the price difference?",
        "the return instructions say 'include all original packaging' but i recycled it. is that a problem?",
        "do you deliver on sundays? the tracking says out for delivery sunday but the estimated date was monday.",
        "i need to know if this blender is loud. my apartment has thin walls. any dB rating in the specs?",
        "can i combine multiple 'save for later' items into one checkout?",
        "the product reviews seem fake. all 5 stars with the same wording. is there a way to filter verified purchases?",
        "does your subscription service let me skip months? or am i locked in?",
        "i ordered something as a gift but forgot to check 'this is a gift'. can i add gift options now?",
        "whats your policy on damaged packaging but intact product? worth keeping or should i return?",
        "do you price adjust items that drop in price within 30 days of purchase?",
        "can i return something i bought 6 months ago? the return window must have passed but its defective.",
        "the 'deal of the day' ended at midnight but the timer showed 4 hours left. what happened?",
        "is there a way to see when an item will be back in stock?",
        "my order has 'free delivery' but also a 'delivery fee' line item. is that a bug?",
        "do you have a student discount for non-students who are taking online courses?",
        "the product image shows one thing but the title says something different. which one am i getting?",
        "how long does international shipping usually take to australia?",
        "can i change the payment method on an order that's already processing?",
        "is the packaging recyclable? trying to reduce waste.",
        "do you notify sellers when i leave a negative review? worried about retaliation.",
        "the 'subscribe and save' discount is 5% for 1 item but 15% for 5. what if i want 3 items?",
        "can i add a gift message after i already placed the order?",
        "whats the difference between 'amazon choice' and 'editors pick' badges?",
        "i need this for a gift by friday. says in stock but will it actually arrive in time?",
        "do you offer expedited returns? like next-day pickup instead of waiting for a label?",
        "the product says ships from and sold by amazon but the seller info shows a third party. contradictory.",
        "how do i know if something is refurbished vs new? the listing isnt clear.",
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
    """Generate realistic synthetic support data when HuggingFace is unavailable.

    Uses a diverse pool of ~50 base messages per intent (no mechanical prefix/suffix
    variations) to avoid near-duplicate inflation in evaluation scores.
    """
    tweets: List[Tweet] = []
    idx = 0
    seen_texts = set()

    random.seed(42)
    intent_list = list(_SYNTHETIC_MESSAGES.keys())

    for intent in intent_list:
        msgs = _SYNTHETIC_MESSAGES[intent]
        replies = _SYNTHETIC_REPLIES[intent]

        for i, base_msg in enumerate(msgs):
            if base_msg in seen_texts:
                continue
            seen_texts.add(base_msg)

            reply = replies[i % len(replies)]
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
    """Load REAL Twitter support conversations from HuggingFace, filter to one brand.

    Falls back to realistic synthetic Amazon data if HF is unreachable.
    """
    from datasets import load_dataset

    print(f"[dataset] Loading full dataset from HuggingFace...")
    try:
        ds = load_dataset("gorkemsevinc/Customer_Support_on_Twitter", split="train")
        print(f"[dataset] Full dataset: {len(ds):,} rows")

        print(f"[dataset] Filtering to brand: {brand}...")
        brand_rows = [row for row in ds if row.get('company', '') == brand]
        print(f"[dataset] {brand} conversations: {len(brand_rows):,}")

        if len(brand_rows) == 0:
            raise ValueError(f"No rows found for brand '{brand}'.")

        random.seed(42)
        random.shuffle(brand_rows)
        brand_rows = brand_rows[:max_conversations]
        print(f"[dataset] Using {len(brand_rows):,} conversations")
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

        first_support = next((t['text'] for t in turns if t['role'] == 'support'), "")

        for i, turn in enumerate(turns):
            if turn['role'] != 'customer':
                continue
            text = turn['text'].strip()
            if not text or len(text) < 5:
                continue
            if text in seen_texts:
                continue
            seen_texts.add(text)

            intent = _classify_intent_keywords(text)

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

    print(f"[dataset] Parsed {len(tweets):,} unique customer messages across "
          f"{len(set(t.conversation_id for t in tweets)):,} conversations")
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
