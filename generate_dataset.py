"""
generate_dataset.py
===================
Synthetic customer-support triage dataset generator.

WHY synthetic?
  For a 2.5-hour classroom lab we want a dataset that is (a) reproducible on any
  machine with zero downloads, (b) free of privacy/PII concerns, and (c) shaped
  EXACTLY like the task we are teaching (ticket -> structured triage). Real
  support logs are messy, proprietary, and slow to clean. A seeded synthetic
  generator gives every student the identical data so results are comparable.

WHAT it produces:
  data/train.jsonl, data/val.jsonl, data/test.jsonl
  Each line is one JSON record:
    {
      "ticket":          "<raw customer message>",
      "category":        one of 6 fixed categories,
      "priority":        one of [Low, Medium, High, Urgent],
      "suggested_reply": "<a polite, on-brand draft reply>"
    }

HOW we keep classes balanced:
  Each ticket body is combined with a varying PRODUCT and a varying OPENER so a
  small number of templates expands into many unique, natural-sounding tickets.
  We then CAP each category to the same target count, so the model sees every
  class roughly equally (otherwise it learns to always guess the majority class).

HOW to swap in YOUR data:
  Replace this whole file with your own loader that writes the SAME JSONL schema
  (the four keys above). Everything downstream in the notebook only depends on
  those keys, so your real data flows through unchanged.
"""

import json
import os
import random
from collections import Counter, defaultdict

random.seed(42)  # WHY: determinism -> every student gets the same dataset.

CATEGORIES = [
    "Billing", "Technical Issue", "Account & Login",
    "Shipping & Delivery", "Refund & Return", "Product Inquiry",
]
PRIORITIES = ["Low", "Medium", "High", "Urgent"]

NAMES = ["Alex", "Priya", "Sam", "Jordan", "Maria", "Wei", "Tom", "Lena", "Omar", "Nina"]
PRODUCTS = ["Pro plan", "mobile app", "web dashboard", "API", "Starter plan", "analytics add-on"]
OPENERS = ["", "Hi, ", "Hello team, ", "Hey there, ", "Quick one - ", "Good morning, "]
AMOUNTS = ["$19", "$49", "$99", "$149"]
OIDS = ["#10241", "#55012", "#48190", "#73265", "#39907"]

# Each template: (body, priority, reply). Bodies/replies use named fields:
#   {p} product, {p2} other product, {a} amount, {a2} amount2, {o} order id, {name}
# Every body references {p} (or {a}/{o}) so PRODUCT/OPENER variation yields many
# unique tickets per category -> we can cap each class to the same size.
TEMPLATES = {
    "Billing": [
        ("I was charged {a} twice this month for my {p}. Please refund the duplicate charge.",
         "High",
         "Hi {name}, I'm sorry about the duplicate {a} charge on your {p}. I've escalated this to our billing team and the extra charge will be refunded to your original payment method within 5-7 business days. You'll get an email confirmation once it's processed."),
        ("Why did my {p} subscription jump from {a} to {a2} this month?",
         "Medium",
         "Hi {name}, great question. The change from {a} to {a2} on your {p} reflects a plan or usage adjustment. I've attached an itemized breakdown and I'm happy to walk you through it or help you switch plans."),
        ("Can you send me an invoice for my last {p} payment of {a}?",
         "Low",
         "Hi {name}, absolutely. I've emailed the invoice for your {a} {p} payment to your account address. You can also download past invoices anytime under Settings > Billing."),
    ],
    "Technical Issue": [
        ("The {p} keeps crashing every time I open the reports page.",
         "High",
         "Hi {name}, sorry you're hitting crashes on the reports page in the {p}. Could you share your version and a screenshot of the error? Meanwhile please try clearing the cache and reloading; I've alerted engineering to investigate."),
        ("I'm getting a 500 error when calling the {p}.",
         "High",
         "Hi {name}, thanks for flagging the 500 error on the {p}. Could you send the request timestamp and endpoint so we can pull the logs? I've opened a ticket with our backend team and we'll follow up shortly."),
        ("The {p} loads very slowly in the afternoons.",
         "Medium",
         "Hi {name}, thanks for letting us know the {p} feels slow at peak times. We're reviewing capacity in those windows. Could you share your region and rough time of day so we can correlate it with our metrics?"),
    ],
    "Account & Login": [
        ("I can't log in to the {p} - it says my password is wrong even after I reset it.",
         "High",
         "Hi {name}, sorry for the login trouble on the {p}. Password resets can take a minute to propagate; please retry and check caps-lock. If it still fails, I can send a secure magic-link to your verified email to get you back in."),
        ("I'm locked out of the {p} after too many attempts and have a client demo in 1 hour.",
         "Urgent",
         "Hi {name}, I understand the urgency with your demo. I've cleared the lockout on your {p} account and sent a one-time secure login link to your email - you should be back in within a couple of minutes. Reply here if anything still blocks you."),
        ("How do I change the email address on my {p} account?",
         "Low",
         "Hi {name}, happy to help. You can update your email under Settings > Profile; we'll send a confirmation link to the new address to verify it. Let me know if you'd like me to start the change for you."),
    ],
    "Shipping & Delivery": [
        ("My {p} order {o} was supposed to arrive 5 days ago and there's still no update.",
         "High",
         "Hi {name}, I'm sorry your order {o} is delayed. I've checked with the carrier and flagged it for investigation; if it isn't delivered within 48 hours we'll ship a free replacement or issue a refund - your choice. I'll keep you posted."),
        ("Can I change the delivery address for my {p} order {o}?",
         "Medium",
         "Hi {name}, I can help with that. If order {o} hasn't shipped I can update the address now - please reply with the new one. If it's already in transit, I'll request a redirect with the carrier."),
        ("When will my {p} order {o} ship?",
         "Low",
         "Hi {name}, your order {o} is being prepared and should ship within 1-2 business days. You'll receive a tracking link by email as soon as it's on its way."),
    ],
    "Refund & Return": [
        ("The {p} I received is defective and I want a full refund.",
         "High",
         "Hi {name}, I'm sorry the {p} arrived defective. You're fully covered - I've started a refund and emailed a prepaid return label. The refund posts to your original payment method once the carrier scans the item."),
        ("I returned my {p} 2 weeks ago but still haven't received my {a} refund.",
         "High",
         "Hi {name}, apologies for the wait on your {a} refund. I can see the {p} return was received; I've escalated to finance to release it today and you'll get a confirmation email. Thank you for your patience."),
        ("I changed my mind about my {p} purchase - how do I return it?",
         "Low",
         "Hi {name}, no problem at all - you're within our return window. Reply 'yes' and I'll email a prepaid label and step-by-step instructions; your refund is issued once we receive the {p}."),
    ],
    "Product Inquiry": [
        ("Does the {p} support exporting data to CSV?",
         "Low",
         "Hi {name}, yes - the {p} supports CSV export from any data table via the Export button (top-right). Happy to share a short guide or hop on a quick call if useful."),
        ("What's the difference between the {p} and the {p2}?",
         "Low",
         "Hi {name}, great question. The {p} and {p2} differ mainly in limits and advanced features; I've attached a side-by-side comparison and I'm glad to recommend the best fit for your use case."),
        ("Do you offer a discount for annual billing on the {p}?",
         "Medium",
         "Hi {name}, yes - switching the {p} to annual billing saves roughly two months versus monthly. I can apply the annual rate to your account now if you'd like; just say the word."),
    ],
}

# A handful of explicit URGENT outliers so every priority class is represented.
URGENT_EXTRAS = [
    ("URGENT: our production integration with your {p} is completely down and we're losing orders right now.",
     "Technical Issue", "Urgent",
     "Hi {name}, this is now our top priority. I've paged on-call engineering about the {p} outage and opened a P1 incident. I'll post status updates every 15 minutes until it's resolved."),
    ("URGENT: I was double-billed {a} on my {p} AND charged a late fee I don't recognize - fix today please.",
     "Billing", "Urgent",
     "Hi {name}, I'm sorry - that's not the experience we want. I've reversed the duplicate {a} {p} charge and removed the unrecognized late fee; both reflect within 1-2 business days and I'll email written confirmation now."),
    ("URGENT: my {p} order {o} for a live event tomorrow still hasn't shipped - I need it expedited now.",
     "Shipping & Delivery", "Urgent",
     "Hi {name}, given the event deadline I've upgraded order {o} to overnight shipping at no charge and confirmed it leaves our warehouse today. You'll get a tracking link within the hour."),
]


def fill(s, p):
    p2 = random.choice([x for x in PRODUCTS if x != p])
    return s.format(p=p, p2=p2,
                    a=random.choice(AMOUNTS), a2=random.choice(AMOUNTS),
                    o=random.choice(OIDS), name="{name}")


def build_records(target_per_cat=58):
    by_cat = defaultdict(list)

    # Expand templates combinatorially: PRODUCT x OPENER -> many unique tickets.
    for cat, templates in TEMPLATES.items():
        pool = []
        for body, pri, reply in templates:
            for p in PRODUCTS:
                for op in OPENERS:
                    ticket = op + fill(body, p)
                    rep = fill(reply, p)
                    pool.append((ticket, cat, pri, rep))
        # De-duplicate on ticket text, then cap to the per-category target.
        seen, uniq = set(), []
        for t in pool:
            if t[0] in seen:
                continue
            seen.add(t[0])
            uniq.append(t)
        random.shuffle(uniq)
        by_cat[cat] = uniq[:target_per_cat]

    # Add the urgent outliers.
    for ticket, cat, pri, reply in URGENT_EXTRAS:
        p = random.choice(PRODUCTS)
        by_cat[cat].append((fill(ticket, p), cat, pri, fill(reply, p)))

    # Flatten, attach a random customer name, shuffle.
    records = []
    for cat in CATEGORIES:
        for ticket, c, pri, reply in by_cat[cat]:
            name = random.choice(NAMES)
            records.append({
                "ticket": ticket,
                "category": c,
                "priority": pri,
                "suggested_reply": reply.replace("{name}", name),
            })
    random.shuffle(records)
    return records


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    records = build_records()

    n = len(records)
    n_test = min(36, n // 10)
    n_val = min(36, n // 10)
    test = records[:n_test]
    val = records[n_test:n_test + n_val]
    train = records[n_test + n_val:]

    write_jsonl(os.path.join(out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(out_dir, "val.jsonl"), val)
    write_jsonl(os.path.join(out_dir, "test.jsonl"), test)

    print(f"Total unique records: {n}")
    print(f"  train: {len(train)}   val: {len(val)}   test: {len(test)}")
    print("\nExample record:")
    print(json.dumps(train[0], indent=2, ensure_ascii=False))
    print("\nCategory counts (all):", dict(Counter(r["category"] for r in records)))
    print("Priority counts (all):", dict(Counter(r["priority"] for r in records)))


if __name__ == "__main__":
    main()
