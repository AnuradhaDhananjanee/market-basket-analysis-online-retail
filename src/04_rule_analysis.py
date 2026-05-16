"""
Step 4: Rule Analysis & Interpretation
Market Basket Analysis - Online Retail Dataset
MSc Data Science | Data Mining Assignment
"""

import pandas as pd
import re
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

rules = pd.read_csv(os.path.join(OUTPUT_DIR, "all_rules.csv"))
top_rules = pd.read_csv(os.path.join(OUTPUT_DIR, "top_rules.csv"))
freq_items = pd.read_csv(os.path.join(OUTPUT_DIR, "frequent_itemsets.csv"))

print(f"Total rules: {len(rules)}")
print(f"High-quality rules (lift>=2): {len(top_rules)}")

# ──────────────────────────────────────────────
# 1. Rule Statistics Summary
# ──────────────────────────────────────────────
print("\n-- Rule Metrics Summary --")
for metric in ["support", "confidence", "lift"]:
    print(f"{metric.capitalize():15s}  min={rules[metric].min():.3f}  "
          f"mean={rules[metric].mean():.3f}  max={rules[metric].max():.3f}")

# ──────────────────────────────────────────────
# 2. Itemset Size Distribution
# ──────────────────────────────────────────────
def frozenset_len(s):
    # Count items in frozenset string like "frozenset({'a', 'b'})"
    items = re.findall(r"'([^']+)'", str(s))
    return len(items)

freq_items["itemset_size"] = freq_items["itemsets"].apply(frozenset_len)
print("\n-- Itemset Size Distribution --")
print(freq_items["itemset_size"].value_counts().sort_index())

# ──────────────────────────────────────────────
# 3. Interesting Rules for Client Presentation
# ──────────────────────────────────────────────
def parse_frozenset(s):
    items = re.findall(r"'([^']+)'", str(s))
    return ", ".join(items)

# All rules already filtered at lift>=2 in cycle 3, apply confidence filter
interesting = rules[
    (rules["lift"] > 3) & (rules["confidence"] > 0.5) & (rules["support"] > 0.02)
].copy()

interesting["ant"] = interesting["antecedents"].apply(parse_frozenset)
interesting["con"] = interesting["consequents"].apply(parse_frozenset)
interesting["rule"] = interesting["ant"] + "  ->  " + interesting["con"]
interesting = interesting.sort_values("lift", ascending=False)

print(f"\n-- Interesting Rules (lift>3, conf>0.5, supp>0.02): {len(interesting)} rules --")
pd.set_option("display.max_colwidth", 70)
cols = ["rule", "support", "confidence", "lift"]
print(interesting[cols].head(20).to_string(index=False))

interesting[cols].to_csv(os.path.join(OUTPUT_DIR, "interesting_rules.csv"), index=False)
print("\nSaved: outputs/interesting_rules.csv")

# ──────────────────────────────────────────────
# 4. Confidence Bands
# ──────────────────────────────────────────────
bins = [0, 0.3, 0.5, 0.7, 1.01]
labels = ["Low (0-0.3)", "Medium (0.3-0.5)", "High (0.5-0.7)", "Very High (>0.7)"]
rules["conf_band"] = pd.cut(rules["confidence"], bins=bins, labels=labels)
print("\n-- Confidence Band Distribution --")
print(rules["conf_band"].value_counts())

# ──────────────────────────────────────────────
# 5. Actionability score (lift x confidence)
# ──────────────────────────────────────────────
rules["actionability"] = rules["lift"] * rules["confidence"]
top10 = rules.nlargest(10, "actionability").copy()
top10["ant"] = top10["antecedents"].apply(parse_frozenset)
top10["con"] = top10["consequents"].apply(parse_frozenset)
top10["rule"] = top10["ant"] + "  ->  " + top10["con"]
print("\n-- Top 10 Most Actionable Rules (lift x confidence) --")
print(top10[["rule", "support", "confidence", "lift", "actionability"]].to_string(index=False))
