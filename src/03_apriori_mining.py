"""
Step 3: Association Rule Mining using Apriori Algorithm
Market Basket Analysis - Online Retail Dataset
MSc Data Science | Data Mining Assignment
"""

import pandas as pd
import time
import os
from mlxtend.frequent_patterns import apriori, association_rules

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Load Basket Matrix
# ──────────────────────────────────────────────
print("Loading basket matrix...")
basket = pd.read_csv(os.path.join(OUTPUT_DIR, "basket_matrix.csv"), index_col=0)
basket = basket.astype(bool)
print(f"Basket matrix: {basket.shape[0]} transactions x {basket.shape[1]} items")

# ──────────────────────────────────────────────
# Pre-filter to items with support >= 0.02 (200 items)
# This keeps computation tractable while retaining meaningful items
# ──────────────────────────────────────────────
item_support = basket.mean()
basket = basket.loc[:, item_support >= 0.02]
print(f"Filtered basket to {basket.shape[1]} items with support >= 0.02")

# ──────────────────────────────────────────────
# CYCLE 1 - Broad sweep
# Purpose: Understand the support landscape
# ──────────────────────────────────────────────
print("\n-- Cycle 1: Broad sweep (min_support=0.03) --")
t0 = time.time()
freq_items_c1 = apriori(basket, min_support=0.03, use_colnames=True)
t1 = time.time()
print(f"Frequent itemsets found: {len(freq_items_c1)}  |  Time: {t1-t0:.1f}s")

rules_c1 = association_rules(freq_items_c1, metric="lift", min_threshold=1.0)
print(f"Rules generated: {len(rules_c1)}")
print(f"Lift range: {rules_c1['lift'].min():.2f} - {rules_c1['lift'].max():.2f}")

# ──────────────────────────────────────────────
# CYCLE 2 - Lower support for more rules
# Purpose: Find granular rules with confidence filter
# ──────────────────────────────────────────────
print("\n-- Cycle 2: Lower support (min_support=0.02, confidence>=0.3) --")
t0 = time.time()
freq_items_c2 = apriori(basket, min_support=0.02, use_colnames=True)
t1 = time.time()
print(f"Frequent itemsets found: {len(freq_items_c2)}  |  Time: {t1-t0:.1f}s")

rules_c2 = association_rules(freq_items_c2, metric="confidence", min_threshold=0.3)
print(f"Rules generated: {len(rules_c2)}")

# ──────────────────────────────────────────────
# CYCLE 3 - Final refined pass (lift >= 2)
# Purpose: High-quality rules for client presentation
# ──────────────────────────────────────────────
print("\n-- Cycle 3: High-quality rules (lift >= 2) --")
rules_c3 = rules_c2[rules_c2["lift"] >= 2.0].copy()
rules_c3 = rules_c3.sort_values("lift", ascending=False)
print(f"High-quality rules: {len(rules_c3)}")

# ──────────────────────────────────────────────
# Add readable columns
# ──────────────────────────────────────────────
rules_c3["antecedents_str"] = rules_c3["antecedents"].apply(lambda x: ", ".join(list(x)))
rules_c3["consequents_str"] = rules_c3["consequents"].apply(lambda x: ", ".join(list(x)))
rules_c3["rule"] = rules_c3["antecedents_str"] + "  ->  " + rules_c3["consequents_str"]

# ──────────────────────────────────────────────
# Save outputs
# ──────────────────────────────────────────────
freq_items_c2.to_csv(os.path.join(OUTPUT_DIR, "frequent_itemsets.csv"), index=False)
rules_c2.to_csv(os.path.join(OUTPUT_DIR, "all_rules.csv"), index=False)
rules_c3.to_csv(os.path.join(OUTPUT_DIR, "top_rules.csv"), index=False)

print("\nSaved:")
print("  outputs/frequent_itemsets.csv")
print("  outputs/all_rules.csv")
print("  outputs/top_rules.csv")

# ──────────────────────────────────────────────
# Show Top 15 Rules
# ──────────────────────────────────────────────
pd.set_option("display.max_colwidth", 60)
print("\n-- Top 15 Rules by Lift --")
display_cols = ["rule", "support", "confidence", "lift"]
print(rules_c3[display_cols].head(15).to_string(index=False))
