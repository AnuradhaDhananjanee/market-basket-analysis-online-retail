"""
Step 5: Visualisation of Association Rules
Market Basket Analysis - Online Retail Dataset
MSc Data Science | Data Mining Assignment
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import ast
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

rules = pd.read_csv(os.path.join(OUTPUT_DIR, "all_rules.csv"))
sns.set_theme(style="whitegrid")

# ──────────────────────────────────────────────
# 1. Support vs Confidence scatter (coloured by Lift)
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(
    rules["support"],
    rules["confidence"],
    c=rules["lift"],
    cmap="YlOrRd",
    alpha=0.7,
    s=40,
    edgecolors="none",
)
plt.colorbar(sc, ax=ax, label="Lift")
ax.set_xlabel("Support", fontsize=12)
ax.set_ylabel("Confidence", fontsize=12)
ax.set_title("Association Rules: Support vs Confidence (colour = Lift)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "viz_scatter_rules.png"), dpi=150)
plt.close()
print("Saved: viz_scatter_rules.png")

# ──────────────────────────────────────────────
# 2. Top 15 Rules by Lift – horizontal bar chart
# ──────────────────────────────────────────────
def parse_frozenset(s):
    s = str(s).replace("frozenset(", "").rstrip(")")
    return ", ".join(ast.literal_eval(s))

rules["ant"] = rules["antecedents"].apply(parse_frozenset)
rules["con"] = rules["consequents"].apply(parse_frozenset)
rules["rule_label"] = rules["ant"].str[:35] + " → " + rules["con"].str[:35]

top15 = rules.nlargest(15, "lift")

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(top15["rule_label"], top15["lift"], color="steelblue")
ax.set_xlabel("Lift", fontsize=12)
ax.set_title("Top 15 Association Rules by Lift", fontsize=13, fontweight="bold")
ax.invert_yaxis()
for bar, val in zip(bars, top15["lift"]):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "viz_top15_lift.png"), dpi=150)
plt.close()
print("Saved: viz_top15_lift.png")

# ──────────────────────────────────────────────
# 3. Lift Distribution Histogram
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(rules["lift"], bins=40, color="coral", edgecolor="white")
ax.axvline(1, color="black", linestyle="--", linewidth=1.2, label="Lift = 1 (independence)")
ax.axvline(rules["lift"].mean(), color="navy", linestyle="--", linewidth=1.2, label=f"Mean lift = {rules['lift'].mean():.2f}")
ax.set_xlabel("Lift", fontsize=12)
ax.set_ylabel("Number of Rules", fontsize=12)
ax.set_title("Distribution of Lift Values Across All Rules", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "viz_lift_distribution.png"), dpi=150)
plt.close()
print("Saved: viz_lift_distribution.png")

# ──────────────────────────────────────────────
# 4. Heatmap: Support × Confidence bins
# ──────────────────────────────────────────────
rules["supp_bin"] = pd.cut(rules["support"], bins=5)
rules["conf_bin"] = pd.cut(rules["confidence"], bins=5)
pivot = rules.groupby(["supp_bin", "conf_bin"], observed=False)["lift"].mean().unstack()

fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues", ax=ax, linewidths=0.5)
ax.set_title("Average Lift by Support × Confidence Bins", fontsize=13, fontweight="bold")
ax.set_xlabel("Confidence Range")
ax.set_ylabel("Support Range")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "viz_heatmap_lift.png"), dpi=150)
plt.close()
print("Saved: viz_heatmap_lift.png")

print("\nAll visualisations saved to outputs/")
