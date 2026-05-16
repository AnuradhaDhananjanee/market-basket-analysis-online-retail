"""
Step 2: Exploratory Data Analysis (EDA)
Market Basket Analysis - Online Retail Dataset
MSc Data Science | Data Mining Assignment
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(OUTPUT_DIR, "cleaned_data.csv"), parse_dates=["InvoiceDate"])
print(f"Loaded cleaned data: {df.shape}")

sns.set_theme(style="whitegrid", palette="muted")

# ──────────────────────────────────────────────
# 1. Top 20 Best-Selling Products
# ──────────────────────────────────────────────
top_products = (
    df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(20)
)

fig, ax = plt.subplots(figsize=(12, 6))
top_products.plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Top 20 Best-Selling Products (UK)", fontsize=14, fontweight="bold")
ax.set_xlabel("Total Quantity Sold")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_top20_products.png"), dpi=150)
plt.close()
print("Saved: eda_top20_products.png")

# ──────────────────────────────────────────────
# 2. Monthly Sales Trend
# ──────────────────────────────────────────────
df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")
monthly = df.groupby("YearMonth")["Quantity"].sum()

fig, ax = plt.subplots(figsize=(12, 4))
monthly.plot(ax=ax, marker="o", color="darkorange")
ax.set_title("Monthly Sales Volume (UK)", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Total Quantity Sold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_monthly_trend.png"), dpi=150)
plt.close()
print("Saved: eda_monthly_trend.png")

# ──────────────────────────────────────────────
# 3. Transaction Size Distribution
# ──────────────────────────────────────────────
items_per_invoice = df.groupby("InvoiceNo")["Description"].nunique()

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(items_per_invoice[items_per_invoice < 50], bins=40, color="teal", edgecolor="white")
ax.set_title("Distribution of Basket Size (# Unique Items per Invoice)", fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Unique Items")
ax.set_ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_basket_size.png"), dpi=150)
plt.close()
print("Saved: eda_basket_size.png")

# ──────────────────────────────────────────────
# 4. Revenue by Country (top 10, non-UK)
# ──────────────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(OUTPUT_DIR), "data", "Online_Retail.xlsx")
df_all = pd.read_excel(DATA_FILE, dtype={"InvoiceNo": str})
df_all = df_all[~df_all["InvoiceNo"].str.startswith("C")]
df_all = df_all[(df_all["Quantity"] > 0) & (df_all["UnitPrice"] > 0)]
df_all["Revenue"]     = df_all["Quantity"] * df_all["UnitPrice"]
df_all["InvoiceDate"] = pd.to_datetime(df_all["InvoiceDate"])

top_countries = (
    df_all[df_all["Country"] != "United Kingdom"]
    .groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(10, 5))
top_countries.plot(kind="bar", ax=ax, color="mediumseagreen")
ax.set_title("Top 10 Countries by Revenue (excl. UK)", fontsize=13, fontweight="bold")
ax.set_xlabel("Country")
ax.set_ylabel("Revenue (£)")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_revenue_by_country.png"), dpi=150)
plt.close()
print("Saved: eda_revenue_by_country.png")

# ──────────────────────────────────────────────
# # 5. Monthly Revenue Trend (UK)
# ──────────────────────────────────────────────

try:
    df_all["InvoiceDate"] = pd.to_datetime(df_all["InvoiceDate"])
    df_uk = df_all[df_all["Country"] == "United Kingdom"].copy()
    df_uk["YearMonth"] = df_uk["InvoiceDate"].dt.to_period("M")
    monthly_rev = df_uk.groupby("YearMonth")["Revenue"].sum()
    monthly_rev.index = monthly_rev.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.bar(
        monthly_rev.index, monthly_rev.values, width=20, color="#E7D731", edgecolor="none", alpha=0.88
    )
    ax.plot(
        monthly_rev.index, monthly_rev.values, color="#C00000", marker="o", markersize=4, linewidth=1.5
    )
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Total Revenue (£)", fontsize=11)
    ax.set_title(
        "Monthly Revenue Trend - UK (Dec 2010 to Dec 2011)",
        fontsize=13,
        fontweight="bold"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "eda_revenue_by_month.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

except Exception as e:
    print("ERROR in Monthly Revenue Trend section:")
    print(e)



# ──────────────────────────────────────────────
# 6. Print EDA summary
# ──────────────────────────────────────────────
print(f"\n── EDA Summary ────────────────────────────────")
print(f"Avg basket size:   {items_per_invoice.mean():.2f} unique items")
print(f"Median basket:     {items_per_invoice.median():.0f} items")
print(f"Max basket:        {items_per_invoice.max()} items")
print(f"\nTop 5 products:\n{top_products.head()}")
print("\nEDA complete.")
