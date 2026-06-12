"""
Step 1: Data Preprocessing
Market Basket Analysis - Online Retail Dataset
MSc Data Science | Data Mining Assignment
"""

import pandas as pd
import numpy as np
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
# ──────────────────────────────────────────────
# 1. Load Data
# ──────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Online_Retail.xlsx")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_excel(DATA_PATH)
print(f"Raw shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# ──────────────────────────────────────────────
# 2. Remove Cancellations (InvoiceNo starting with 'C')
# ──────────────────────────────────────────────
initial_count = len(df)
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
print(f"\nRemoved {initial_count - len(df)} cancellation rows.")

# ──────────────────────────────────────────────
# 3. Drop rows with missing CustomerID or Description
# ──────────────────────────────────────────────
df.dropna(subset=["CustomerID", "Description"], inplace=True)
print(f"After dropping missing CustomerID/Description: {df.shape}")

# ──────────────────────────────────────────────
# 4. Remove rows with non-positive Quantity or UnitPrice
# ──────────────────────────────────────────────
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]
print(f"After removing non-positive Quantity/UnitPrice: {df.shape}")

# ──────────────────────────────────────────────
# 5. Standardise types
# ──────────────────────────────────────────────
df["InvoiceNo"] = df["InvoiceNo"].astype(str)
df["StockCode"] = df["StockCode"].astype(str)
df["Description"] = df["Description"].str.strip().str.upper()
df["CustomerID"] = df["CustomerID"].astype(int)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# ──────────────────────────────────────────────
# 6. Focus on UK transactions (largest market)
# ──────────────────────────────────────────────
df_uk = df[df["Country"] == "United Kingdom"].copy()
print(f"\nUK-only transactions: {df_uk.shape}")

# ──────────────────────────────────────────────
# 7. Build Basket Matrix (InvoiceNo × Item)
# ──────────────────────────────────────────────
print("\nBuilding basket matrix...")
basket = (
    df_uk.groupby(["InvoiceNo", "Description"])["Quantity"]
    .sum()
    .unstack(fill_value=0)
)

# Encode: 1 if item was bought, 0 otherwise
def encode(x):
    return 1 if x > 0 else 0

basket_encoded = basket.map(encode)
print(f"Basket matrix shape: {basket_encoded.shape}")

# ──────────────────────────────────────────────
# 8. Save cleaned data and basket
# ──────────────────────────────────────────────
df_uk.to_csv(os.path.join(OUTPUT_DIR, "cleaned_data.csv"), index=False)
basket_encoded.to_csv(os.path.join(OUTPUT_DIR, "basket_matrix.csv"))

print("\nPreprocessing complete. Saved:")
print("  outputs/cleaned_data.csv")
print("  outputs/basket_matrix.csv")

# ──────────────────────────────────────────────
# 9. Summary Statistics
# ──────────────────────────────────────────────
print(f"\n===== Summary =====")
print(f"Total transactions (UK):  {df_uk['InvoiceNo'].nunique()}")
print(f"Unique products:          {df_uk['Description'].nunique()}")
print(f"Unique customers:         {df_uk['CustomerID'].nunique()}")
print(f"Date range:               {df_uk['InvoiceDate'].min().date()} to {df_uk['InvoiceDate'].max().date()}")
print(f"Avg items per invoice:    {basket_encoded.sum(axis=1).mean():.2f}")
