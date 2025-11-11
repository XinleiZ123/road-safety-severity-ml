"""
data_preparation.py
-------------------
This script extracts a manageable sample of the US_Accidents_March23 dataset
for exploratory analysis and modeling. It performs column selection, 
three-level severity recoding, and stratified sampling.

Notes:
- Missing values are not imputed here; handled later during modeling.
- Only low-missing, relevant columns are retained.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# ---- 1. Read a subset of useful columns ----
usecols = [
    "ID", "Severity", "Start_Time", "State", "City", "County",
    "Temperature(F)", "Visibility(mi)", "Weather_Condition",
    "Sunrise_Sunset", "Description",
    "Distance(mi)", "Start_Lat", "Start_Lng",
    "Amenity", "Bump", "Crossing", "Junction", "Traffic_Signal"
]

print("Reading dataset (subset of columns)...")
df = pd.read_csv("../data/US_Accidents_March23.csv", usecols=usecols)

# ---- 2. Simplify Severity into 3 levels ----
print("Simplifying severity levels...")
df["Severity"] = df["Severity"].map({
    1: "Minor",
    2: "Moderate",
    3: "Severe",
    4: "Severe"
})
df["Severity"] = pd.Categorical(df["Severity"], categories=["Minor", "Moderate", "Severe"], ordered=True)

# ---- 3. Stratified sampling (balanced classes) ----
print("Performing stratified sampling for class balance...")
df_small, _ = train_test_split(
    df,
    stratify=df["Severity"],
    train_size=0.3,  # keep about 30% of total, adjust if needed
    random_state=42
)


# ---- 4. Save clean sampled dataset ----
output_path = r"data/accidents_clean.csv"
df_small .to_csv(output_path, index=False)

print("\nDataset saved to:", output_path)
print("Rows:", len(df_small))
print("Class distribution:\n", df_small["Severity"].value_counts())
