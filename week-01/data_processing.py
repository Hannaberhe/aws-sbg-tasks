import pandas as pd
import numpy as np


def clean_data(df):
    cleaned = df.copy()

    # Convert scores into numbers
    cleaned["score"] = pd.to_numeric(
        cleaned["score"], errors="coerce"
    )

    # Remove missing scores
    cleaned = cleaned.dropna(subset=["score"])

    # Keep scores between 0 and 100
    cleaned = cleaned[
        cleaned["score"].between(0, 100)
    ]

    # Remove duplicate records
    cleaned = cleaned.drop_duplicates()

    # Reset row numbers
    cleaned = cleaned.reset_index(drop=True)

    return cleaned


raw_data = [
    {"student": "Amina", "score": 85},
    {"student": "Bekele", "score": 70},
    {"student": "Selam", "score": None},
    {"student": "Amina", "score": 85},
    {"student": "Dawit", "score": 110},
    {"student": "Hana", "score": 95},
    {"student": "Mekdes", "score": 60},
    {"student": "Yonas", "score": 78},
]


data = pd.DataFrame(raw_data)

cleaned_data = clean_data(data)

scores = cleaned_data["score"].to_numpy()

print("SMALL PYTHON DATA-PROCESSING APP")
print("-" * 35)

print("Original number of records:", len(data))
print("Cleaned number of records:", len(cleaned_data))
print("Records removed:", len(data) - len(cleaned_data))

print("Average score:", round(np.mean(scores), 2))
print("Highest score:", np.max(scores))
print("Lowest score:", np.min(scores))
print("Median score:", np.median(scores))
print("Standard deviation:", np.std(scores))

print("\nCleaned dataset:")
print(cleaned_data)

print("\nData processing completed successfully!")