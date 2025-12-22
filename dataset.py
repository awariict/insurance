import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
np.random.seed(42)

# Number of rows
n_rows = 240

# Possible categories
age_categories = ["16-25", "26-39", "40-64", "65+"]
gender_categories = ["male", "female"]
race_categories = ["majority", "minority"]
driving_exp = ["0-9y", "10-19y", "20-29y", "30y+"]
education_levels = ["none", "high school", "university", "postgraduate"]
income_levels = ["poverty", "working class", "middle class", "upper class"]
vehicle_type = ["sedan", "SUV", "sports car", "truck"]

# Generate data
data = {
    "ID": range(1, n_rows + 1),
    "AGE": np.random.choice(age_categories, n_rows),
    "GENDER": np.random.choice(gender_categories, n_rows),
    "RACE": np.random.choice(race_categories, n_rows),
    "DRIVING_EXPERIENCE": np.random.choice(driving_exp, n_rows),
    "EDUCATION": np.random.choice(education_levels, n_rows),
    "INCOME": np.random.choice(income_levels, n_rows),
    "CREDIT_SCORE": np.random.randint(300, 850, n_rows),  # realistic credit score range
    "VEHICLE_OWNERSHIP": np.random.choice([0, 1], n_rows),
    "VEHICLE_YEAR": np.random.choice(["before 2015", "after 2015"], n_rows),
    "MARRIED": np.random.choice([0, 1], n_rows),
    "CHILDREN": np.random.choice([0, 1, 2, 3, np.nan], n_rows),  # some missing
    "POSTAL_CODE": np.random.randint(100001, 999999, n_rows),
    "ANNUAL_MILEAGE": np.random.choice(
        [5000, 10000, 15000, 20000, 25000, np.nan], n_rows
    ),
    "VEHICLE_TYPE": np.random.choice(vehicle_type, n_rows),
    "SPEEDING_VIOLATIONS": np.random.poisson(1, n_rows),  # skewed toward fewer violations
    "DUIS": np.random.poisson(0.2, n_rows),  # mostly 0, few > 1
    "PAST_ACCIDENTS": np.random.poisson(0.5, n_rows),  # mostly 0 or 1
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Generate OUTCOME based on risk factors
def generate_outcome(row):
    risk = 0
    risk += row["SPEEDING_VIOLATIONS"] * 0.2
    risk += row["DUIS"] * 0.5
    risk += row["PAST_ACCIDENTS"] * 0.3
    if row["CREDIT_SCORE"] < 500:
        risk += 0.3
    if row["ANNUAL_MILEAGE"] is not np.nan and row["ANNUAL_MILEAGE"] > 20000:
        risk += 0.2
    return 1 if random.random() < risk else 0

df["OUTCOME"] = df.apply(generate_outcome, axis=1)

# Show dataset preview
print(df.head())

# Save to CSV
df.to_csv("simulated_insurance_data.csv", index=False)
print("\nSimulated dataset with 240 rows saved as 'simulated_insurance_data.csv'")
