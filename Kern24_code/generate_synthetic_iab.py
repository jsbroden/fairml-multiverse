#!/usr/bin/env python3
"""
Synthetic Data Generator that Mimics the IAB SIAB Dataset

This script creates a synthetic version of the IAB data used in Fair Algorithmic Profiling.
Inspired by the processing in 01_setup.py  and the subsequent training/evaluation files –,
it constructs a DataFrame with:
  - An "id" column (sequential identifier)
  - "year" (ranging from 2010 to 2016)
  - A "dummy" column (a placeholder variable)
  - "ltue" (a binary outcome for long‐term unemployment)
  - 160 feature columns (columns 4–163 in the original code), where the first three are:
      • "frau1" (binary indicator for female)
      • "maxdeutsch1" (binary indicator for high German language proficiency)
      • "maxdeutsch.Missing." (indicator for missing language data)
    and the remaining 157 features are synthetic numeric values.
    
The outcome “ltue” is simulated via a logistic model that incorporates a small effect of protected attributes,
a year effect (to mimic changes over time), and contributions from a couple of randomly generated features.
This synthetic dataset is then saved as CSV in a folder named "data".
"""

import os
import numpy as np
import pandas as pd

def sigmoid(x):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-x))

def generate_synthetic_iab(n_per_year=1000, random_state=42):
    """
    Generate synthetic IAB-like data.
    
    Parameters:
        n_per_year (int): Number of samples per year.
        random_state (int): Seed for reproducibility.
    
    Returns:
        pd.DataFrame: Synthetic SIAB dataset.
    """
    np.random.seed(random_state)
    years = np.arange(2010, 2017)  # 2010 to 2016 (inclusive)
    data_rows = []
    next_id = 1
    
    # Define feature names for the 160 feature columns.
    # The first three are special protected attributes.
    feature_names = ["frau1", "maxdeutsch1", "maxdeutsch.Missing."] + \
                    [f"f{i}" for i in range(3, 160)]  # Total length = 3 + 157 = 160

    # For each year, generate n_per_year observations
    for yr in years:
        for _ in range(n_per_year):
            row = {}
            row["id"] = next_id
            next_id += 1
            row["year"] = yr
            # A dummy column (could be any placeholder, here a random float)
            row["dummy"] = np.random.rand()
            
            # --- Generate protected features (part of the 160 features) ---
            # Simulate "frau1": binary, 1 = female, 0 = male, with probability 0.5
            frau1 = np.random.binomial(1, 0.5)
            # Simulate "maxdeutsch1": binary, e.g. 1 = proficient, 0 = not proficient; p = 0.7
            maxdeutsch1 = np.random.binomial(1, 0.7)
            # Simulate "maxdeutsch.Missing.": indicator of missing language data; p = 0.1
            maxdeutsch_missing = np.random.binomial(1, 0.1)
            
            features = [frau1, maxdeutsch1, maxdeutsch_missing]
            # --- Generate remaining 157 synthetic features (normal variables) ---
            # These features mimic a diverse set of administrative attributes
            remaining_feats = np.random.normal(loc=0, scale=1, size=(157,)).tolist()
            features.extend(remaining_feats)
            
            # Build a dict for the features (using the names defined above)
            for feat_name, feat_val in zip(feature_names, features):
                row[feat_name] = feat_val
            
            # --- Simulate outcome "ltue" (long-term unemployment) ---
            # We'll use a logistic model with a base intercept and small contributions:
            # base intercept: -2
            # effect of being female (frau1): +0.5
            # effect of language proficiency: +0.2 * maxdeutsch1
            # effect of missing language data: -0.3 * maxdeutsch.Missing.
            # year effect: decrease risk by 0.05 per year above 2010
            # include a small contribution from two other synthetic features, say "f3" and "f4"
            # Note: "f3" and "f4" come from our remaining features.
            linear_predictor = (
                -2 +
                0.5 * frau1 +
                0.2 * maxdeutsch1 +
                -0.3 * maxdeutsch_missing +
                -0.05 * (yr - 2010) +
                0.1 * row.get("f3", 0) +      # if f3 exists in the dict
                -0.1 * row.get("f4", 0)
            )
            prob = sigmoid(linear_predictor)
            ltue = np.random.binomial(1, prob)
            row["ltue"] = ltue
            
            data_rows.append(row)
    
    # Define the order of columns:
    # We want the DataFrame with columns: "id", "year", "dummy", "ltue", and then the 160 feature columns.
    col_order = ["id", "year", "dummy", "ltue"] + feature_names
    df = pd.DataFrame(data_rows)[col_order]
    
    return df

def main():
    # Generate the synthetic IAB data.
    # Here we generate n_per_year samples for each year from 2010 to 2016.
    df = generate_synthetic_iab(n_per_year=40000, random_state=42)
    
    # Create the "data" folder if it doesn't exist.
    os.makedirs("data", exist_ok=True)
    
    # Save the synthetic dataset as CSV in the "data" folder.
    output_path = os.path.join("data", "siab.csv")
    df.to_csv(output_path, index=False)
    print(f"Synthetic IAB data saved to {output_path}")

if __name__ == "__main__":
    main()
