#!/usr/bin/env python3
"""
Synthetic Data Generator for Long-Term Unemployment (LTU) Binary Classification

This script creates a synthetic dataset inspired by the profiling setup in Kern et al. (2024).
It generates features such as age, gender, nationality, number of previous unemployment episodes,
last job duration, education, and several additional features (e.g., number of moves, total employment
duration, job search duration, wage, skill level, industry, region, benefit duration, training participation,
and number of previous jobs). The binary outcome LTU is simulated using a logistic model with adjustments
for multiple factors. The generated CSV file is saved in a folder named "data" within the repository.

Usage:
    python generate_synthetic_data.py
"""

import os
import numpy as np
import pandas as pd

def sigmoid(x):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-x))

def generate_synthetic_data(n_samples=1000, random_state=42):
    """Generate a synthetic dataset for LTU risk prediction.
    
    Parameters:
        n_samples (int): Number of samples to generate.
        random_state (int): Seed for reproducibility.
    
    Returns:
        pd.DataFrame: A dataframe containing synthetic features and the binary LTU outcome.
    """
    np.random.seed(random_state)
    
    # Basic features
    age = np.random.randint(18, 66, size=n_samples)  # Age between 18 and 65
    gender = np.random.choice(["Male", "Female"], size=n_samples, p=[0.5, 0.5])
    nationality = np.random.choice(["German", "Non-German"], size=n_samples, p=[0.7, 0.3])
    num_unemployment = np.random.poisson(lam=1.5, size=n_samples)  # Number of previous unemployment episodes
    last_job_duration = np.maximum(0, np.random.normal(loc=3, scale=2, size=n_samples))  # in years
    education = np.random.choice(["Low", "Medium", "High"], size=n_samples, p=[0.3, 0.5, 0.2])
    
    # Additional features (at least 10 extra)
    num_moves = np.random.poisson(lam=2, size=n_samples)  # Number of residential moves
    total_employment_duration = np.maximum(0, np.random.normal(loc=8, scale=4, size=n_samples))  # in years
    job_search_duration = np.maximum(0, np.random.normal(loc=6, scale=2, size=n_samples))  # in weeks
    wage = np.maximum(0, np.random.normal(loc=3000, scale=800, size=n_samples))  # in euros
    # Skill level: categorical variable independent of education
    skill_level = np.random.choice(["Low", "Medium", "High"], size=n_samples, p=[0.3, 0.5, 0.2])
    # Industry: several options with specified probabilities
    industry = np.random.choice(["Manufacturing", "Services", "Tech", "Other"], size=n_samples, 
                                p=[0.25, 0.45, 0.20, 0.10])
    # Region: urbanicity
    region = np.random.choice(["Urban", "Suburban", "Rural"], size=n_samples, p=[0.5, 0.3, 0.2])
    # Benefit duration: total months receiving unemployment benefits
    benefit_duration = np.random.exponential(scale=3, size=n_samples).astype(int)
    # Training participation: whether the job seeker participated in a training program (binary)
    training_participation = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    # Number of previous jobs (as an integer)
    number_of_jobs = np.random.poisson(lam=3, size=n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "nationality": nationality,
        "num_unemployment": num_unemployment,
        "last_job_duration": last_job_duration,
        "education": education,
        "num_moves": num_moves,
        "total_employment_duration": total_employment_duration,
        "job_search_duration": job_search_duration,
        "wage": wage,
        "skill_level": skill_level,
        "industry": industry,
        "region": region,
        "benefit_duration": benefit_duration,
        "training_participation": training_participation,
        "number_of_jobs": number_of_jobs
    })
    
    # For simulation in the logistic model, convert some categorical variables to numeric
    # Nationality: German = 0, Non-German = 1.
    df["nationality_num"] = df["nationality"].map({"German": 0, "Non-German": 1})
    # For education, create dummy variables: "Low" and "High" (Medium as the baseline).
    df["edu_low"] = (df["education"] == "Low").astype(int)
    df["edu_high"] = (df["education"] == "High").astype(int)
    # For skill level, map Low = 1, Medium = 0, High = -1.
    df["skill_numeric"] = df["skill_level"].map({"Low": 1, "Medium": 0, "High": -1})
    # For industry, map coefficients: Manufacturing = 0.2, Services = 0.0, Tech = -0.1, Other = 0.0.
    industry_map = {"Manufacturing": 0.2, "Services": 0.0, "Tech": -0.1, "Other": 0.0}
    df["industry_coef"] = df["industry"].map(industry_map)
    # For region, map: Urban = 0.0, Suburban = 0.0, Rural = 0.1.
    region_map = {"Urban": 0.0, "Suburban": 0.0, "Rural": 0.1}
    df["region_coef"] = df["region"].map(region_map)
    
    # Simulate LTU outcome using a logistic model.
    # Coefficients (arbitrarily chosen for simulation) based on domain intuition:
    # - Intercept: -3
    # - Age: +0.03 per year
    # - Nationality: +0.5 if Non-German
    # - Number of previous unemployment episodes: +0.2 per episode
    # - Education: +0.8 effect if low, -0.5 if high (Medium = baseline)
    # - Last job duration: -0.1 per year (longer job duration implies lower LTU risk)
    # Additional features effects:
    # - Num_moves: +0.05 per move
    # - Total employment duration: -0.02 per year (more experience reduces risk)
    # - Job search duration: +0.1 per week
    # - Wage: +0.0002 per euro (higher wage might indicate a prior stable job, but here we simulate as risk factor)
    # - Skill level: coefficient based on skill_numeric
    # - Industry: use industry_coef directly
    # - Region: add region_coef
    # - Benefit duration: +0.05 per month
    # - Training participation: +0.2 if participated
    # - Number of jobs: +0.1 per job
    linear_combination = (
        -3 +
        0.03 * df["age"] +
        0.5 * df["nationality_num"] +
        0.2 * df["num_unemployment"] +
        0.8 * df["edu_low"] -
        0.5 * df["edu_high"] -
        0.1 * df["last_job_duration"] +
        0.05 * df["num_moves"] -
        0.02 * df["total_employment_duration"] +
        0.1 * df["job_search_duration"] +
        0.0002 * df["wage"] +
        0.3 * df["skill_numeric"] +
        df["industry_coef"] +
        df["region_coef"] +
        0.05 * df["benefit_duration"] +
        0.2 * df["training_participation"] +
        0.1 * df["number_of_jobs"]
    )
    
    # Add Gaussian noise to the linear predictor
    linear_combination += np.random.normal(scale=0.5, size=n_samples)
    
    # Convert the linear combination to probabilities using the sigmoid function.
    probability = sigmoid(linear_combination)
    
    # Draw binary outcomes using the computed probabilities.
    df["LTU"] = (np.random.rand(n_samples) < probability).astype(int)
    
    # Remove helper columns used for simulation.
    df.drop(columns=["nationality_num", "edu_low", "edu_high", "skill_numeric", "industry_coef", "region_coef"], inplace=True)
    
    return df

def main():
    # Generate the synthetic dataset.
    data = generate_synthetic_data(n_samples=1000, random_state=42)
    
    # Create the "data" folder if it doesn't exist.
    os.makedirs("data", exist_ok=True)
    
    # Save the dataset as a CSV file in the "data" folder.
    data_path = os.path.join("data", "synthetic_data.csv")
    data.to_csv(data_path, index=False)
    print(f"Synthetic data saved to {data_path}")

if __name__ == "__main__":
    main()
    