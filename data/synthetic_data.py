"""
Synthetic dataset generator for Academic Burnout Detection.
Generates anonymized student records with realistic feature distributions.
"""

import numpy as np
import pandas as pd
import os

def generate_dataset(n_samples: int = 2500, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    records = []

    # Class distribution: Low=40%, Medium=35%, High=25%
    class_counts = {
        "Low": int(n_samples * 0.40),
        "Medium": int(n_samples * 0.35),
        "High": n_samples - int(n_samples * 0.40) - int(n_samples * 0.35),
    }

    def anon_id(i):
        return f"STU-{str(i+1).zfill(5)}"

    idx = 0
    for label, count in class_counts.items():
        for _ in range(count):
            if label == "Low":
                row = {
                    "student_id": anon_id(idx),
                    "login_frequency": np.random.randint(5, 14),          # logins/week
                    "avg_study_duration": round(np.random.uniform(2.5, 5.0), 2),  # hrs/day
                    "assignment_delay_days": round(np.random.uniform(0, 1.5), 2),
                    "forum_participation": np.random.randint(3, 12),
                    "quiz_score_avg": round(np.random.uniform(72, 98), 2),
                    "resource_access_count": np.random.randint(8, 20),
                    "missed_deadlines_pct": round(np.random.uniform(0, 10), 2),
                    "session_break_freq": round(np.random.uniform(1, 3), 2),
                    "burnout_risk": 0,  # Low
                }
            elif label == "Medium":
                row = {
                    "student_id": anon_id(idx),
                    "login_frequency": np.random.randint(3, 8),
                    "avg_study_duration": round(np.random.uniform(1.5, 3.5), 2),
                    "assignment_delay_days": round(np.random.uniform(1.5, 4.0), 2),
                    "forum_participation": np.random.randint(1, 5),
                    "quiz_score_avg": round(np.random.uniform(50, 75), 2),
                    "resource_access_count": np.random.randint(3, 10),
                    "missed_deadlines_pct": round(np.random.uniform(10, 30), 2),
                    "session_break_freq": round(np.random.uniform(3, 6), 2),
                    "burnout_risk": 1,  # Medium
                }
            else:  # High
                row = {
                    "student_id": anon_id(idx),
                    "login_frequency": np.random.randint(0, 4),
                    "avg_study_duration": round(np.random.uniform(0, 2.0), 2),
                    "assignment_delay_days": round(np.random.uniform(4.0, 10.0), 2),
                    "forum_participation": np.random.randint(0, 2),
                    "quiz_score_avg": round(np.random.uniform(20, 55), 2),
                    "resource_access_count": np.random.randint(0, 4),
                    "missed_deadlines_pct": round(np.random.uniform(30, 80), 2),
                    "session_break_freq": round(np.random.uniform(6, 12), 2),
                    "burnout_risk": 2,  # High
                }
            records.append(row)
            idx += 1

    df = pd.DataFrame(records)
    # Shuffle rows
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_dataset()
    df.to_csv("data/student_data.csv", index=False)
    print(f"[OK] Generated {len(df)} records -> data/student_data.csv")
    print(df["burnout_risk"].value_counts().rename({0: "Low", 1: "Medium", 2: "High"}))
