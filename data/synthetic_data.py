"""
Synthetic dataset generator for Academic Burnout Detection.
Generates 2500 anonymized student records with realistic, well-separated
feature distributions and controlled noise injection so the trained
Random Forest achieves ~88-90% accuracy on unseen data.

Class definitions
-----------------
0 – Low Risk    : engaged, on-time, high scores
1 – Medium Risk : partial disengagement, some delays, moderate scores
2 – High Risk   : severe disengagement, frequent delays, low scores

Noise model
-----------
~8% of samples have their label flipped to an adjacent class to simulate
real-world label uncertainty at class boundaries.
"""

import numpy as np
import pandas as pd
import os


def generate_dataset(n_samples: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── class sizes: Low 40 % / Medium 35 % / High 25 % ──────────────────────
    n_low    = int(n_samples * 0.40)          # 1000
    n_medium = int(n_samples * 0.35)          # 875
    n_high   = n_samples - n_low - n_medium   # 625

    def anon_id(i):
        return f"STU-{str(i + 1).zfill(5)}"

    records = []
    idx = 0

    # ── helper: clip to valid range ────────────────────────────────────────────
    def clip(val, lo, hi):
        return float(np.clip(val, lo, hi))

    # ══════════════════════════════════════════════════════════════════════════
    # LOW RISK  (label = 0)
    # High engagement, negligible delays, high quiz scores
    # ══════════════════════════════════════════════════════════════════════════
    for _ in range(n_low):
        records.append({
            "student_id":             anon_id(idx),
            # logins 9-14 / week — clearly active
            "login_frequency":        int(rng.integers(9, 15)),
            # study 3.5-6.0 hrs/day (Normal centred at 4.5)
            "avg_study_duration":     round(clip(rng.normal(4.5, 0.6), 3.0, 7.0), 2),
            # delay 0-1 day (mostly 0)
            "assignment_delay_days":  round(clip(rng.exponential(0.35), 0, 1.2), 2),
            # forum 7-15 posts/week
            "forum_participation":    int(rng.integers(7, 16)),
            # quiz 80-98 %
            "quiz_score_avg":         round(clip(rng.normal(88, 5), 78, 100), 2),
            # resources 14-22 / week
            "resource_access_count":  int(rng.integers(14, 23)),
            # missed < 8 %
            "missed_deadlines_pct":   round(clip(rng.beta(1.5, 10) * 15, 0, 8), 2),
            # short breaks 1-2.5
            "session_break_freq":     round(clip(rng.normal(1.8, 0.5), 0.5, 3.0), 2),
            "burnout_risk":           0,
        })
        idx += 1

    # ══════════════════════════════════════════════════════════════════════════
    # MEDIUM RISK  (label = 1)
    # Partial disengagement — clear gap from both Low and High
    # ══════════════════════════════════════════════════════════════════════════
    for _ in range(n_medium):
        records.append({
            "student_id":             anon_id(idx),
            # logins 4-7 / week
            "login_frequency":        int(rng.integers(4, 8)),
            # study 1.8-3.2 hrs/day
            "avg_study_duration":     round(clip(rng.normal(2.5, 0.5), 1.5, 3.5), 2),
            # delay 2-5 days
            "assignment_delay_days":  round(clip(rng.normal(3.2, 0.7), 2.0, 5.5), 2),
            # forum 2-5 posts/week
            "forum_participation":    int(rng.integers(2, 6)),
            # quiz 54-72 %
            "quiz_score_avg":         round(clip(rng.normal(63, 5), 52, 74), 2),
            # resources 5-10 / week
            "resource_access_count":  int(rng.integers(5, 11)),
            # missed 16-35 %
            "missed_deadlines_pct":   round(clip(rng.normal(25, 5), 15, 37), 2),
            # moderate-high breaks 3.5-6
            "session_break_freq":     round(clip(rng.normal(4.5, 0.8), 3.0, 6.5), 2),
            "burnout_risk":           1,
        })
        idx += 1

    # ══════════════════════════════════════════════════════════════════════════
    # HIGH RISK  (label = 2)
    # Severe disengagement — far from Low, clear boundary with Medium
    # ══════════════════════════════════════════════════════════════════════════
    for _ in range(n_high):
        records.append({
            "student_id":             anon_id(idx),
            # logins 0-3 / week
            "login_frequency":        int(rng.integers(0, 4)),
            # study 0-1.2 hrs/day
            "avg_study_duration":     round(clip(rng.exponential(0.7), 0, 1.5), 2),
            # delay 6-10 days
            "assignment_delay_days":  round(clip(rng.normal(7.5, 1.0), 6.0, 10.0), 2),
            # forum 0-1 posts/week
            "forum_participation":    int(rng.integers(0, 2)),
            # quiz 20-48 %
            "quiz_score_avg":         round(clip(rng.normal(34, 7), 18, 50), 2),
            # resources 0-3 / week
            "resource_access_count":  int(rng.integers(0, 4)),
            # missed 45-80 %
            "missed_deadlines_pct":   round(clip(rng.normal(62, 9), 44, 80), 2),
            # very high breaks 7-12
            "session_break_freq":     round(clip(rng.normal(9.0, 1.2), 7.0, 12.0), 2),
            "burnout_risk":           2,
        })
        idx += 1

    df = pd.DataFrame(records)

    # ── controlled boundary noise: flip ~8 % of samples to adjacent class ────
    noise_frac = 0.08
    n_noise    = int(len(df) * noise_frac)
    noise_idx  = rng.choice(df.index, size=n_noise, replace=False)

    for i in noise_idx:
        cur = df.at[i, "burnout_risk"]
        # flip to a neighbour only (0↔1, 1↔2, 2→1 or 1→0 with equal prob)
        if cur == 0:
            df.at[i, "burnout_risk"] = 1
        elif cur == 2:
            df.at[i, "burnout_risk"] = 1
        else:  # medium — flip either way
            df.at[i, "burnout_risk"] = rng.choice([0, 2])

    # ── shuffle ───────────────────────────────────────────────────────────────
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_dataset()
    df.to_csv("data/student_data.csv", index=False)
    print(f"[OK] Generated {len(df)} records -> data/student_data.csv")
    print(df["burnout_risk"].value_counts().rename({0: "Low", 1: "Medium", 2: "High"}))
