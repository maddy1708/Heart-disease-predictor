"""
Dataset Generator for Heart Attack Risk Prediction
Generates a realistic clinical dataset of 8,763 patient records with 26 Kaggle-compatible columns.
Incorporates multi-factorial clinical cardiology risk modeling with strong predictive signals.
"""

import math
import random
import csv
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "heart_attack_prediction.csv"


def generate_dataset(num_records: int = 8763, output_file: Path = OUTPUT_PATH):
    random.seed(42)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "Patient ID", "Age", "Sex", "Cholesterol", "Blood Pressure", "Heart Rate",
        "Diabetes", "Family History", "Smoking", "Obesity", "Alcohol Consumption",
        "Exercise Hours Per Week", "Diet", "Previous Heart Problems", "Medication Use",
        "Stress Level", "Sedentary Hours Per Day", "Income", "BMI", "Triglycerides",
        "Physical Activity Days Per Week", "Sleep Hours Per Day", "Country", "Continent",
        "Hemisphere", "Heart Attack Risk"
    ]

    countries = ["United States", "United Kingdom", "Canada", "Germany", "France",
                 "Australia", "India", "China", "Brazil", "Japan", "Italy", "Spain"]
    continents = ["North America", "Europe", "Asia", "South America", "Australia"]
    hemispheres = ["Northern Hemisphere", "Southern Hemisphere"]

    rows = []
    positive_count = 0

    for i in range(num_records):
        patient_id = f"PID{i+10001:05d}"
        age = random.randint(22, 85)
        sex = random.choice(["Male", "Female"])

        systolic = int(random.gauss(135, 22))
        systolic = max(90, min(200, systolic))
        diastolic = int(random.gauss(85, 12))
        diastolic = max(60, min(120, diastolic))
        bp_str = f"{systolic}/{diastolic}"

        cholesterol = int(random.gauss(230, 45))
        cholesterol = max(120, min(400, cholesterol))

        triglycerides = int(random.gauss(200, 80))
        triglycerides = max(50, min(750, triglycerides))

        heart_rate = random.randint(50, 110)
        diabetes = 1 if random.random() < 0.30 else 0
        family_history = 1 if random.random() < 0.45 else 0
        smoking = 1 if random.random() < 0.25 else 0
        obesity = 1 if random.random() < 0.40 else 0
        alcohol = 1 if random.random() < 0.50 else 0

        diet_rand = random.random()
        if diet_rand < 0.30:
            diet = "Healthy"
        elif diet_rand < 0.75:
            diet = "Average"
        else:
            diet = "Unhealthy"

        prev_heart_problems = 1 if random.random() < 0.25 else 0
        medication_use = 1 if random.random() < 0.40 else 0
        stress_level = random.randint(1, 10)
        sedentary_hours = round(random.uniform(2.0, 12.0), 2)
        exercise_hours = round(random.uniform(0.0, 15.0), 2)

        bmi = round(max(16.0, min(42.0, random.gauss(27.5, 5.0))), 2)
        phys_activity_days = random.randint(0, 7)
        sleep_hours = random.randint(4, 10)
        income = random.randint(25000, 250000)

        country = random.choice(countries)
        continent = random.choice(continents)
        hemisphere = random.choice(hemispheres)

        # Multi-factorial Cardiology Risk Modeling
        diet_score = 0.85 if diet == "Unhealthy" else (-0.65 if diet == "Healthy" else 0.0)
        sex_score = 0.45 if sex == "Male" else 0.0

        z = (
            -3.8
            + 0.048 * (age - 50)
            + 0.028 * (systolic - 120)
            + 0.016 * (diastolic - 80)
            + 0.009 * (cholesterol - 200)
            + 0.0035 * (triglycerides - 150)
            + 1.05 * diabetes
            + 0.95 * prev_heart_problems
            + 0.75 * family_history
            + 0.70 * smoking
            + 0.45 * obesity
            + 0.06 * (bmi - 25.0)
            + 0.09 * (stress_level - 5)
            + 0.06 * (sedentary_hours - 6.0)
            - 0.09 * exercise_hours
            + diet_score
            + sex_score
        )

        prob = 1.0 / (1.0 + math.exp(-z))
        risk = 1 if random.random() < prob else 0
        if risk == 1:
            positive_count += 1

        rows.append([
            patient_id, age, sex, cholesterol, bp_str, heart_rate,
            diabetes, family_history, smoking, obesity, alcohol,
            exercise_hours, diet, prev_heart_problems, medication_use,
            stress_level, sedentary_hours, income, bmi, triglycerides,
            phys_activity_days, sleep_hours, country, continent,
            hemisphere, risk
        ])

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Generating realistic dataset -> {output_file}...")
    print(f"Dataset generated successfully!")
    print(f"Total records: {len(rows)}")
    print(f"Positive risk cases: {positive_count} ({positive_count / len(rows) * 100:.2f}%)")
    print(f"File size: {output_file.stat().st_size:,} bytes")


if __name__ == "__main__":
    generate_dataset()
