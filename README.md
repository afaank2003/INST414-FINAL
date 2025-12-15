# Predicting Tomorrow’s Hazardous Waves at the Mouth of the Patapsco

This repository contains the code and data for my INST414 final project.  
The goal is to use a simple machine learning model to predict whether **noon tomorrow** at the mouth of the Patapsco River will be **hazardous** or **calm**, based on today’s buoy measurements.

The final project extends an earlier Module 6 assignment by adding a larger multi-year dataset and applying Module 7 evaluation techniques, especially careful model evaluation and decision threshold tuning.

---

## Motivating question and stakeholder

**Question**

> Given today’s conditions at the SN_OCEAN buoy, will noon tomorrow be hazardous (significant wave height ≥ 0.3 m) or calm?

**Stakeholder**

A **recreational kayaker or small-boat boater** who wants a simple “go / no-go” signal for noon tomorrow. For this person, **missing a hazardous day is worse** than cancelling on a calm day. The project therefore focuses on **increasing hazardous recall**, even if that lowers overall accuracy.

---

## Repository contents

All files live in the top level of the repo:

- `SN_OCEAN_2010-2025.csv`  
  Raw NOAA CBIBS data from the **SN_OCEAN** buoy, with many readings per day from 2010 through 2025.

- `final_module_pandas_script.py`  
  Data preparation script. Reads the raw CSV and produces a strict-noon daily dataset with one record per day at **12:00:00 UTC**.

- `final_module_script.py`  
  Main modeling and evaluation script. Builds features from the noon-only data, trains a Random Forest classifier, compares against a Dummy baseline, and tunes the probability threshold to increase hazardous recall.

- `final_module_confusion_matrix.png`  
  Confusion matrix for the tuned Random Forest on the test set (threshold = 0.285).

- `new_random_forest_bar_inst414.png`  
  Bar chart of Random Forest feature importances.

- `README.md`  
  This file.

---

## Data and preprocessing

The raw data come from the NOAA Chesapeake Bay Interpretive Buoy System (CBIBS), **SN_OCEAN** buoy. Important columns include:

- `Time (UTC)`
- `Temperature`
- `Salinity`
- `Significant wave height`
- `Wave period`
- `Wave from direction`
- `North surface currents`
- `East surface currents`

### Step 1: strict noon filtering

`final_module_pandas_script.py` does the following:

1. Reads `SN_OCEAN_2010-2025.csv` with pandas.
2. Parses `Time (UTC)` using `pd.to_datetime`.
3. Keeps only rows where the time is exactly **12:00:00** (strict noon).
4. Drops rows with missing values in the core wave and current variables.
5. Adds a `datetime` column with the parsed timestamp.
6. Writes the resulting noon-only dataset to a new CSV (the output path is defined at the top of the script).

This produces a daily time series with one noon record per day for the years where noon data exist.

---

## Modeling and evaluation

`final_module_script.py` assumes the noon-only CSV created by the previous step.

The script:

1. Loads the strict-noon dataset.
2. Sorts by `datetime` and defines:
   - `is_hazardous_today` based on significant wave height ≥ 0.3 m.
   - `tomorrow_is_hazardous` using a one-day forward shift of that label.
   - `tomorrow_wave_height` for analysis only.
3. Drops the last row, which lacks a “tomorrow”.
4. Constructs features from **today’s** measurements:
   - `Temperature`, `Salinity`, `Significant wave height`, `Wave period`
   - Sine and cosine of wave direction (`wave_dir_sin`, `wave_dir_cos`)
   - `North surface currents`, `East surface currents`
   - `month`, `day_of_year`
5. Splits the data into **60% train, 20% validation, 20% test** (stratified by the hazardous label).

### Models

The script trains and evaluates three key configurations:

1. **Dummy baseline (always calm)**  
   - `DummyClassifier(strategy="most_frequent")`  
   - Provides a reference accuracy and shows how a trivial model behaves on hazardous recall.

2. **Random Forest, default threshold 0.50**  
   - `RandomForestClassifier(n_estimators=200, class_weight="balanced")`  
   - Trained on the training set.  
   - Evaluated using accuracy, precision, recall, F1, and the confusion matrix.  
   - Achieves perfect accuracy on the training set and about **0.64** accuracy on the test set, with hazardous recall around **0.35**.

3. **Random Forest, tuned probability threshold**  
   - Uses `predict_proba` on the **validation** set and `precision_recall_curve` to choose a probability threshold.  
   - The policy is to select the **highest threshold that still reaches a target hazardous recall** (for example, 0.70) on validation.  
   - In this run, the chosen threshold is **0.285**.  
   - This threshold is then fixed and evaluated once on the test set.

### Key test results (current run)

On the held-out test set (226 calm days, 130 hazardous days):

- **Dummy baseline (always calm)**  
  - Accuracy: 0.635  
  - Hazardous recall: 0.000

- **Random Forest, default threshold 0.50**  
  - Accuracy: 0.640  
  - Confusion matrix: `[[182, 44], [84, 46]]`  
  - Hazardous recall: 46 / 130 ≈ 0.35

- **Random Forest, tuned threshold 0.285**  
  - Accuracy: 0.567  
  - Confusion matrix: `[[106, 120], [34, 96]]`  
  - Hazardous recall: 96 / 130 ≈ 0.74  
  - False alarm rate on calm days: 120 / 226 ≈ 0.53  

The tuned threshold **cuts missed hazardous days from 84 to 34**, at the cost of more false alarms on calm days. This is intentional and reflects the stakeholder’s preference for catching hazardous days rather than maximizing overall accuracy.

Feature importances (shown in `new_random_forest_bar_inst414.png`) indicate that:

- Today’s **significant wave height, temperature, and salinity** are the most informative features.
- Wave direction (sine and cosine), day of year, and surface currents also contribute.

---

## How to run the code

1. Install Python 3 and the required packages. A minimal set is:

   ```bash
   pip install pandas numpy scikit-learn matplotlib
Ensure SN_OCEAN_2010-2025.csv is in the same folder as the scripts, or update the file paths at the top of each script.

Run the data preparation script:

bash
Copy code
python final_module_pandas_script.py
This will create the noon-only CSV at the path defined inside the script.

Run the modeling and evaluation script:

bash
Copy code
python final_module_script.py
This prints metrics for the Dummy baseline, the default Random Forest, and the tuned-threshold Random Forest, and generates the confusion matrix and feature-importance plots.

Limitations
This is an instructional project, not an operational marine safety tool. Important limitations include:

The model uses data from a single buoy as a proxy for conditions across the estuary.

It assumes that the relationship between today’s and tomorrow’s conditions is stable over time.

It uses only today’s noon measurements and calendar features, not full weather forecasts.

At the tuned threshold, more than half of the “hazardous” predictions are false alarms, which may cause warning fatigue for some users.
