# 🏏 IPL First-Innings Score Intelligence

> **An interactive machine-learning dashboard for estimating an IPL first-innings score from Powerplay performance.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://ipl-score-intelligence.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Model-Linear%20Regression-0A66C2)](https://scikit-learn.org/stable/modules/linear_model.html)
[![Data](https://img.shields.io/badge/IPL%20Matches-1%2C169-138A72)](https://ipl-score-intelligence.streamlit.app)

## 🌐 Live Demo

### [🚀 Open IPL Score Intelligence](https://ipl-score-intelligence.streamlit.app)

The deployed application provides an interactive analytics cockpit for exploring Powerplay patterns, evaluating the trained model, and generating first-innings score estimates.

---

## 🎯 Project Overview

The goal of this project is to estimate an IPL team's **final first-innings score using information available after the first six overs (Powerplay)**.

The project follows a complete machine-learning workflow:

```text
Raw IPL Ball-by-Ball Data
          ↓
Data Cleaning & Transformation
          ↓
Match-Level Feature Engineering
          ↓
Exploratory Data Analysis
          ↓
Feature Scaling
          ↓
Linear Regression
          ↓
Model Evaluation
          ↓
Interactive Streamlit Dashboard
```

Only **one machine-learning model** is used in the project: **Linear Regression**.

---

## 🧠 Machine Learning Approach

### Target

**Final first-innings score**

### Input Features

The model uses six Powerplay features:

| Feature | Meaning |
|---|---|
| `powerplay_runs` | Runs scored during the first six overs |
| `powerplay_fours` | Number of fours during the Powerplay |
| `powerplay_sixes` | Number of sixes during the Powerplay |
| `powerplay_dot_balls` | Dot balls during the Powerplay |
| `powerplay_wickets` | Wickets lost during the Powerplay |
| `powerplay_extras` | Extras during the Powerplay |

The features are standardized using **StandardScaler** before model training and prediction.

---

## 📊 Dataset & Evaluation

The prepared project contains **1,169 IPL matches** after match-level processing.

| Metric | Result |
|---|---:|
| Raw rows | 278,205 |
| Raw columns | 64 |
| Processed matches | 1,169 |
| Training samples | 935 |
| Testing samples | 234 |
| MSE | 568.5461 |
| RMSE | 23.8442 runs |
| R² | 0.4267 |

### What the metrics mean

- **MSE** measures the average squared prediction error.
- **RMSE** puts the prediction error back into runs; the model's RMSE is about **23.84 runs** on the prepared test split.
- **R²** indicates the proportion of target variation explained by the model; the prepared test split gives an R² of about **0.43**.

The model should therefore be treated as an **analytical estimate**, not an exact score predictor.

---

## 🖥️ Interactive Streamlit Dashboard

The deployed application is designed as an **IPL Command Center** rather than a basic prediction form.

### 1. Command Dashboard

- Project and model overview
- Match-count and model KPI cards
- Powerplay vs final-score relationship
- Final-score distribution
- High-level model insights

### 2. Score Predictor

Interactive controls allow users to change the six Powerplay inputs and immediately see:

- Predicted first-innings score
- Prediction planning range based on model RMSE
- Powerplay profile visualization
- Feature-level contribution view
- Preset scenarios for different Powerplay styles

### 3. EDA Explorer

Interactive visual analysis of the processed dataset, including distributions, relationships, and correlations.

### 4. Model Evaluation

Model diagnostics including:

- MSE
- RMSE
- R²
- Actual vs predicted values
- Prediction residuals

### 5. Data Explorer

Users can explore the processed match-level dataset with filters and download the filtered data as CSV.

---

## 🎨 UI & Experience

The application uses a futuristic dark analytics theme with:

- Command-center navigation
- Interactive Plotly visualizations
- KPI cards
- Live prediction controls
- Score gauge
- Powerplay radar profile
- Responsive dashboard layout
- Data filtering and CSV export

The objective is to make the ML workflow understandable and interactive rather than presenting only a static model output.

---

## 📁 Repository Structure

```text
IPL-First-Innings-Score-Predictor/
│
├── app.py
├── ipl_ml_project.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── ipl_project_output/
│   ├── IPL_ML_Processed.csv
│   ├── descriptive_statistics.csv
│   ├── results.json
│   ├── linear_regression_model.joblib
│   ├── standard_scaler.joblib
│   └── visualization files
│
└── docs/
    └── IPL_ML_Project_Report.pdf
```

---

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/SoumityaPandey/IPL-First-Innings-Score-Predictor.git
cd IPL-First-Innings-Score-Predictor
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start Streamlit

```bash
python -m streamlit run app.py
```

The application will open at:

```text
http://localhost:8501
```

---

## 🚀 Deployment

The application is deployed using **Streamlit Community Cloud** and connected to the GitHub repository.

### Deployment configuration

```text
Repository: SoumityaPandey/IPL-First-Innings-Score-Predictor
Branch: main
Main file: app.py
Live URL: https://ipl-score-intelligence.streamlit.app
```

After making changes locally:

```bash
git add .
git commit -m "Update Streamlit app"
git push
```

The connected Streamlit deployment can then rebuild from the updated GitHub source.

---

## 📦 Data Note

The original raw `IPL.csv` ball-by-ball dataset is **not included in this GitHub-ready deployment package** because the file is larger than GitHub's individual-file limit and is not required to run the deployed application.

The repository includes the processed match-level dataset and the trained model artifacts required by the Streamlit app.

If the original raw dataset is available locally, the preprocessing/training script can be used to reproduce the project outputs.

---

## ⚠️ Limitations

- Linear Regression assumes a linear relationship between the selected Powerplay features and the final score.
- The prepared test split has an R² of approximately 0.43, so a substantial portion of score variation remains unexplained.
- An RMSE of about 23.84 runs means predictions can deviate considerably from actual scores.
- Only six Powerplay features are used.
- The model does not explicitly account for factors such as venue, pitch conditions, batting lineup, bowling quality, match situation, weather, or tactical decisions.
- The prediction range shown in the dashboard is an error-based planning range, not a formal probability interval.

---

## 🔮 Future Enhancements

Possible future improvements include:

- Compare Linear Regression with additional regression models.
- Add venue and team-level information.
- Incorporate historical team performance.
- Add pitch and weather information where available.
- Add time-aware validation for season-based evaluation.
- Track model performance across different IPL seasons.
- Add model explainability with more detailed feature analysis.
- Build a richer match-scenario simulator.

---

## 📄 Project Report

The detailed university project report is available here:

[`docs/IPL_ML_Project_Report.pdf`](docs/IPL_ML_Project_Report.pdf)

---

## ⭐ Project Highlights

**Machine Learning:** Linear Regression + StandardScaler  
**Dataset:** 1,169 processed IPL matches  
**Interface:** Streamlit + Plotly  
**Deployment:** Streamlit Community Cloud  
**Prediction Target:** Final first-innings score  
**Live Demo:** https://ipl-score-intelligence.streamlit.app
