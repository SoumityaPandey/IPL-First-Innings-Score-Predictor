# 🏏 IPL First-Innings Score Predictor

A machine-learning project that predicts an IPL first-innings final score from Powerplay statistics using **Linear Regression** and provides an interactive **Streamlit** web app.

## 🎯 Objective

Predict the final first-innings score using information available after the first six overs (Powerplay).

## 🤖 Machine Learning Model

Only one ML model is used:

**Linear Regression**

The model uses six Powerplay features:

- Powerplay runs
- Powerplay fours
- Powerplay sixes
- Powerplay dot balls
- Powerplay wickets
- Powerplay extras

Features are standardized with `StandardScaler` before training/prediction.

## 📊 Evaluation Metrics

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

Current evaluation from the prepared project outputs:

| Metric | Value |
|---|---:|
| MSE | 568.5461 |
| RMSE | 23.8442 runs |
| R² | 0.4267 |
| Training samples | 935 |
| Testing samples | 234 |

## 🖥️ Streamlit App

The app contains five sections:

1. **Dashboard** — project overview and Powerplay vs final-score relationship
2. **EDA Explorer** — distributions, boxplots, and correlation matrix
3. **Model Evaluation** — MSE, RMSE, R², and actual-vs-predicted plot
4. **Score Predictor** — interactive prediction from Powerplay inputs
5. **Data Explorer** — browse and download the processed match-level dataset

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

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The prepared Streamlit app does **not** need the original raw `IPL.csv` to run because the processed dataset and trained model artifacts are already included in `ipl_project_output/`.

To reproduce the complete preprocessing and training pipeline from the raw ball-by-ball data, place the original `IPL.csv` beside `ipl_ml_project.py` and run:

```bash
python ipl_ml_project.py
```

## 🚀 Deploy on Streamlit Community Cloud

1. Create a new public GitHub repository.
2. Upload the files in this project folder.
3. Open Streamlit Community Cloud.
4. Create a new app from the GitHub repository.
5. Select branch `main`.
6. Set the main file to `app.py`.
7. Deploy.

After deployment, future changes can be published with:

```bash
git add .
git commit -m "Update Streamlit app"
git push
```

## ⚠️ Dataset Note

The original `IPL.csv` raw ball-by-ball dataset is intentionally not included in this GitHub-ready package because it is over GitHub's 100 MB individual-file limit and is not required for the deployed Streamlit app. The processed match-level dataset and trained model artifacts required by the app are included.

## 📌 Limitations

- Linear Regression captures only a linear relationship between Powerplay statistics and final score.
- The model has an R² of approximately 0.43 on the prepared test split, so predictions should be treated as estimates rather than exact scores.
- The model uses only six Powerplay features and does not include every factor that can affect an IPL innings.

## Enhanced Streamlit UI

The latest version includes a futuristic interactive analytics cockpit with:

- Command-center dashboard with KPI cards
- Interactive Plotly charts and hover details
- Live first-innings score prediction
- Prediction gauge and RMSE-based planning band
- Powerplay profile radar visualization
- Feature contribution view from the trained Linear Regression model
- Conservative, balanced, and aggressive prediction presets
- Interactive EDA explorer with distributions and correlation heatmap
- Actual-vs-predicted and residual diagnostics
- Season, batting-team, and score filters in the data explorer
- Filtered CSV download
- Responsive dark futuristic styling
