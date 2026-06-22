# 🏏 IPL Cricket Team Selector — DS + OR

> **DS Layer**: Role-weighted performance scoring model &nbsp;|&nbsp; **OR Layer**: ILP 0-1 Knapsack under salary cap + role constraints

## 🎯 Problem Statement
Pick the best 11 players for a cricket team from a pool of 40 candidates — maximising total performance score while staying within a salary cap (₹90Cr) and satisfying mandatory role quotas (batsmen, bowlers, all-rounders, wicket-keeper).

## 🏗️ Architecture
```
Player Stats (batting avg, SR, bowling avg, economy, form...)
        │
        ▼
  Role-Weighted Scoring Model    ← DS Layer
  (MinMax normalisation + role-specific weights)
        │
        ▼
  0-1 ILP Team Selector          ← OR Layer
  Maximise: Σ score_i · x_i
  Subject to: 11 players, salary cap, role quotas
        │
        ▼
  Radar Charts + Selection Table
```

## 📦 Tech Stack
| Layer | Tool |
|-------|------|
| Data Science | `scikit-learn` MinMaxScaler, `pandas`, `numpy` |
| Optimisation | `PuLP` CBC — Binary ILP |
| Visualisation | `plotly` radar charts |
| Data Source | ESPNCricinfo / Kaggle IPL dataset (synthetic here) |

## 🚀 Quick Start
```bash
pip install -r requirements.txt
python cricket_selector.py
streamlit run app.py
```

## 📊 Constraints Modelled
| Constraint | Value |
|------------|-------|
| Squad size | Exactly 11 |
| Salary cap | ≤ ₹90 Crore |
| Batsmen | ≥ 3 |
| Bowlers | ≥ 3 |
| All-Rounders | ≥ 1 |
| Wicket-Keeper | = 1 |

## 📁 Files
```
├── cricket_selector.py    # Full pipeline
├── app.py                 # Streamlit dashboard
├── requirements.txt
└── README.md
```
