"""
IPL Cricket Team Selector — DS + OR
=====================================
DS Layer  : Player performance scoring model (weighted feature regression)
OR Layer  : 0-1 Knapsack / ILP — pick best XI under salary cap + role constraints
"""

import numpy as np
import pandas as pd
import pulp
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import random

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# 1. SYNTHETIC PLAYER DATA  (replace with ESPNCricinfo scrape)
# ─────────────────────────────────────────────
ROLES = ["Batsman", "Bowler", "All-Rounder", "Wicket-Keeper"]

def generate_players(n=40):
    players = []
    for i in range(n):
        role = random.choice(ROLES)
        p = {
            "player_id": i,
            "name": f"Player_{i:02d}",
            "team": random.choice(["MI","CSK","RCB","KKR","DC","PBKS","RR","SRH"]),
            "role": role,
            "salary_cr": round(random.uniform(0.5, 16.0), 2),   # IPL salary in Crores
            # Batting stats
            "batting_avg":     round(random.uniform(15, 60), 1)  if role in ["Batsman","All-Rounder","Wicket-Keeper"] else round(random.uniform(5, 25), 1),
            "strike_rate":     round(random.uniform(100, 190), 1) if role in ["Batsman","All-Rounder","Wicket-Keeper"] else round(random.uniform(60, 130), 1),
            "consistency":     round(random.uniform(0.3, 0.9), 2),
            # Bowling stats (only meaningful for bowlers/all-rounders)
            "bowling_avg":     round(random.uniform(18, 45), 1)  if role in ["Bowler","All-Rounder"] else None,
            "economy_rate":    round(random.uniform(6.0, 12.0),2) if role in ["Bowler","All-Rounder"] else None,
            "wickets_season":  random.randint(5, 30)              if role in ["Bowler","All-Rounder"] else 0,
            # Fielding
            "fielding_rating": round(random.uniform(5, 10), 1),
            # Recent form (last 5 matches average score)
            "recent_form":     round(random.uniform(0.3, 1.0), 2),
        }
        players.append(p)
    return pd.DataFrame(players)

players = generate_players(40)
print(f"[DATA] {len(players)} players loaded")
print(players[["name","role","salary_cr"]].head(8).to_string(index=False))

# ─────────────────────────────────────────────
# 2. DS LAYER — PERFORMANCE SCORE MODEL
# ─────────────────────────────────────────────
def compute_score(df):
    df = df.copy()
    df["bowling_avg"]   = df["bowling_avg"].fillna(50)    # penalise non-bowlers
    df["economy_rate"]  = df["economy_rate"].fillna(12)

    scaler = MinMaxScaler()
    # Normalise each stat to [0,1]
    bat_avg_n  = scaler.fit_transform(df[["batting_avg"]])
    sr_n       = scaler.fit_transform(df[["strike_rate"]])
    bowl_avg_n = 1 - scaler.fit_transform(df[["bowling_avg"]])   # lower is better
    econ_n     = 1 - scaler.fit_transform(df[["economy_rate"]])  # lower is better
    wkts_n     = scaler.fit_transform(df[["wickets_season"]])
    field_n    = scaler.fit_transform(df[["fielding_rating"]])
    form_n     = df[["recent_form"]].values
    cons_n     = df[["consistency"]].values

    # Role-specific weights
    scores = []
    for idx, row in df.iterrows():
        i = df.index.get_loc(idx)
        if row.role == "Batsman":
            s = 0.35*bat_avg_n[i] + 0.30*sr_n[i] + 0.15*form_n[i] + 0.10*cons_n[i] + 0.10*field_n[i]
        elif row.role == "Bowler":
            s = 0.30*bowl_avg_n[i] + 0.30*econ_n[i] + 0.20*wkts_n[i] + 0.10*form_n[i] + 0.10*field_n[i]
        elif row.role == "All-Rounder":
            s = 0.20*bat_avg_n[i] + 0.15*sr_n[i] + 0.20*bowl_avg_n[i] + 0.15*econ_n[i] + 0.15*form_n[i] + 0.15*wkts_n[i]
        else:  # Wicket-Keeper
            s = 0.40*bat_avg_n[i] + 0.25*sr_n[i] + 0.20*cons_n[i] + 0.15*form_n[i]
        scores.append(float(s))
    df["performance_score"] = [round(s*100, 2) for s in scores]
    return df

players = compute_score(players)
print(f"\n[SCORE] Top 10 players by performance score:")
print(players.nlargest(10,"performance_score")[["name","role","salary_cr","performance_score"]].to_string(index=False))

# ─────────────────────────────────────────────
# 3. OR LAYER — TEAM SELECTION ILP
#    Maximise: Σ performance_score_i * x_i
#    Subject to:
#      Σ x_i = 11                (exactly 11 players)
#      Σ salary_i * x_i ≤ CAP   (salary cap)
#      Σ x_i [role=Batsman] ≥ 3
#      Σ x_i [role=Bowler]  ≥ 3
#      Σ x_i [role=All-Rounder] ≥ 1
#      Σ x_i [role=WK]      = 1
# ─────────────────────────────────────────────
SALARY_CAP = 90.0   # Crores (realistic IPL cap-like)

I = list(players.index)
score  = players["performance_score"].to_dict()
salary = players["salary_cr"].to_dict()
role_d = players["role"].to_dict()

prob = pulp.LpProblem("CricketXI", pulp.LpMaximize)
x = pulp.LpVariable.dicts("select", I, cat="Binary")

# Objective
prob += pulp.lpSum(score[i] * x[i] for i in I)

# Constraints
prob += pulp.lpSum(x[i] for i in I) == 11
prob += pulp.lpSum(salary[i] * x[i] for i in I) <= SALARY_CAP
prob += pulp.lpSum(x[i] for i in I if role_d[i] == "Batsman")        >= 3
prob += pulp.lpSum(x[i] for i in I if role_d[i] == "Bowler")         >= 3
prob += pulp.lpSum(x[i] for i in I if role_d[i] == "All-Rounder")    >= 1
prob += pulp.lpSum(x[i] for i in I if role_d[i] == "Wicket-Keeper")  == 1

prob.solve(pulp.PULP_CBC_CMD(msg=0))
print(f"\n[OR] Status: {pulp.LpStatus[prob.status]}")
print(f"[OR] Team score: {pulp.value(prob.objective):.2f}")

selected = players[[pulp.value(x[i]) == 1 for i in I]].copy()
selected = selected.sort_values("performance_score", ascending=False)
print(f"\n[OR] Selected XI (salary cap ₹{SALARY_CAP}Cr):")
print(selected[["name","role","salary_cr","performance_score"]].to_string(index=False))
print(f"\n[OR] Total salary used: ₹{selected['salary_cr'].sum():.2f}Cr / ₹{SALARY_CAP}Cr")

selected.to_csv("selected_xi.csv", index=False)

# ─────────────────────────────────────────────
# 4. RADAR CHART VISUALISATION
# ─────────────────────────────────────────────
def radar_chart(player_row):
    role = player_row["role"]
    cats = ["Batting Avg","Strike Rate","Consistency","Recent Form","Fielding"]
    values = [
        min(player_row["batting_avg"]/60, 1)*100,
        min(player_row["strike_rate"]/190, 1)*100,
        player_row["consistency"]*100,
        player_row["recent_form"]*100,
        player_row["fielding_rating"]*10,
    ]
    fig = go.Figure(go.Scatterpolar(r=values+[values[0]], theta=cats+[cats[0]],
                                    fill="toself", name=player_row["name"]))
    fig.update_layout(title=f"{player_row['name']} — {role}", polar=dict(radialaxis=dict(range=[0,100])))
    return fig

fig = radar_chart(selected.iloc[0])
fig.write_html("top_player_radar.html")
print("[VIZ] Saved → top_player_radar.html")
print("[DONE]")
