"""
Streamlit Dashboard — Cricket Team Selector
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import pulp
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import random

st.set_page_config(page_title="IPL Team Selector", layout="wide")
st.title("🏏 IPL Cricket Team Selector — DS + OR")
st.markdown("**DS Layer**: Role-weighted performance scoring &nbsp;|&nbsp; **OR Layer**: ILP under salary cap + role quotas")

ROLES = ["Batsman", "Bowler", "All-Rounder", "Wicket-Keeper"]

with st.sidebar:
    st.header("⚙️ Squad Settings")
    n_players  = st.slider("Player pool size", 20, 60, 40)
    salary_cap = st.slider("Salary cap (₹ Crore)", 50, 150, 90)
    min_bat    = st.slider("Min Batsmen", 2, 5, 3)
    min_bowl   = st.slider("Min Bowlers", 2, 5, 3)
    min_ar     = st.slider("Min All-Rounders", 1, 3, 1)
    if "run_optimizer" not in st.session_state:
    st.session_state.run_optimizer = False

   if st.button("🚀 Run Optimizer", type="primary"):
    st.session_state.run_optimizer = True

def generate_players(n, seed=42):
    random.seed(seed); np.random.seed(seed)
    players = []
    for i in range(n):
        role = random.choice(ROLES)
        players.append({
            "player_id": i, "name": f"Player_{i:02d}", "role": role,
            "team": random.choice(["MI","CSK","RCB","KKR","DC","PBKS","RR","SRH"]),
            "salary_cr": round(random.uniform(0.5, 16.0), 2),
            "batting_avg": round(random.uniform(15,60),1) if role != "Bowler" else round(random.uniform(5,25),1),
            "strike_rate": round(random.uniform(100,190),1) if role != "Bowler" else round(random.uniform(60,130),1),
            "consistency": round(random.uniform(0.3,0.9),2),
            "bowling_avg": round(random.uniform(18,45),1) if role in ["Bowler","All-Rounder"] else 50,
            "economy_rate": round(random.uniform(6,12),2) if role in ["Bowler","All-Rounder"] else 12,
            "wickets_season": random.randint(5,30) if role in ["Bowler","All-Rounder"] else 0,
            "fielding_rating": round(random.uniform(5,10),1),
            "recent_form": round(random.uniform(0.3,1.0),2),
        })
    return pd.DataFrame(players)

def compute_score(df):
    df = df.copy()
    scaler = MinMaxScaler()
    bat_n = scaler.fit_transform(df[["batting_avg"]]).ravel()
    sr_n = scaler.fit_transform(df[["strike_rate"]]).ravel()
    ba_n = (1 - scaler.fit_transform(df[["bowling_avg"]])).ravel()
    ec_n = (1 - scaler.fit_transform(df[["economy_rate"]])).ravel()
    wk_n = scaler.fit_transform(df[["wickets_season"]]).ravel()
    fi_n = scaler.fit_transform(df[["fielding_rating"]]).ravel()
    scores = []
    for idx in range(len(df)):
        r = df.iloc[idx]
        if r.role == "Batsman":    s = 0.35*bat_n[idx]+0.30*sr_n[idx]+0.15*r.recent_form+0.10*r.consistency+0.10*fi_n[idx]
        elif r.role == "Bowler":   s = 0.30*ba_n[idx]+0.30*ec_n[idx]+0.20*wk_n[idx]+0.10*r.recent_form+0.10*fi_n[idx]
        elif r.role == "All-Rounder": s = 0.20*bat_n[idx]+0.15*sr_n[idx]+0.20*ba_n[idx]+0.15*ec_n[idx]+0.15*r.recent_form+0.15*wk_n[idx]
        else: s = 0.40*bat_n[idx]+0.25*sr_n[idx]+0.20*r.consistency+0.15*r.recent_form
        scores.append(round(s * 100, 2))
    df["score"] = scores
    return df

if st.session_state.run_optimizer:
    players = compute_score(generate_players(n_players))
    I = list(players.index)
    score_d  = players["score"].to_dict()
    salary_d = players["salary_cr"].to_dict()
    role_d   = players["role"].to_dict()

    prob = pulp.LpProblem("XI", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", I, cat="Binary")
    prob += pulp.lpSum(score_d[i]*x[i] for i in I)
    prob += pulp.lpSum(x[i] for i in I) == 11
    prob += pulp.lpSum(salary_d[i]*x[i] for i in I) <= salary_cap
    prob += pulp.lpSum(x[i] for i in I if role_d[i]=="Batsman") >= min_bat
    prob += pulp.lpSum(x[i] for i in I if role_d[i]=="Bowler") >= min_bowl
    prob += pulp.lpSum(x[i] for i in I if role_d[i]=="All-Rounder") >= min_ar
    prob += pulp.lpSum(x[i] for i in I if role_d[i]=="Wicket-Keeper") == 1
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[prob.status]
    selected = players[[bool(pulp.value(x[i])==1) for i in I]].sort_values("score", ascending=False)
    total_salary = selected["salary_cr"].sum()
    team_score = selected["score"].sum()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Status", status)
    c2.metric("Team Score", f"{team_score:.1f}")
    c3.metric("Salary Used", f"₹{total_salary:.1f}Cr")
    c4.metric("Salary Remaining", f"₹{salary_cap-total_salary:.1f}Cr")

    role_colors = {"Batsman":"#4682B4","Bowler":"#E8503A","All-Rounder":"#2ECC71","Wicket-Keeper":"#F39C12"}
    selected["color"] = selected["role"].map(role_colors)

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("🏆 Selected XI")
        st.dataframe(selected[["name","role","team","salary_cr","score"]].reset_index(drop=True),
                     use_container_width=True)
    with col2:
        st.subheader("📊 Role Distribution")
        role_counts = selected["role"].value_counts().reset_index()
        role_counts.columns = ["Role","Count"]
        fig_pie = go.Figure(go.Pie(labels=role_counts.Role, values=role_counts.Count,
                                   marker_colors=[role_colors[r] for r in role_counts.Role]))
        fig_pie.update_layout(height=300, margin=dict(t=20,b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("🕸️ Top Player Radar")
    top = selected.iloc[0]
    cats = ["Batting Avg","Strike Rate","Consistency","Recent Form","Fielding"]
    vals = [min(top.batting_avg/60,1)*100, min(top.strike_rate/190,1)*100,
            top.consistency*100, top.recent_form*100, top.fielding_rating*10]
    fig_r = go.Figure(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]], fill="toself",
                                       marker_color="#5364FF"))
    fig_r.update_layout(title=f"{top['name']} — {top['role']}", height=350,
                        polar=dict(radialaxis=dict(range=[0,100])))
    st.plotly_chart(fig_r, use_container_width=True)
else:
    st.info("👈 Set squad parameters and click **Select Best XI**")
