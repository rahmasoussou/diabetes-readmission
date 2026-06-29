"""
Dashboard ClinAI v4 — Mode clair/sombre + historique corrigé
"""
import os, requests, streamlit as st, pandas as pd, plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

API_URL = f"http://{os.environ.get('API_HOST','ml-service')}:{os.environ.get('API_PORT','8000')}"

st.set_page_config(page_title="ClinAI · Réhospitalisation", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

# ── Session state ─────────────────────────────────────────────────
if "token"     not in st.session_state: st.session_state.token     = None
if "username"  not in st.session_state: st.session_state.username  = None
if "dark_mode" not in st.session_state: st.session_state.dark_mode = False
if "hist_offset" not in st.session_state: st.session_state.hist_offset = 0

D = st.session_state.dark_mode

# ── Palette ───────────────────────────────────────────────────────
if D:
    BG       = "#0d1117"; SIDEBAR  = "#161b22"; CARD    = "#161b22"
    TEXT     = "#e6edf3"; MUTED    = "#8b949e"; BORDER  = "#21262d"
    ACCENT   = "#1f6feb"; ACCENT2  = "#388bfd"; INPUT   = "#161b22"
    IBORDER  = "#30363d"; PLOTBG   = "rgba(0,0,0,0)"; GRIDC = "#21262d"
    RBHIGH   = "#1f1315"; RBMED    = "#171208"; RBLOW   = "#0d1f17"
    ABWARN   = "#171208"; ABINFO   = "#0d1626"; ABOK    = "#0d1f17"; ABERR="#1f1315"
    TCWARN   = "#e3b341"; TCINFO   = "#79c0ff"; TCOK    = "#56d364"; TCERR="#ffa198"
    BCWARN   = "#d29922"; BCINFO   = "#1f6feb"; BCOK    = "#3fb950"; BCERR="#f85149"
    HERO1    = "#0d1f3c"; HERO2    = "#0d2a4a"
else:
    BG       = "#F7F9FC"; SIDEBAR  = "#FFFFFF"; CARD    = "#FFFFFF"
    TEXT     = "#1a1a2e"; MUTED    = "#64748B"; BORDER  = "#E2E8F0"
    ACCENT   = "#1B4FD8"; ACCENT2  = "#1440B0"; INPUT   = "#FFFFFF"
    IBORDER  = "#CBD5E1"; PLOTBG   = "rgba(0,0,0,0)"; GRIDC = "#F1F5F9"
    RBHIGH   = "#FEF2F2"; RBMED    = "#FFFBEB"; RBLOW   = "#F0FDF4"
    ABWARN   = "#FFFBEB"; ABINFO   = "#EFF6FF"; ABOK    = "#F0FDF4"; ABERR="#FEF2F2"
    TCWARN   = "#92400E"; TCINFO   = "#1e40af"; TCOK    = "#14532D"; TCERR="#991B1B"
    BCWARN   = "#FCD34D"; BCINFO   = "#93C5FD"; BCOK    = "#86EFAC"; BCERR="#FCA5A5"
    HERO1    = "#1B4FD8"; HERO2    = "#0EA5E9"

PLOTLY_THEME = dict(
    paper_bgcolor=PLOTBG, plot_bgcolor=PLOTBG,
    font=dict(color=MUTED, family="DM Sans"),
    xaxis=dict(gridcolor=GRIDC, linecolor=BORDER, tickfont=dict(size=11)),
    yaxis=dict(gridcolor=GRIDC, linecolor=BORDER, tickfont=dict(size=11)),
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{{font-family:'DM Sans',sans-serif!important;color:{TEXT};}}
.stApp{{background:{BG};}}
.block-container{{padding-top:1.5rem;padding-bottom:2rem;}}
#MainMenu,footer,header{{visibility:hidden;}}
section[data-testid="stSidebar"]{{background:{SIDEBAR}!important;border-right:1px solid {BORDER}!important;box-shadow:2px 0 12px rgba(0,0,0,0.05);}}
section[data-testid="stSidebar"]>div{{padding:1.5rem 1.2rem;}}
.stTabs [data-baseweb="tab-list"]{{background:{CARD};border-radius:12px;padding:5px;gap:4px;border:1px solid {BORDER};box-shadow:0 1px 4px rgba(0,0,0,0.06);}}
.stTabs [data-baseweb="tab"]{{background:transparent;color:{MUTED};border-radius:8px;font-weight:500;font-size:0.9rem;padding:0.55rem 1.3rem;border:none!important;}}
.stTabs [aria-selected="true"]{{background:{ACCENT}!important;color:#fff!important;box-shadow:0 2px 8px rgba(27,79,216,0.3);}}
.streamlit-expanderHeader{{background:{CARD}!important;border:1px solid {BORDER}!important;border-radius:10px!important;color:{TEXT}!important;font-weight:600!important;font-size:0.9rem!important;}}
.streamlit-expanderContent{{background:{BG}!important;border:1px solid {BORDER}!important;border-top:none!important;border-radius:0 0 10px 10px!important;}}
.stSelectbox>div>div,.stNumberInput>div>div>input,.stTextInput>div>div>input,.stPasswordInput>div>div>input{{background:{INPUT}!important;border:1.5px solid {IBORDER}!important;border-radius:8px!important;color:{TEXT}!important;font-family:'DM Sans',sans-serif!important;font-size:0.9rem!important;}}
.stButton>button{{background:{ACCENT}!important;color:#fff!important;border:none!important;border-radius:10px!important;font-weight:600!important;font-size:0.92rem!important;padding:0.65rem 1.4rem!important;box-shadow:0 2px 8px rgba(27,79,216,0.25)!important;transition:all 0.2s!important;}}
.stButton>button:hover{{background:{ACCENT2}!important;transform:translateY(-1px);}}
.stDownloadButton>button{{background:{BG}!important;color:{MUTED}!important;border:1.5px solid {BORDER}!important;border-radius:8px!important;font-weight:500!important;box-shadow:none!important;}}
.stDataFrame{{border:1px solid {BORDER}!important;border-radius:12px!important;overflow:hidden!important;}}
[data-testid="stMetricLabel"]{{color:{MUTED}!important;font-size:0.78rem!important;font-weight:500!important;text-transform:uppercase;letter-spacing:0.05em;}}
[data-testid="stMetricValue"]{{color:{ACCENT}!important;font-size:1.9rem!important;font-weight:700!important;font-family:'DM Serif Display',serif!important;}}
.section-label{{color:{MUTED};font-size:0.72rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.75rem;margin-top:0.25rem;}}
.clinai-hero{{background:linear-gradient(135deg,{HERO1} 0%,{HERO2} 100%);border-radius:16px;padding:1.6rem 2rem;color:white;margin-bottom:1.5rem;box-shadow:0 4px 20px rgba(27,79,216,0.2);}}
.risk-high{{background:{RBHIGH};border:1.5px solid {BCERR};border-left:4px solid #EF4444;color:{"#991B1B" if not D else "#ffa198"};padding:1rem 1.2rem;border-radius:10px;}}
.risk-medium{{background:{RBMED};border:1.5px solid {BCWARN};border-left:4px solid #F59E0B;color:{TCWARN};padding:1rem 1.2rem;border-radius:10px;}}
.risk-low{{background:{RBLOW};border:1.5px solid {BCOK};border-left:4px solid #22C55E;color:{TCOK};padding:1rem 1.2rem;border-radius:10px;}}
.alert-warn{{background:{ABWARN};border:1.5px solid {BCWARN};border-radius:10px;padding:1rem 1.2rem;color:{TCWARN};font-size:0.9rem;margin:0.6rem 0;}}
.alert-info{{background:{ABINFO};border:1.5px solid {BCINFO};border-radius:10px;padding:1rem 1.2rem;color:{TCINFO};font-size:0.9rem;margin:0.6rem 0;}}
.alert-ok{{background:{ABOK};border:1.5px solid {BCOK};border-radius:10px;padding:1rem 1.2rem;color:{TCOK};font-size:0.9rem;margin:0.6rem 0;}}
.alert-err{{background:{ABERR};border:1.5px solid {BCERR};border-radius:10px;padding:1rem 1.2rem;color:{TCERR};font-size:0.9rem;margin:0.6rem 0;}}
</style>
""", unsafe_allow_html=True)

def api_headers(): return {"Authorization": f"Bearer {st.session_state.token}"}
def handle_401(r):
    if r.status_code == 401:
        st.session_state.token = None; st.rerun()

def authenticate(u, p):
    try:
        r = requests.post(f"{API_URL}/token", json={"username":u,"password":p}, timeout=5)
        return r.json().get("access_token") if r.status_code==200 else None
    except: return None

# ── LOGIN ─────────────────────────────────────────────────────────
if not st.session_state.token:
    col = st.columns([1,1.2,1])[1]
    with col:
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:20px;padding:2.5rem 2rem;
                    box-shadow:0 8px 40px rgba(0,0,0,0.10);text-align:center;margin-top:8vh;">
          <div style="font-size:2.5rem;margin-bottom:0.5rem;">🩺</div>
          <div style="font-family:'DM Serif Display',serif;font-size:1.8rem;color:{ACCENT};">ClinAI</div>
          <div style="color:{MUTED};font-size:0.85rem;margin-top:0.2rem;">Système de prédiction de réhospitalisation</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        with st.form("login"):
            st.markdown(f'<p style="color:{MUTED};font-size:0.85rem;font-weight:500;margin-bottom:0.2rem;">Identifiant praticien</p>', unsafe_allow_html=True)
            username = st.text_input("", placeholder="medecin", label_visibility="collapsed")
            st.markdown(f'<p style="color:{MUTED};font-size:0.85rem;font-weight:500;margin-bottom:0.2rem;margin-top:0.6rem;">Mot de passe</p>', unsafe_allow_html=True)
            password = st.text_input("", type="password", label_visibility="collapsed")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.form_submit_button("Accéder au tableau de bord", use_container_width=True):
                t = authenticate(username, password)
                if t:
                    st.session_state.token = t; st.session_state.username = username; st.rerun()
                else:
                    st.markdown('<div class="alert-err">❌ Identifiants incorrects ou service indisponible.</div>', unsafe_allow_html=True)
    st.stop()

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="margin-bottom:1.2rem;display:flex;align-items:center;gap:0.7rem;">
      <span style="font-size:1.8rem;">🩺</span>
      <div>
        <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;color:{ACCENT};">ClinAI</div>
        <div style="color:{MUTED};font-size:0.75rem;">Réhospitalisation · Diabète</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Toggle dark mode
    mode_label = "☀️ Mode clair" if D else "🌙 Mode sombre"
    if st.button(mode_label, use_container_width=True, key="toggle_dark"):
        st.session_state.dark_mode = not D; st.rerun()

    st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:1rem 0;">', unsafe_allow_html=True)
    st.markdown(f'<div style="background:{"#0d1f3c" if D else "#EFF6FF"};border-radius:10px;padding:0.75rem 1rem;margin-bottom:1.2rem;"><div style="color:{MUTED};font-size:0.78rem;">👤 Connecté</div><div style="color:{ACCENT};font-weight:700;">{st.session_state.username}</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="section-label">Modèle actif</div>', unsafe_allow_html=True)
    try:
        r = requests.get(f"{API_URL}/model/info", headers=api_headers(), timeout=3)
        if r.status_code == 200:
            m = r.json()
            c1,c2 = st.columns(2)
            c1.metric("AUC-ROC", m.get("auc_roc","—"))
            c2.metric("AUC-PR",  m.get("auc_pr","—"))
            if m.get("f1_score"): st.metric("F1-Score", m.get("f1_score","—"))
            st.markdown(f'<div style="color:{MUTED};font-size:0.75rem;font-family:JetBrains Mono,monospace;margin-top:0.4rem;">{m.get("feature_count","?")} features · {m.get("n_train",0):,} patients</div>', unsafe_allow_html=True)
    except: st.markdown(f'<div style="color:{MUTED};font-size:0.82rem;">Infos modèle indisponibles</div>', unsafe_allow_html=True)

    st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:1rem 0;">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-label">Activité</div>', unsafe_allow_html=True)
    try:
        r2 = requests.get(f"{API_URL}/stats", headers=api_headers(), timeout=3)
        if r2.status_code==200:
            s = r2.json()
            st.metric("Prédictions totales", f"{s['total']:,}")
            if s["total"]>0:
                pct = s["eleve"]/s["total"]*100
                color = "#EF4444" if pct>30 else "#F59E0B" if pct>15 else "#22C55E"
                st.markdown(f'<div style="color:{color};font-size:0.88rem;font-weight:600;">{pct:.1f}% risque élevé</div>', unsafe_allow_html=True)
    except: pass

    st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:1rem 0;">', unsafe_allow_html=True)
    if st.button("Déconnexion", use_container_width=True): st.session_state.token=None; st.rerun()

# ── HERO ──────────────────────────────────────────────────────────
st.markdown(f"""
<div class="clinai-hero">
  <div style="display:flex;align-items:center;gap:1rem;">
    <span style="font-size:2.2rem;">🩺</span>
    <div>
      <div style="font-family:'DM Serif Display',serif;font-size:1.7rem;font-weight:400;line-height:1.2;">Prédiction de Réhospitalisation</div>
      <div style="opacity:0.82;font-size:0.88rem;margin-top:0.35rem;">Score de risque à 30 jours · XGBoost + SHAP · 130 hôpitaux US · 1999–2008</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_history, tab_stats = st.tabs(["  🔍 Nouvelle prédiction  ","  📋 Historique  ","  📊 Statistiques  "])

# ══════════════════════════════════════════════════════════════════
# ONGLET 1 — PRÉDICTION
# ══════════════════════════════════════════════════════════════════
with tab_predict:
    col_form, col_result = st.columns([1.05,1], gap="large")
    with col_form:
        st.markdown('<div class="section-label">Données cliniques du patient</div>', unsafe_allow_html=True)
        with st.expander("👤  Démographie", expanded=True):
            age = st.slider("Âge (années)", 0, 100, 65)
            g1,g2 = st.columns(2)
            gender = g1.selectbox("Genre", ["Female","Male","Unknown"])
            race   = g2.selectbox("Origine", ["Caucasian","AfricanAmerican","Hispanic","Asian","Other","Unknown"])
        with st.expander("🏥  Séjour hospitalier", expanded=True):
            h1,h2 = st.columns(2)
            time_hosp = h1.number_input("Durée (jours)",   min_value=1, max_value=30,  value=5)
            num_meds  = h2.number_input("Médicaments",     min_value=0, max_value=100, value=15)
            h3,h4 = st.columns(2)
            num_lab  = h3.number_input("Procédures labo",  min_value=0, max_value=130, value=40)
            num_proc = h4.number_input("Autres procédures",min_value=0, max_value=10,  value=1)
            num_diag = st.number_input("Nombre de diagnostics", min_value=1, max_value=16, value=7)
        with st.expander("📅  Historique 12 mois", expanded=True):
            p1,p2,p3 = st.columns(3)
            num_out  = p1.number_input("Ambulatoire", min_value=0, max_value=50, value=0)
            num_emer = p2.number_input("Urgences",    min_value=0, max_value=50, value=0)
            num_inp  = p3.number_input("Hospitalisé", min_value=0, max_value=20, value=1)
        with st.expander("🔬  Résultats cliniques", expanded=True):
            r1,r2 = st.columns(2)
            a1c     = r1.selectbox("HbA1c",           ["None","Norm",">7",">8"])
            glucose = r2.selectbox("Glucose sérique", ["None","Norm",">200",">300"])
            r3,r4 = st.columns(2)
            change_meds  = r3.selectbox("Changement médication", ["No","Ch"])
            diabetes_med = r4.selectbox("Anti-diabétiques",      ["Yes","No"])
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍  Calculer le score de risque", use_container_width=True)

    with col_result:
        st.markdown('<div class="section-label">Résultat & Analyse</div>', unsafe_allow_html=True)
        if predict_btn:
            payload = {
                "age_num":age,"gender":gender,"race":race,
                "time_in_hospital":time_hosp,"num_medications":num_meds,
                "num_lab_procedures":num_lab,"num_procedures":num_proc,
                "number_diagnoses":num_diag,"num_outpatient":num_out,
                "num_emergency":num_emer,"num_inpatient":num_inp,
                "a1c_result":a1c,"glucose_serum":glucose,
                "change_in_meds":change_meds,"diabetes_meds":diabetes_med,
            }
            with st.spinner("Analyse en cours..."):
                try:
                    resp = requests.post(f"{API_URL}/predict", json=payload, headers=api_headers(), timeout=10)
                    handle_401(resp)
                    if resp.status_code == 200:
                        result = resp.json(); score = result["risk_score"]; level = result["risk_level"]
                        color_map = {"ÉLEVÉ":"#EF4444","MODÉRÉ":"#F59E0B","FAIBLE":"#22C55E"}
                        gauge_color = color_map.get(level,"#1B4FD8")
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number", value=round(score*100,1),
                            number={"suffix":"%","font":{"size":46,"color":gauge_color,"family":"DM Serif Display"}},
                            gauge={"axis":{"range":[0,100],"tickcolor":BORDER,"tickfont":{"color":MUTED,"size":11}},
                                   "bar":{"color":gauge_color,"thickness":0.22},"bgcolor":CARD,
                                   "bordercolor":BORDER,"borderwidth":1,
                                   "steps":[{"range":[0,30],"color":"#F0FDF4" if not D else "#0d1f17"},
                                            {"range":[30,50],"color":"#FFFBEB" if not D else "#171208"},
                                            {"range":[50,100],"color":"#FEF2F2" if not D else "#1f1315"}],
                                   "threshold":{"line":{"color":"#EF4444","width":2},"value":50}}))
                        fig.update_layout(height=220,margin=dict(t=20,b=0,l=20,r=20),paper_bgcolor=PLOTBG,plot_bgcolor=PLOTBG,font=dict(family="DM Sans"))
                        st.plotly_chart(fig, use_container_width=True)
                        css = {"ÉLEVÉ":"risk-high","MODÉRÉ":"risk-medium","FAIBLE":"risk-low"}[level]
                        ico = {"ÉLEVÉ":"🔴","MODÉRÉ":"🟡","FAIBLE":"🟢"}[level]
                        st.markdown(f'<div class="{css}"><b style="font-size:1rem;">{ico} Risque {level}</b><span style="float:right;font-family:JetBrains Mono,monospace;font-size:0.9rem;">{score:.1%}</span></div>', unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                        if level=="ÉLEVÉ":
                            st.markdown('<div class="alert-warn">⚠️ <b>Suivi renforcé recommandé</b><br><span style="font-size:0.85rem;">Consultation de sortie dédiée · Rappel à 48h · Vérification observance</span></div>', unsafe_allow_html=True)
                        elif level=="MODÉRÉ":
                            st.markdown('<div class="alert-info">ℹ️ <b>Surveillance modérée</b><br><span style="font-size:0.85rem;">Appel de suivi à J+7 · Vérifier l\'observance médicamenteuse</span></div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="alert-ok">✅ <b>Protocole de sortie standard</b><br><span style="font-size:0.85rem;">Aucune mesure supplémentaire requise</span></div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="section-label" style="margin-top:1.2rem;">Facteurs déterminants (SHAP)</div>', unsafe_allow_html=True)
                        factors_df = pd.DataFrame(list(result["top_factors"].items()),columns=["Facteur","Impact"]).sort_values("Impact",key=abs,ascending=True)
                        fig2 = go.Figure(go.Bar(x=factors_df["Impact"],y=factors_df["Facteur"],orientation="h",
                            marker_color=["#EF4444" if v>0 else "#22C55E" for v in factors_df["Impact"]],marker_line_width=0,
                            text=[f"{v:+.4f}" for v in factors_df["Impact"]],textposition="outside",
                            textfont=dict(size=11,color=MUTED,family="JetBrains Mono")))
                        fig2.update_layout(height=250,margin=dict(t=5,b=5,l=5,r=80),xaxis_title="Impact",xaxis_zeroline=True,xaxis_zerolinecolor=BORDER,**PLOTLY_THEME)
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.markdown(f'<div class="alert-err">❌ Erreur API {resp.status_code} — {resp.text[:200]}</div>', unsafe_allow_html=True)
                except requests.exceptions.ConnectionError:
                    st.markdown('<div class="alert-err">🔌 API ml-service inaccessible.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:{CARD};border:1.5px dashed {BORDER};border-radius:14px;padding:3rem 2rem;text-align:center;margin-top:0.5rem;">
              <div style="font-size:2.5rem;margin-bottom:1rem;">🩺</div>
              <div style="color:{MUTED};font-size:0.95rem;line-height:1.6;">Remplis les données patient à gauche<br>puis clique sur <b style="color:{ACCENT};">Calculer le score de risque</b></div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# ONGLET 2 — HISTORIQUE
# ══════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown('<div class="section-label">Historique des prédictions</div>', unsafe_allow_html=True)
    f1,f2,f3 = st.columns([2,2,1])
    filtre = f1.selectbox("Niveau de risque",["Tous","ÉLEVÉ","MODÉRÉ","FAIBLE"],key="hf")
    nb_l   = f2.selectbox("Lignes par page",[25,50,100],index=1,key="hl")
    f3.markdown("<br>", unsafe_allow_html=True)
    if f3.button("↺ Actualiser",key="rh"): st.session_state.hist_offset=0
    niveau_param = None if filtre=="Tous" else filtre
    try:
        params = {"limit":nb_l,"offset":st.session_state.hist_offset}
        if niveau_param: params["risk_level"]=niveau_param
        r = requests.get(f"{API_URL}/predictions",headers=api_headers(),params=params,timeout=5)
        handle_401(r)
        if r.status_code==200:
            data=r.json(); total=data["total"]; items=data["results"]
            if total==0:
                st.markdown('<div class="alert-info">Aucune prédiction enregistrée. Calculez un score dans l\'onglet Nouvelle prédiction.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="color:{MUTED};font-size:0.82rem;margin-bottom:0.75rem;">{total:,} prédiction(s) au total</div>', unsafe_allow_html=True)
                df_h = pd.DataFrame(items)
                df_h["predicted_at"] = pd.to_datetime(df_h["predicted_at"]).dt.strftime("%d/%m/%Y %H:%M")
                df_h["risk_score"]   = (df_h["risk_score"]*100).round(1).astype(str)+"%"
                df_h["risk_level"]   = df_h["risk_level"].map({"ÉLEVÉ":"🔴 ÉLEVÉ","MODÉRÉ":"🟡 MODÉRÉ","FAIBLE":"🟢 FAIBLE"})
                st.dataframe(df_h[["id","predicted_at","risk_score","risk_level","model_version","requested_by"]].rename(columns={
                    "id":"ID","predicted_at":"Date & heure","risk_score":"Score","risk_level":"Niveau","model_version":"Modèle","requested_by":"Praticien"}),
                    use_container_width=True,hide_index=True)
                pa,pb,pc = st.columns([1,2,1])
                with pa:
                    if st.button("← Précédent") and st.session_state.hist_offset>0:
                        st.session_state.hist_offset-=nb_l; st.rerun()
                with pb:
                    shown=min(st.session_state.hist_offset+nb_l,total)
                    st.markdown(f'<div style="text-align:center;color:{MUTED};padding-top:0.5rem;">{shown}/{total:,}</div>',unsafe_allow_html=True)
                with pc:
                    if st.button("Suivant →") and st.session_state.hist_offset+nb_l<total:
                        st.session_state.hist_offset+=nb_l; st.rerun()
                csv=df_h.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Exporter CSV",csv,f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv","text/csv")
        else:
            st.markdown(f'<div class="alert-err">Erreur {r.status_code}</div>',unsafe_allow_html=True)
    except requests.exceptions.ConnectionError:
        st.markdown('<div class="alert-err">🔌 API ml-service inaccessible.</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# ONGLET 3 — STATISTIQUES
# ══════════════════════════════════════════════════════════════════
with tab_stats:
    st.markdown('<div class="section-label">Tableau de bord statistique</div>', unsafe_allow_html=True)
    col_r,_ = st.columns([1,5])
    col_r.button("↺ Actualiser",key="rs")
    try:
        r = requests.get(f"{API_URL}/stats",headers=api_headers(),timeout=5)
        handle_401(r)
        if r.status_code==200:
            s=r.json()
            if s["total"]==0:
                st.markdown('<div class="alert-info">Effectuez des prédictions pour voir les statistiques.</div>',unsafe_allow_html=True)
            else:
                k1,k2,k3,k4,k5=st.columns(5)
                k1.metric("Total",f"{s['total']:,}")
                k2.metric("🔴 Élevé", str(s['eleve']),  f"{s['eleve']/s['total']*100:.1f}%")
                k3.metric("🟡 Modéré",str(s['modere']), f"{s['modere']/s['total']*100:.1f}%")
                k4.metric("🟢 Faible",str(s['faible']), f"{s['faible']/s['total']*100:.1f}%")
                k5.metric("Score moyen",f"{s['score_moyen']*100:.1f}%")
                st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:1rem 0;">',unsafe_allow_html=True)
                cl,cr=st.columns(2)
                with cl:
                    st.markdown('<div class="section-label">Répartition des niveaux</div>',unsafe_allow_html=True)
                    fig_d=go.Figure(go.Pie(labels=["Élevé","Modéré","Faible"],values=[s["eleve"],s["modere"],s["faible"]],
                        hole=0.62,marker_colors=["#EF4444","#F59E0B","#22C55E"],textinfo="label+percent",
                        hoverinfo="label+value",textfont=dict(family="DM Sans",size=12)))
                    fig_d.update_layout(height=280,margin=dict(t=10,b=10,l=10,r=10),showlegend=False,**PLOTLY_THEME)
                    st.plotly_chart(fig_d,use_container_width=True)
                with cr:
                    st.markdown('<div class="section-label">Facteurs les plus influents</div>',unsafe_allow_html=True)
                    if s["top_factors_global"]:
                        df_f=pd.DataFrame(s["top_factors_global"])
                        fig_f=go.Figure(go.Bar(x=df_f["mean_abs_shap"],y=df_f["feature"],orientation="h",
                            marker_color=ACCENT,marker_line_width=0,opacity=0.85,
                            text=df_f["mean_abs_shap"].round(4),textposition="outside",
                            textfont=dict(size=10,color=MUTED,family="JetBrains Mono")))
                        fig_f.update_layout(height=280,margin=dict(t=5,b=5,l=5,r=70),
                            paper_bgcolor=PLOTBG,plot_bgcolor=PLOTBG,
                            font=dict(color=MUTED,family="DM Sans"),
                            xaxis=dict(title="|SHAP| moyen",gridcolor=GRIDC,linecolor=BORDER),
                            yaxis=dict(categoryorder="total ascending",gridcolor=GRIDC,linecolor=BORDER))
                        st.plotly_chart(fig_f,use_container_width=True)
                if s["trend"]:
                    st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:0.5rem 0 1rem;">',unsafe_allow_html=True)
                    st.markdown('<div class="section-label">Activité — 30 derniers jours</div>',unsafe_allow_html=True)
                    df_t=pd.DataFrame(s["trend"]); df_t["jour"]=pd.to_datetime(df_t["jour"])
                    fig_t=go.Figure()
                    fig_t.add_trace(go.Bar(x=df_t["jour"],y=df_t["nb"],name="Total",marker_color="#93C5FD",opacity=0.6,marker_line_width=0))
                    fig_t.add_trace(go.Bar(x=df_t["jour"],y=df_t["nb_eleve"],name="Risque élevé",marker_color="#EF4444",opacity=0.85,marker_line_width=0))
                    fig_t.add_trace(go.Scatter(x=df_t["jour"],y=df_t["score_moyen"]*100,name="Score moyen (%)",yaxis="y2",
                        mode="lines+markers",line=dict(color="#F59E0B",width=2),marker=dict(size=4)))
                    fig_t.update_layout(height=300,barmode="overlay",
                        paper_bgcolor=PLOTBG,plot_bgcolor=PLOTBG,
                        font=dict(color=MUTED,family="DM Sans"),
                        xaxis=dict(gridcolor=GRIDC,linecolor=BORDER),
                        yaxis=dict(title="Prédictions",gridcolor=GRIDC),
                        yaxis2=dict(title="Score moyen (%)",overlaying="y",side="right",range=[0,100],gridcolor="rgba(0,0,0,0)"),
                        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,bgcolor="rgba(0,0,0,0)",font=dict(size=11)),
                        margin=dict(t=40,b=20,l=10,r=10))
                    st.plotly_chart(fig_t,use_container_width=True)
    except requests.exceptions.ConnectionError:
        st.markdown('<div class="alert-err">🔌 API ml-service inaccessible.</div>',unsafe_allow_html=True)