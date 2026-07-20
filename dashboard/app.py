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
    model_meta_info = {}
    try:
        r = requests.get(f"{API_URL}/model/info", headers=api_headers(), timeout=3)
        if r.status_code == 200:
            m = r.json()
            model_meta_info = m
            c1,c2 = st.columns(2)
            c1.metric("AUC-ROC", m.get("auc_roc","—"))
            c2.metric("AUC-PR",  m.get("auc_pr","—"))
            if m.get("f1_score"): st.metric("F1-Score", m.get("f1_score","—"))
            n_total = m.get("n_fit",0) + m.get("n_calib",0) + m.get("n_test",0)
            st.markdown(f'<div style="color:{MUTED};font-size:0.75rem;font-family:JetBrains Mono,monospace;margin-top:0.4rem;">{m.get("feature_count","?")} features · {n_total:,} patients</div>', unsafe_allow_html=True)
            if m.get("calibration_method"):
                st.markdown(f'<div style="color:{MUTED};font-size:0.72rem;margin-top:0.2rem;">Probabilités calibrées ({m.get("calibration_method")})</div>', unsafe_allow_html=True)
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
    st.markdown(f'<div class="section-label">Seuils de risque</div>', unsafe_allow_html=True)
    st.caption("Ajuste les seuils selon ta tolérance clinique")
    # Préremplis à partir du seuil optimisé par train.py (model_meta["best_threshold"]),
    # PAS de valeurs fixes 50%/30% : après calibration, un modèle entraîné sur une
    # classe positive rare (~9% ici) produit des scores réalistes bien en dessous
    # de 50% même pour les patients à risque réel. Avec un seuil fixe à 50%, quasi
    # aucun patient n'atteindrait jamais "ÉLEVÉ", quel que soit son profil clinique.
    _best_thr = model_meta_info.get("best_threshold")
    if _best_thr is not None:
        default_eleve_pct  = max(round(_best_thr * 100), 5)
        default_modere_pct = max(round(_best_thr * 100 * 0.6), 2)
    else:
        default_eleve_pct, default_modere_pct = 50, 30
    seuil_eleve  = st.slider("Seuil risque élevé (%)",  min_value=5,  max_value=80, value=default_eleve_pct,  step=1, key="seuil_eleve")
    seuil_modere = st.slider("Seuil risque modéré (%)", min_value=1, max_value=max(seuil_eleve-1,2), value=min(default_modere_pct,seuil_eleve-1), step=1, key="seuil_modere")
    if _best_thr is not None:
        st.caption(f"Seuil optimisé par le modèle (max F1) : {_best_thr*100:.1f}%")
    st.markdown(f'<div style="background:{"#0d1626" if D else "#EFF6FF"};border-radius:8px;padding:0.6rem 0.8rem;font-size:0.78rem;color:{MUTED};margin-top:0.3rem;">Faible : 0-{seuil_modere}% | Modéré : {seuil_modere}-{seuil_eleve}% | Élevé : {seuil_eleve}-100%</div>', unsafe_allow_html=True)
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

tab_predict, tab_history, tab_stats, tab_compare, tab_whatif, tab_modelcard = st.tabs(["  🔍 Nouvelle prédiction  ","  📋 Historique  ","  📊 Statistiques  ","  ⚖️ Comparaison  ","  🧪 What-if  ","  📄 Model Card  "])

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
                        result = resp.json(); score = result["risk_score"]
                        # Seuils personnalisés
                        seuil_e = st.session_state.get("seuil_eleve", 50) / 100
                        seuil_m = st.session_state.get("seuil_modere", 30) / 100
                        if score >= seuil_e:   level = "ÉLEVÉ"
                        elif score >= seuil_m: level = "MODÉRÉ"
                        else:                  level = "FAIBLE"
                        color_map = {"ÉLEVÉ":"#EF4444","MODÉRÉ":"#F59E0B","FAIBLE":"#22C55E"}
                        gauge_color = color_map.get(level,"#1B4FD8")
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number", value=round(score*100,1),
                            domain={"x": [0, 1], "y": [0, 1]},
                            number={"suffix":"%","font":{"size":36,"color":gauge_color,"family":"DM Serif Display"}},
                            gauge={"axis":{"range":[0,100],"tickcolor":BORDER,"tickfont":{"color":MUTED,"size":11}},
                                   "bar":{"color":gauge_color,"thickness":0.22},"bgcolor":CARD,
                                   "bordercolor":BORDER,"borderwidth":1,
                                   "steps":[{"range":[0,30],"color":"#F0FDF4" if not D else "#0d1f17"},
                                            {"range":[30,50],"color":"#FFFBEB" if not D else "#171208"},
                                            {"range":[50,100],"color":"#FEF2F2" if not D else "#1f1315"}],
                                   "threshold":{"line":{"color":"#EF4444","width":2},"value":st.session_state.get("seuil_eleve",50)}}))
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
                        # PDF button
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown('<div class="section-label">Rapport</div>', unsafe_allow_html=True)
                        try:
                            pdf_resp = requests.post(f"{API_URL}/predict/pdf", json=payload, headers=api_headers(), timeout=15)
                            if pdf_resp.status_code == 200:
                                st.download_button(label="📄 Télécharger le rapport PDF", data=pdf_resp.content, file_name=f"rapport_ClinAI_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", mime="application/pdf", use_container_width=True)
                            else:
                                st.markdown('<div class="alert-warn">⚠️ Rapport PDF indisponible.</div>', unsafe_allow_html=True)
                        except Exception:
                            st.markdown('<div class="alert-warn">⚠️ Rapport PDF indisponible.</div>', unsafe_allow_html=True)
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
                # ── SHAP Global étendu ───────────────────────────────
                if s["top_factors_global"]:
                    st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:0.5rem 0 1rem;">',unsafe_allow_html=True)
                    st.markdown('<div class="section-label">Analyse SHAP globale — Impact moyen sur toutes les prédictions</div>',unsafe_allow_html=True)
                    st.caption("Montre quelles features ont le plus influencé les prédictions en moyenne. Plus la barre est longue, plus la feature est importante.")

                    df_shap = pd.DataFrame(s["top_factors_global"])
                    df_shap = df_shap.sort_values("mean_abs_shap", ascending=True)
                    df_shap["feature_clean"] = df_shap["feature"].str.replace("_", " ").str.title()

                    # Couleur dégradée selon importance
                    max_val = df_shap["mean_abs_shap"].max()
                    colors_shap = [f"rgba(27,79,216,{0.3 + 0.7*(v/max_val):.2f})" for v in df_shap["mean_abs_shap"]]

                    fig_shap = go.Figure()
                    fig_shap.add_trace(go.Bar(
                        x=df_shap["mean_abs_shap"],
                        y=df_shap["feature_clean"],
                        orientation="h",
                        marker_color=colors_shap,
                        marker_line_width=0,
                        text=[f"{v:.4f}" for v in df_shap["mean_abs_shap"]],
                        textposition="outside",
                        textfont=dict(size=10, color=MUTED, family="JetBrains Mono"),
                        hovertemplate="<b>%{y}</b><br>Impact SHAP moyen : %{x:.4f}<extra></extra>",
                    ))

                    # Ligne médiane
                    median_val = float(df_shap["mean_abs_shap"].median())
                    fig_shap.add_vline(
                        x=median_val,
                        line_dash="dash",
                        line_color=MUTED,
                        annotation_text=f"Médiane ({median_val:.4f})",
                        annotation_position="top",
                        annotation_font_size=9,
                        annotation_font_color=MUTED,
                    )

                    fig_shap.update_layout(
                        height=max(350, len(df_shap)*30),
                        margin=dict(t=30, b=20, l=10, r=80),
                        paper_bgcolor=PLOTBG,
                        plot_bgcolor=PLOTBG,
                        font=dict(color=MUTED, family="DM Sans"),
                        xaxis=dict(title="|SHAP| moyen", gridcolor=GRIDC, linecolor=BORDER, tickfont=dict(size=10)),
                        yaxis=dict(gridcolor=GRIDC, linecolor=BORDER, tickfont=dict(size=10)),
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)

                    # Tableau détaillé
                    with st.expander("📋 Voir le tableau détaillé"):
                        df_display = df_shap[["feature_clean","mean_abs_shap"]].copy()
                        df_display.columns = ["Feature", "|SHAP| moyen"]
                        df_display["|SHAP| moyen"] = df_display["|SHAP| moyen"].round(5)
                        df_display["Rang"] = range(len(df_display), 0, -1)
                        df_display = df_display[["Rang","Feature","|SHAP| moyen"]].sort_values("Rang")
                        st.dataframe(df_display, use_container_width=True, hide_index=True)

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

# ══════════════════════════════════════════════════════════════════
# ONGLET 4 — COMPARAISON DE DEUX PATIENTS
# ══════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown('<div class="section-label">Comparaison de deux profils patients</div>', unsafe_allow_html=True)
    st.caption("Saisissez les données de deux patients pour comparer leurs scores de risque et facteurs déterminants.")

    col_p1, col_sep, col_p2 = st.columns([1, 0.05, 1], gap="small")

    def patient_form(col, label, key_prefix):
        with col:
            st.markdown(f'<div style="background:{"#161b22" if D else "#EFF6FF"};border-radius:10px;padding:1rem 1.2rem;border:1px solid {BORDER};margin-bottom:1rem;"><b style="color:{ACCENT};">{label}</b></div>', unsafe_allow_html=True)
            age   = st.slider("Âge", 0, 100, 65, key=f"{key_prefix}_age")
            g1,g2 = st.columns(2)
            gender = g1.selectbox("Genre",   ["Female","Male","Unknown"], key=f"{key_prefix}_gender")
            race   = g2.selectbox("Origine", ["Caucasian","AfricanAmerican","Hispanic","Asian","Other","Unknown"], key=f"{key_prefix}_race")
            h1,h2 = st.columns(2)
            time_h = h1.number_input("Durée séjour (j)", 1, 30, 5,  key=f"{key_prefix}_time")
            n_meds = h2.number_input("Médicaments",      0, 100, 15, key=f"{key_prefix}_meds")
            h3,h4 = st.columns(2)
            n_lab  = h3.number_input("Procédures labo", 0, 130, 40, key=f"{key_prefix}_lab")
            n_proc = h4.number_input("Procédures",      0, 10,  1,  key=f"{key_prefix}_proc")
            n_diag = st.number_input("Diagnostics",     1, 16,  7,  key=f"{key_prefix}_diag")
            p1c,p2c,p3c = st.columns(3)
            n_out  = p1c.number_input("Ambul.", 0, 50, 0, key=f"{key_prefix}_out")
            n_emer = p2c.number_input("Urg.",   0, 50, 0, key=f"{key_prefix}_emer")
            n_inp  = p3c.number_input("Hosp.",  0, 20, 1, key=f"{key_prefix}_inp")
            r1,r2 = st.columns(2)
            a1c    = r1.selectbox("HbA1c",          ["None","Norm",">7",">8"], key=f"{key_prefix}_a1c")
            glucose= r2.selectbox("Glucose",        ["None","Norm",">200",">300"], key=f"{key_prefix}_gluc")
            r3,r4 = st.columns(2)
            change = r3.selectbox("Chgt médication", ["No","Ch"],  key=f"{key_prefix}_change")
            diab   = r4.selectbox("Anti-diab.",      ["Yes","No"], key=f"{key_prefix}_diab")
            return {
                "age_num":age,"gender":gender,"race":race,
                "time_in_hospital":time_h,"num_medications":n_meds,
                "num_lab_procedures":n_lab,"num_procedures":n_proc,
                "number_diagnoses":n_diag,"num_outpatient":n_out,
                "num_emergency":n_emer,"num_inpatient":n_inp,
                "a1c_result":a1c,"glucose_serum":glucose,
                "change_in_meds":change,"diabetes_meds":diab,
            }

    p1_data = patient_form(col_p1, "👤 Patient A", "p1")
    with col_sep:
        st.markdown(f'<div style="border-left:2px dashed {BORDER};height:100%;margin:0 auto;width:1px;"></div>', unsafe_allow_html=True)
    p2_data = patient_form(col_p2, "👤 Patient B", "p2")

    st.markdown("<br>", unsafe_allow_html=True)
    compare_btn = st.button("⚖️ Comparer les deux patients", use_container_width=True, key="compare_btn")

    if compare_btn:
        with st.spinner("Analyse en cours..."):
            try:
                batch_payload = {"patients": [
                    {**p1_data, "patient_label": "A"},
                    {**p2_data, "patient_label": "B"},
                ]}
                rb = requests.post(f"{API_URL}/predict/batch", json=batch_payload, headers=api_headers(), timeout=15)

                if rb.status_code == 200:
                    res1, res2 = rb.json()
                    s1 = res1["risk_score"]; s2 = res2["risk_score"]

                    seuil_e = st.session_state.get("seuil_eleve", 50) / 100
                    seuil_m = st.session_state.get("seuil_modere", 30) / 100

                    def get_level(score):
                        if score >= seuil_e:   return "ÉLEVÉ"
                        elif score >= seuil_m: return "MODÉRÉ"
                        else:                  return "FAIBLE"

                    l1 = get_level(s1); l2 = get_level(s2)
                    color_map = {"ÉLEVÉ":"#EF4444","MODÉRÉ":"#F59E0B","FAIBLE":"#22C55E"}
                    css_map   = {"ÉLEVÉ":"risk-high","MODÉRÉ":"risk-medium","FAIBLE":"risk-low"}
                    ico_map   = {"ÉLEVÉ":"🔴","MODÉRÉ":"🟡","FAIBLE":"🟢"}

                    st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:1rem 0;">', unsafe_allow_html=True)
                    st.markdown('<div class="section-label">Résultats de comparaison</div>', unsafe_allow_html=True)

                    # Scores côte à côte
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        fig1 = go.Figure(go.Indicator(
                            mode="gauge+number", value=round(s1*100,1),
                            domain={"x":[0,1],"y":[0,1]},
                            number={"suffix":"%","font":{"size":38,"color":color_map[l1]}},
                            title={"text":"Patient A","font":{"size":13,"color":MUTED}},
                            gauge={"axis":{"range":[0,100],"tickcolor":BORDER},
                                   "bar":{"color":color_map[l1],"thickness":0.22},
                                   "bgcolor":CARD,"bordercolor":BORDER,
                                   "steps":[{"range":[0,seuil_m*100],"color":"#F0FDF4" if not D else "#0d1f17"},
                                            {"range":[seuil_m*100,seuil_e*100],"color":"#FFFBEB" if not D else "#171208"},
                                            {"range":[seuil_e*100,100],"color":"#FEF2F2" if not D else "#1f1315"}],
                                   "threshold":{"line":{"color":"#EF4444","width":2},"value":seuil_e*100}}))
                        fig1.update_layout(height=280,margin=dict(t=40,b=20,l=30,r=30),paper_bgcolor=PLOTBG,plot_bgcolor=PLOTBG)
                        st.plotly_chart(fig1, use_container_width=True)
                        st.markdown(f'<div class="{css_map[l1]}" style="text-align:center;"><b>{ico_map[l1]} Risque {l1}</b> — {s1:.1%}</div>', unsafe_allow_html=True)

                    with rc2:
                        fig2 = go.Figure(go.Indicator(
                            mode="gauge+number", value=round(s2*100,1),
                            domain={"x":[0,1],"y":[0,1]},
                            number={"suffix":"%","font":{"size":38,"color":color_map[l2]}},
                            title={"text":"Patient B","font":{"size":13,"color":MUTED}},
                            gauge={"axis":{"range":[0,100],"tickcolor":BORDER},
                                   "bar":{"color":color_map[l2],"thickness":0.22},
                                   "bgcolor":CARD,"bordercolor":BORDER,
                                   "steps":[{"range":[0,seuil_m*100],"color":"#F0FDF4" if not D else "#0d1f17"},
                                            {"range":[seuil_m*100,seuil_e*100],"color":"#FFFBEB" if not D else "#171208"},
                                            {"range":[seuil_e*100,100],"color":"#FEF2F2" if not D else "#1f1315"}],
                                   "threshold":{"line":{"color":"#EF4444","width":2},"value":seuil_e*100}}))
                        fig2.update_layout(height=280,margin=dict(t=40,b=20,l=30,r=30),paper_bgcolor=PLOTBG,plot_bgcolor=PLOTBG)
                        st.plotly_chart(fig2, use_container_width=True)
                        st.markdown(f'<div class="{css_map[l2]}" style="text-align:center;"><b>{ico_map[l2]} Risque {l2}</b> — {s2:.1%}</div>', unsafe_allow_html=True)

                    # Verdict
                    st.markdown("<br>", unsafe_allow_html=True)
                    diff = abs(s1 - s2) * 100
                    if s1 > s2:
                        st.markdown(f'<div class="alert-warn">⚖️ <b>Patient A est plus à risque</b> que Patient B — différence de {diff:.1f} points de pourcentage.</div>', unsafe_allow_html=True)
                    elif s2 > s1:
                        st.markdown(f'<div class="alert-warn">⚖️ <b>Patient B est plus à risque</b> que Patient A — différence de {diff:.1f} points de pourcentage.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="alert-info">⚖️ Les deux patients ont le <b>même niveau de risque</b>.</div>', unsafe_allow_html=True)

                    # Comparaison SHAP
                    st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:1rem 0;">', unsafe_allow_html=True)
                    st.markdown('<div class="section-label">Comparaison des facteurs SHAP</div>', unsafe_allow_html=True)

                    f1_data = res1["top_factors"]; f2_data = res2["top_factors"]
                    all_features = list(set(list(f1_data.keys()) + list(f2_data.keys())))
                    v1 = [f1_data.get(f, 0) for f in all_features]
                    v2 = [f2_data.get(f, 0) for f in all_features]

                    fig_comp = go.Figure()
                    fig_comp.add_trace(go.Bar(name="Patient A", x=all_features, y=v1,
                        marker_color="#1B4FD8", opacity=0.85, marker_line_width=0))
                    fig_comp.add_trace(go.Bar(name="Patient B", x=all_features, y=v2,
                        marker_color="#F59E0B", opacity=0.85, marker_line_width=0))
                    fig_comp.update_layout(
                        barmode="group", height=300,
                        margin=dict(t=10,b=20,l=10,r=10),
                        paper_bgcolor=PLOTBG, plot_bgcolor=PLOTBG,
                        font=dict(color=MUTED, family="DM Sans"),
                        xaxis=dict(gridcolor=GRIDC, linecolor=BORDER, tickangle=-30),
                        yaxis=dict(title="Impact SHAP", gridcolor=GRIDC, linecolor=BORDER),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                   xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)

                else:
                    st.markdown('<div class="alert-err">❌ Erreur lors de la comparaison.</div>', unsafe_allow_html=True)

            except requests.exceptions.ConnectionError:
                st.markdown('<div class="alert-err">🔌 API ml-service inaccessible.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:{CARD};border:1.5px dashed {BORDER};border-radius:14px;
                    padding:2rem;text-align:center;margin-top:1rem;">
          <div style="font-size:2rem;margin-bottom:0.8rem;">⚖️</div>
          <div style="color:{MUTED};font-size:0.9rem;line-height:1.6;">
            Remplis les données des deux patients<br>
            puis clique sur <b style="color:{ACCENT};">Comparer les deux patients</b>
          </div>
        </div>""", unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════════
# ONGLET 5 — SIMULATEUR WHAT-IF
# ══════════════════════════════════════════════════════════════════
with tab_whatif:
    st.markdown('<div class="section-label">Patient de référence</div>', unsafe_allow_html=True)
    st.caption("Définis un patient à gauche, puis ajuste les curseurs de scénario à droite — le risque se recalcule automatiquement.")

    col_base, col_sim = st.columns([1, 1.1], gap="large")

    with col_base:
        with st.expander("👤 Profil du patient", expanded=True):
            wi_age = st.slider("Âge (années)", 0, 100, 65, key="wi_age")
            wg1, wg2 = st.columns(2)
            wi_gender = wg1.selectbox("Genre", ["Female","Male","Unknown"], key="wi_gender")
            wi_race   = wg2.selectbox("Origine", ["Caucasian","AfricanAmerican","Hispanic","Asian","Other","Unknown"], key="wi_race")
            wh1, wh2 = st.columns(2)
            wi_time_hosp = wh1.number_input("Durée (jours)", min_value=1, max_value=30, value=5, key="wi_time_hosp")
            wi_num_meds  = wh2.number_input("Médicaments", min_value=0, max_value=100, value=15, key="wi_num_meds")
            wh3, wh4 = st.columns(2)
            wi_num_lab  = wh3.number_input("Procédures labo", min_value=0, max_value=130, value=40, key="wi_num_lab")
            wi_num_proc = wh4.number_input("Autres procédures", min_value=0, max_value=10, value=1, key="wi_num_proc")
            wi_num_diag = st.number_input("Nombre de diagnostics", min_value=1, max_value=16, value=7, key="wi_num_diag")
            wp1, wp2, wp3 = st.columns(3)
            wi_num_out  = wp1.number_input("Ambulatoire", min_value=0, max_value=50, value=0, key="wi_num_out")
            wi_num_emer = wp2.number_input("Urgences", min_value=0, max_value=50, value=0, key="wi_num_emer")
            wi_num_inp  = wp3.number_input("Hospitalisé", min_value=0, max_value=20, value=1, key="wi_num_inp")
            wr1, wr2 = st.columns(2)
            wi_a1c     = wr1.selectbox("HbA1c", ["None","Norm",">7",">8"], key="wi_a1c")
            wi_glucose = wr2.selectbox("Glucose sérique", ["None","Norm",">200",">300"], key="wi_glucose")
            wr3, wr4 = st.columns(2)
            wi_change_meds  = wr3.selectbox("Changement médication", ["No","Ch"], key="wi_change_meds")
            wi_diabetes_med = wr4.selectbox("Anti-diabétiques", ["Yes","No"], key="wi_diabetes_med")

    with col_sim:
        st.markdown('<div class="section-label">Scénario "What-if"</div>', unsafe_allow_html=True)
        st.caption("Simule une intervention clinique sur ce patient.")

        wi_delta_meds = st.slider("Δ Médicaments (déprescription si négatif)", -10, 10, 0, key="wi_delta_meds")
        wi_delta_hosp = st.slider("Δ Durée de séjour (jours)", -5, 5, 0, key="wi_delta_hosp")
        wi_add_outpatient = st.checkbox("➕ Ajouter un suivi ambulatoire (+1 visite)", key="wi_add_outpatient")
        wi_override_a1c = st.selectbox("Nouveau contrôle HbA1c (optionnel)", ["(inchangé)","None","Norm",">7",">8"], key="wi_override_a1c")
        wi_override_change = st.selectbox("Changement de médication (optionnel)", ["(inchangé)","No","Ch"], key="wi_override_change")

        sim_num_meds  = max(0, wi_num_meds + wi_delta_meds)
        sim_time_hosp = max(1, wi_time_hosp + wi_delta_hosp)
        sim_num_out   = wi_num_out + (1 if wi_add_outpatient else 0)
        sim_a1c       = wi_a1c if wi_override_a1c == "(inchangé)" else wi_override_a1c
        sim_change    = wi_change_meds if wi_override_change == "(inchangé)" else wi_override_change

        baseline_payload = {
            "age_num": wi_age, "gender": wi_gender, "race": wi_race,
            "time_in_hospital": wi_time_hosp, "num_medications": wi_num_meds,
            "num_lab_procedures": wi_num_lab, "num_procedures": wi_num_proc,
            "number_diagnoses": wi_num_diag, "num_outpatient": wi_num_out,
            "num_emergency": wi_num_emer, "num_inpatient": wi_num_inp,
            "a1c_result": wi_a1c, "glucose_serum": wi_glucose,
            "change_in_meds": wi_change_meds, "diabetes_meds": wi_diabetes_med,
            "patient_label": "Référence",
        }
        scenario_payload = {
            "age_num": wi_age, "gender": wi_gender, "race": wi_race,
            "time_in_hospital": sim_time_hosp, "num_medications": sim_num_meds,
            "num_lab_procedures": wi_num_lab, "num_procedures": wi_num_proc,
            "number_diagnoses": wi_num_diag, "num_outpatient": sim_num_out,
            "num_emergency": wi_num_emer, "num_inpatient": wi_num_inp,
            "a1c_result": sim_a1c, "glucose_serum": wi_glucose,
            "change_in_meds": sim_change, "diabetes_meds": wi_diabetes_med,
            "patient_label": "Scénario",
        }

        try:
            resp = requests.post(
                f"{API_URL}/predict/batch",
                json={"patients": [baseline_payload, scenario_payload]},
                headers=api_headers(), timeout=10,
            )
            handle_401(resp)
            if resp.status_code == 200:
                res_ref, res_sim = resp.json()
                s_ref = res_ref["risk_score"]; s_sim = res_sim["risk_score"]
                delta = (s_sim - s_ref) * 100

                st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:1rem 0;">', unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Risque de référence", f"{s_ref:.1%}")
                m2.metric("Risque du scénario", f"{s_sim:.1%}", delta=f"{delta:+.1f} pts", delta_color="inverse")
                direction = "réduit" if delta < 0 else "augmenté" if delta > 0 else "inchangé"
                m3.markdown(f'<div style="padding-top:0.6rem;color:{MUTED};font-size:0.85rem;">Le scénario a <b>{direction}</b> le risque.</div>', unsafe_allow_html=True)

                fig_wi = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=round(s_sim*100,1),
                    delta={"reference": round(s_ref*100,1),
                           "increasing":{"color":"#EF4444"}, "decreasing":{"color":"#22C55E"}},
                    number={"suffix":"%","font":{"size":34}},
                    gauge={"axis":{"range":[0,100],"tickcolor":BORDER}, "bar":{"color":"#F59E0B","thickness":0.22},
                           "bgcolor":CARD, "bordercolor":BORDER,
                           "threshold":{"line":{"color":"#1B4FD8","width":3}, "value":round(s_ref*100,1)}},
                ))
                fig_wi.update_layout(height=260, margin=dict(t=30,b=10,l=30,r=30), paper_bgcolor=PLOTBG, plot_bgcolor=PLOTBG)
                st.plotly_chart(fig_wi, use_container_width=True)
                st.caption("La ligne bleue indique le score de référence ; la jauge affiche le score du scénario avec le delta.")

                st.markdown('<div class="section-label">Facteurs qui ont le plus changé (SHAP)</div>', unsafe_allow_html=True)
                f_ref = res_ref["top_factors"]; f_sim = res_sim["top_factors"]
                all_feats = list(set(list(f_ref.keys()) + list(f_sim.keys())))
                v_ref = [f_ref.get(f,0) for f in all_feats]
                v_sim = [f_sim.get(f,0) for f in all_feats]
                fig_diff = go.Figure()
                fig_diff.add_trace(go.Bar(name="Référence", x=all_feats, y=v_ref, marker_color="#94A3B8"))
                fig_diff.add_trace(go.Bar(name="Scénario", x=all_feats, y=v_sim, marker_color="#F59E0B"))
                fig_diff.update_layout(barmode="group", height=280, margin=dict(t=10,b=20,l=10,r=10),
                                         paper_bgcolor=PLOTBG, plot_bgcolor=PLOTBG,
                                         font=dict(color=MUTED, family="DM Sans"),
                                         xaxis=dict(tickangle=-30, gridcolor=GRIDC, linecolor=BORDER),
                                         yaxis=dict(title="Impact SHAP", gridcolor=GRIDC, linecolor=BORDER),
                                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_diff, use_container_width=True)
            else:
                st.markdown(f'<div class="alert-err">❌ Erreur API {resp.status_code}</div>', unsafe_allow_html=True)
        except requests.exceptions.ConnectionError:
            st.markdown('<div class="alert-err">🔌 API ml-service inaccessible.</div>', unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════════
# ONGLET 6 — MODEL CARD
# ══════════════════════════════════════════════════════════════════
with tab_modelcard:
    st.markdown('<div class="section-label">Fiche modèle (Model Card)</div>', unsafe_allow_html=True)
    st.caption("Documentation synthétique du modèle actif : usage, données, performance, limites.")

    if not model_meta_info:
        st.markdown('<div class="alert-warn">⚠️ Informations du modèle indisponibles.</div>', unsafe_allow_html=True)
    else:
        m = model_meta_info
        n_total = m.get("n_fit",0) + m.get("n_calib",0) + m.get("n_test",0)

        # ── Usage prévu ─────────────────────────────────────────
        st.markdown(f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:1.3rem 1.5rem;margin-bottom:1rem;">'
                     f'<b style="color:{ACCENT};font-size:1rem;">🎯 Usage prévu</b>'
                     f'<p style="color:{TEXT};font-size:0.9rem;line-height:1.6;margin-top:0.6rem;margin-bottom:0;">'
                     f'Outil d\'aide à la décision clinique estimant le risque de réhospitalisation '
                     f'sous 30 jours pour un patient diabétique hospitalisé. Destiné à <b>compléter</b> le jugement '
                     f'clinique de l\'équipe soignante — priorisation du suivi post-sortie — et non à s\'y substituer. '
                     f'Ne doit pas être utilisé comme seul critère de décision médicale.</p></div>',
                     unsafe_allow_html=True)

        # ── Données d'entraînement ──────────────────────────────
        st.markdown(f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:1.3rem 1.5rem;margin-bottom:1rem;">'
                     f'<b style="color:{ACCENT};font-size:1rem;">📊 Données d\'entraînement</b></div>', unsafe_allow_html=True)
        d1,d2,d3,d4 = st.columns(4)
        d1.metric("Rencontres totales", f"{m.get('n_encounters_total', n_total):,}")
        d2.metric("Patients uniques", f"{m.get('n_patients_total','—'):,}" if m.get('n_patients_total') else "—")
        d3.metric("Features", m.get("feature_count","—"))
        d4.metric("Version", m.get("version","—"))
        st.markdown(f'<div style="color:{MUTED};font-size:0.82rem;margin-top:0.3rem;">'
                     f'Source : Diabetes 130-US Hospitals (1999–2008), 130 hôpitaux américains. '
                     f'Split : {m.get("split_method","train/test standard")}. '
                     f'Hash de traçabilité du dataset : <code>{m.get("data_hash","—")}</code>.</div>',
                     unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Performance (métriques honnêtes) ────────────────────
        st.markdown(f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:1.3rem 1.5rem;margin-bottom:1rem;">'
                     f'<b style="color:{ACCENT};font-size:1rem;">📈 Performance</b>'
                     f'<p style="color:{MUTED};font-size:0.82rem;margin-top:0.4rem;">'
                     f'Chiffres mesurés par validation croisée GroupKFold (5 plis), split groupé par patient — '
                     f'aucune fuite de données entre les patients d\'entraînement et de test.</p></div>',
                     unsafe_allow_html=True)
        p1,p2,p3,p4 = st.columns(4)
        p1.metric("AUC-ROC", m.get("groupkfold5_auc_mean", m.get("auc_roc","—")),
                   f"± {m.get('groupkfold5_auc_std','—')}" if m.get("groupkfold5_auc_std") else None)
        p2.metric("AUC-PR", m.get("groupkfold5_ap_mean", m.get("auc_pr","—")),
                   f"± {m.get('groupkfold5_ap_std','—')}" if m.get("groupkfold5_ap_std") else None)
        p3.metric("F1-Score", m.get("f1_score","—"))
        p4.metric("Recall top 10% risque", f"{m.get('top10_capture_rate','—')}")
        st.markdown(f'<div style="color:{MUTED};font-size:0.82rem;margin-top:0.3rem;">'
                     f'Le recall top 10% signifie : parmi les 10% de patients jugés les plus à risque par le modèle, '
                     f'{float(m.get("top10_capture_rate",0))*100:.0f}% des vraies réadmissions sous 30 jours sont capturées — '
                     f'soit environ {float(m.get("top10_capture_rate",0))/0.10:.1f}× mieux qu\'un tri aléatoire.</div>',
                     unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # ── Analyse de seuil interactive ─────────────────────────
        if m.get("threshold_analysis"):
            st.markdown(f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:1.3rem 1.5rem;margin-bottom:1rem;">'
                         f'<b style="color:{ACCENT};font-size:1rem;">⚖️ Analyse de seuil</b>'
                         f'<p style="color:{MUTED};font-size:0.82rem;margin-top:0.4rem;">'
                         f'Compromis precision / recall selon le seuil choisi, mesuré sur le jeu de test '
                         f'(jamais utilisé pour entraîner le modèle).</p></div>', unsafe_allow_html=True)

            df_thr = pd.DataFrame(m["threshold_analysis"])

            fig_thr = go.Figure()
            fig_thr.add_trace(go.Scatter(x=df_thr["threshold"], y=df_thr["precision"],
                mode="lines+markers", name="Precision", line=dict(color="#1B4FD8", width=2), marker=dict(size=5)))
            fig_thr.add_trace(go.Scatter(x=df_thr["threshold"], y=df_thr["recall"],
                mode="lines+markers", name="Recall", line=dict(color="#F59E0B", width=2), marker=dict(size=5)))
            fig_thr.add_vline(x=m.get("best_threshold", 0.146), line_dash="dash", line_color="#EF4444",
                annotation_text=f"Seuil actif ({m.get('best_threshold',0):.3f})", annotation_font_size=10,
                annotation_font_color="#EF4444")
            fig_thr.update_layout(height=300, margin=dict(t=20,b=20,l=10,r=10),
                paper_bgcolor=PLOTBG, plot_bgcolor=PLOTBG, font=dict(color=MUTED, family="DM Sans"),
                xaxis=dict(title="Seuil de décision", gridcolor=GRIDC, linecolor=BORDER),
                yaxis=dict(title="Score", range=[0,1], gridcolor=GRIDC, linecolor=BORDER),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_thr, use_container_width=True)

            thr_pick = st.select_slider(
                "Explorer un seuil précis",
                options=df_thr["threshold"].tolist(),
                value=min(df_thr["threshold"], key=lambda x: abs(x - m.get("best_threshold", 0.146))),
                key="thr_explore",
            )
            row = df_thr[df_thr["threshold"] == thr_pick].iloc[0]
            t1,t2,t3 = st.columns(3)
            t1.metric("Precision", f"{row['precision']:.1%}")
            t2.metric("Recall", f"{row['recall']:.1%}")
            t3.metric("Patients alertés", f"{row['n_alerted']:,} ({row['pct_alerted']:.1%})")
            st.caption(f"À ce seuil : sur 100 alertes déclenchées, {row['precision']*100:.0f} correspondent à une "
                       f"vraie réadmission ; sur 100 vraies réadmissions, {row['recall']*100:.0f} sont détectées ; "
                       f"{row['pct_alerted']*100:.1f}% des patients du jeu de test seraient signalés à l'équipe.")

            st.markdown("<br>", unsafe_allow_html=True)
        # ── Limites ──────────────────────────────────────────────
        st.markdown('<div class="alert-warn">'
                     '⚠️ <b>Limites connues</b><br>'
                     '<span style="font-size:0.85rem;line-height:1.6;">'
                     '• Performance modeste (AUC ≈ 0,64) — cohérente avec ce que ce type de données permet honnêtement, '
                     'mais insuffisante pour un usage diagnostique autonome.<br>'
                     '• Environ 79% des vraies réadmissions ne figurent pas dans le top 10% des patients jugés à risque.<br>'
                     '• Données historiques (États-Unis, 1999–2008) : la généralisation à un autre contexte de soins, '
                     'un autre pays ou une période plus récente n\'est pas garantie.<br>'
                     '• Le modèle ne capture pas les facteurs sociaux, psychologiques ou d\'accès aux soins, '
                     'connus pour influencer la réhospitalisation.'
                     '</span></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Précautions éthiques ─────────────────────────────────
        st.markdown('<div class="alert-info">'
                     'ℹ️ <b>Précautions d\'usage et considérations éthiques</b><br>'
                     '<span style="font-size:0.85rem;line-height:1.6;">'
                     '• Le modèle inclut des variables démographiques (âge, genre, origine) : une vigilance est requise '
                     'pour s\'assurer que la performance reste comparable entre sous-groupes, et non discriminante.<br>'
                     '• Une prédiction de risque élevé ne doit jamais, seule, restreindre l\'accès d\'un patient à des soins '
                     'ou ressources — elle doit au contraire déclencher une attention accrue.<br>'
                     '• Toute prédiction est tracée (journal d\'audit) et attribuable au praticien qui l\'a demandée.<br>'
                     '• Les probabilités affichées sont calibrées (méthode isotonic) : un score de 20% signifie '
                     'qu\'environ 1 patient sur 5 avec ce score sera réellement réhospitalisé.'
                     '</span></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div style="color:{MUTED};font-size:0.75rem;text-align:right;">'
                     f'Fiche générée dynamiquement depuis model_meta.json — dernière version : {m.get("version","—")}</div>',
                     unsafe_allow_html=True)