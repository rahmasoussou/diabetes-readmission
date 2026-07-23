"""
Dashboard ClinAI – Design clinique, version finale (impression retirée, style affiné)
"""
import os, requests, streamlit as st, pandas as pd, plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

API_URL = f"http://{os.environ.get('API_HOST','ml-service')}:{os.environ.get('API_PORT','8000')}"

st.set_page_config(page_title="ClinAI · Réhospitalisation", page_icon="⚕️", layout="wide", initial_sidebar_state="expanded")

# ── Session state ─────────────────────────────────────────────────
if "token"     not in st.session_state: st.session_state.token     = None
if "username"  not in st.session_state: st.session_state.username  = None
if "dark_mode" not in st.session_state: st.session_state.dark_mode = False
if "hist_offset" not in st.session_state: st.session_state.hist_offset = 0

D = st.session_state.dark_mode


# PALETTE CLINIQUE (teal profond, fond sauge, gris doux)

if D:
    BG       = "#1a1c1e"; SIDEBAR  = "#222527"; CARD    = "#222527"
    TEXT     = "#e4e6e8"; MUTED    = "#9aa0a6"; BORDER  = "#2f3336"
    ACCENT   = "#0B7B7B"; ACCENT2  = "#0D9488"; INPUT   = "#1e2022"
    IBORDER  = "#383c40"; PLOTBG   = "rgba(0,0,0,0)"; GRIDC = "#2a2d2f"
    RISK_HIGH_BG = "#2a1e1e"; RISK_HIGH_BORDER = "#5c3a3a"; RISK_HIGH_TXT = "#e9b3b3"
    RISK_MED_BG  = "#2a2618"; RISK_MED_BORDER  = "#5c4a2e"; RISK_MED_TXT  = "#e9d3a0"
    RISK_LOW_BG  = "#1a2a20"; RISK_LOW_BORDER  = "#2e5c3a"; RISK_LOW_TXT  = "#a3d9a5"
else:
    BG       = "#F2F5F1"; SIDEBAR  = "#FFFFFF"; CARD    = "#FFFFFF"
    TEXT     = "#1c2a28"; MUTED    = "#5c6a66"; BORDER  = "#d4dad5"
    ACCENT   = "#0B7B7B"; ACCENT2  = "#0D9488"; INPUT   = "#FFFFFF"
    IBORDER  = "#c2cdc6"; PLOTBG   = "rgba(0,0,0,0)"; GRIDC = "#e6ece7"
    RISK_HIGH_BG = "#fdf2f2"; RISK_HIGH_BORDER = "#f5c6c6"; RISK_HIGH_TXT = "#a13b3b"
    RISK_MED_BG  = "#fef9ee"; RISK_MED_BORDER  = "#f5d68a"; RISK_MED_TXT  = "#8a6d20"
    RISK_LOW_BG  = "#f2faf2"; RISK_LOW_BORDER  = "#a3d9a5"; RISK_LOW_TXT  = "#2a6b34"

PLOTLY_THEME = dict(
    paper_bgcolor=PLOTBG, plot_bgcolor=PLOTBG,
    font=dict(color=MUTED, family="IBM Plex Sans"),
    xaxis=dict(gridcolor=GRIDC, linecolor=BORDER, tickfont=dict(size=11)),
    yaxis=dict(gridcolor=GRIDC, linecolor=BORDER, tickfont=dict(size=11)),
)


# CSS – Typo Fraunces / IBM Plex, finitions cliniques

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: {TEXT};
}}
.stApp {{ background: {BG}; }}
.block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}
#MainMenu, footer, header {{ visibility: hidden; }}
section[data-testid="stSidebar"] {{
    background: {SIDEBAR} !important;
    border-right: 1px solid {BORDER} !important;
    box-shadow: 2px 0 12px rgba(0,0,0,0.03);
}}
.stTabs [data-baseweb="tab-list"] {{
    background: {CARD}; border-radius: 12px; padding: 5px; gap: 4px;
    border: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent; color: {MUTED}; border-radius: 8px;
    font-weight: 500; font-size: 0.9rem; padding: 0.55rem 1.3rem; border: none !important;
    transition: all 0.15s;
}}
.stTabs [aria-selected="true"] {{
    background: {ACCENT} !important; color: #fff !important;
    box-shadow: 0 2px 8px rgba(11,123,123,0.3);
}}
.stSelectbox>div>div, .stNumberInput>div>div>input,
.stTextInput>div>div>input, .stPasswordInput>div>div>input {{
    background: {INPUT} !important; border: 1.5px solid {IBORDER} !important;
    border-radius: 8px !important; color: {TEXT} !important;
    font-family: 'IBM Plex Sans', sans-serif !important; font-size: 0.9rem !important;
}}
.stButton>button {{
    background: {ACCENT} !important; color: #fff !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.92rem !important; padding: 0.65rem 1.4rem !important;
    box-shadow: 0 2px 8px rgba(11,123,123,0.25) !important;
    transition: all 0.2s !important;
}}
.stButton>button:hover {{
    background: {ACCENT2} !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(11,123,123,0.35) !important;
}}
.stDownloadButton>button {{
    background: {BG} !important; color: {MUTED} !important;
    border: 1.5px solid {BORDER} !important; border-radius: 8px !important;
    font-weight: 500 !important; box-shadow: none !important;
    transition: all 0.15s;
}}
.stDownloadButton>button:hover {{ border-color: {ACCENT} !important; color: {ACCENT} !important; }}
[data-testid="stMetricLabel"] {{
    color: {MUTED} !important; font-size: 0.78rem !important;
    font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.05em;
}}
[data-testid="stMetricValue"] {{
    color: {ACCENT} !important; font-family: 'Fraunces', serif !important;
    font-size: 2rem !important; font-weight: 500 !important;
}}
.section-label {{
    color: {MUTED}; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    margin-bottom: 0.75rem; margin-top: 0.25rem;
}}
.card {{
    background: {CARD}; border: 1px solid {BORDER};
    border-radius: 12px; padding: 1.4rem 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s, transform 0.15s;
    margin-bottom: 1rem;
}}
.card:hover {{
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    transform: translateY(-1px);
}}
.hero-clinical {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-left: 6px solid {ACCENT};
    border-radius: 0 12px 12px 0;
    padding: 1.4rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}}
/* Plage de référence */
.risk-bar-container {{
    background: {CARD}; border: 1px solid {BORDER};
    border-radius: 10px; padding: 1.2rem 1.4rem; margin: 0.8rem 0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.03);
}}
.risk-bar {{ position: relative; height: 14px; background: #e9ecef;
    border-radius: 7px; margin: 0.5rem 0 0.3rem; overflow: visible;
}}
.risk-bar-fill-low {{ position: absolute; left:0; top:0; height:100%;
    background: #86c89a; border-radius: 7px 0 0 7px;
}}
.risk-bar-fill-med {{ position: absolute; top:0; height:100%;
    background: #e3c46f;
}}
.risk-bar-fill-high {{ position: absolute; top:0; height:100%;
    background: #e8948a; border-radius: 0 7px 7px 0;
}}
.risk-bar-marker {{
    position: absolute; top: -7px; width: 4px; height: 28px;
    background: {ACCENT}; border-radius: 2px; box-shadow: 0 0 6px rgba(0,0,0,0.15);
    transform: translateX(-2px); z-index: 2;
}}
.risk-bar-marker.secondary {{
    background: #6c757d; width: 3px; height: 22px; top: -4px;
    box-shadow: 0 0 4px rgba(0,0,0,0.1);
}}
.risk-legend {{
    display: flex; justify-content: space-between;
    font-size: 0.75rem; color: {MUTED}; font-family: 'IBM Plex Mono', monospace;
}}
.risk-level-badge {{
    font-weight: 600; font-size: 0.9rem;
    padding: 0.4rem 1rem; border-radius: 6px;
    display: inline-block;
}}
.high {{ background: {RISK_HIGH_BG}; border: 1px solid {RISK_HIGH_BORDER}; color: {RISK_HIGH_TXT}; }}
.medium {{ background: {RISK_MED_BG}; border: 1px solid {RISK_MED_BORDER}; color: {RISK_MED_TXT}; }}
.low {{ background: {RISK_LOW_BG}; border: 1px solid {RISK_LOW_BORDER}; color: {RISK_LOW_TXT}; }}
.alert-box {{
    border-radius: 8px; padding: 0.9rem 1.2rem; font-size: 0.85rem;
    margin: 0.6rem 0;
}}
.alert-high {{ background: {RISK_HIGH_BG}; border: 1px solid {RISK_HIGH_BORDER}; color: {RISK_HIGH_TXT}; }}
.alert-med {{ background: {RISK_MED_BG}; border: 1px solid {RISK_MED_BORDER}; color: {RISK_MED_TXT}; }}
.alert-low {{ background: {RISK_LOW_BG}; border: 1px solid {RISK_LOW_BORDER}; color: {RISK_LOW_TXT}; }}
</style>
""", unsafe_allow_html=True)

ECG_SVG = """
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"
     style="vertical-align:middle;margin-right:6px;">
  <path d="M3 12H6L9 3L15 21L18 12H21" stroke="currentColor" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

def api_headers(): return {"Authorization": f"Bearer {st.session_state.token}"}
def handle_401(r):
    if r.status_code == 401:
        st.session_state.token = None; st.rerun()

def authenticate(u, p):
    try:
        r = requests.post(f"{API_URL}/token", json={"username":u,"password":p}, timeout=5)
        return r.json().get("access_token") if r.status_code==200 else None
    except: return None

def render_risk_bar(score, seuil_m, seuil_e, secondary=None):
    low_pct = seuil_m*100; med_pct = (seuil_e-seuil_m)*100; high_start = seuil_e*100
    marker_left = score*100
    sec_marker = ""
    if secondary is not None:
        sec_left = secondary*100
        sec_marker = f'<div class="risk-bar-marker secondary" style="left:{sec_left}%;"></div>'
    return f"""
    <div class="risk-bar-container">
      <div style="display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:0.8rem;">
        <span style="color:{MUTED};">Faible</span><span style="color:{MUTED};">Modéré</span><span style="color:{MUTED};">Élevé</span>
      </div>
      <div class="risk-bar">
        <div class="risk-bar-fill-low" style="width:{low_pct}%;"></div>
        <div class="risk-bar-fill-med" style="left:{low_pct}%;width:{med_pct}%;"></div>
        <div class="risk-bar-fill-high" style="left:{high_start}%;width:{100-high_start}%;"></div>
        <div class="risk-bar-marker" style="left:{marker_left}%;"></div>
        {sec_marker}
      </div>
      <div class="risk-legend">
        <span>0%</span><span>{seuil_m*100:.0f}%</span><span>{seuil_e*100:.0f}%</span><span>100%</span>
      </div>
    </div>"""


# LOGIN

if not st.session_state.token:
    col = st.columns([1,1.2,1])[1]
    with col:
        st.markdown(f"""
        <div class="card" style="text-align:center;margin-top:8vh;">
          <div style="font-size:2rem;">{ECG_SVG.replace('currentColor', ACCENT)}</div>
          <div style="font-family:'Fraunces',serif;font-size:2rem;color:{ACCENT};">ClinAI</div>
          <div style="color:{MUTED};font-size:0.85rem;">Système de prédiction de réhospitalisation</div>
        </div>""", unsafe_allow_html=True)
        with st.form("login"):
            st.markdown(f'<p style="color:{MUTED};font-size:0.85rem;font-weight:500;">Identifiant praticien</p>', unsafe_allow_html=True)
            username = st.text_input("", placeholder="medecin", label_visibility="collapsed")
            st.markdown(f'<p style="color:{MUTED};font-size:0.85rem;font-weight:500;margin-top:0.6rem;">Mot de passe</p>', unsafe_allow_html=True)
            password = st.text_input("", type="password", label_visibility="collapsed")
            if st.form_submit_button("Accéder au tableau de bord", use_container_width=True):
                t = authenticate(username, password)
                if t:
                    st.session_state.token = t; st.session_state.username = username; st.rerun()
                else:
                    st.markdown('<div class="alert-box alert-high">Identifiants incorrects ou service indisponible.</div>', unsafe_allow_html=True)
    st.stop()


# SIDEBAR

with st.sidebar:
    st.markdown(f"""
    <div style="margin-bottom:1.2rem;display:flex;align-items:center;gap:0.7rem;">
      <span>{ECG_SVG.replace('currentColor', ACCENT)}</span>
      <div><div style="font-family:'Fraunces',serif;font-size:1.4rem;color:{ACCENT};">ClinAI</div>
      <div style="color:{MUTED};font-size:0.75rem;">Réhospitalisation · Diabète</div></div>
    </div>""", unsafe_allow_html=True)

    mode_label = "☀️ Mode clair" if D else "🌙 Mode sombre"
    if st.button(mode_label, use_container_width=True, key="toggle_dark"):
        st.session_state.dark_mode = not D; st.rerun()

    st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:1rem 0;">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card" style="padding:0.75rem 1rem;margin-bottom:1.2rem;">
      <div style="color:{MUTED};font-size:0.78rem;">Connecté</div>
      <div style="color:{ACCENT};font-weight:700;">{st.session_state.username}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Modèle actif</div>', unsafe_allow_html=True)
    model_meta_info = {}
    try:
        r = requests.get(f"{API_URL}/model/info", headers=api_headers(), timeout=3)
        if r.status_code == 200:
            m = r.json(); model_meta_info = m
            c1,c2 = st.columns(2)
            c1.metric("AUC-ROC", m.get("auc_roc","—")); c2.metric("AUC-PR", m.get("auc_pr","—"))
            if m.get("f1_score"): st.metric("F1-Score", m.get("f1_score","—"))
            n_total = m.get("n_fit",0)+m.get("n_calib",0)+m.get("n_test",0)
            st.markdown(f'<div style="color:{MUTED};font-size:0.75rem;font-family:IBM Plex Mono,monospace;margin-top:0.4rem;">{m.get("feature_count","?")} features · {n_total:,} patients</div>', unsafe_allow_html=True)
    except: pass

    st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:1rem 0;">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Seuils de risque</div>', unsafe_allow_html=True)
    _best_thr = model_meta_info.get("best_threshold")
    if _best_thr is not None:
        default_eleve_pct = max(round(_best_thr*100),5); default_modere_pct = max(round(_best_thr*100*0.6),2)
    else:
        default_eleve_pct, default_modere_pct = 50,30
    seuil_eleve = st.slider("Seuil risque élevé (%)", 5,80,default_eleve_pct,1,key="seuil_eleve")
    seuil_modere = st.slider("Seuil risque modéré (%)",1,max(seuil_eleve-1,2),min(default_modere_pct,seuil_eleve-1),1,key="seuil_modere")
    if _best_thr is not None: st.caption(f"Seuil optimisé (max F1) : {_best_thr*100:.1f}%")
    st.markdown(f"""<div class="card" style="padding:0.6rem 0.8rem;font-size:0.78rem;color:{MUTED};margin-top:0.3rem;">
      Faible : 0-{seuil_modere}% | Modéré : {seuil_modere}-{seuil_eleve}% | Élevé : {seuil_eleve}-100%</div>""", unsafe_allow_html=True)

    if st.button("Déconnexion", use_container_width=True): st.session_state.token = None; st.rerun()


# HERO

st.markdown(f"""
<div class="hero-clinical">
  <div style="display:flex;align-items:center;gap:1rem;">
    <span style="font-size:2rem;">{ECG_SVG.replace('currentColor', ACCENT)}</span>
    <div>
      <div style="font-family:'Fraunces',serif;font-size:1.6rem;font-weight:500;color:{ACCENT};">Prédiction de Réhospitalisation</div>
      <div style="color:{MUTED};font-size:0.85rem;">Score de risque à 30 jours · XGBoost + SHAP · 130 hôpitaux US · 1999–2008</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


# ONGLETS

tab_predict, tab_history, tab_stats, tab_compare, tab_whatif, tab_modelcard = st.tabs([
    " Nouvelle prédiction"," Historique"," Statistiques"," Comparaison"," What‑if"," Model Card"
])

# ONGLET 1 — PRÉDICTION (sans bouton impression)

with tab_predict:
    col_form, col_result = st.columns([1.05,1], gap="large")
    with col_form:
        st.markdown('<div class="section-label">Données cliniques du patient</div>', unsafe_allow_html=True)
        with st.expander("Profil démographique", expanded=True):
            age = st.slider("Âge (années)", 0, 100, 65)
            g1,g2 = st.columns(2)
            gender = g1.selectbox("Genre", ["Female","Male","Unknown"])
            race   = g2.selectbox("Origine", ["Caucasian","AfricanAmerican","Hispanic","Asian","Other","Unknown"])
        with st.expander("Séjour hospitalier", expanded=True):
            h1,h2 = st.columns(2)
            time_hosp = h1.number_input("Durée (jours)", 1,30,5)
            num_meds  = h2.number_input("Médicaments", 0,100,15)
            h3,h4 = st.columns(2)
            num_lab  = h3.number_input("Procédures labo", 0,130,40)
            num_proc = h4.number_input("Autres procédures", 0,10,1)
            num_diag = st.number_input("Nombre de diagnostics", 1,16,7)
        with st.expander("Historique 12 mois", expanded=True):
            p1,p2,p3 = st.columns(3)
            num_out  = p1.number_input("Ambulatoire", 0,50,0)
            num_emer = p2.number_input("Urgences", 0,50,0)
            num_inp  = p3.number_input("Hospitalisé", 0,20,1)
        with st.expander("Résultats cliniques", expanded=True):
            r1,r2 = st.columns(2)
            a1c     = r1.selectbox("HbA1c", ["None","Norm",">7",">8"])
            glucose = r2.selectbox("Glucose sérique", ["None","Norm",">200",">300"])
            r3,r4 = st.columns(2)
            change_meds  = r3.selectbox("Changement médication", ["No","Ch"])
            diabetes_med = r4.selectbox("Anti-diabétiques", ["Yes","No"])
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Calculer le score de risque", use_container_width=True)

    with col_result:
        st.markdown('<div class="section-label">Résultat d\'analyse</div>', unsafe_allow_html=True)
        if predict_btn:
            payload = {
                "age_num":age,"gender":gender,"race":race,"time_in_hospital":time_hosp,
                "num_medications":num_meds,"num_lab_procedures":num_lab,
                "num_procedures":num_proc,"number_diagnoses":num_diag,
                "num_outpatient":num_out,"num_emergency":num_emer,"num_inpatient":num_inp,
                "a1c_result":a1c,"glucose_serum":glucose,"change_in_meds":change_meds,
                "diabetes_meds":diabetes_med,
            }
            with st.spinner("Analyse en cours..."):
                try:
                    resp = requests.post(f"{API_URL}/predict", json=payload, headers=api_headers(), timeout=10)
                    handle_401(resp)
                    if resp.status_code == 200:
                        result = resp.json(); score = result["risk_score"]
                        seuil_e = st.session_state.get("seuil_eleve",50)/100
                        seuil_m = st.session_state.get("seuil_modere",30)/100
                        level = "ÉLEVÉ" if score>=seuil_e else ("MODÉRÉ" if score>=seuil_m else "FAIBLE")

                        st.markdown(f"""
                        <div class="card" style="border-left: 4px solid {ACCENT};">
                          <div style="display:flex;justify-content:space-between;align-items:center;">
                            <div>
                              <div style="font-family:'Fraunces',serif;font-size:1.2rem;color:{ACCENT};">Risque de réhospitalisation à 30 jours</div>
                              <div style="color:{MUTED};font-size:0.8rem;">Modèle ClinAI v{model_meta_info.get('version','?')} · Calibration isotonic</div>
                            </div>
                            <div style="text-align:right;">
                              <div style="font-family:'Fraunces',serif;font-size:2.4rem;color:{ACCENT};">{score:.1%}</div>
                              <span class="risk-level-badge { {'ÉLEVÉ':'high','MODÉRÉ':'medium','FAIBLE':'low'}[level] }">{level}</span>
                            </div>
                          </div>
                        </div>""", unsafe_allow_html=True)
                        st.markdown(render_risk_bar(score, seuil_m, seuil_e), unsafe_allow_html=True)
                        st.markdown(f"""<div style="color:{MUTED};font-size:0.8rem;margin-top:0.2rem;">
                          Seuils cliniques : faible &lt; {seuil_m*100:.0f}% · modéré {seuil_m*100:.0f}–{seuil_e*100:.0f}% · élevé &gt; {seuil_e*100:.0f}%
                        </div>""", unsafe_allow_html=True)

                        if level == "ÉLEVÉ":
                            st.markdown('<div class="alert-box alert-high">⚠ Suivi renforcé recommandé · Consultation de sortie dédiée · Rappel à 48h</div>', unsafe_allow_html=True)
                        elif level == "MODÉRÉ":
                            st.markdown('<div class="alert-box alert-med">ℹ Surveillance modérée · Appel de suivi à J+7 · Vérifier observance</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="alert-box alert-low">✓ Protocole de sortie standard · Aucune mesure supplémentaire requise</div>', unsafe_allow_html=True)

                        st.markdown('<div class="section-label" style="margin-top:1.2rem;">Facteurs déterminants (SHAP)</div>', unsafe_allow_html=True)
                        factors_df = pd.DataFrame(list(result["top_factors"].items()),columns=["Facteur","Impact"]).sort_values("Impact",key=abs,ascending=True)
                        fig2 = go.Figure(go.Bar(
                            x=factors_df["Impact"], y=factors_df["Facteur"], orientation="h",
                            marker_color=["#e8948a" if v>0 else "#86c89a" for v in factors_df["Impact"]],
                            text=[f"{v:+.4f}" for v in factors_df["Impact"]], textposition="outside",
                            textfont=dict(size=11,color=MUTED,family="IBM Plex Mono")))
                        fig2.update_layout(height=250, margin=dict(t=5,b=5,l=5,r=80),
                                           xaxis_title="Impact", xaxis_zeroline=True, xaxis_zerolinecolor=BORDER,
                                           **PLOTLY_THEME)
                        st.plotly_chart(fig2, use_container_width=True)

                        # Bouton PDF uniquement
                        try:
                            pdf_resp = requests.post(f"{API_URL}/predict/pdf", json=payload, headers=api_headers(), timeout=15)
                            if pdf_resp.status_code == 200:
                                st.download_button("⬇ Télécharger le rapport PDF", data=pdf_resp.content,
                                                   file_name=f"rapport_ClinAI_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                                   mime="application/pdf", use_container_width=True)
                            else:
                                st.markdown('<div class="alert-box alert-med">Rapport PDF indisponible.</div>', unsafe_allow_html=True)
                        except:
                            st.markdown('<div class="alert-box alert-med">Rapport PDF indisponible.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="alert-box alert-high">Erreur API {resp.status_code}</div>', unsafe_allow_html=True)
                except requests.exceptions.ConnectionError:
                    st.markdown('<div class="alert-box alert-high">API ml-service inaccessible.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="card" style="text-align:center;padding:3rem 2rem;">
              <div style="font-size:2rem;margin-bottom:1rem;">⚕️</div>
              <div style="color:{MUTED};">Remplis les données patient à gauche<br>puis clique sur <b>Calculer le score de risque</b></div>
            </div>""", unsafe_allow_html=True)


# ONGLET 2 — HISTORIQUE

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
                st.markdown('<div class="alert-box alert-med">Aucune prédiction enregistrée.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="color:{MUTED};font-size:0.82rem;">{total:,} prédiction(s)</div>', unsafe_allow_html=True)
                df_h = pd.DataFrame(items)
                df_h["predicted_at"] = pd.to_datetime(df_h["predicted_at"]).dt.strftime("%d/%m/%Y %H:%M")
                df_h["risk_score"]   = (df_h["risk_score"]*100).round(1).astype(str)+"%"
                df_h["risk_level"]   = df_h["risk_level"].map({"ÉLEVÉ":"■ ÉLEVÉ","MODÉRÉ":"■ MODÉRÉ","FAIBLE":"■ FAIBLE"})
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
            st.markdown(f'<div class="alert-box alert-high">Erreur {r.status_code}</div>', unsafe_allow_html=True)
    except requests.exceptions.ConnectionError:
        st.markdown('<div class="alert-box alert-high">API ml-service inaccessible.</div>', unsafe_allow_html=True)


# ONGLET 3 — STATISTIQUES

with tab_stats:
    st.markdown('<div class="section-label">Tableau de bord statistique</div>', unsafe_allow_html=True)
    col_r,_ = st.columns([1,5]); col_r.button("↺ Actualiser",key="rs")
    try:
        r = requests.get(f"{API_URL}/stats",headers=api_headers(),timeout=5)
        handle_401(r)
        if r.status_code==200:
            s=r.json()
            if s["total"]==0:
                st.markdown('<div class="alert-box alert-med">Effectuez des prédictions pour voir les statistiques.</div>', unsafe_allow_html=True)
            else:
                k1,k2,k3,k4,k5=st.columns(5)
                k1.metric("Total",f"{s['total']:,}")
                k2.metric("■ Élevé", str(s['eleve']),  f"{s['eleve']/s['total']*100:.1f}%")
                k3.metric("■ Modéré",str(s['modere']), f"{s['modere']/s['total']*100:.1f}%")
                k4.metric("■ Faible",str(s['faible']), f"{s['faible']/s['total']*100:.1f}%")
                k5.metric("Score moyen",f"{s['score_moyen']*100:.1f}%")
                st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:1rem 0;">', unsafe_allow_html=True)
                cl,cr=st.columns(2)
                with cl:
                    st.markdown('<div class="section-label">Répartition des niveaux</div>', unsafe_allow_html=True)
                    fig_d=go.Figure(go.Pie(labels=["Élevé","Modéré","Faible"],values=[s["eleve"],s["modere"],s["faible"]],
                        hole=0.62,marker_colors=["#e8948a","#e3c46f","#86c89a"],textinfo="label+percent",
                        hoverinfo="label+value",textfont=dict(family="IBM Plex Sans",size=12)))
                    fig_d.update_layout(height=280,margin=dict(t=10,b=10,l=10,r=10),showlegend=False,**PLOTLY_THEME)
                    st.plotly_chart(fig_d,use_container_width=True)
                with cr:
                    st.markdown('<div class="section-label">Facteurs les plus influents</div>', unsafe_allow_html=True)
                    if s["top_factors_global"]:
                        df_f=pd.DataFrame(s["top_factors_global"])
                        fig_f=go.Figure(go.Bar(x=df_f["mean_abs_shap"],y=df_f["feature"],orientation="h",
                            marker_color=ACCENT,marker_line_width=0,opacity=0.85,
                            text=df_f["mean_abs_shap"].round(4),textposition="outside",
                            textfont=dict(size=10,color=MUTED,family="IBM Plex Mono")))
                        fig_f.update_layout(height=280,margin=dict(t=5,b=5,l=5,r=70),
                            paper_bgcolor=PLOTBG,plot_bgcolor=PLOTBG,
                            font=dict(color=MUTED,family="IBM Plex Sans"),
                            xaxis=dict(title="|SHAP| moyen",gridcolor=GRIDC,linecolor=BORDER),
                            yaxis=dict(categoryorder="total ascending",gridcolor=GRIDC,linecolor=BORDER))
                        st.plotly_chart(fig_f,use_container_width=True)

                if s["top_factors_global"]:
                    st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:0.5rem 0 1rem;">', unsafe_allow_html=True)
                    st.markdown('<div class="section-label">Analyse SHAP globale</div>', unsafe_allow_html=True)
                    df_shap = pd.DataFrame(s["top_factors_global"]).sort_values("mean_abs_shap", ascending=True)
                    df_shap["feature_clean"] = df_shap["feature"].str.replace("_", " ").str.title()
                    max_val = df_shap["mean_abs_shap"].max()
                    colors_shap = [f"rgba(11,123,123,{0.3 + 0.7*(v/max_val):.2f})" for v in df_shap["mean_abs_shap"]]
                    fig_shap = go.Figure(go.Bar(
                        x=df_shap["mean_abs_shap"], y=df_shap["feature_clean"], orientation="h",
                        marker_color=colors_shap, text=[f"{v:.4f}" for v in df_shap["mean_abs_shap"]],
                        textposition="outside", textfont=dict(size=10,color=MUTED,family="IBM Plex Mono")))
                    median_val = float(df_shap["mean_abs_shap"].median())
                    fig_shap.add_vline(x=median_val, line_dash="dash", line_color=MUTED,
                                       annotation_text=f"Médiane ({median_val:.4f})", annotation_font_size=9)
                    fig_shap.update_layout(height=max(350, len(df_shap)*30), margin=dict(t=30,b=20,l=10,r=80),
                                           **PLOTLY_THEME)
                    st.plotly_chart(fig_shap, use_container_width=True)
                    with st.expander("📋 Tableau détaillé"):
                        st.dataframe(df_shap[["feature_clean","mean_abs_shap"]].rename(columns={"feature_clean":"Feature","mean_abs_shap":"|SHAP| moyen"}),
                                     use_container_width=True, hide_index=True)

                if s["trend"]:
                    st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:0.5rem 0 1rem;">', unsafe_allow_html=True)
                    st.markdown('<div class="section-label">Activité — 30 derniers jours</div>', unsafe_allow_html=True)
                    df_t=pd.DataFrame(s["trend"]); df_t["jour"]=pd.to_datetime(df_t["jour"])
                    fig_t=go.Figure()
                    fig_t.add_trace(go.Bar(x=df_t["jour"],y=df_t["nb"],name="Total",marker_color="#86c89a",opacity=0.6))
                    fig_t.add_trace(go.Bar(x=df_t["jour"],y=df_t["nb_eleve"],name="Risque élevé",marker_color="#e8948a"))
                    fig_t.add_trace(go.Scatter(x=df_t["jour"],y=df_t["score_moyen"]*100,name="Score moyen (%)",
                        yaxis="y2", mode="lines+markers",line=dict(color="#e3c46f",width=2), marker=dict(size=5)))
                    merged_theme = PLOTLY_THEME.copy()
                    merged_theme["yaxis"] = {**PLOTLY_THEME.get("yaxis", {}), "title": "Nombre de prédictions", "gridcolor": GRIDC, "linecolor": BORDER}
                    merged_theme["yaxis2"] = dict(title="Score moyen (%)", overlaying="y", side="right", range=[0,100],
                                                   gridcolor="rgba(0,0,0,0)", linecolor=BORDER)
                    fig_t.update_layout(height=320,barmode="overlay",
                        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(size=10)),
                        margin=dict(t=20, b=20, l=10, r=70), **merged_theme)
                    st.plotly_chart(fig_t,use_container_width=True)
    except requests.exceptions.ConnectionError:
        st.markdown('<div class="alert-box alert-high">API ml-service inaccessible.</div>', unsafe_allow_html=True)


# ONGLET 4 — COMPARAISON

with tab_compare:
    st.markdown('<div class="section-label">Comparaison de deux profils patients</div>', unsafe_allow_html=True)
    col_p1, col_sep, col_p2 = st.columns([1, 0.05, 1], gap="small")
    def patient_form(col, label, key_prefix):
        with col:
            st.markdown(f"""<div class="card" style="padding:1rem 1.2rem;margin-bottom:1rem;"><b style="color:{ACCENT};">{label}</b></div>""", unsafe_allow_html=True)
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
                "age_num":age,"gender":gender,"race":race,"time_in_hospital":time_h,"num_medications":n_meds,
                "num_lab_procedures":n_lab,"num_procedures":n_proc,"number_diagnoses":n_diag,
                "num_outpatient":n_out,"num_emergency":n_emer,"num_inpatient":n_inp,
                "a1c_result":a1c,"glucose_serum":glucose,"change_in_meds":change,"diabetes_meds":diab,
            }
    p1_data = patient_form(col_p1, "Patient A", "p1")
    with col_sep: st.markdown(f'<div style="border-left:2px dashed {BORDER};height:100%;margin:0 auto;width:1px;"></div>', unsafe_allow_html=True)
    p2_data = patient_form(col_p2, "Patient B", "p2")
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Comparer les deux patients", use_container_width=True, key="compare_btn"):
        with st.spinner("Analyse en cours..."):
            try:
                batch_payload = {"patients": [{**p1_data, "patient_label": "A"}, {**p2_data, "patient_label": "B"}]}
                rb = requests.post(f"{API_URL}/predict/batch", json=batch_payload, headers=api_headers(), timeout=15)
                if rb.status_code == 200:
                    res1, res2 = rb.json(); s1=res1["risk_score"]; s2=res2["risk_score"]
                    seuil_e = st.session_state.get("seuil_eleve",50)/100; seuil_m = st.session_state.get("seuil_modere",30)/100
                    def get_level(s): return "ÉLEVÉ" if s>=seuil_e else ("MODÉRÉ" if s>=seuil_m else "FAIBLE")
                    l1=get_level(s1); l2=get_level(s2)
                    badge_map = {"ÉLEVÉ":"high","MODÉRÉ":"medium","FAIBLE":"low"}
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.markdown(render_risk_bar(s1, seuil_m, seuil_e), unsafe_allow_html=True)
                        st.markdown(f'<span class="risk-level-badge {badge_map[l1]}">{s1:.1%} — {l1}</span>', unsafe_allow_html=True)
                    with rc2:
                        st.markdown(render_risk_bar(s2, seuil_m, seuil_e), unsafe_allow_html=True)
                        st.markdown(f'<span class="risk-level-badge {badge_map[l2]}">{s2:.1%} — {l2}</span>', unsafe_allow_html=True)
                    diff = abs(s1-s2)*100
                    if s1>s2: st.markdown(f'<div class="alert-box alert-med">Patient A plus à risque que B — différence de {diff:.1f} points.</div>', unsafe_allow_html=True)
                    elif s2>s1: st.markdown(f'<div class="alert-box alert-med">Patient B plus à risque que A — différence de {diff:.1f} points.</div>', unsafe_allow_html=True)
                    else: st.markdown(f'<div class="alert-box alert-low">Les deux patients ont le même niveau de risque.</div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-label">Comparaison des facteurs SHAP</div>', unsafe_allow_html=True)
                    f1_data=res1["top_factors"]; f2_data=res2["top_factors"]
                    all_features=list(set(list(f1_data.keys())+list(f2_data.keys())))
                    v1=[f1_data.get(f,0) for f in all_features]; v2=[f2_data.get(f,0) for f in all_features]
                    fig_comp=go.Figure()
                    fig_comp.add_trace(go.Bar(name="Patient A", x=all_features, y=v1, marker_color=ACCENT))
                    fig_comp.add_trace(go.Bar(name="Patient B", x=all_features, y=v2, marker_color="#e3c46f"))
                    merged_theme=PLOTLY_THEME.copy(); merged_theme["xaxis"]={**PLOTLY_THEME.get("xaxis",{}),"tickangle":-30}
                    fig_comp.update_layout(barmode="group",height=300,margin=dict(t=10,b=20,l=10,r=10),**merged_theme)
                    st.plotly_chart(fig_comp,use_container_width=True)
                else: st.markdown('<div class="alert-box alert-high">Erreur lors de la comparaison.</div>', unsafe_allow_html=True)
            except requests.exceptions.ConnectionError: st.markdown('<div class="alert-box alert-high">API ml-service inaccessible.</div>', unsafe_allow_html=True)
    else: st.markdown(f"""<div class="card" style="text-align:center;padding:2rem;"><div style="font-size:2rem;">⚖️</div><div style="color:{MUTED};">Remplis les deux profils puis clique sur <b>Comparer les deux patients</b></div></div>""", unsafe_allow_html=True)


# ONGLET 5 — WHAT-IF

with tab_whatif:
    st.markdown('<div class="section-label">Patient de référence</div>', unsafe_allow_html=True)
    col_base, col_sim = st.columns([1, 1.1], gap="large")
    with col_base:
        with st.expander("Profil du patient", expanded=True):
            wi_age = st.slider("Âge", 0, 100, 65, key="wi_age")
            wg1,wg2=st.columns(2); wi_gender=wg1.selectbox("Genre",["Female","Male","Unknown"],key="wi_gender"); wi_race=wg2.selectbox("Origine",["Caucasian","AfricanAmerican","Hispanic","Asian","Other","Unknown"],key="wi_race")
            wh1,wh2=st.columns(2); wi_time_hosp=wh1.number_input("Durée (jours)",1,30,5,key="wi_time_hosp"); wi_num_meds=wh2.number_input("Médicaments",0,100,15,key="wi_num_meds")
            wh3,wh4=st.columns(2); wi_num_lab=wh3.number_input("Procédures labo",0,130,40,key="wi_num_lab"); wi_num_proc=wh4.number_input("Autres procédures",0,10,1,key="wi_num_proc")
            wi_num_diag=st.number_input("Diagnostics",1,16,7,key="wi_num_diag")
            wp1,wp2,wp3=st.columns(3); wi_num_out=wp1.number_input("Ambulatoire",0,50,0,key="wi_num_out"); wi_num_emer=wp2.number_input("Urgences",0,50,0,key="wi_num_emer"); wi_num_inp=wp3.number_input("Hospitalisé",0,20,1,key="wi_num_inp")
            wr1,wr2=st.columns(2); wi_a1c=wr1.selectbox("HbA1c",["None","Norm",">7",">8"],key="wi_a1c"); wi_glucose=wr2.selectbox("Glucose",["None","Norm",">200",">300"],key="wi_glucose")
            wr3,wr4=st.columns(2); wi_change_meds=wr3.selectbox("Changement médication",["No","Ch"],key="wi_change_meds"); wi_diabetes_med=wr4.selectbox("Anti-diabétiques",["Yes","No"],key="wi_diabetes_med")
    with col_sim:
        st.markdown('<div class="section-label">Scénario "What-if"</div>', unsafe_allow_html=True)
        wi_delta_meds=st.slider("Δ Médicaments",-10,10,0,key="wi_delta_meds"); wi_delta_hosp=st.slider("Δ Durée de séjour",-5,5,0,key="wi_delta_hosp")
        wi_add_outpatient=st.checkbox("Ajouter un suivi ambulatoire (+1 visite)",key="wi_add_outpatient")
        wi_override_a1c=st.selectbox("Nouveau contrôle HbA1c",["(inchangé)","None","Norm",">7",">8"],key="wi_override_a1c")
        wi_override_change=st.selectbox("Changement de médication",["(inchangé)","No","Ch"],key="wi_override_change")
        sim_num_meds=max(0,wi_num_meds+wi_delta_meds); sim_time_hosp=max(1,wi_time_hosp+wi_delta_hosp)
        sim_num_out=wi_num_out+(1 if wi_add_outpatient else 0)
        sim_a1c=wi_a1c if wi_override_a1c=="(inchangé)" else wi_override_a1c
        sim_change=wi_change_meds if wi_override_change=="(inchangé)" else wi_override_change
        baseline_payload={"age_num":wi_age,"gender":wi_gender,"race":wi_race,"time_in_hospital":wi_time_hosp,"num_medications":wi_num_meds,"num_lab_procedures":wi_num_lab,"num_procedures":wi_num_proc,"number_diagnoses":wi_num_diag,"num_outpatient":wi_num_out,"num_emergency":wi_num_emer,"num_inpatient":wi_num_inp,"a1c_result":wi_a1c,"glucose_serum":wi_glucose,"change_in_meds":wi_change_meds,"diabetes_meds":wi_diabetes_med,"patient_label":"Référence"}
        scenario_payload={"age_num":wi_age,"gender":wi_gender,"race":wi_race,"time_in_hospital":sim_time_hosp,"num_medications":sim_num_meds,"num_lab_procedures":wi_num_lab,"num_procedures":wi_num_proc,"number_diagnoses":wi_num_diag,"num_outpatient":sim_num_out,"num_emergency":wi_num_emer,"num_inpatient":wi_num_inp,"a1c_result":sim_a1c,"glucose_serum":wi_glucose,"change_in_meds":sim_change,"diabetes_meds":wi_diabetes_med,"patient_label":"Scénario"}
        try:
            resp=requests.post(f"{API_URL}/predict/batch",json={"patients":[baseline_payload,scenario_payload]},headers=api_headers(),timeout=10)
            handle_401(resp)
            if resp.status_code==200:
                res_ref,res_sim=resp.json(); s_ref=res_ref["risk_score"]; s_sim=res_sim["risk_score"]
                delta=(s_sim-s_ref)*100; seuil_e=st.session_state.get("seuil_eleve",50)/100; seuil_m=st.session_state.get("seuil_modere",30)/100
                m1,m2,m3=st.columns(3); m1.metric("Référence",f"{s_ref:.1%}"); m2.metric("Scénario",f"{s_sim:.1%}",delta=f"{delta:+.1f} pts",delta_color="inverse")
                direction="réduit" if delta<0 else ("augmenté" if delta>0 else "inchangé")
                m3.markdown(f'<div style="padding-top:0.6rem;color:{MUTED};">Le scénario a <b>{direction}</b> le risque.</div>',unsafe_allow_html=True)
                st.markdown(render_risk_bar(s_sim,seuil_m,seuil_e,secondary=s_ref),unsafe_allow_html=True)
                st.caption("Marqueur teal = scénario · marqueur gris = référence")
                st.markdown('<div class="section-label">Facteurs qui ont le plus changé (SHAP)</div>',unsafe_allow_html=True)
                f_ref=res_ref["top_factors"]; f_sim=res_sim["top_factors"]
                all_feats=list(set(list(f_ref.keys())+list(f_sim.keys())))
                v_ref=[f_ref.get(f,0) for f in all_feats]; v_sim=[f_sim.get(f,0) for f in all_feats]
                fig_diff=go.Figure()
                fig_diff.add_trace(go.Bar(name="Référence",x=all_feats,y=v_ref,marker_color="#94A3B8"))
                fig_diff.add_trace(go.Bar(name="Scénario",x=all_feats,y=v_sim,marker_color="#e3c46f"))
                merged_theme=PLOTLY_THEME.copy(); merged_theme["xaxis"]={**PLOTLY_THEME.get("xaxis",{}),"tickangle":-30}
                fig_diff.update_layout(barmode="group",height=280,margin=dict(t=10,b=20,l=10,r=10),**merged_theme)
                st.plotly_chart(fig_diff,use_container_width=True)
            else: st.markdown(f'<div class="alert-box alert-high">Erreur API {resp.status_code}</div>',unsafe_allow_html=True)
        except requests.exceptions.ConnectionError: st.markdown('<div class="alert-box alert-high">API ml-service inaccessible.</div>',unsafe_allow_html=True)


# ONGLET 6 — MODEL CARD

with tab_modelcard:
    st.markdown('<div class="section-label">Fiche modèle (Model Card)</div>', unsafe_allow_html=True)
    if not model_meta_info:
        st.markdown('<div class="alert-box alert-med">Informations du modèle indisponibles.</div>', unsafe_allow_html=True)
    else:
        m=model_meta_info; n_total=m.get("n_fit",0)+m.get("n_calib",0)+m.get("n_test",0)
        st.markdown(f"""<div class="card"><b style="color:{ACCENT};">⚕ Usage prévu</b><p style="color:{TEXT};font-size:0.9rem;line-height:1.6;margin-top:0.6rem;">Aide à la décision clinique estimant le risque de réhospitalisation sous 30 jours pour patient diabétique hospitalisé. Destiné à compléter le jugement de l’équipe soignante — ne remplace pas une évaluation médicale complète.</p></div>
        <div class="card"><b style="color:{ACCENT};">📊 Données d’entraînement</b></div>""",unsafe_allow_html=True)
        d1,d2,d3,d4=st.columns(4)
        d1.metric("Rencontres",f"{m.get('n_encounters_total',n_total):,}"); d2.metric("Patients",f"{m.get('n_patients_total','—'):,}" if m.get('n_patients_total') else "—")
        d3.metric("Features",m.get("feature_count","—")); d4.metric("Version",m.get("version","—"))
        st.markdown(f"""<div style="color:{MUTED};font-size:0.82rem;margin-top:0.3rem;">Source : Diabetes 130-US Hospitals (1999–2008). Split : {m.get("split_method","standard")}. Hash dataset : <code>{m.get("data_hash","—")}</code>.</div>""",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown(f"""<div class="card"><b style="color:{ACCENT};">📈 Performance</b><p style="color:{MUTED};font-size:0.82rem;">GroupKFold 5 plis, sans fuite patient.</p></div>""",unsafe_allow_html=True)
        p1,p2,p3,p4=st.columns(4)
        p1.metric("AUC-ROC",m.get("groupkfold5_auc_mean",m.get("auc_roc","—")),f"± {m.get('groupkfold5_auc_std','—')}" if m.get("groupkfold5_auc_std") else None)
        p2.metric("AUC-PR",m.get("groupkfold5_ap_mean",m.get("auc_pr","—")),f"± {m.get('groupkfold5_ap_std','—')}" if m.get("groupkfold5_ap_std") else None)
        p3.metric("F1",m.get("f1_score","—")); p4.metric("Recall top 10%",m.get("top10_capture_rate","—"))
        if m.get("threshold_analysis"):
            st.markdown(f"""<div class="card"><b style="color:{ACCENT};">⚖ Analyse de seuil</b></div>""",unsafe_allow_html=True)
            df_thr=pd.DataFrame(m["threshold_analysis"]); fig_thr=go.Figure()
            fig_thr.add_trace(go.Scatter(x=df_thr["threshold"],y=df_thr["precision"],mode="lines+markers",name="Precision",line=dict(color=ACCENT,width=2)))
            fig_thr.add_trace(go.Scatter(x=df_thr["threshold"],y=df_thr["recall"],mode="lines+markers",name="Recall",line=dict(color="#e3c46f",width=2)))
            fig_thr.add_vline(x=m.get("best_threshold",0.146),line_dash="dash",line_color="#e8948a",annotation_text=f"Seuil actif ({m.get('best_threshold',0):.3f})")
            merged_theme_thr=PLOTLY_THEME.copy(); merged_theme_thr["xaxis"]={**PLOTLY_THEME.get("xaxis",{}),"title":"Seuil"}; merged_theme_thr["yaxis"]={**PLOTLY_THEME.get("yaxis",{}),"title":"Score","range":[0,1]}
            fig_thr.update_layout(height=300,margin=dict(t=20,b=20,l=10,r=10),**merged_theme_thr)
            st.plotly_chart(fig_thr,use_container_width=True)
            thr_pick=st.select_slider("Explorer un seuil",options=df_thr["threshold"].tolist(),value=min(df_thr["threshold"],key=lambda x:abs(x-m.get("best_threshold",0.146))))
            row=df_thr[df_thr["threshold"]==thr_pick].iloc[0]; t1,t2,t3=st.columns(3)
            t1.metric("Precision",f"{row['precision']:.1%}"); t2.metric("Recall",f"{row['recall']:.1%}"); t3.metric("Patients alertés",f"{row['n_alerted']:,} ({row['pct_alerted']:.1%})")
        st.markdown(f"""<br><div class="alert-box alert-high"><b>⚠ Limites</b><br><span style="font-size:0.85rem;">• AUC modeste (~0.64) — usage en complément uniquement.<br>• Données historiques US 1999-2008, généralisation non garantie.<br>• Facteurs sociaux/psychologiques absents.</span></div>
        <div class="alert-box alert-med"><b>ℹ Précautions éthiques</b><br><span style="font-size:0.85rem;">• Variables démographiques incluses → vigilance équité.<br>• Toute prédiction tracée et attribuée au praticien.<br>• Probabilités calibrées (isotonic).</span></div>
        <div style="color:{MUTED};font-size:0.75rem;text-align:right;margin-top:1rem;">Généré depuis model_meta.json — v{m.get("version","—")}</div>""",unsafe_allow_html=True)