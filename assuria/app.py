"""
Assuria — Gestion intelligente de sinistres automobiles
Modeles :
  - best_clf_responsabilite.pkl : LogisticRegression MiniLM 384d
  - best_clf_type_sinistre.pkl  : LogisticRegression MiniLM 384d
  - model.pkl                   : XGBoost (Optuna) — 13 features
  - encoders.pkl                : LabelEncoders par colonne categorielle
"""
print("APP VERSION TEST")
import re
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from groq import Groq

warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Assuria – Sinistres Auto",
    layout="wide",
    initial_sidebar_state="collapsed",
)


import os
from dotenv import load_dotenv

load_dotenv()  # charge .env à la racine

api_key = os.getenv("GROQ_API_KEY")
print("API KEY LOADED:", api_key[:6] if api_key else None)  # debug court

from groq import Groq
groq_client = Groq(api_key=api_key)

FEATURE_NAMES = [
    "policy_state", "policy_deductable", "auto_make", "auto_year",
    "incident_type", "collision_type", "incident_severity", "incident_state",
    "incident_hour_of_the_day", "number_of_vehicles_involved",
    "bodily_injuries", "witnesses", "police_report_available"
]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Lora:ital,wght@0,600;1,400&display=swap');
:root {
    --navy:#0F2B5B;--blue:#1A56DB;--blue-h:#1447C0;--sky:#3B82F6;
    --pale:#EFF6FF;--pale2:#DBEAFE;--white:#FFFFFF;
    --g50:#F8FAFC;--g100:#F1F5F9;--g200:#E2E8F0;--g400:#94A3B8;--g600:#475569;--g800:#1E293B;
    --green:#059669;--green-bg:#ECFDF5;--red:#DC2626;--red-bg:#FEF2F2;
    --amber:#D97706;--amber-bg:#FFFBEB;
    --r:12px;--r-sm:8px;
    --sh:0 1px 4px rgba(15,43,91,.07),0 1px 2px rgba(15,43,91,.05);
    --sh-md:0 4px 16px rgba(15,43,91,.10);--sh-lg:0 12px 36px rgba(15,43,91,.14);
}
html,body,[data-testid="stAppViewContainer"]{font-family:'Plus Jakarta Sans',sans-serif!important;background:#ECF2FC!important;color:var(--g800)!important;}
[data-testid="stHeader"]{display:none!important;}
section[data-testid="stSidebar"]{display:none!important;}
.block-container{padding:0!important;max-width:100%!important;}
.main .block-container{padding:1.5rem 2rem 4rem!important;max-width:1300px!important;margin:auto!important;}
.topbar{background:var(--navy);display:flex;align-items:center;justify-content:space-between;padding:.9rem 2.5rem;margin-bottom:1.75rem;border-radius:var(--r);box-shadow:var(--sh-md);}
.brand-name{font-family:'Lora',serif;font-size:1.75rem;color:#fff;font-weight:600;letter-spacing:-.3px;}
.brand-sub{font-size:.72rem;color:#93C5FD;letter-spacing:.12em;text-transform:uppercase;margin-top:1px;}
.card{background:var(--white);border-radius:var(--r);padding:1.75rem 2rem;box-shadow:var(--sh);margin-bottom:1.25rem;}
.card-title{font-family:'Lora',serif;font-size:1.3rem;color:var(--navy);margin:0 0 .2rem;}
.section-line{width:32px;height:3px;background:var(--blue);border-radius:3px;margin:.4rem 0 1.2rem;}
.hero{background:linear-gradient(135deg,var(--navy) 0%,#1A3A7A 50%,var(--blue) 100%);border-radius:var(--r);padding:2.5rem 3rem;margin-bottom:1.75rem;position:relative;overflow:hidden;box-shadow:var(--sh-lg);}
.hero-title{font-family:'Lora',serif;font-size:2.2rem;color:#fff;line-height:1.2;margin:0 0 .6rem;}
.hero-sub{font-size:.95rem;color:#BFDBFE;margin:0;}
.tile-wrap{display:grid;grid-template-columns:repeat(3,1fr);gap:1.1rem;margin-bottom:1.5rem;}
.tile{background:var(--white);border-radius:var(--r);padding:1.75rem 1.5rem;box-shadow:var(--sh);border:1.5px solid transparent;text-align:center;transition:all .2s ease;}
.tile:hover{border-color:var(--sky);box-shadow:var(--sh-md);transform:translateY(-2px);}
.tile-ico{width:54px;height:54px;background:var(--pale2);border-radius:50%;display:flex;align-items:center;justify-content:center;margin:0 auto 1rem;}
.tile-ico svg{width:24px;height:24px;stroke:var(--blue);fill:none;stroke-width:1.8;}
.tile-t{font-weight:700;font-size:.95rem;color:var(--navy);margin-bottom:.35rem;}
.tile-d{font-size:.8rem;color:var(--g400);line-height:1.55;}
.stTextArea textarea,.stTextInput input,.stSelectbox select,.stNumberInput input{font-family:'Plus Jakarta Sans',sans-serif!important;font-size:.88rem!important;border:1.5px solid var(--g200)!important;border-radius:var(--r-sm)!important;background:var(--g50)!important;color:var(--g800)!important;}
label[data-testid="stWidgetLabel"]>div>p{font-family:'Plus Jakarta Sans',sans-serif!important;font-size:.82rem!important;font-weight:600!important;color:var(--g600)!important;}
.stButton>button{background:var(--blue)!important;color:#fff!important;border:none!important;border-radius:var(--r-sm)!important;font-family:'Plus Jakarta Sans',sans-serif!important;font-weight:700!important;font-size:.88rem!important;padding:.6rem 1.6rem!important;transition:all .18s!important;}
.stButton>button:hover{background:var(--blue-h)!important;transform:translateY(-1px)!important;box-shadow:0 4px 14px rgba(26,86,219,.3)!important;}
.res-box{border-radius:var(--r);padding:1.5rem 1.75rem;margin:1.1rem 0 .5rem;}
.res-box-blue{background:var(--pale);border:1.5px solid var(--pale2);}
.res-box-green{background:var(--green-bg);border:1.5px solid #A7F3D0;}
.res-box-red{background:var(--red-bg);border:1.5px solid #FECACA;}
.res-label{font-size:.72rem;text-transform:uppercase;letter-spacing:.1em;font-weight:700;margin-bottom:.3rem;}
.res-value{font-family:'Lora',serif;font-size:1.6rem;line-height:1.1;}
.res-note{font-size:.78rem;color:var(--g400);margin-top:.4rem;}
.est-row{display:flex;justify-content:space-between;align-items:center;padding:.6rem 0;border-bottom:1px solid var(--g100);font-size:.88rem;}
.est-row:last-child{border:none;}
.est-lbl{color:var(--g600);}
.est-val{font-weight:600;color:var(--navy);}
.est-total{display:flex;justify-content:space-between;align-items:center;padding-top:1rem;border-top:2px solid var(--pale2);margin-top:.5rem;}
.est-total-lbl{font-weight:700;font-size:.95rem;}
.est-total-val{font-family:'Lora',serif;font-size:2rem;color:var(--blue);}
.chat-wrap{background:var(--g50);border:1.5px solid var(--g100);border-radius:var(--r);padding:1.1rem;min-height:300px;max-height:420px;overflow-y:auto;margin-bottom:1rem;display:flex;flex-direction:column;gap:.7rem;}
.msg-user{display:flex;justify-content:flex-end;}
.bubble-user{background:var(--blue);color:#fff;padding:.65rem 1rem;border-radius:16px 16px 3px 16px;max-width:78%;font-size:.86rem;line-height:1.5;}
.msg-ai{display:flex;gap:.6rem;align-items:flex-start;}
.ai-av{width:32px;height:32px;border-radius:50%;background:var(--pale2);display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.75rem;color:var(--blue);flex-shrink:0;}
.bubble-ai{background:var(--white);border:1.5px solid var(--g100);color:var(--g800);padding:.65rem 1rem;border-radius:3px 16px 16px 16px;max-width:82%;font-size:.86rem;line-height:1.55;box-shadow:var(--sh);}
.transcription-info{background:var(--pale);border:1.5px solid var(--pale2);border-radius:var(--r-sm);padding:.9rem 1.1rem;font-size:.82rem;color:var(--g600);margin-bottom:1rem;line-height:1.6;}
.transcription-info strong{color:var(--navy);}
.ph{margin-bottom:1.5rem;}
.ph h1{font-family:'Lora',serif;font-size:1.8rem;color:var(--navy);margin:0 0 .2rem;}
.ph p{font-size:.85rem;color:var(--g400);margin:0;}
</style>
""", unsafe_allow_html=True)


# ── Chargement modeles ──
@st.cache_resource(show_spinner="Chargement des modeles...")
def load_models():
    clf_resp = joblib.load("best_clf_responsabilite.pkl")
    clf_type = joblib.load("best_clf_type_sinistre.pkl")
    clf_est  = joblib.load("model.pkl")
    encoders = joblib.load("encoders.pkl")
    return clf_resp, clf_type, clf_est, encoders

@st.cache_resource(show_spinner="Chargement de MiniLM...")
def load_minilm():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

clf_resp, clf_type, clf_est, encoders = load_models()
minilm = load_minilm()

# Valeurs valides selon les encodeurs du notebook
VALID = {col: list(le.classes_) for col, le in encoders.items()}

def encode_input(df):
    """Encode les colonnes categorielle avec les encodeurs du notebook."""
    df = df.copy()
    for col, le in encoders.items():
        if col in df.columns:
            val = str(df[col].iloc[0])
            if val not in le.classes_:
                val = le.classes_[0]
            df[col] = le.transform([val])
    return df

# ── Preprocessing texte ──
MOTS_BRUIT = ["bonjour","bonsoir","allo","allô","salut","euh","hum","ben","pis","genre","alors"]

def nettoyer(texte):
    if not texte:
        return ""
    for mot in MOTS_BRUIT:
        texte = re.sub(rf"\b{mot}\b", "", texte, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", texte).strip()

def embed(texte):
    return minilm.encode([nettoyer(texte)])

# ── Session state ──
for key, val in [("page","accueil"),("chat_history",[]),("dernier_sinistre",{})]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Navigation ──
st.markdown("""
<div class="topbar">
  <div>
    <div class="brand-name">Assuria</div>
    <div class="brand-sub">Votre Expert en Sinistres et Assurance</div>
  </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Accueil", use_container_width=True, key="n1"):
        st.session_state.page = "accueil"; st.rerun()
with col2:
    if st.button("Analyse de Sinistre", use_container_width=True, key="n2"):
        st.session_state.page = "analyse"; st.rerun()
with col3:
    if st.button("Estimation des Couts", use_container_width=True, key="n3"):
        st.session_state.page = "estimation"; st.rerun()
with col4:
    if st.button("Assistance IA", use_container_width=True, key="n4"):
        st.session_state.page = "chat"; st.rerun()

st.markdown("<br>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE : ACCUEIL
# ════════════════════════════════════════════════════════════
if st.session_state.page == "accueil":

    st.markdown("""
    <div class="hero">
      <div class="hero-title">Gestion de Sinistres<br>Automobile au Canada</div>
      <div class="hero-sub">Declarez votre sinistre en francais, notre IA analyse votre transcription,
      determine la responsabilite, identifie le type de sinistre et estime les couts.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tile-wrap">
      <div class="tile">
        <div class="tile-ico"><svg viewBox="0 0 24 24"><path d="M12 20h9M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4L16.5 3.5z" stroke-linecap="round" stroke-linejoin="round"/></svg></div>
        <div class="tile-t">Analyse de Sinistre</div>
        <div class="tile-d">Decrivez l'accident en francais. Notre modele MiniLM detecte automatiquement le type de sinistre et la responsabilite.</div>
      </div>
      <div class="tile">
        <div class="tile-ico"><svg viewBox="0 0 24 24"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6" stroke-linecap="round" stroke-linejoin="round"/></svg></div>
        <div class="tile-t">Estimation des Couts</div>
        <div class="tile-d">Entrez les informations de l'incident. Notre modele XGBoost predit le montant total du sinistre.</div>
      </div>
      <div class="tile">
        <div class="tile-ico"><svg viewBox="0 0 24 24"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" stroke-linecap="round" stroke-linejoin="round"/></svg></div>
        <div class="tile-t">Assistance IA</div>
        <div class="tile-d">Posez vos questions en francais a notre agent IA specialise en droit quebecois des assurances.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Types de sinistres pris en charge</div><div class="section-line"></div>', unsafe_allow_html=True)
    cols = st.columns(6)
    for i, (t, d) in enumerate([
        ("Collision","Accident avec un autre vehicule"),
        ("Collision avec animal","Chevreuil, orignal, etc."),
        ("Bris de glace","Pare-brise, vitre laterale"),
        ("Vol","Vol du vehicule complet"),
        ("Vandalisme","Dommages intentionnels"),
        ("Feu","Incendie du vehicule"),
    ]):
        with cols[i]:
            st.markdown(f"**{t}**")
            st.caption(d)
    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE : ANALYSE DE SINISTRE
# ════════════════════════════════════════════════════════════
elif st.session_state.page == "analyse":

    st.markdown("""
    <div class="ph"><h1>Analyse de Sinistre</h1>
    <p>Decrivez l'accident en francais — notre IA detecte le type et la responsabilite</p></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Description de l\'accident</div><div class="section-line"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="transcription-info">
      <strong>Comment remplir ce champ :</strong> Decrivez l'accident comme si vous telephoniez
      a votre assureur. Utilisez vos propres mots, en francais.
    </div>
    """, unsafe_allow_html=True)

    transcription = st.text_area(
        "Description de l'accident",
        height=160,
        placeholder="Ex: J'etais en train de tourner a gauche et j'ai heurte un autre vehicule...",
        key="transcription_input"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Analyser le Sinistre", use_container_width=True):
        if not transcription.strip():
            st.warning("Veuillez d'abord decrire l'accident.")
        else:
            with st.spinner("Analyse en cours..."):
                try:
                    vec        = embed(transcription)
                    pred_resp  = clf_resp.predict(vec)[0]
                    prob_resp  = clf_resp.predict_proba(vec)[0]
                    pred_type  = clf_type.predict(vec)[0]
                    prob_type  = clf_type.predict_proba(vec)[0]

                    responsible  = (pred_resp == "YES")
                    conf_resp    = float(max(prob_resp)) * 100
                    conf_type    = float(max(prob_type)) * 100
                    classes_type = clf_type.classes_

                    st.session_state.dernier_sinistre.update({
                        "description": transcription[:200],
                        "responsable": "Responsable" if responsible else "Non responsable",
                        "type":        pred_type,
                        "conf_resp":   conf_resp,
                        "conf_type":   conf_type,
                    })

                    col_a, col_b = st.columns(2)
                    with col_a:
                        box_cls = "res-box-red" if responsible else "res-box-green"
                        label_c = "#DC2626" if responsible else "#059669"
                        st.markdown(f"""
                        <div class="res-box {box_cls}">
                          <div class="res-label" style="color:{label_c}">Responsabilite</div>
                          <div class="res-value" style="color:{label_c}">{"Responsable" if responsible else "Non responsable"}</div>
                          <div class="res-note">Confiance : {conf_resp:.0f}%</div>
                        </div>""", unsafe_allow_html=True)
                    with col_b:
                        st.markdown(f"""
                        <div class="res-box res-box-blue">
                          <div class="res-label" style="color:#1A56DB">Type de sinistre detecte</div>
                          <div class="res-value" style="color:#0F2B5B">{pred_type}</div>
                          <div class="res-note">Confiance : {conf_type:.0f}%</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown('<div class="card" style="margin-top:.5rem">', unsafe_allow_html=True)
                    st.markdown('<div class="card-title">Probabilites par type</div><div class="section-line"></div>', unsafe_allow_html=True)
                    for classe, prob in sorted(zip(classes_type, prob_type), key=lambda x: x[1], reverse=True):
                        is_top = (classe == pred_type)
                        color  = "#1A56DB" if is_top else "#94A3B8"
                        st.markdown(f"""
                        <div style="margin-bottom:.6rem;">
                          <div style="display:flex;justify-content:space-between;margin-bottom:.2rem;">
                            <span style="font-size:.82rem;font-weight:{'700' if is_top else '400'};color:{'#0F2B5B' if is_top else '#475569'}">{classe}</span>
                            <span style="font-size:.82rem;font-weight:600;color:{color}">{prob*100:.1f}%</span>
                          </div>
                          <div style="background:#E2E8F0;border-radius:4px;height:7px;">
                            <div style="background:{color};width:{prob*100:.1f}%;border-radius:4px;height:7px;"></div>
                          </div>
                        </div>""", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    if responsible:
                        st.error("Notre modele indique que vous pourriez etre considere responsable. Contactez votre assureur rapidement.")
                    else:
                        st.success("Notre modele indique que vous n'etes probablement pas responsable. Signalez quand meme l'incident.")

                except Exception as e:
                    st.error(f"Erreur lors de l'analyse : {e}")


# ════════════════════════════════════════════════════════════
# PAGE : ESTIMATION DES COUTS
# ════════════════════════════════════════════════════════════
elif st.session_state.page == "estimation":

    st.markdown("""
    <div class="ph"><h1>Estimation des Couts</h1>
    <p>Calcul du montant total prevu par notre modele XGBoost</p></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Police d\'assurance et vehicule</div><div class="section-line"></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        policy_state      = st.selectbox("Province de la police", VALID["policy_state"])
        policy_deductable = st.selectbox("Franchise ($)", [500, 1000, 2000])
    with c2:
        auto_make = st.selectbox("Marque vehicule", VALID["auto_make"])
        auto_year = st.number_input("Annee vehicule", 1990, 2025, 2020)
    with c3:
        incident_state = st.selectbox("Province de l'incident", VALID["incident_state"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Details de l\'incident</div><div class="section-line"></div>', unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    with c4:
        incident_type     = st.selectbox("Type d'incident", VALID["incident_type"])
        collision_type    = st.selectbox("Type de collision", VALID["collision_type"])
        incident_severity = st.selectbox("Gravite des dommages", VALID["incident_severity"])
    with c5:
        incident_hour = st.slider("Heure de l'incident (0-23)", 0, 23, 14)
        nb_vehicles   = st.number_input("Nombre de vehicules impliques", 1, 10, 1)
    with c6:
        bodily_injuries = st.number_input("Nombre de blesses corporels", 0, 10, 0)
        witnesses       = st.number_input("Nombre de temoins", 0, 20, 0)
        police_report   = st.selectbox("Rapport de police disponible", VALID["police_report_available"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ✅ BOUTON ET CALCUL ICI — dans le elif estimation
    if st.button("Calculer l'Estimation", use_container_width=True):
        with st.spinner("Calcul en cours..."):
            try:
                input_df = pd.DataFrame([{
                    "policy_state":                policy_state,
                    "policy_deductable":           policy_deductable,
                    "auto_make":                   auto_make,
                    "auto_year":                   auto_year,
                    "incident_type":               incident_type,
                    "collision_type":              collision_type,
                    "incident_severity":           incident_severity,
                    "incident_state":              incident_state,
                    "incident_hour_of_the_day":    incident_hour,
                    "number_of_vehicles_involved": nb_vehicles,
                    "bodily_injuries":             bodily_injuries,
                    "witnesses":                   witnesses,
                    "police_report_available":     police_report,
                }])[FEATURE_NAMES]

                for col, le in encoders.items():
                    if col in input_df.columns:
                        input_df[col] = input_df[col].apply(
                            lambda x: x if x in le.classes_ else le.classes_[0]
                        )
                        input_df[col] = le.transform(input_df[col])

                input_df = input_df.astype(float)

                pred_log = clf_est.predict(input_df)[0]
                total    = float(np.expm1(pred_log))

                reparation  = total * 0.52
                pieces      = total * 0.27
                main_oeuvre = total * 0.15
                autres      = total * 0.06

                st.markdown(f"""
                <div class="res-box res-box-blue">
                  <div class="res-label" style="color:#1A56DB">Estimation Totale du Sinistre</div>
                  <div class="res-value" style="color:#0F2B5B">{total:,.0f} $</div>
                  <div class="res-note">Prediction XGBoost | Entrainement sur log(montant+1)</div>
                </div>""", unsafe_allow_html=True)

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Ventilation estimee</div><div class="section-line"></div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="est-row"><span class="est-lbl">Reparation carrosserie</span><span class="est-val">{reparation:,.0f} $</span></div>
                <div class="est-row"><span class="est-lbl">Pieces et materiaux</span><span class="est-val">{pieces:,.0f} $</span></div>
                <div class="est-row"><span class="est-lbl">Main d'oeuvre</span><span class="est-val">{main_oeuvre:,.0f} $</span></div>
                <div class="est-row"><span class="est-lbl">Frais divers</span><span class="est-val">{autres:,.0f} $</span></div>
                <div class="est-total">
                  <span class="est-total-lbl">Total predit</span>
                  <span class="est-total-val">{total:,.0f} $</span>
                </div>""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                if incident_severity == "Total Loss":
                    st.error("Gravite : Perte totale. Le vehicule sera probablement declare irreparable.")
                elif incident_severity == "Major Damage":
                    st.warning("Gravite : Dommages importants. Une expertise detaillee est recommandee.")
                elif incident_severity == "Minor Damage":
                    st.success("Gravite : Dommages mineurs. La reparation devrait etre rapide.")

                st.session_state.dernier_sinistre["estimation"] = f"{total:,.0f} $"

            except Exception as e:
                st.error(f"Erreur lors de l'estimation : {e}")


# ════════════════════════════════════════════════════════════
# PAGE : ASSISTANCE IA — Groq llama-3.3-70b-versatile
# ════════════════════════════════════════════════════════════
elif st.session_state.page == "chat":

    st.markdown("""
    <div class="ph"><h1>Assistance IA</h1>
    <p>Posez vos questions en francais a notre agent specialise en sinistres et assurances auto au Canada</p></div>
    """, unsafe_allow_html=True)

    ctx     = st.session_state.dernier_sinistre
    ctx_str = ""
    if ctx:
        ctx_str = (
            f"Responsabilite={ctx.get('responsable','?')} (confiance {ctx.get('conf_resp',0):.0f}%), "
            f"Type={ctx.get('type','?')} (confiance {ctx.get('conf_type',0):.0f}%), "
            f"Estimation={ctx.get('estimation','non calculee')}. "
            f"Description: {ctx.get('description','?')}"
        )
        st.markdown(f"""
        <div class="transcription-info">
          <strong>Contexte de votre dossier :</strong> {ctx.get('responsable','?')} — Type : {ctx.get('type','?')} — Estimation : {ctx.get('estimation','non calculee')}
        </div>""", unsafe_allow_html=True)

    SYSTEM = f"""Tu es un expert en sinistres automobiles et en droit des assurances au Canada,
specialise dans le droit quebecois. Tu t'appelles Assuria.
Tu reponds UNIQUEMENT en francais. Tes reponses sont precises, concises (3-5 phrases),
professionnelles et bienveillantes.
Tu peux citer le Code civil du Quebec (CCQ), la Loi sur l'assurance automobile (SAAQ),
le regime sans egard (no-fault) quebecois, et les pratiques des assureurs canadiens.
Tu ne donnes pas de conseils juridiques formels mais tu aides a comprendre les situations.
{('Contexte du sinistre : ' + ctx_str) if ctx_str else ''}"""

    chat_html = '<div class="chat-wrap">'
    if not st.session_state.chat_history:
        chat_html += """
        <div class="msg-ai"><div class="ai-av">A</div>
          <div class="bubble-ai">Bonjour! Je suis Assuria, votre assistante specialisee en sinistres
          automobiles au Canada. Comment puis-je vous aider?</div>
        </div>"""
    for m in st.session_state.chat_history:
        if m["role"] == "user":
            chat_html += f'<div class="msg-user"><div class="bubble-user">{m["content"]}</div></div>'
        else:
            chat_html += f'<div class="msg-ai"><div class="ai-av">A</div><div class="bubble-ai">{m["content"]}</div></div>'
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

    col_i, col_b = st.columns([5, 1])
    with col_i:
        user_msg = st.text_input("Votre question", label_visibility="collapsed",
                                 placeholder="Posez votre question en francais...", key="user_input")
    with col_b:
        send = st.button("Envoyer", use_container_width=True, key="send_btn")

    st.markdown("**Questions frequentes :**")
    qs = [
        "Suis-je responsable si je frappe un animal?",
        "Comment fonctionne le regime sans egard au Quebec?",
        "Quels delais pour declarer un sinistre?",
        "Mon assureur peut-il refuser ma reclamation?",
        "Comment contester une decision de mon assureur?",
        "Le vol de mon vehicule est-il couvert automatiquement?",
    ]
    q_cols = st.columns(3)
    for i, q in enumerate(qs):
        with q_cols[i % 3]:
            if st.button(q, key=f"q{i}", use_container_width=True):
                user_msg = q
                send = True

    if send and user_msg.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        with st.spinner("Assuria reflechit..."):
            try:
                messages = [{"role": "system", "content": SYSTEM}]
                for m in st.session_state.chat_history:
                    role = "assistant" if m["role"] == "assistant" else "user"
                    messages.append({"role": role, "content": m["content"]})
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    max_tokens=512,
                    temperature=0.4,
                )
                answer = response.choices[0].message.content
            except Exception as e:
                answer = f"Desolee, une erreur s'est produite : {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("Effacer la conversation"):
            st.session_state.chat_history = []
            st.rerun()
