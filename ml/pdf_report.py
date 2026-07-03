"""
Générateur de rapport PDF — ClinAI
====================================
Génère un rapport PDF professionnel pour une prédiction patient.
Utilise reportlab pour la création du PDF.

Installer : pip install reportlab matplotlib
"""

import io
import json
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image
)
from reportlab.platypus import KeepTogether


# ─── Couleurs ClinAI ──────────────────────────────────────────────
BLUE      = colors.HexColor("#1B4FD8")
BLUE_LIGHT= colors.HexColor("#EFF6FF")
RED       = colors.HexColor("#EF4444")
RED_LIGHT = colors.HexColor("#FEF2F2")
ORANGE    = colors.HexColor("#F59E0B")
ORANGE_LT = colors.HexColor("#FFFBEB")
GREEN     = colors.HexColor("#22C55E")
GREEN_LT  = colors.HexColor("#F0FDF4")
GRAY      = colors.HexColor("#64748B")
GRAY_LT   = colors.HexColor("#F8FAFC")
DARK      = colors.HexColor("#1a1a2e")
BORDER    = colors.HexColor("#E2E8F0")
WHITE     = colors.white


def _gauge_image(score: float, level: str) -> io.BytesIO:
    """Génère une image de jauge matplotlib."""
    color_map = {"ÉLEVÉ": "#EF4444", "MODÉRÉ": "#F59E0B", "FAIBLE": "#22C55E"}
    gauge_color = color_map.get(level, "#1B4FD8")

    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("white")

    # Fond de la jauge
    theta_bg = np.linspace(np.pi, 0, 200)
    ax.fill_between(theta_bg, 0.6, 1.0, color="#F1F5F9", zorder=1)

    # Zones colorées
    zones = [(np.pi, np.pi * 0.7, "#dcfce7"),
             (np.pi * 0.7, np.pi * 0.5, "#fef9c3"),
             (np.pi * 0.5, 0, "#fee2e2")]
    for start, end, color in zones:
        theta = np.linspace(start, end, 100)
        ax.fill_between(theta, 0.6, 1.0, color=color, zorder=2)

    # Aiguille
    angle = np.pi * (1 - score)
    ax.annotate("", xy=(angle, 0.85), xytext=(angle, 0.0),
                arrowprops=dict(arrowstyle="-|>", color=gauge_color,
                               lw=2.5, mutation_scale=15), zorder=5)
    ax.plot(0, 0, "o", color=gauge_color, markersize=8, zorder=6)

    # Score au centre
    ax.text(0, -0.15, f"{score*100:.1f}%", ha="center", va="center",
            fontsize=20, fontweight="bold", color=gauge_color,
            transform=ax.transData)

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, np.pi)
    ax.axis("off")
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", transparent=False)
    plt.close(fig)
    buf.seek(0)
    return buf


def _shap_image(top_factors: dict) -> io.BytesIO:
    """Génère un graphique SHAP horizontal."""
    items = sorted(top_factors.items(), key=lambda x: abs(x[1]))
    features = [k.replace("_", " ") for k, v in items]
    values   = [v for k, v in items]
    colors_bar = ["#EF4444" if v > 0 else "#22C55E" for v in values]

    fig, ax = plt.subplots(figsize=(6, max(2.5, len(features) * 0.5)))
    fig.patch.set_facecolor("white")
    bars = ax.barh(features, values, color=colors_bar, edgecolor="none", height=0.6)

    for bar, val in zip(bars, values):
        ax.text(val + (0.001 if val >= 0 else -0.001),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}", va="center",
                ha="left" if val >= 0 else "right",
                fontsize=9, color="#475569")

    ax.axvline(0, color="#CBD5E1", linewidth=1)
    ax.set_xlabel("|Impact SHAP|", fontsize=9, color="#64748B")
    ax.tick_params(labelsize=9, colors="#475569")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#E2E8F0")

    # Légende
    pos_patch = mpatches.Patch(color="#EF4444", label="Augmente le risque")
    neg_patch = mpatches.Patch(color="#22C55E", label="Réduit le risque")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right",
              fontsize=8, framealpha=0.8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", transparent=False)
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_pdf(
    patient_data: dict,
    risk_score: float,
    risk_level: str,
    top_factors: dict,
    model_version: str,
    requested_by: str,
    model_auc: float = None,
) -> bytes:
    """
    Génère un rapport PDF complet pour une prédiction.
    Retourne les bytes du PDF.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    story  = []

    # ── Styles personnalisés ──────────────────────────────────────
    title_style = ParagraphStyle("ClinTitle",
        fontSize=22, fontName="Helvetica-Bold",
        textColor=BLUE, alignment=TA_LEFT, spaceAfter=2)

    subtitle_style = ParagraphStyle("ClinSub",
        fontSize=10, fontName="Helvetica",
        textColor=GRAY, alignment=TA_LEFT, spaceAfter=0)

    section_style = ParagraphStyle("ClinSection",
        fontSize=11, fontName="Helvetica-Bold",
        textColor=DARK, spaceBefore=12, spaceAfter=6)

    body_style = ParagraphStyle("ClinBody",
        fontSize=9, fontName="Helvetica",
        textColor=DARK, leading=14)

    small_style = ParagraphStyle("ClinSmall",
        fontSize=8, fontName="Helvetica",
        textColor=GRAY, alignment=TA_CENTER)

    # ── EN-TÊTE ───────────────────────────────────────────────────
    header_data = [[
        Paragraph("<b>ClinAI</b>", ParagraphStyle("H",
            fontSize=20, fontName="Helvetica-Bold", textColor=WHITE)),
        Paragraph(
            f"<b>Rapport de prédiction</b><br/>"
            f"<font size='8'>Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}</font><br/>"
            f"<font size='8'>Praticien : {requested_by} · Modèle : {model_version}</font>",
            ParagraphStyle("HR", fontSize=9, fontName="Helvetica",
                          textColor=WHITE, alignment=TA_RIGHT))
    ]]
    header_table = Table(header_data, colWidths=[8*cm, 9*cm])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), BLUE),
        ("PADDING",    (0,0), (-1,-1), 14),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [BLUE]),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.4*cm))

    # ── RÉSULTAT PRINCIPAL ────────────────────────────────────────
    level_colors = {"ÉLEVÉ": (RED, RED_LIGHT), "MODÉRÉ": (ORANGE, ORANGE_LT), "FAIBLE": (GREEN, GREEN_LT)}
    lc, lc_bg = level_colors.get(risk_level, (BLUE, BLUE_LIGHT))

    emoji_map = {"ÉLEVÉ": "▲", "MODÉRÉ": "●", "FAIBLE": "▼"}
    reco_map = {
        "ÉLEVÉ":  "Suivi renforcé recommandé — Consultation de sortie dédiée · Rappel à 48h · Vérification de l'observance",
        "MODÉRÉ": "Surveillance modérée — Appel de suivi à J+7 · Vérifier l'observance médicamenteuse",
        "FAIBLE": "Protocole de sortie standard — Aucune mesure supplémentaire requise",
    }

    # Jauge
    gauge_buf = _gauge_image(risk_score, risk_level)
    gauge_img = Image(gauge_buf, width=7*cm, height=3.8*cm)

    result_left = [
        [Paragraph("<b>SCORE DE RISQUE DE RÉHOSPITALISATION</b>",
                   ParagraphStyle("RL", fontSize=9, fontName="Helvetica-Bold",
                                  textColor=GRAY, spaceAfter=4))],
        [gauge_img],
        [Paragraph(
            f"<b>{emoji_map.get(risk_level,'')} Risque {risk_level}</b>",
            ParagraphStyle("RLV", fontSize=14, fontName="Helvetica-Bold",
                          textColor=lc, alignment=TA_CENTER))],
    ]

    result_right = [
        [Paragraph("<b>RECOMMANDATION CLINIQUE</b>",
                   ParagraphStyle("RR", fontSize=9, fontName="Helvetica-Bold",
                                  textColor=GRAY, spaceAfter=4))],
        [Paragraph(reco_map.get(risk_level, ""),
                   ParagraphStyle("RRC", fontSize=10, fontName="Helvetica",
                                  textColor=DARK, leading=16))],
    ]

    tl = Table(result_left,  colWidths=[8*cm])
    tl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), lc_bg),
        ("PADDING",    (0,0), (-1,-1), 10),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("BOX",        (0,0), (-1,-1), 1, lc),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [lc_bg]),
    ]))

    tr = Table(result_right, colWidths=[9*cm])
    tr.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), GRAY_LT),
        ("PADDING",    (0,0), (-1,-1), 12),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("BOX",        (0,0), (-1,-1), 1, BORDER),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [GRAY_LT]),
    ]))

    combined = Table([[tl, tr]], colWidths=[8.5*cm, 9.5*cm], hAlign="LEFT")
    combined.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))
    story.append(combined)
    story.append(Spacer(1, 0.4*cm))

    # ── DONNÉES PATIENT ───────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    story.append(Paragraph("Données cliniques du patient", section_style))

    patient_rows = [
        ["Âge", f"{patient_data.get('age_num', '—')} ans",
         "Genre", patient_data.get('gender', '—')],
        ["Origine", patient_data.get('race', '—'),
         "Durée séjour", f"{patient_data.get('time_in_hospital', '—')} jours"],
        ["Médicaments", str(patient_data.get('num_medications', '—')),
         "Procédures labo", str(patient_data.get('num_lab_procedures', '—'))],
        ["Diagnostics", str(patient_data.get('number_diagnoses', '—')),
         "Hospitalisations préc.", str(patient_data.get('num_inpatient', '—'))],
        ["Urgences préc.", str(patient_data.get('num_emergency', '—')),
         "Ambulatoire préc.", str(patient_data.get('num_outpatient', '—'))],
        ["HbA1c", patient_data.get('a1c_result', '—'),
         "Glucose sérique", patient_data.get('glucose_serum', '—')],
        ["Changement médication", patient_data.get('change_in_meds', '—'),
         "Anti-diabétiques", patient_data.get('diabetes_meds', '—')],
    ]

    header_row = [
        Paragraph("<b>Paramètre</b>", ParagraphStyle("PH", fontSize=8, fontName="Helvetica-Bold", textColor=WHITE)),
        Paragraph("<b>Valeur</b>",    ParagraphStyle("PH", fontSize=8, fontName="Helvetica-Bold", textColor=WHITE)),
        Paragraph("<b>Paramètre</b>", ParagraphStyle("PH", fontSize=8, fontName="Helvetica-Bold", textColor=WHITE)),
        Paragraph("<b>Valeur</b>",    ParagraphStyle("PH", fontSize=8, fontName="Helvetica-Bold", textColor=WHITE)),
    ]

    table_data = [header_row] + [
        [Paragraph(str(c), ParagraphStyle("TC", fontSize=8, fontName="Helvetica", textColor=DARK)) for c in row]
        for row in patient_rows
    ]

    patient_table = Table(table_data, colWidths=[4.5*cm, 4*cm, 4.5*cm, 4*cm])
    patient_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), BLUE),
        ("BACKGROUND",    (0,1), (-1,-1), WHITE),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, GRAY_LT]),
        ("PADDING",       (0,0), (-1,-1), 7),
        ("GRID",          (0,0), (-1,-1), 0.5, BORDER),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 0.3*cm))

    # ── FACTEURS SHAP ─────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    story.append(Paragraph("Facteurs déterminants (SHAP)", section_style))
    story.append(Paragraph(
        "Les valeurs positives (rouge) augmentent le risque de réhospitalisation. "
        "Les valeurs négatives (vert) le réduisent.",
        ParagraphStyle("SHAPDesc", fontSize=8, fontName="Helvetica",
                      textColor=GRAY, spaceAfter=6)))

    shap_buf = _shap_image(top_factors)
    shap_img = Image(shap_buf, width=15*cm, height=max(4*cm, len(top_factors)*0.8*cm))
    story.append(shap_img)
    story.append(Spacer(1, 0.3*cm))

    # ── PIED DE PAGE ──────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "⚕ ClinAI — Outil d'aide à la décision clinique · "
        "Ce rapport ne remplace pas le jugement médical du praticien · "
        f"Dataset : Strack et al. 2014, BioMed Research International · "
        f"AUC-ROC modèle : {model_auc:.3f}" if model_auc is not None else "AUC-ROC modèle : N/A",
        small_style))

    doc.build(story)
    buf.seek(0)
    return buf.read()