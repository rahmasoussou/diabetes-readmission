# Synthèse des expérimentations — pistes d'amélioration du modèle

Suite aux pistes proposées, deux expériences ont été menées, chacune
documentée qu'elle soit concluante ou non.

---

## Expérience 1 — LabelEncoder vs One-Hot Encoding

**Hypothèse testée :** le `LabelEncoder` actuel impose un ordre artificiel
sur des variables nominales (`race`, `gender`, `a1c_result`,
`glucose_serum`, `change_in_meds`, `diabetes_meds`) qui n'en ont pas.
Remplacer par du One-Hot Encoding devrait éliminer ce biais.

**Protocole :** même split (80/20, `random_state=42`), mêmes
hyperparamètres XGBoost, seule la représentation des variables nominales
change entre les deux versions.

| | AUC-ROC | AUC-PR | Nb features |
|---|---|---|---|
| A — LabelEncoder (actuel) | 0.6118 | 0.1385 | 35 |
| B — One-Hot (testé) | 0.6126 | 0.1413 | 50 |
| **Δ** | **+0.0008** | **+0.0028** | +15 |

**Conclusion : résultat négatif.** L'écart est largement dans le bruit
statistique (l'écart-type observé en validation croisée sur ce modèle est
de 0.0058, soit 7× plus grand que ce delta). Le LabelEncoder n'est **pas**
le facteur limitant la performance. Le One-Hot ajoute 15 colonnes pour un
gain non significatif — pas justifié de l'adopter en l'état.

Script : `ml/experiments/exp_encoding.py`

---

## Expérience 2 — Toutes les rencontres + split groupé par patient

**Hypothèse testée :** la déduplication actuelle (1 rencontre gardée par
patient) jette environ 32% des lignes brutes. Garder toutes les
rencontres, à condition d'un split groupé par `patient_nbr` (aucun
patient à cheval entre train et test, vérifié explicitement), devrait
donner plus de signal au modèle.

**Protocole :** mêmes hyperparamètres XGBoost. Version A : dédup
actuelle + split aléatoire classique. Version B : toutes les rencontres
(99 343 lignes après nettoyage, contre 69 990 dédupliquées) + split
groupé (`GroupShuffleSplit` sur `patient_nbr`, chevauchement vérifié = 0).

| | AUC-ROC | AUC-PR | Train | Test |
|---|---|---|---|---|
| A — dédup + split aléatoire (actuel) | 0.6150 | 0.1429 | 55 992 | 13 998 |
| B — toutes rencontres + split groupé (testé) | **0.6424** | **0.2071** | 79 541 | 19 802 |
| **Δ** | **+0.0274** | **+0.0642** | +42% de données | — |

**Point de vigilance identifié et vérifié :** la prévalence (taux de
réhospitalisation à 30 jours) diffère entre les deux jeux de test — 8.98%
en version A contre 11.19% en version B (les patients avec plusieurs
rencontres ont un taux de réhospitalisation plus élevé, logique
cliniquement : ce sont les patients avec un historique hospitalier plus
chargé). Or l'AUC-PR est mécaniquement sensible à la prévalence — un
classifieur aléatoire obtiendrait déjà un AUC-PR égal à la prévalence.

**Décomposition honnête du gain d'AUC-PR :**
- Gain "gratuit" attendu du seul effet de prévalence : ≈ +0.022
- Gain total observé : +0.0642
- Gain net après retrait de l'effet de prévalence : ≈ **+0.042**

**L'AUC-ROC n'est pas affecté par la prévalence** (il ne dépend que du
classement relatif des scores, pas du taux de base) : le **+0.0274 observé
est donc un signal propre**, non biaisé par la question de prévalence.

**Conclusion : résultat positif, avec nuance documentée.** Le gain est réel
mais plus modeste que les chiffres bruts ne le suggèrent. La piste est la
plus prometteuse des deux testées.

Scripts : `ml/experiments/exp_group_split.py`,
`ml/experiments/exp_prevalence_check.py`

---

## Recommandation

1. **Abandonner** la piste One-Hot Encoding (gain non significatif,
   coût en complexité pour rien).
2. **Approfondir** la piste "toutes rencontres + split groupé" : passer
   d'un split unique à un vrai `GroupKFold` à 5 plis pour confirmer la
   stabilité du gain (le split unique actuel donne une seule estimation,
   pas encore une moyenne robuste comme le `cross_val_score` déjà utilisé
   pour le modèle de production).
3. Si le gain se confirme en validation croisée groupée, l'intégrer dans
   `train.py`/`etl/pipeline.py` de production — ce qui impliquera de
   revoir la déduplication dans l'ETL, et de vérifier l'impact sur la
   calibration (le jeu de calibration devra, lui aussi, être découpé par
   groupe de patient).
