"""
Vérification complémentaire — prévalence (taux de positifs) dans les
deux versions de l'expérience #2, pour s'assurer que le gain d'AUC-PR
observé ne vient pas simplement d'un taux de base différent entre les
jeux de test des versions A et B (l'AUC-PR est sensible à la prévalence).
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/etl")

df = pd.read_csv("/app/data/raw/diabetic_data.csv", low_memory=False)
df = df.replace("?", np.nan)
df = df.drop(columns=[c for c in ["weight", "payer_code", "medical_specialty", "examide", "citoglipton"] if c in df.columns])
df = df[~df["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21])]
df["readmitted_30"] = (df["readmitted"] == "<30").astype(int)

df_dedup = df.drop_duplicates(subset="patient_nbr", keep="first")

print("Prévalence dédup, toutes lignes (version A avant split)     :",
      round(df_dedup["readmitted_30"].mean() * 100, 2), "%")
print("Prévalence toutes rencontres, toutes lignes (version B avant split) :",
      round(df["readmitted_30"].mean() * 100, 2), "%")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_b, te_b = next(gss.split(df, df["readmitted_30"], groups=df["patient_nbr"]))
print("Prévalence TEST version B (groupé, ce qui compte pour l'AUC-PR)     :",
      round(df.iloc[te_b]["readmitted_30"].mean() * 100, 2), "%")

_, te_a = train_test_split(
    np.arange(len(df_dedup)), test_size=0.2, random_state=42,
    stratify=df_dedup["readmitted_30"],
)
print("Prévalence TEST version A (dédup, ce qui compte pour l'AUC-PR)      :",
      round(df_dedup.iloc[te_a]["readmitted_30"].mean() * 100, 2), "%")
