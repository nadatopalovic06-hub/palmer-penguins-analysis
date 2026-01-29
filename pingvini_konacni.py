# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 19:36:45 2025

@author: nadat
"""
# penguins_classification_clean.py
# Klasifikacija pingvina – čisto rešenje bez data leakage

import pandas as pd
#rad sa tabelama
import matplotlib.pyplot as plt
#crtanje grafika
import seaborn as sns
#lepsi grafikoni baziran na matplotlib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
#sklearn=biblioteka za masinsko ucenje
# 1. UČITAVANJE PODATAKA
penguins = pd.read_csv("C:\\Users\\nadat\\Desktop\\penguins.txt")
#cita fajl i pravi tabelu (DataFrame)
print("Prvih 5 redova:")
print(penguins.head())
#prikazuje prvih pet redova (po defaultu) da vidimo dal se sve dobro ucitalo
print("\nBroj pingvina po vrsti:")
print(penguins['species'].value_counts())
#Broji koliko ima pingvina svake vrste da bi kladse bile balansirane
# 2. VIZUALIZACIJE (EDA)

# Pairplot (numeričke osobine)
sns.pairplot(
    penguins.dropna(),
    hue="species",
    diag_kind="kde",
    markers=['o', 's', 'D'])
plt.suptitle("Odnos numeričkih karakteristika po vrsti", y=1.02)
plt.show()
#crta svaki numericki atribut protiv svakog, boje=vrste pingvina
# Bar chart: ostrvo vs vrsta
plt.figure(figsize=(8, 6))
sns.countplot(data=penguins, x="island", hue="species")
#koliko pingvina svake vrste ima na svakom ostrvu
plt.title("Raspodela pingvina po ostrvu i vrsti")
plt.show()

# Heatmap korelacija
plt.figure(figsize=(8, 6))
sns.heatmap(
    penguins.select_dtypes(include="number").corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm")
plt.title("Korelacija numeričkih osobina")
plt.show()
#koliko su nam numericke kolone povezane 1=jaka veza, 0=slaba veza, -1 obrnuta veza
# Boxplot dužine peraja
plt.figure(figsize=(8, 6))
sns.boxplot(data=penguins, x="species", y="flipper_length_mm")
plt.title("Dužina peraja po vrsti")
plt.show()
#"poredi duzinu peraja po vrsti, ako se boxplotovi ne preklapaju
#"puno odlican prediktor""
# 3. PCA 2D PROJEKCIJA (sa skaliranjem)

penguins_clean = penguins.dropna().copy()
#brisemo redove bez vrednosti jer ML modeli ne znaju da rade sa pravim vrednostima
numerical_cols = [
    'bill_length_mm',
    'bill_depth_mm',
    'flipper_length_mm',
    'body_mass_g']

X_num = penguins_clean[numerical_cols]
X_num_scaled = StandardScaler().fit_transform(X_num)
#da bi sve kolone imale slican opseg
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_num_scaled)
#pcasmanji 4 dimenzije na dve da mozemo crtati
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=penguins_clean['species'])
plt.title("PCA 2D projekcija numeričkih osobina")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
# 4. PREPROCESIRANJE ZA MODELE
# One-hot encoding za kategorijske promenljive
penguins_model = pd.get_dummies(
    penguins_clean,
    columns=['island', 'sex'],
    drop_first=True)
#pretvara tekst u boje
X = penguins_model.drop(columns=['species'])
#x ulaz
y = penguins_model['species']
#y ono sto predvidjamo
# 5. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,#20% podataka ide u test 80% ostaje za train
    stratify=y,
    random_state=42)#sluzi da bi bilo nasumicno
#TRAIN → za učenje modela
#TEST → za proveru koliko je model dobar
# 6. LOGISTIČKA REGRESIJA (PIPELINE)
pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=500))])

pipe_lr.fit(X_train, y_train)
y_pred_lr = pipe_lr.predict(X_test)

print("\n=== LOGISTIČKA REGRESIJA ===")
print("Tačnost:", accuracy_score(y_test, y_pred_lr))
print("\nClassification report:")
print(classification_report(y_test, y_pred_lr))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred_lr))

cv_lr = cross_val_score(pipe_lr, X, y, cv=5)
print("Prosečna CV tačnost (5-fold):", cv_lr.mean())

# 7. POREĐENJE VIŠE MODELA I PREDIKTORA

feature_sets = {
    "Sve kolone": X.columns.tolist(),
    "Bez pola": [c for c in X.columns if not c.startswith('sex_')],
    "Bez ostrva": [c for c in X.columns if not c.startswith('island_')],
    "Samo numerički": numerical_cols}

models = {
    "Logistička regresija": LogisticRegression(max_iter=500),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5)}

rezultati = []

for set_name, cols in feature_sets.items():
    X_tmp = penguins_model[cols]
    y_tmp = y

    for model_name, model in models.items():

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)])

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_tmp, y_tmp,
            test_size=0.2,
            stratify=y_tmp,
            random_state=42)

        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        cv_mean = cross_val_score(pipe, X_tmp, y_tmp, cv=5).mean()

        rezultati.append({
            "Model": model_name,
            "Prediktori": set_name,
            "Tačnost (test)": acc,
            "Prosečna CV tačnost": cv_mean})

rezultati_df = pd.DataFrame(rezultati).sort_values(
    by="Prosečna CV tačnost",
    ascending=False)

print("\n=== UPOREDNI REZULTATI MODELA ===")
print(rezultati_df)

najbolji = rezultati_df.iloc[0]
print("\n=== NAJBOLJI MODEL PO CV ===")
print(f"Model: {najbolji['Model']}")
print(f"Prediktori: {najbolji['Prediktori']}")
print(f"Test tačnost: {najbolji['Tačnost (test)']:.3f}")
print(f"CV tačnost: {najbolji['Prosečna CV tačnost']:.3f}")

