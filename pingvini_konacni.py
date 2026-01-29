uins"""
Penguin Species Classification
Author: Nada Topalović

This project demonstrates an end-to-end machine learning workflow:
- Exploratory Data Analysis (EDA)
- Feature preprocessing and encoding
- Dimensionality reduction (PCA)
- Model training using pipelines
- Model evaluation with cross-validation

Special attention is paid to preventing data leakage.
"""
# Data handling and visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Machine learning tools
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# Load dataset
penguins = pd.read_csv("penguins.txt")
# Quick inspection of the dataset
print("First five rows:")
print(penguins.head())
print("\nNumber of samples per species:")
print(penguins['species'].value_counts())


# Exploratory Data Analysis (EDA)
# Pairwise relationships between numerical features, colored by species
sns.pairplot(
    penguins.dropna(),
    hue="species",
    diag_kind="kde",
    markers=['o', 's', 'D'])
plt.suptitle("Pairwise relationships between numerical features", y=1.02)
plt.show()
plt.figure(figsize=(8, 6))
# Distribution of species across islands
sns.countplot(data=penguins, x="island", hue="species")
plt.title("Species distribution across islands")
plt.show()

plt.figure(figsize=(8, 6))
# Correlation matrix of numerical features
sns.heatmap(
    penguins.select_dtypes(include="number").corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm")
plt.title("Correlation of numerical features")
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(data=penguins, x="species", y="flipper_length_mm")
plt.title("Flipper length by species")
plt.show()

# Remove missing values
penguins_clean = penguins.dropna().copy()
# Select numerical features
numerical_cols = [
    'bill_length_mm',
    'bill_depth_mm',
    'flipper_length_mm',
    'body_mass_g']

X_num = penguins_clean[numerical_cols]
X_num_scaled = StandardScaler().fit_transform(X_num)

# PCA for 2D geometric visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_num_scaled)
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=penguins_clean['species'])
plt.title("2D PCA projection of numerical features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# One-hot encode categorical features
penguins_model = pd.get_dummies(
    penguins_clean,
    columns=['island', 'sex'],
    drop_first=True)
X = penguins_model.drop(columns=['species'])
y = penguins_model['species']

# Stratified train-test split to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42)

# Pipeline ensures scaling is applied only on training data
# Prevents data leakage during evaluation
pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=500))])

pipe_lr.fit(X_train, y_train)
y_pred_lr = pipe_lr.predict(X_test)

print("\n=== LOGISTIC REGRESSION ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\nClassification report:")
print(classification_report(y_test, y_pred_lr))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred_lr))

cv_lr = cross_val_score(pipe_lr, X, y, cv=5)
print("Mean CV accuracy (5-fold):", cv_lr.mean())


feature_sets = {
    "All feautures": X.columns.tolist(),
    "Without sex": [c for c in X.columns if not c.startswith('sex_')],
    "Without island": [c for c in X.columns if not c.startswith('island_')],
    "Numerical only": numerical_cols}

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
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

print("\n=== MODEL COMPARISON RESULTS ===")
print(rezultati_df)

najbolji = rezultati_df.iloc[0]
print("\n=== BEST MODEL BY CV ===")
print(f"Model: {najbolji['Model']}")
print(f"Prediktori: {najbolji['Prediktori']}")
print(f"Test tačnost: {najbolji['Tačnost (test)']:.3f}")
print(f"CV tačnost: {najbolji['Prosečna CV tačnost']:.3f}")




