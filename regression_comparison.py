
# 1. Importation des bibliothèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score


# 2. Chargement des données
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Dimensions des données :", X.shape)
print("\nAperçu des données :")
print(X.head())


# 3. Division des données
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. Définition des modèles
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": make_pipeline(StandardScaler(), Ridge()),
    "Lasso": make_pipeline(StandardScaler(), Lasso(max_iter=10000)),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    ),
    "SVR": make_pipeline(StandardScaler(), SVR())
}


# 5. Entraînement et évaluation
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results.append({
        "Modèle": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })


# 6. Affichage des résultats
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="RMSE")

print("\nComparaison des modèles :")
print(results_df.round(3).to_string(index=False))

results_df.to_csv("resultats_comparaison.csv", index=False)


# 7. Visualisation des performances
plt.figure(figsize=(10, 6))
plt.bar(results_df["Modèle"], results_df["RMSE"])
plt.title("Comparaison des modèles de régression")
plt.xlabel("Modèles")
plt.ylabel("RMSE")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("comparaison_modeles.png")
plt.show()
