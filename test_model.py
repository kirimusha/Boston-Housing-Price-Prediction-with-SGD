import joblib
import pandas as pd

data = pd.read_csv('housing.csv', sep=r'\s+', header=None)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
scaler = joblib.load("scaler.pkl")
model = joblib.load("boston_model.pkl")

X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)

# Вывод первых 5 предсказаний
print("Тестовые значения vs Предсказания:")
for true, pred in zip(y[:5], y_pred[:5]):
    print(f"Реальное: ${true:.2f} | Предсказанное: ${pred:.2f}")