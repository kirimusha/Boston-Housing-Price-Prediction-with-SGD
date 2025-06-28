import pandas as pd
import joblib

model = joblib.load("boston_model.pkl")
scaler = joblib.load("scaler.pkl")

# Пример данных. Формат: [CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]
new_data = pd.DataFrame([[
    0.00632, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296.0, 15.3, 396.9, 4.98
]])

new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)

print(f"Предсказанная цена дома: ${prediction[0]:.2f}")