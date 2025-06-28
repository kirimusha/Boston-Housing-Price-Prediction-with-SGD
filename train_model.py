import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

data = pd.read_csv('housing.csv', sep=r'\s+', header=None)
X = data.iloc[:, :-1]  # признаки
y = data.iloc[:, -1]   # PRICE

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = SGDRegressor(
    loss='squared_error',  
    learning_rate='constant',
    eta0=0.01,
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"R² на обучении: {train_score:.3f}")
print(f"R² на тесте: {test_score:.3f}")

joblib.dump(model, "boston_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Модель и scaler сохранены!")