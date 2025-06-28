# Boston Housing Price Prediction with SGD

Проект по предсказанию цен на жильё в Бостоне с использованием градиентного спуска.

## 📌 О проекте

Простая, но эффективная реализация линейной регрессии с SGD (Stochastic Gradient Descent) для предсказания цен на жильё на основе классического датасета Boston Housing.

## 📌 Команды

1) python -m venv optim_env
2) optim_env\Scripts\activate
3) pip install -r requirements.txt
4) python train_model.py
   На выходе получите:
     Модель (boston_model.pkl)
     Нормализатор (scaler.pkl)
     Метрики R² на обучении и тесте
6) python predict.py
7) python test_model.py

