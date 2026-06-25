# Справочник по моделям и метрикам временных рядов

## Содержание
1. [Модели прогнозирования](#модели-прогнозирования)
2. [Метрики качества](#метрики-качества)
3. [Backtesting методы](#backtesting-методы)
4. [Рекомендации по выбору](#рекомендации-по-выбору)

---

## Модели прогнозирования

### 1. Baseline модели

#### 1.1. Naive Forecast
**Описание:** Прогноз = последнее наблюдение

**Формула:** $\hat{y}_{t+1} = y_t$

**Применение:** Отправная точка для сравнения

**Код:**
```python
naive_pred = np.full(len(test), train.iloc[-1])
```

---

#### 1.2. Seasonal Naive
**Описание:** Прогноз = значение из прошлого сезона

**Формула:** $\hat{y}_{t+s} = y_t$ (где s - период сезонности)

**Применение:** Данные с сильной сезонностью

**Код:**
```python
seasonal_naive_pred = train.iloc[-(period-i)]
```

---

#### 1.3. Moving Average
**Описание:** Прогноз = среднее за окно

**Формула:** $\hat{y}_{t+1} = \frac{1}{k}\sum_{i=0}^{k-1}y_{t-i}$

**Применение:** Сглаживание шума

**Код:**
```python
ma_pred = train.tail(window).mean()
```

---

### 2. Статистические модели

#### 2.1. ARIMA (AutoRegressive Integrated Moving Average)

**Описание:** Комбинация авторегрессии, интегрирования и скользящего среднего

**Параметры:**
- **p** - порядок авторегрессии (AR)
- **d** - порядок дифференцирования (I)
- **q** - порядок скользящего среднего (MA)

**Формула:**
$$\phi(B)(1-B)^d y_t = \theta(B)\epsilon_t$$

где:
- $\phi(B)$ - полином AR
- $\theta(B)$ - полином MA
- $B$ - оператор сдвига назад
- $\epsilon_t$ - белый шум

**Применение:** Одномерные стационарные ряды

**Выбор параметров:**
- **p:** из PACF (значимые лаги)
- **d:** порядок дифференцирования для стационарности (обычно 0, 1 или 2)
- **q:** из ACF (значимые лаги)

**Код:**
```python
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(train, order=(p, d, q))
fitted = model.fit()
forecast = fitted.forecast(steps=len(test))
```

---

#### 2.2. SARIMA (Seasonal ARIMA)

**Описание:** ARIMA с учетом сезонности

**Параметры:** (p,d,q)(P,D,Q,s)
- **(p,d,q)** - несезонные параметры
- **(P,D,Q)** - сезонные параметры
- **s** - период сезонности

**Формула:**
$$\phi(B)\Phi(B^s)(1-B)^d(1-B^s)^D y_t = \theta(B)\Theta(B^s)\epsilon_t$$

**Применение:** Данные с сезонностью

**Код:**
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(train, order=(p,d,q), seasonal_order=(P,D,Q,s))
fitted = model.fit()
forecast = fitted.forecast(steps=len(test))
```

---

#### 2.3. Holt-Winters (Exponential Smoothing)

**Описание:** Экспоненциальное сглаживание с трендом и сезонностью

**Типы:**
- **Аддитивная:** для постоянной амплитуды сезонности
- **Мультипликативная:** для растущей амплитуды сезонности

**Формулы (аддитивная):**
- Уровень: $l_t = \alpha(y_t - s_{t-m}) + (1-\alpha)(l_{t-1} + b_{t-1})$
- Тренд: $b_t = \beta(l_t - l_{t-1}) + (1-\beta)b_{t-1}$
- Сезонность: $s_t = \gamma(y_t - l_t) + (1-\gamma)s_{t-m}$

**Применение:** Данные с трендом и сезонностью

**Код:**
```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing
model = ExponentialSmoothing(train, seasonal_periods=12, trend='add', seasonal='add')
fitted = model.fit()
forecast = fitted.forecast(steps=len(test))
```

---

### 3. Машинное обучение

#### 3.1. Linear Regression

**Описание:** Линейная регрессия с лаговыми признаками

**Формула:** $\hat{y}_t = \beta_0 + \beta_1 x_{t-1} + \beta_2 x_{t-2} + ... + \beta_p x_{t-p}$

**Применение:** Базовая ML модель для временных рядов

**Код:**
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
forecast = model.predict(X_test)
```

---

#### 3.2. Random Forest

**Описание:** Ансамбль деревьев решений

**Параметры:**
- **n_estimators** - количество деревьев
- **max_depth** - максимальная глубина
- **min_samples_split** - минимум для разбиения

**Применение:** Нелинейные зависимости, важность признаков

**Код:**
```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
forecast = model.predict(X_test)
```

---

#### 3.3. Gradient Boosting

**Описание:** Последовательное построение деревьев с коррекцией ошибок

**Параметры:**
- **n_estimators** - количество деревьев
- **learning_rate** - скорость обучения
- **max_depth** - глубина деревьев

**Применение:** Высокая точность, сложные паттерны

**Варианты:** XGBoost, LightGBM, CatBoost

**Код:**
```python
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)
forecast = model.predict(X_test)
```

---

### 4. Глубокое обучение

#### 4.1. LSTM (Long Short-Term Memory)

**Описание:** Рекуррентная нейронная сеть с памятью

**Архитектура:**
- Входной слой: последовательность длины seq_length
- LSTM слои: с dropout для регуляризации
- Выходной слой: Dense(1)

**Применение:** Сложные нелинейные зависимости, долгосрочные паттерны

**Код:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32)
forecast = model.predict(X_test)
```

---

## Метрики качества

### 1. MAE (Mean Absolute Error)

**Формула:** 
$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**Интерпретация:** Средняя абсолютная ошибка в единицах измерения

**Преимущества:**
- Легко интерпретируется
- Робастна к выбросам

**Недостатки:**
- Не штрафует большие ошибки сильнее

**Код:**
```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
```

---

### 2. MSE (Mean Squared Error)

**Формула:**
$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**Интерпретация:** Средняя квадратичная ошибка

**Преимущества:**
- Сильно штрафует большие ошибки
- Дифференцируема (удобна для оптимизации)

**Недостатки:**
- Чувствительна к выбросам
- Единицы измерения в квадрате

**Код:**
```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)
```

---

### 3. RMSE (Root Mean Squared Error)

**Формула:**
$$RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

**Интерпретация:** Корень из MSE, в единицах измерения

**Преимущества:**
- В тех же единицах, что и целевая переменная
- Штрафует большие ошибки

**Недостатки:**
- Чувствительна к выбросам

**Код:**
```python
import numpy as np
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

---

### 4. MAPE (Mean Absolute Percentage Error)

**Формула:**
$$MAPE = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

**Интерпретация:** Средняя абсолютная процентная ошибка

**Преимущества:**
- Легко интерпретируется (в процентах)
- Масштабонезависима

**Недостатки:**
- Не определена при $y_i = 0$
- Асимметрична (больше штрафует недооценку)

**Код:**
```python
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

---

### 5. R² (Coefficient of Determination)

**Формула:**
$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

**Интерпретация:** Доля объясненной дисперсии (0 до 1, может быть отрицательным)

**Преимущества:**
- Показывает качество модели относительно среднего
- Масштабонезависима

**Недостатки:**
- Может быть отрицательной для плохих моделей
- Не всегда подходит для временных рядов

**Код:**
```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
```

---

### Сравнение метрик

| Метрика | Единицы | Чувствительность к выбросам | Интерпретация |
|---------|---------|----------------------------|---------------|
| MAE | Исходные | Низкая | Средняя ошибка |
| MSE | Квадрат исходных | Высокая | Квадрат ошибки |
| RMSE | Исходные | Высокая | Типичная ошибка |
| MAPE | Проценты | Средняя | % ошибка |
| R² | Безразмерная | Средняя | Качество модели |

---

## Backtesting методы

### 1. Train-Test Split

**Описание:** Простое разделение на обучение и тест

**Применение:** Быстрая оценка

**Недостатки:** Одна точка оценки

**Код:**
```python
train_size = int(len(data) * 0.8)
train = data[:train_size]
test = data[train_size:]
```

---

### 2. Walk-Forward Validation

**Описание:** Последовательное расширение обучающей выборки

**Схема:**
```
Fold 1: [Train] [Test]
Fold 2: [Train----] [Test]
Fold 3: [Train---------] [Test]
```

**Применение:** Имитация реального использования

**Код:**
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(data):
    train = data[train_idx]
    test = data[test_idx]
    # обучение и оценка
```

---

### 3. Rolling Window

**Описание:** Фиксированный размер обучающей выборки

**Схема:**
```
Fold 1: [Train] [Test]
Fold 2:   [Train] [Test]
Fold 3:     [Train] [Test]
```

**Применение:** Когда старые данные теряют актуальность

**Код:**
```python
window_size = 100
for i in range(window_size, len(data) - test_size):
    train = data[i-window_size:i]
    test = data[i:i+test_size]
    # обучение и оценка
```

---

## Рекомендации по выбору

### Выбор модели

**Начинайте с простого:**
1. Baseline (Naive, Seasonal Naive)
2. Статистические (ARIMA, SARIMA)
3. ML (Random Forest, Gradient Boosting)
4. Deep Learning (LSTM)

**Критерии выбора:**

| Характеристика данных | Рекомендуемая модель |
|----------------------|---------------------|
| Линейный тренд, без сезонности | ARIMA, Linear Regression |
| Сезонность | SARIMA, Holt-Winters |
| Нелинейные зависимости | Random Forest, Gradient Boosting |
| Множественные сезонности | LSTM, Prophet |
| Экзогенные переменные | SARIMAX, ML модели |
| Малый объем данных | ARIMA, SARIMA |
| Большой объем данных | LSTM, Gradient Boosting |

---

### Выбор метрики

**Критерии выбора:**

| Ситуация | Рекомендуемая метрика |
|----------|----------------------|
| Нужна интерпретируемость | MAE, RMSE |
| Есть выбросы | MAE |
| Важны большие ошибки | RMSE, MSE |
| Сравнение разных масштабов | MAPE, R² |
| Оптимизация модели | MSE (дифференцируема) |

**Рекомендация:** Используйте несколько метрик для комплексной оценки!

---

### Выбор backtesting

| Ситуация | Метод |
|----------|-------|
| Быстрая оценка | Train-Test Split |
| Надежная оценка | Walk-Forward |
| Нестационарные данные | Rolling Window |
| Малый объем данных | Leave-One-Out |

---

## Практические советы

### 1. Последовательность действий
1. Начните с EDA и baseline
2. Проверьте стационарность
3. Постройте несколько моделей
4. Сравните по метрикам
5. Примените backtesting
6. Анализируйте остатки
7. Выберите лучшую модель

### 2. Оптимизация гиперпараметров

```python
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.1, 0.3]
}

tscv = TimeSeriesSplit(n_splits=5)
grid = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
```

### 3. Ансамблирование

```python
# Простое усреднение
ensemble_pred = (pred1 + pred2 + pred3) / 3

# Взвешенное усреднение
weights = [0.5, 0.3, 0.2]
ensemble_pred = weights[0]*pred1 + weights[1]*pred2 + weights[2]*pred3

# Stacking
from sklearn.ensemble import StackingRegressor
estimators = [('rf', rf_model), ('gb', gb_model)]
stacking = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
```

---

## Заключение

Выбор модели и метрик зависит от:
- Характеристик данных
- Бизнес-требований
- Доступных ресурсов
- Интерпретируемости

**Золотое правило:** Всегда сравнивайте несколько подходов и выбирайте на основе объективных метрик и backtesting!

