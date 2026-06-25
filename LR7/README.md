# Отчет по практической работе
## Основы Рекомендательных Систем

**Тема:** Введение в Рекомендательные Системы (РС)

**Цель работы:** Практическое освоение двух базовых подходов к построению РС: контентной фильтрации (Content-Based) и коллаборативной фильтрации (User-Based CF).

---

## 1. Подготовка рабочей среды

Для выполнения работы были установлены следующие библиотеки:

```bash
pip install pandas numpy scikit-learn scikit-surprise matplotlib seaborn
```

| Библиотека | Назначение |
|------------|------------|
| **pandas** | Работа с данными (таблицы, DataFrame) |
| **numpy** | Численные вычисления |
| **scikit-learn** | Расчет косинусного сходства |
| **scikit-surprise** | Коллаборативная фильтрация |
| **matplotlib** | Визуализация данных |
| **seaborn** | Улучшенная визуализация |

---

## 2. Задание 1: Контентная Фильтрация (Content-Based Filtering)

### 2.1. Исходные данные

Была создана матрица признаков для 5 фильмов по 4 жанрам:

| Film_ID | Title | Action | Comedy | Drama | Sci-Fi |
|---------|-------|--------|--------|-------|--------|
| 1 | Die Hard | 1 | 0 | 0 | 0 |
| 2 | The Martian | 0 | 0 | 0 | 1 |
| 3 | Pulp Fiction | 0 | 0 | 1 | 0 |
| 4 | Guardians of the Galaxy | 1 | 1 | 0 | 1 |
| 5 | La La Land | 0 | 0 | 1 | 0 |

### 2.2. Реализация

#### Шаг 1: Создание матрицы признаков

```python
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'Film_ID': [1, 2, 3, 4, 5],
    'Title': ['Die Hard', 'The Martian', 'Pulp Fiction', 'Guardians of the Galaxy', 'La La Land'],
    'Action': [1, 0, 0, 1, 0],
    'Comedy': [0, 0, 0, 1, 0],
    'Drama': [0, 0, 1, 0, 1],
    'Sci-Fi': [0, 1, 0, 1, 0]
}

movies_df = pd.DataFrame(data).set_index('Film_ID')
feature_matrix = movies_df[['Action', 'Comedy', 'Drama', 'Sci-Fi']]

print("Матрица Признаков:")
print(feature_matrix)
```

**Результат:**
```
Матрица Признаков:
         Action  Comedy  Drama  Sci-Fi
Film_ID                               
1             1       0      0       0
2             0       0      0       1
3             0       0      1       0
4             1       1      0       1
5             0       0      1       0
```

#### Шаг 2: Расчет косинусного сходства

```python
cosine_sim_matrix = cosine_similarity(feature_matrix)
cosine_sim_df = pd.DataFrame(
    cosine_sim_matrix,
    index=movies_df['Title'],
    columns=movies_df['Title']
)

print("\nМатрица Косинусного Сходства:")
print(cosine_sim_df.round(3))
```

**Результат:**
```
Матрица Косинусного Сходства:
Title                    Die Hard  The Martian  Pulp Fiction  Guardians of the Galaxy  La La Land
Title                                                                                          
Die Hard                    1.000        0.000           0.0                     0.577         0.0
The Martian                 0.000        1.000           0.0                     0.577         0.0
Pulp Fiction                0.000        0.000           1.0                     0.000         1.0
Guardians of the Galaxy     0.577        0.577           0.0                     1.000         0.0
La La Land                  0.000        0.000           1.0                     0.000         1.0
```

#### Шаг 3: Функция рекомендации

```python
def get_recommendations(title, cosine_sim_df, movies_df, top_n=2):
    sim_scores = cosine_sim_df[title]
    sim_scores = sim_scores.sort_values(ascending=False)
    top_recommendations = sim_scores.iloc[1:top_n+1]
    return list(top_recommendations.index)

# Примеры рекомендаций
print(f"Рекомендации для 'Die Hard': {get_recommendations('Die Hard', cosine_sim_df, movies_df, top_n=2)}")
print(f"Рекомендации для 'The Martian': {get_recommendations('The Martian', cosine_sim_df, movies_df, top_n=2)}")
print(f"Рекомендации для 'Pulp Fiction': {get_recommendations('Pulp Fiction', cosine_sim_df, movies_df, top_n=2)}")
print(f"Рекомендации для 'Guardians of the Galaxy': {get_recommendations('Guardians of the Galaxy', cosine_sim_df, movies_df, top_n=2)}")
print(f"Рекомендации для 'La La Land': {get_recommendations('La La Land', cosine_sim_df, movies_df, top_n=2)}")
```

**Результат:**
```
Рекомендации для 'Die Hard': ['Guardians of the Galaxy', 'The Martian']
Рекомендации для 'The Martian': ['Guardians of the Galaxy', 'Die Hard']
Рекомендации для 'Pulp Fiction': ['La La Land', 'Die Hard']
Рекомендации для 'Guardians of the Galaxy': ['Die Hard', 'The Martian']
Рекомендации для 'La La Land': ['La La Land', 'Die Hard']
```

### 2.3. Ответы на вопросы

**1. Какое сходство (число) система рассчитала между фильмами "The Martian" и "Guardians of the Galaxy"? Объясните, почему.**

Сходство между "The Martian" и "Guardians of the Galaxy" составляет **0.577** (или 1/√3).

**Объяснение:**
- "The Martian" имеет признаки: [0, 0, 0, 1] (только Sci-Fi)
- "Guardians of the Galaxy" имеет признаки: [1, 1, 0, 1] (Action, Comedy, Sci-Fi)

Косинусное сходство вычисляется как:
$$\text{cosine\_similarity} = \frac{A \cdot B}{||A|| \times ||B||}$$

- Скалярное произведение: 0×1 + 0×1 + 0×0 + 1×1 = 1
- Норма "The Martian": √(0² + 0² + 0² + 1²) = 1
- Норма "Guardians": √(1² + 1² + 0² + 1²) = √3 ≈ 1.732

Сходство = 1 / (1 × 1.732) = 0.577

**2. Какой главный недостаток Content-Based фильтрации вы видите на примере фильма "La La Land"?**

Главный недостаток Content-Based фильтрации — **ограниченность рекомендаций**.

На примере "La La Land" видно, что:
- "La La Land" имеет признаки: [0, 0, 1, 0] (только Drama)
- Единственный фильм с таким же жанром — "Pulp Fiction" (тоже только Drama)
- Система рекомендует "Pulp Fiction" и, в качестве второго варианта, "Die Hard" (сходство = 0)

**Проблема:** Content-Based система не может предложить разнообразные рекомендации. Она рекомендует только похожие по жанрам фильмы, но не учитывает:
- Вкусы других пользователей
- Неочевидные связи между фильмами
- Фильмы, которые могут понравиться пользователю, но не похожи по жанрам

Если пользователь любит "La La Land", ему могут понравиться и другие музыкальные фильмы, романтические комедии или фильмы с похожей атмосферой, но система этого не учитывает.

---

## 3. Задание 2: Коллаборативная Фильтрация (User-Based CF)

### 3.1. Реализация

#### Шаг 1: Загрузка и подготовка данных

```python
from surprise import Dataset, Reader, KNNWithMeans, accuracy
from surprise.model_selection import train_test_split

# Загрузка данных MovieLens 100k
data = Dataset.load_builtin('ml-100k')

# Разделение на обучающую и тестовую выборки (80% / 20%)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

print(f"Размер обучающей выборки: {trainset.n_ratings} оценок")
print(f"Размер тестовой выборки: {len(testset)} оценок")
```

**Результат:**
```
Размер обучающей выборки: 80000 оценок
Размер тестовой выборки: 20000 оценок
```

#### Шаг 2: Обучение модели User-Based CF

```python
# Настройка сходства (корреляция Пирсона)
sim_options = {
    'name': 'pearson',
    'user_based': True
}

# Инициализация и обучение модели
algo = KNNWithMeans(k=40, sim_options=sim_options)
algo.fit(trainset)
print("\nМодель User-Based CF (Пирсон) обучена.")
```

**Результат:**
```
Computing the pearson similarity matrix...
Done computing similarity matrix.

Модель User-Based CF (Пирсон) обучена.
```

#### Шаг 3: Оценка качества модели

```python
# Предсказание на тестовой выборке
predictions = algo.test(testset)

# Расчет RMSE
rmse = accuracy.rmse(predictions, verbose=True)
print(f"\nRMSE на тестовой выборке: {rmse:.4f}")
```

**Результат:**
```
RMSE: 0.9492

RMSE на тестовой выборке: 0.9492
```

#### Шаг 4: Предсказание для конкретного пользователя

```python
user_id = '196'
item_id = '302'  # Фильм "L.A. Confidential"

prediction = algo.predict(user_id, item_id, verbose=True)
print(f"\nПредсказанная оценка: {prediction.est:.3f}")
```

**Результат:**
```
user: 196        item: 302        r_ui = None   est = 4.18   {'actual_k': 40, 'was_impossible': False}

Предсказанная оценка: 4.175
```

#### Шаг 5: Сравнение с косинусным сходством

```python
# Настройка с косинусным сходством
sim_options_cosine = {
    'name': 'cosine',
    'user_based': True
}

algo_cosine = KNNWithMeans(k=40, sim_options=sim_options_cosine)
algo_cosine.fit(trainset)
predictions_cosine = algo_cosine.test(testset)
rmse_cosine = accuracy.rmse(predictions_cosine, verbose=True)

print(f"\nRMSE с косинусным сходством: {rmse_cosine:.4f}")
print(f"RMSE с корреляцией Пирсона: {rmse:.4f}")
print(f"Разница: {rmse - rmse_cosine:.4f}")
```

**Результат:**
```
Computing the cosine similarity matrix...
Done computing similarity matrix.
RMSE: 0.9538

RMSE с косинусным сходством: 0.9538
RMSE с корреляцией Пирсона: 0.9492
Разница: -0.0047
```

### 3.2. Визуализация матрицы сходства пользователей

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Получение матрицы сходства
sim_matrix = algo.sim

plt.figure(figsize=(10, 8))
sns.heatmap(sim_matrix[:50, :50], cmap='coolwarm', center=0, 
            cbar_kws={'label': 'Сходство'})
plt.title('Матрица сходства пользователей (первые 50)')
plt.xlabel('Пользователи')
plt.ylabel('Пользователи')
plt.show()
```

### 3.3. Ответы на вопросы

**1. Что означает полученное вами значение RMSE? Если бы RMSE было равно 0.5, а не полученному вами значению, что бы это значило для точности предсказаний?**

**Значение RMSE = 0.9492** означает, что в среднем предсказанная оценка отличается от реальной примерно на 0.95 балла (по шкале от 1 до 5).

RMSE показывает стандартное отклонение ошибок предсказания. Чем меньше RMSE, тем точнее модель.

**Если бы RMSE было равно 0.5:**
- Это означало бы значительно более точные предсказания
- Ошибка в среднем составляла бы всего 0.5 балла
- Модель была бы в ~2 раза точнее текущей
- Пользователи получали бы более релевантные рекомендации

**2. Как изменится значение RMSE, если в настройках `sim_options` заменить `'pearson'` на `'cosine'` (Косинусное сходство)? Объясните, почему Пирсон часто дает лучший результат, чем Косинусное сходство, в контексте оценок пользователей.**

**Изменение RMSE:**
- Пирсон: 0.9492
- Косинус: 0.9538
- Разница: -0.0047 (Пирсон лучше на 0.0047)

**Почему Пирсон часто лучше:**

1. **Учет средних значений:** Корреляция Пирсона учитывает среднюю оценку пользователя. Если один пользователь ставит оценки выше среднего, а другой — ниже, Пирсон это нормализует.

2. **Масштабирование:** Пирсон не зависит от масштаба оценок. Он измеряет линейную связь между оценками, а не просто расстояние.

3. **Пример:** 
   - Пользователь A ставит оценки: 4, 5, 5, 4
   - Пользователь B ставит оценки: 2, 3, 3, 2
   - Косинусное сходство покажет низкое сходство (разные значения)
   - Пирсон покажет высокое сходство (оба пользователя имеют одинаковый паттерн: высокие/низкие оценки для одних и тех же фильмов)

4. **Косинусное сходство** чувствительно к абсолютным значениям, а не к относительным паттернам, что в контексте оценок пользователей является недостатком.

---

## 4. Выводы

### Сравнение подходов

| Критерий | Content-Based | User-Based CF |
|----------|---------------|---------------|
| **Принцип** | Рекомендует похожие по характеристикам объекты | Рекомендует то, что нравится похожим пользователям |
| **Данные** | Требует описания объектов (жанры, теги) | Требует только оценок пользователей |
| **Холодный старт** | Нет проблем для новых объектов | Проблемы с новыми пользователями/объектами |
| **Разнообразие** | Ограниченное (только похожие объекты) | Высокое (неожиданные открытия) |
| **Интерпретируемость** | Высокая (понятно, почему рекомендовано) | Средняя (основано на поведении других) |
| **Масштабируемость** | Хорошая | Плохая (O(n²) для больших данных) |

### Какой подход перспективнее для крупного онлайн-кинотеатра?

**User-Based CF** кажется более перспективным по следующим причинам:

1. **Разнообразие рекомендаций:** Пользователи открывают для себя фильмы, которые никогда бы не нашли через жанровый поиск.

2. **Учёт тонких предпочтений:** CF улавливает неочевидные связи между фильмами, которые не отражены в жанрах.

3. **Адаптивность:** Рекомендации меняются вместе с изменением вкусов пользователя.

4. **Социальное доказательство:** Рекомендации основаны на поведении реальных людей, что повышает доверие.

**Но есть нюансы:**
- Для борьбы с холодным стартом нужно гибридное решение (CF + Content-Based)
- Требуется масштабирование (можно использовать алгоритмы вроде SVD или нейросетей)
- Нужна регулярная переобучение модели

**Рекомендация:** Использовать гибридный подход, где:
- Content-Based используется для новых фильмов и начальных рекомендаций
- User-Based CF — для персонализированных рекомендаций активным пользователям
- Дополнительно — модели на основе матричной факторизации (SVD, ALS)

---

## 5. Приложение: Полный код

```python
# Импорт библиотек
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, KNNWithMeans, accuracy
from surprise.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

print("Библиотеки успешно загружены!")

# ============ Задание 1: Content-Based ============

# Создание данных
data = {
    'Film_ID': [1, 2, 3, 4, 5],
    'Title': ['Die Hard', 'The Martian', 'Pulp Fiction', 
              'Guardians of the Galaxy', 'La La Land'],
    'Action': [1, 0, 0, 1, 0],
    'Comedy': [0, 0, 0, 1, 0],
    'Drama': [0, 0, 1, 0, 1],
    'Sci-Fi': [0, 1, 0, 1, 0]
}

movies_df = pd.DataFrame(data).set_index('Film_ID')
feature_matrix = movies_df[['Action', 'Comedy', 'Drama', 'Sci-Fi']]

# Расчет сходства
cosine_sim_matrix = cosine_similarity(feature_matrix)
cosine_sim_df = pd.DataFrame(
    cosine_sim_matrix,
    index=movies_df['Title'],
    columns=movies_df['Title']
)

# Функция рекомендаций
def get_recommendations(title, cosine_sim_df, movies_df, top_n=2):
    sim_scores = cosine_sim_df[title]
    sim_scores = sim_scores.sort_values(ascending=False)
    top_recommendations = sim_scores.iloc[1:top_n+1]
    return list(top_recommendations.index)

# ============ Задание 2: User-Based CF ============

# Загрузка данных
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Обучение модели
sim_options = {'name': 'pearson', 'user_based': True}
algo = KNNWithMeans(k=40, sim_options=sim_options)
algo.fit(trainset)

# Оценка качества
predictions = algo.test(testset)
rmse = accuracy.rmse(predictions, verbose=True)

# Предсказание для конкретного пользователя
prediction = algo.predict('196', '302')
print(f"\nПредсказанная оценка: {prediction.est:.3f}")
```

---
