# Исследование оттока клиентов банка
### Инструменты и навыки
Python: `numpy, pandas, scipy, sklearn, lightgbm, matplotlib, seaborn`  
Предобработка, EDA, статистический анализ, feature engineering, обучение с учителем, бинарная классификация

## Задача
Банк хочет проанализировать поведение клиентов и расторжение договоров с банком. Необходимо построить модель для прогнозирования вероятности ухода клиента из банка в ближайшее время.

### Данные
В наличии были исторические данные о поведении клиентов. Для каждого клиента были известны:
* уникальный идентификатор и фамилия
* пол, возраст, страна проживания
* баланс на счёте, предполагаемая зарплата, количество недвижимости
* кредитный рейтинг
* количество продуктов банка, наличие кредитной карты
* активность клиента
* факт ухода из банка

### Описание проекта
1. Выполнила предобработку и feature engineering. Провела подробный EDA: проанализировала параметры, определяющие уход клиента из банка.
2. Объединила в pipeline предобработку и обучение модели. Создала класс GridSearch для подбора на кросс-валидации помимо стандартных гиперпараметров способа балансировки классов и порога классификации. Построила несколько моделей бинарной классификации на основе алгоритмов LogisticRegression, RandomForest и GB, подобрала на кросс-валидации оптимальные гиперпараметры и выбрала лучшую модель. 

### Основной результат
Итоговая F1-мера лучшей модели  сотавляет 0.65 (для константной модели F1 = 0.34). Доля верно классифицированных клиентов 86%.
