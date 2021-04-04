# Выбор региона для разработки нефтяных месторождений

### Инструменты и навыки
Python: `numpy, pandas, scipy, sklearn, catboost, lightgbm, matplotlib, seaborn`  
Предобработка, EDA, feature engineering, bootstrap, обучение с учителем, кластеризация

## Задача

Необходимо обучить модель, предсказывающую объём запасов нефти в скважине по данным о  её качестве. Добывающая компания планирует с помощью модели выбирать скважины с наибольшими предсказанными запасами нефти. Заказчику важно, чтобы результаты модели были интерпретируемы.  

Кроме того, компании нужно выбрать наиболее перспективный регион для развития. На выбор есть три региона: в каждом регионе компания собирает данные о качестве нефти на 500 случайных скважинах и на основании предсказаний модели выбирает 200 скважин для разработки. Необходимо определить, в каком регионе добыча принесёт наибольшую прибыль и оценить риск убытков. 

### Данные
В наличии были характеристики пробы нефти и объём запасов нефти в скважинах трёх регионов. 
 
### Описание проекта
1. Выполнила предобработку и исследовательский анализ данных. Провела feature engineering: кластеризовала скважины и добавила новый признак — метку кластера.
2. Создала pipeline предобработки данных и обучения модели линейной регрессии. Для каждого региона подобрала на кросс-валидации оптимальные гиперпараметры и нашла предсказания на тестовой выборке. 
3. Техникой bootstrap рассчитала ожидаемую прибыль компании от развития каждого региона, оценила доверительный интервал прибыли и риск убытков компании. Исходя из полученных результатов, выбрала наиболее перспективный регион для развития.

### Основной результат
Ожидаемая выручка компании при разработке рекомендованных моделью скважин в выбранном регионе на 25% выше по сравнению с выручкой при разработке случайных скважин в этом регионе.