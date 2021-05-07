# Определение температуры стали

### Инструменты и навыки
Python: `numpy, pandas, scipy, sklearn, lightgbm, matplotlib, seaborn`  
Предобработка, EDA, статистический анализ, feature engineering, обучение с учителем, анализ остатков
## Задача

Необходимо построить модель, предсказывающую температуру стали после легирования. Металлургический комбинат планирует с помощью модели оптимизировать производство и уменьшить потребление электроэнергии на этапе обработки стали. 

### Данные
В наличии были данные об обработке 3200 партий стали:
* нагрев: время начала и окончания нагрева, активная и реактивная мощность (каждую партию нагревают в среднем 4 раза)
* легирование: время подачи ряда сыпучих и проволочных материалов, а также их объёмы (в каждую партию добавляют в среднем 5 легирующих примесей)
* замеры температуры: время замера и само измеренное значение (температура партии замеряется в среднем 4 раза в ходе обработки)
* данные о продувке стали инертным газом
 
### Описание проекта
1. Выполнила предобработку сильно загрянённых данных. Провела feature engineering и подробный исследовательский анализ данных на основе статистических и графических методов. 
2. Создала pipeline предобработки данных и обучения модели. Для двух моделей машинного обучения (линейной регрессиии и градиентного бустинга) подобрала на кросс-валидации оптимальные гиперпараметры. Выбрала лучшую модель и оценила её качество на тестовой выборке. Провела подробный анализ весов признаков и остатков модели, на основе которого приняла решение о дальнейшей доработке модели.
3. Изменила подход к добавлению объектов в обучающую выборку и обучила модель на обновлённой выборке. Снова подобрала гиперпараметры моделей, оценила качество на тестовой выборке и проанализирорвала предсказания новой модели. Сделала вывод о целесообразности применения рассмотренного подхода.

### Основной результат
Средняя абсолютная ошибка выбранной модели составляет 5.4$^{\circ}$C, что почти в 2 раза меньше по сравнению с константной моделью (MAE = 9.6$^{\circ}$C).