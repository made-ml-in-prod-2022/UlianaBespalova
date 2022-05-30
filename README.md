# ДЗ2
## Беспалова Ульяна, Технопарк, ML-21

**Датасет:**

https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci


**Установка зависимостей:**
~~~
pip install -r requirements.txt
pip install -e .
~~~

**Экспорт переменных среды и запуск сервера:**
~~~
export PATH_TO_MODEL_LR=online_inference/models/model_lr.pkl
export PATH_TO_MODEL_RF=online_inference/models/model_rf.pkl

uvicorn --app-dir=online_inference app:app --host 0.0.0.0 --port 8000
~~~

**Тестирование:**
~~~
python -m online_inference.tests
~~~

**Скрипт:**
~~~
python -m online_inference.script --config=config/config_get_data.yaml
~~~
где config - путь до конфига с параметрами скачивания данных с S3


**Локальный запуск докера:**
~~~
docker build -t hw2:v1 .
docker run -p 8000:8000 hw2:v1
~~~

**Запуск с помощью DokerHub:**
~~~
docker pull firthefir/hw2:v1
docker run -p 8000:8000 firthefir/hw2:v1
~~~



Архитектура проекта
==============================

    ├── LICENSE
    ├── Makefile           
    ├── README.md          
    ├── data               <- Папка c результатами выполнения скрипта
    │
    ├── notebooks          <- Ноутбук с разведочным анализом данных
    │
    ├── configs            <- Конфиги с параметрами моделей
    │
    ├── requirements.txt   <- Файл с зависимостями
    │
    ├── ml_project         <- Исходный код проекта
    │   ├── __init__.py    
    │   │
    │   ├── data           <- Скрипт для загрузки и разделения данных
    │   │   └── get_data.py
    │   │
    │   ├── features       <- Трансформер для обработки датасета
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Скрипт для работы с моделью
    │   │   └── model_fit_predict.py
    │   │
    │   ├── enities        <- Параметры модели  датасета в формате DataClass
    │   │   ├── features_params.py
    │   │   ├── model_params.py
    │   │   └── split_params.py
    │   │
    │   └── tests          <- Юнит тесты отдельных модулей и тест пайплайна для обучения модели
    |
    ├── online_inference   <- Код для онлайн-использования моделей
    │   ├── app.py         <- Точка входа: обработчики запросов и валидация
    |   |
    │   ├── script.py      <- Скрипт для создания запросов серверу
    │   |
    │   ├── tests.py    
    │   │
    │   ├── data_utils    
    │   │   └── data_utils.py  <- Утилиты для работы с моделью
    │   │
    │   ├── data           <- Директория для скаченных датасетов
    │   │  
    │   └── models         <- Модели, используемые онлайн
    │  
    ├── Dockerfile
    │
    └── setup.py            


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


### Задание


В прошлом ДЗ вы обучили модель для решения задачи классификации(по умолчанию использовался датасет https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci), ваше следующее задание, это обернуть ее в вид, пригодный для использования в режиме онлайн

Весь код должен находиться в том же репозитории, но в отдельной папке online_inference. 

**Основная часть**

1) + Оберните inference вашей модели в rest сервис на FastAPI, должен быть endpoint /predict (3 балла)
2) + Напишите endpoint /health (1 балл), должен возращать 200, если ваша модель готова к работе (такой чек особенно актуален если делаете доп задание про скачивание из хранилища) 
3) + Напишите unit тест для /predict  (3 балла) (https://fastapi.tiangolo.com/tutorial/testing/, https://flask.palletsprojects.com/en/1.1.x/testing/)

4) + Напишите скрипт, который будет делать запросы к вашему сервису -- 2 балла

5) + Напишите dockerfile, соберите на его основе образ и запустите локально контейнер(docker build, docker run), внутри контейнера должен запускать сервис, написанный в предущем пункте, закоммитьте его, напишите в readme корректную команду сборки (4 балл)

6) + Опубликуйте образ в https://hub.docker.com/, используя docker push (вам потребуется зарегистрироваться) (+2 балла)

7) + Напишите в readme корректные команды docker pull/run, которые должны привести к тому, что локально поднимется на inference ваша модель (1 балл)
   Убедитесь, что вы можете протыкать его скриптом из пункта 3

8) + Проведите самооценку (распишите в реквесте какие пункты выполнили и на сколько баллов, укажите сумму баллов) -- 1 балл


**Дополнительная часть**: 
1) + Ваш сервис скачивает модель из S3 или любого другого хранилища при старте, путь для скачивания передается через переменные окружения (+2 доп балла)

2) Оптимизируйте размер docker image (+2 доп балла)

3) + Сделайте валидацию входных данных https://pydantic-docs.helpmanual.io/usage/validators/
