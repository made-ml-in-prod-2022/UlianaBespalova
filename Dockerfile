FROM python:3.6.10

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ml_project /ml_project
COPY setup.py /setup.py
RUN pip install -e .


COPY online_inference /online_inference

WORKDIR /online_inference/

ENV PATH_TO_MODEL_LR="models/model_lr.pkl"
ENV PATH_TO_MODEL_RF="models/model_rf.pkl"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]