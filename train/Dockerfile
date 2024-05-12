FROM python:3.10
LABEL authors="egorgeller"

WORKDIR /usr/src/app/train-app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "model_train.py"]