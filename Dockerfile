FROM python:3.8.10-slim

ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache -r requirements.txt

COPY . .

EXPOSE 8888

CMD [ "python", "main.py" ]