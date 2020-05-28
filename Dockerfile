# docker build -t classify-url  .
FROM python:3.5.2

MAINTAINER trungdq
COPY . /app/
# COPY search_server.py /app/search_server.py
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install gunicorn
CMD gunicorn --bind=0.0.0.0:1995 --workers=1 classify_service:app
