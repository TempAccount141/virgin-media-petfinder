FROM python:3.9.19-slim-bullseye

WORKDIR /app

COPY requirements.txt /app
#COPY pip.conf /etc/pip.conf

RUN pip install -r requirements.txt

COPY run.py /app

RUN chmod 777 /app/run.py

ENTRYPOINT ["python", "run.py"]
