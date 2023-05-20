FROM python:3.10.6-buster

WORKDIR /app

COPY requirements_prod.txt requirements_prod.txt
RUN pip install -r requirements_prod.txt

COPY pill_pic pill_pic
COPY setup.py setup.py
RUN pip install .

# CMD ["uvicorn", "pill_pic.fast:app", "--host", "0.0.0.0", "--port", "8000"]
CMD uvicorn pill_pic.fast:app --host 0.0.0.0 --port $PORT
