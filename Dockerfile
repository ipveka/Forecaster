FROM python:3.12.0

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r frozen_requirements.txt

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
