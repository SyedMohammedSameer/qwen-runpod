FROM runpod/base:0.4.0-cuda11.8.0

WORKDIR /app

COPY handler.py /app/
COPY main.py /app/
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt


CMD ["python", "main.py"]