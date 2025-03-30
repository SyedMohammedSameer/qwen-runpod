FROM runpod/base:0.4.0-cuda11.8.0
# Install Python
RUN apt update && apt install -y python3 python3-pip

# Create a symlink so `python` points to `python3`
RUN ln -s /usr/bin/python3 /usr/bin/python
WORKDIR /app

COPY handler.py /app/
COPY main.py /app/
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt


CMD ["python", "main.py"]