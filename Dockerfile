ARG VERSION=${VERSION:-3.12}

FROM python:${VERSION}

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y libgl1

EXPOSE 8000

COPY . .
