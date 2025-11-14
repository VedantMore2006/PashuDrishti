# Dockerfile for PashuDrishti (use with Render web service)
FROM python:3.11-slim

# set env
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

WORKDIR /app

# system deps required for pillow, ffmpeg, and potential CV libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libtiff5-dev \
    libpng-dev \
    libopenjp2-7 \
    libfreetype6-dev \
    libwebp-dev \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# copy requirements & install
COPY requirements.txt /app/requirements.txt

# ensure pip tooling is recent
RUN pip install --upgrade pip setuptools wheel

# install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy app and model (model should be in repo or added via Render secrets/disks)
COPY . /app

# ensure start script is executable
COPY start.sh /start.sh
RUN chmod +x /start.sh

# expose port
EXPOSE ${PORT}

# start
CMD ["/start.sh"]