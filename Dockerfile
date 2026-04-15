FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    xvfb x11vnc fluxbox xdotool \
    tesseract-ocr scrot \
    libglib2.0-0 libnss3 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libasound2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install-deps chromium && playwright install chromium

COPY . .

ENV DISPLAY=:99
ENV SCREEN_WIDTH=1280
ENV SCREEN_HEIGHT=800
ENV SCREEN_DEPTH=24

EXPOSE 8000 5900

CMD ["bash", "-c", "Xvfb :99 -screen 0 ${SCREEN_WIDTH}x${SCREEN_HEIGHT}x${SCREEN_DEPTH} & sleep 1 && fluxbox & x11vnc -display :99 -forever -nopw -rfbport 5900 -quiet & uvicorn app.main:app --host 0.0.0.0 --port 8000"]
