FROM python:3.10-slim

# Install system dependencies for OpenGL, Ursina and X11
RUN apt-get update && apt-get install -y \
    freeglut3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libxrender1 \
    libxext6 \
    libx11-6 \
    libxinerama1 \
    libxcursor1 \
    libxrandr2 \
    libxi6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set PYTHONPATH to include src
ENV PYTHONPATH=/app/src

# Default command is to run the PPO training
CMD ["python", "scripts/train_rl.py"]
