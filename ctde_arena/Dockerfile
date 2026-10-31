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

# Built from the repository root (docker build -f ctde_arena/Dockerfile .): the engine is the shared
# src/marl_arena, and this tree's src/ is a shim that resolves it at ../src (audit item A-11).
WORKDIR /repo/ctde_arena

COPY ctde_arena/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src /repo/src
COPY ctde_arena /repo/ctde_arena

ENV PYTHONPATH=/repo/ctde_arena/src

# Default command is to run the PPO training
CMD ["python", "scripts/train_rl.py"]
