FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Just install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# No COPY . .  ← not needed
CMD ["bash"]
