FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

WORKDIR /workspace

# Just install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# No COPY . .  ← not needed
CMD ["bash"]
