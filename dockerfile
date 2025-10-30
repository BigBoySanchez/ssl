FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

WORKDIR /workspace

# Clone Repo
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/BigBoySanchez/ssl

# Make symlinks to artifacts and data to save space
RUN mkdir -p /workspace/ssl/artifacts /workspace/ssl/data

RUN pip install --no-cache-dir -r /workspace/ssl/requirements.txt

CMD ["bash"]
