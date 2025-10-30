FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

WORKDIR /workspace

# Clone Repo
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/BigBoySanchez/ssl

# Make symlinks to artifacts and data to save space
RUN mkdir -p ../artifacts ../data && \
    ln -s /artifacts /workspace/ssl/artifacts && \
    ln -s /data /workspace/ssl/data

RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash"]
