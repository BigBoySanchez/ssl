# === Stage 1: Clone the SSL repo ===
FROM alpine/git:2.45.2 AS cloner

ARG SSL_COMMIT=main
WORKDIR /src

# Clone the repo and checkout the commit (or branch)
RUN git clone https://github.com/BigBoySanchez/ssl.git . && \
    git checkout $SSL_COMMIT


# === Stage 2: Final runtime image ===
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

WORKDIR /workspace

# Copy code from the previous stage
COPY --from=cloner /src /workspace/ssl

# Create symlinks for shared data/artifacts
RUN mkdir -p /workspace/ssl/artifacts /workspace/ssl/data

# Install dependencies
RUN pip install --no-cache-dir -r /workspace/ssl/requirements.txt

# === W&B setup (optional) ===
ENV WANDB_API_KEY=

CMD ["bash"]

