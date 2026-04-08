# Use official Python 3.11 image as base (required for openenv-core)
FROM python:3.11-slim

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create a non-privileged user (required for Hugging Face Spaces)
RUN useradd -m -u 1000 user
WORKDIR /app

# Copy all files first
COPY . .

# Install dependencies as ROOT to avoid permission issues with /usr/local
RUN uv pip install --system --no-cache .

# Set permissions for the non-privileged user
RUN chown -R user:user /app
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Ensure the app listens on port 7860 as required by Hugging Face
ENV PORT=7860
EXPOSE 7860

# Start the OpenEnv server using the registered entry point
CMD ["uv", "run", "openenv-server"]
