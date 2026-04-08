# Use official Python 3.9 image as base
FROM python:3.9-slim

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create a non-privileged user (required for Hugging Face Spaces)
RUN useradd -m -u 1000 user
WORKDIR /app
RUN chown user:user /app

USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Install dependencies using the lockfile for reproducibility
COPY --chown=user pyproject.toml uv.lock ./
RUN uv pip install --no-cache .

# Copy the rest of the code
COPY --chown=user . .

# Ensure the app listens on port 7860 as required by Hugging Face
ENV PORT=7860
EXPOSE 7860

# Start the OpenEnv server using the registered entry point
CMD ["uv", "run", "openenv-server"]
