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

# Copy all files first so that 'uv pip install .' can find the packages
COPY --chown=user . .

# Install dependencies
# Using --system to install to the global environment
RUN uv pip install --system --no-cache .

# Ensure the app listens on port 7860 as required by Hugging Face
ENV PORT=7860
EXPOSE 7860

# Start the OpenEnv server using the registered entry point
CMD ["uv", "run", "openenv-server"]
