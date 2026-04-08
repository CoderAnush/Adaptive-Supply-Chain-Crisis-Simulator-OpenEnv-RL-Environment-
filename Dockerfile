# Use official Python 3.9 image as base
FROM python:3.9

# Create a non-privileged user (required for Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Install dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the environment code, openenv config, and inference script
COPY --chown=user . /app

# Ensure the app listens on port 7860 as required by Hugging Face
ENV PORT=7860
EXPOSE 7860

# Start the OpenEnv server
CMD ["python", "app.py"]
