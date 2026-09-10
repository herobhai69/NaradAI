# Use a slim, official Python 3.12 Linux image
FROM python:3.12-slim

# Prevent Python from writing .pyc files and force stdout logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for audio processing (ffmpeg) and standard builds
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy uv directly from the official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy ONLY the lockfile and pyproject to cache the heavy dependency downloads
COPY pyproject.toml uv.lock ./

# Install dependencies ONLY (do not attempt to build the project as a library)
RUN uv sync --no-dev --no-install-project

# Copy the rest of your application code into the container
COPY . .

# Activate the virtual environment permanently for this container
ENV PATH="/app/.venv/bin:$PATH"

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]