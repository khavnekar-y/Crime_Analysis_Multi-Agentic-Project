FROM python:3.10-slim

# Prevent Python from creating .pyc files and buffering output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system packages: dos2unix (for line ending fixes), curl (to install Poetry)
RUN apt-get update && apt-get install -y --no-install-recommends \
    dos2unix \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Configure Poetry to NOT create virtual environments
RUN poetry config virtualenvs.create false

# Set our working directory
WORKDIR /app

# Copy your Poetry files first (for efficient Docker layer caching)
# Modify this step in your Dockerfile
COPY pyproject.toml ./
COPY poetry.lock* ./
# Or alternative approach
# RUN if [ ! -f poetry.lock ]; then touch poetry.lock; fi

# Install only the dependencies (no project code yet)
RUN poetry install --no-root --no-interaction --no-ansi

# Now copy everything else into /app
COPY . /app

# Convert all Python files from CRLF to LF (handles Windows line endings)
RUN find /app -name "*.py" -exec dos2unix {} \;

# Optional: check syntax for all .py files at build time
RUN python -m py_compile $(find /app -name "*.py")

# Use Uvicorn to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]