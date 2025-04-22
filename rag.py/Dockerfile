FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt /app/
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy the entire application code
COPY . /app/

# Expose the port the Flask app uses
EXPOSE 8000

# Run your main script
CMD ["python", "rag_api.py"]
