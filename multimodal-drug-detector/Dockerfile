# Use a lightweight Python base image
FROM python:3.9-slim



# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /multimodal_drug_detector

# Copy the current directory contents into the container
COPY . /multimodal_drug_detector

# Install Python dependencies
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Command to run the application
CMD ["python", "drug_abuse_detector_api.py"]