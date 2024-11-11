# use Pytorch base image
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements_asr.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r requirements_asr.txt

# Copy the API code
COPY asr_api.py .

# Make port 8001 available
EXPOSE 8001

# Set environment variable
ENV FLASK_APP=asr_api.py

# Run the application
CMD ["python", "asr_api.py"]