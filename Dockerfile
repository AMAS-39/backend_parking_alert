# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Install system dependencies, including git
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && apt-get clean

# Log installed dependencies for debugging
RUN apt list --installed

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install  -r requirements.txt

# Copy the current directory contents into the container
COPY . .

# Set the environment variables
ENV PORT=8080

# Log installed Python packages for debugging
RUN pip freeze

# Run the application
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]
