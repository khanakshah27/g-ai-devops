# Use Python 3.12 slim images for smaller size
FROM python:3.12-slim

# Install OpenCV system dependencies
# libgl1 and libglib2.0-0 are required for cv2 to work inside Docker
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Upgrade pip and install dependencies
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 8000
EXPOSE 8000

# --- CRITICAL CHANGE ---
# Use Gunicorn instead of "flask run"
# -w 1: Use 1 worker process (sufficient for deep learning models to avoid OOM)
# --threads 4: Allow 4 concurrent threads for file uploads/downloads
# --timeout 120: Allow 120 seconds for processing big videos before timing out
# -b 0.0.0.0:8000: Bind to all network interfaces
CMD ["gunicorn", "-w", "1", "--threads", "4", "-b", "0.0.0.0:8000", "--timeout", "120", "app:app"]


