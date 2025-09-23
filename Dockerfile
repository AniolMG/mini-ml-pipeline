# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt

# Install each package individually, skip if installation fails
RUN set -e; \
    while read pkg; do \
        echo "Installing $pkg..."; \
        pip install --no-cache-dir "$pkg" || echo "Skipping $pkg (failed)"; \
    done < requirements.txt

# Copy the specific saved model folder into the container
# COPY saved_model/ ./saved_model/ # I will use volumes instead

# Copy FastAPI app
COPY src/main.py .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
