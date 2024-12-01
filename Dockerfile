# Use a Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
COPY app/requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

# Copy your application code
COPY app/ .

COPY models /app/models

COPY data /app/data

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "mlapi:app", "--host", "0.0.0.0", "--port", "8000"]
