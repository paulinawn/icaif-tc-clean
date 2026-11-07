# Base image: a slim Linux with Python preinstalled
FROM python:3.14-slim

# Working directory inside the container (created if missing)
WORKDIR /app

# Copy only the dependency lists first, then install them
COPY requirements.txt requirements-dev.txt ./
# no-cache-dir purpose: stops pip from keeping download cache inside image.
RUN pip install --no-cache-dir -r requirements-dev.txt 

# Copy the rest of the project code
COPY . .

# Default command when the container starts: run the test suite
CMD ["python", "-m", "pytest", "-v"]