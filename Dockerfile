# Use Slideflow 2.2.1-torch base image for verified reproducibility
FROM jamesdolezal/slideflow:2.2.1-torch

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY . .

# Set default command (can be overridden)
CMD ["python", "preprocess.py", "--help"]
