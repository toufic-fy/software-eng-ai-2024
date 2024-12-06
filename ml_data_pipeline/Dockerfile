# Use an official Python runtime as the base image
FROM python:3.12
# Set the working directory
WORKDIR /app
# Copy project files
COPY . /app
# Install Poetry and dependencies
RUN pip install pipx && pipx install poetry
RUN ln -s /root/.local/bin/poetry /usr/local/bin/poetry && poetry install
# Expose the port FastAPI will run on
EXPOSE 8000
# Command to run the FastAPI app
CMD ["poetry", "run", "uvicorn", "ml_data_pipeline.main:app"]