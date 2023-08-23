
# Use the specified base image
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy the entire current directory into the container
COPY . .

# Install the dependencies using the install.sh script
RUN pip install dist/Housing_Prediction-0.0.1-py3-none-any.whl
RUN pip install -e .

RUN pip install pytest

# Set the specified label
LABEL maintainer="pushvinder"

CMD ["echo", "House Prediction Container Started"]
CMD ["pytest", "/home/pushvinder/mle_training/tests/unit_tests/test_scripts.py"]

