# Use the official Ubuntu image as the base image
FROM ubuntu:22.04

# Set the working directory in the container
WORKDIR /app

# Copy the code into the container
COPY . .

# Install Python and required packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    cmake \
    make \ 
    vim \
    libomp-dev \ 
    libgmp-dev \ 
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install the required Python packages
RUN pip install -r requirements.txt

# Download the mnist dataset
WORKDIR /app/dataset
RUN git clone https://github.com/wichtounet/mnist.git

# Create a directory named "build"
WORKDIR /app
RUN mkdir build 

# Change the working directory
WORKDIR /app/build

# Run CMake to configure the project
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=OFF

# Compile the project using make
RUN make -j8
