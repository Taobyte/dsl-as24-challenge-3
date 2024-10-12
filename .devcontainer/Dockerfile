FROM continuumio/miniconda3

# Copy the environment file
COPY environment.yaml /tmp/environment.yaml

# Install mamba for faster and more reliable package management
RUN conda install -n base -c conda-forge mamba

# Create the environment using mamba
RUN mamba env create -f /tmp/environment.yaml

# Activate the environment
ENV PATH /opt/conda/envs/myenv/bin:$PATH
