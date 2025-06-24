FROM quay.io/centos/centos:stream10

# By default, listen on port 8080
EXPOSE 8000/tcp
ENV HOST=0.0.0.0
ENV PORT=8000

# Set the working directory in the container
WORKDIR /projects

# Copy the content of the local src directory to the working directory
COPY . .

# Install any dependencies
RUN dnf install which python3 python3-pip -y
RUN \
  if [ -f requirements.txt ]; \
    then pip install -r requirements.txt; \
  elif [ `ls -1q *.txt | wc -l` == 1 ]; \
    then pip install -r *.txt; \
  fi

# Specify the command to run on container start
# Override configuration by overriding /projects/parameters.yaml
ENTRYPOINT ["bash", "-c", "streamlit run --server.address $HOST --server.port $PORT main.py"]
