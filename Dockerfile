FROM continuumio/miniconda3

# install packages
RUN pip install numpy pandas scikit-learn transformers datasets hf_xet

# install torch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Setup work dir and copy src
WORKDIR /usr/src/app
ADD src src

# Run application which writes data to 
RUN chmod +x src/run.sh
ENTRYPOINT ["./src/run.sh"]