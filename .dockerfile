FROM gitpod/workspace-full
FROM python:3.9-slim
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    wget \
    gnupg \
	git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/linux_signing_keyring.gpg   \  
    && echo "deb [signed-by=/usr/share/keyrings/linux_signing_keyring.gpg] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update && apt-get install -y \
    google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# RUN useradd -m -u 1000 user
# USER user

# Create the gitpod user. UID must be 33333.
RUN useradd -l -u 33333 -G sudo -md /home/gitpod -s /bin/bash -p gitpod gitpod
USER gitpod

# ENV PATH="/home/user/.local/bin:$PATH"
ENV PATH="/home/gitpod/.local/bin:$PATH"

WORKDIR /app

# COPY --chown=user ./requirements.txt requirements.txt
COPY --chown=gitpod ./requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

# COPY --chown=user ./app.py app.py
COPY --chown=gitpod ./app.py app.py

EXPOSE 7860
CMD ["python", "app.py"]
