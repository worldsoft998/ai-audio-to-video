#!/bin/sh
#
# FROM not implemented
# Instruction: FROM python:3.9-slim
# git clone https://github.com/worldsoft998/ai-audio-to-video.git
# cd ai-audio-to-video
#
apt-get update && apt-get install -y     ffmpeg     libsm6     libxext6     libxrender-dev     libglib2.0-0     wget     gnupg     && rm -rf /var/lib/apt/lists/*
wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -     && echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list     && apt-get update && apt-get install -y     google-chrome-stable     && rm -rf /var/lib/apt/lists/*
# useradd -m -u 1000 user
#
# USER not implemented
# Instruction: USER user
#
# Switch to the new user and execute commands
# su - user -c 'echo "Hello from myuser" > /home/user/message.txt'
# Alternatively, using sudo (if sudo is installed and configured for the user)
# sudo -u myuser sh -c 'echo "Hello from myuser" > /home/myuser/message.txt'
export PATH="/home/user/.local/bin:\$PATH"
# How to expose a port in google colab
# https://stackoverflow.com/questions/56473719/open-port-on-google-colaboratory-machine
# https://donnysoh.medium.com/running-ollama-on-google-colab-by-exposing-the-colab-port-as-an-# url-2163ac5826f7
# https://www.reddit.com/r/GoogleColab/comments/12wwhxr/accessing_a_web_app_in_google_colab/
# https://gist.github.com/arifsuhan/0566da9a3e0383a48643ba5dfb10cdc0
# https://saturncloud.io/blog/is-there-a-general-way-to-run-web-applications-on-google-colab/
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# https://brightersidetech.com/running-flask-apps-in-google-colab/
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# https://sumanshunankana.hashnode.dev/run-flask-application-on-colab
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# https://www.wangjun.dev/2023/12/use-port-forward-host-server-in-colab/
# port forward
# from google.colab.output import eval_js
# print(eval_js("google.colab.kernel.proxyPort(5000)"))
# EXPOSE not implemented
# Instruction: EXPOSE 7860
#EXPOSE derives ports 22 and 7777 from the container and can not be set using a bash script. This #must be done when creating the container.
#Converting a Dockerfile EXPOSE into a shell script
# git clone https://github.com/worldsoft998/ai-audio-to-video.git
# cd ai-audio-to-video
# mkdir -p /app && cd /app
# cat << __EOFF__ | base64 -d | bunzip2 > requirements.txt
pip install --no-cache-dir --upgrade -r requirements.txt
# #cat << __EOFF__ | base64 -d | bunzip2 > app.py
# Restart session
# CMD not implemented
# Instruction: CMD ["python", "app.py"]
# %cd ai-audio-to-video
# !python app.py