# FROM bitnami/pytorch
# FROM pytorch/pytorch:latest
FROM python:3.8.9-slim

RUN apt-get update \
   #   && apt-get install -y \
      #   libgl1-mesa-glx \
      #   libx11-xcb1 \
     && apt-get clean all \
     && apt-get install libgomp1 \ 
     && rm -r /var/lib/apt/lists/*
     


RUN pip install pipenv
WORKDIR /movie-lens-camile
COPY ["Pipfile", "Pipfile.lock" ,"./"]
# COPY ["requirements.txt" ,"./"]
# RUN pip install -r ./requirements.txt
RUN pipenv install --ignore-pipfile --deploy --system
COPY ["results", "./results/"]
COPY ["app.py",  "./"]
EXPOSE 8501
# Train the model
CMD [ "streamlit", "run", "/movie-lens-camile/app.py", "server.gatherUsageStats", "False"]