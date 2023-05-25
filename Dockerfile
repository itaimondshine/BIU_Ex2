FROM python

COPY . /app

WORKDIR /app

# Add any ots
RUN pip3 install pathlib

RUN pip3 install numpy

ENTRYPOINT ["python", "main.py",  "data/enc.txt"]
