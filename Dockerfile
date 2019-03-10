FROM neubiaswg5/neubias-base

RUN wget http://files.ilastik.org/ilastik-1.3.2-Linux.tar.bz2

RUN tar xjf ilastik-1.3.2-Linux.tar.bz2

RUN mv ilastik-1.3.2-Linux ilastik

ADD PixelObjectClassification.ilp /app/PixelObjectClassification.ilp

ADD wrapper.py /app/wrapper.py

ENTRYPOINT ["python3.6","/app/wrapper.py"]
