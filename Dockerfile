FROM python:3.7-stretch

# ---------------------------------------------------------------------------------------------------------------------
# Install Cytomine python client
RUN git clone https://github.com/cytomine-uliege/Cytomine-python-client.git && \
    cd /Cytomine-python-client && git checkout tags/v2.7.3 && pip install . && \
    rm -r /Cytomine-python-client

# ---------------------------------------------------------------------------------------------------------------------
# Install Neubias-W5-Utilities (annotation exporter, compute metrics, helpers,...)
RUN apt-get update && apt-get install libgeos-dev -y && apt-get clean
RUN git clone https://github.com/Neubias-WG5/biaflows-utilities.git && \
    cd /biaflows-utilities/ && git checkout tags/v0.9.1 && pip install .

# install utilities binaries
RUN chmod +x /biaflows-utilities/bin/*
RUN cp /biaflows-utilities/bin/* /usr/bin/

# cleaning
RUN rm -r /biaflows-utilities

# ---------------------------------------------------------------------------------------------------------------------
# Install ilastik
RUN wget http://files.ilastik.org/ilastik-1.3.2-Linux.tar.bz2
RUN apt-get update && apt-get install -y --no-install-recommends bsdtar
RUN mkdir /app && mkdir /app/ilastik
RUN bsdtar -xjvf ilastik-1.3.2-Linux.tar.bz2 && \
    mv /ilastik-1.3.2-Linux/* /app/ilastik/ && \
    rm /ilastik-1.3.2-Linux.tar.bz2 /ilastik-1.3.2-Linux -r

# ---------------------------------------------------------------------------------------------------------------------
# Install workflow
ADD PixelClassification.ilp /app/PixelClassification.ilp
RUN wget -O /app/RGBPixelClassification.ilp https://biaflows.neubias.org/workflow-files/ilastik_weights.ilp

ADD wrapper.py /app/wrapper.py

ADD descriptor.json /app/descriptor.json

ENTRYPOINT ["python","/app/wrapper.py"]
