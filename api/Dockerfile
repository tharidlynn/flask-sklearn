FROM python:3.7.2-slim

COPY . /app

WORKDIR /app


RUN pip install numpy && \
    pip install flask && \
    pip install gunicorn && \
    pip install scikit-learn
    
ENTRYPOINT [ "gunicorn" ]
CMD ["-w 4", "-b :5000", "app:app"]