FROM python:3.9-slim

RUN pip install --no-cache-dir \
    sagemaker-inference \
    scikit-learn \
    joblib \
    numpy \
    pandas

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

WORKDIR /opt/ml/code

COPY inference.py .

ENV SAGEMAKER_PROGRAM inference.py

EXPOSE 8080

ENTRYPOINT ["python", "-m", "sagemaker_inference"]
