import boto3
import time

region = "us-east-1"

sm = boto3.client("sagemaker", region_name=region)

model_name = "iris-batch-model"
job_name = f"iris-batch-job-{int(time.time())}"

image_uri = "690641652908.dkr.ecr.us-east-1.amazonaws.com/iris-batch-inference:latest"
model_data_url = "s3://my-ml-batch-model-bucket-12345/model/model.tar.gz"
role_arn = "arn:aws:iam::690641652908:role/SageMakerBatchExecutionRole"

input_s3 = "s3://my-ml-batch-model-bucket-12345/batch-input/input.jsonl"
output_s3 = "s3://my-ml-batch-model-bucket-12345/batch-output/"

instance_type = "ml.m5.large"

print("Creating SageMaker model...")

sm.create_model(
    ModelName=model_name,
    PrimaryContainer={
        "Image": image_uri,
        "ModelDataUrl": model_data_url,
    },
    ExecutionRoleArn=role_arn,
)

print("Creating batch transform job...")

sm.create_transform_job(
    TransformJobName=job_name,
    ModelName=model_name,
    TransformInput={
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": input_s3,
            }
        },
        "ContentType": "application/json",
    },
    TransformOutput={
        "S3OutputPath": output_s3,
    },
    TransformResources={
        "InstanceType": instance_type,
        "InstanceCount": 1,
    },
)

print(f"Batch job started: {job_name}")

print("Waiting for job to complete...")

while True:
    status = sm.describe_transform_job(TransformJobName=job_name)["TransformJobStatus"]
    print("Status:", status)
    if status in ["Completed", "Failed", "Stopped"]:
        break
    time.sleep(30)

print("Final status:", status)
