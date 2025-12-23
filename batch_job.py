import boto3
from botocore.exceptions import ClientError
import json
import os
import re
import sys
import time

_ECR_IMAGE_RE = re.compile(
    r"^(?P<account>\d+)\.dkr\.ecr\.(?P<region>[^.]+)\.amazonaws\.com\/(?P<repo>[^:@]+)(?::(?P<tag>[^@]+))?(?:@(?P<digest>.+))?$"
)


def _load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return data


def _env_first(*keys):
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    return None


def _get(cfg, *keys, default=None):
    for key in keys:
        if key in cfg and cfg[key] not in (None, ""):
            return cfg[key]
    return default


def _looks_like_ecr_pull_error(err):
    if not isinstance(err, ClientError):
        return False
    code = err.response.get("Error", {}).get("Code", "")
    message = (err.response.get("Error", {}).get("Message", "") or "").lower()
    if code != "ValidationException":
        return False
    return "cannot pull" in message and "granted pull permission" in message


config_path = None
if len(sys.argv) >= 3 and sys.argv[1] in {"--config", "-c"}:
    config_path = sys.argv[2]
elif os.path.exists("config.json"):
    config_path = "config.json"

cfg = _load_config(config_path) if config_path else {}

region = _get(
    cfg,
    "region",
    default=_env_first("AWS_REGION", "AWS_DEFAULT_REGION") or "us-east-1",
)

sm = boto3.client("sagemaker", region_name=region)

model_name = _get(cfg, "model_name", "modelName", default="iris-batch-model")
job_name = _get(cfg, "job_name", "jobName", default=f"iris-batch-job-{int(time.time())}")

image_uri = _get(cfg, "image_uri", "imageUri", default=_env_first("SM_IMAGE_URI"))
model_data_url = _get(cfg, "model_data_url", "modelDataUrl", default=_env_first("SM_MODEL_DATA_URL"))
role_arn = _get(cfg, "role_arn", "roleArn", default=_env_first("SM_ROLE_ARN"))

input_s3 = _get(cfg, "input_s3", "inputS3", default=_env_first("SM_INPUT_S3"))
output_s3 = _get(cfg, "output_s3", "outputS3", default=_env_first("SM_OUTPUT_S3"))

instance_type = _get(cfg, "instance_type", "instanceType", default="ml.m5.large")
instance_count = int(_get(cfg, "instance_count", "instanceCount", default=1))
poll_seconds = int(_get(cfg, "poll_seconds", "pollSeconds", default=30))
wait_for_completion = not bool(_get(cfg, "no_wait", "noWait", default=False))

missing = []
if not image_uri:
    missing.append("image_uri (or SM_IMAGE_URI)")
if not model_data_url:
    missing.append("model_data_url (or SM_MODEL_DATA_URL)")
if not role_arn:
    missing.append("role_arn (or SM_ROLE_ARN)")
if not input_s3:
    missing.append("input_s3 (or SM_INPUT_S3)")
if not output_s3:
    missing.append("output_s3 (or SM_OUTPUT_S3)")

if missing:
    print("Missing required configuration:", ", ".join(missing))
    print("Create config.json from config.example.json (or pass --config PATH).")
    sys.exit(2)

image_match = _ECR_IMAGE_RE.match(image_uri)
if image_match and image_match.group("region") != region:
    print(
        f"Warning: image URI region is {image_match.group('region')}, but script region is {region}."
    )

print("Creating SageMaker model...")

try:
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data_url,
        },
        ExecutionRoleArn=role_arn,
    )
except ClientError as e:
    if _looks_like_ecr_pull_error(e):
        print("SageMaker could not pull your ECR image with the provided execution role.")
        print(f"Execution role: {role_arn}")
        print(f"Image URI: {image_uri}")
        print("Fix: grant the role ECR pull permissions (and ensure the image/tag exists).")
        print("See README.md for a minimal IAM policy template.")
        sys.exit(1)
    raise

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
        "InstanceCount": instance_count,
    },
)

print(f"Batch job started: {job_name}")

if not wait_for_completion:
    sys.exit(0)

print("Waiting for job to complete...")

while True:
    status = sm.describe_transform_job(TransformJobName=job_name)["TransformJobStatus"]
    print("Status:", status)
    if status in ["Completed", "Failed", "Stopped"]:
        break
    time.sleep(poll_seconds)

print("Final status:", status)
