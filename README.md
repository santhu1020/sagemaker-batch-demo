# sagemaker-batch-demo

Minimal end-to-end demo of **SageMaker Batch Transform** using a custom inference container (ECR) and a scikit-learn Iris model (`model.tar.gz` in S3).

## Prereqs

- AWS credentials configured locally (e.g., `aws configure`)
- Docker installed (for building/pushing the inference image)
- An S3 bucket for model + input/output
- An IAM execution role that SageMaker can assume

## 1) Train and package the model

```bash
python train.py
```

This creates `model.tar.gz`.

## 2) Upload model + input to S3

```bash
export AWS_REGION=us-east-1
export BUCKET=your-unique-bucket-name

aws s3 mb s3://$BUCKET --region $AWS_REGION
aws s3 cp model.tar.gz s3://$BUCKET/model/model.tar.gz
aws s3 cp input.jsonl s3://$BUCKET/batch-input/input.jsonl
```

## 3) Build and push the inference image to ECR

```bash
export AWS_REGION=us-east-1
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export REPO=iris-batch-inference

aws ecr create-repository --repository-name $REPO --region $AWS_REGION || true
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

docker build -t $REPO .
docker tag $REPO:latest $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest
docker push $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest
```

## 4) Create the SageMaker execution role (ECR pull + S3 access)

Create an IAM role with trust to SageMaker:

```bash
aws iam create-role \
  --role-name SageMakerBatchExecutionRole \
  --assume-role-policy-document file://iam/sagemaker-trust-policy.json || true
```

Attach permissions (edit `iam/sagemaker-batch-execution-policy.json` first to set your bucket/account/region):

```bash
aws iam put-role-policy \
  --role-name SageMakerBatchExecutionRole \
  --policy-name SageMakerBatchExecutionPolicy \
  --policy-document file://iam/sagemaker-batch-execution-policy.json
```

## 5) Configure and run the batch job

Copy `config.example.json` to `config.json` and fill in your values, then:

```bash
python batch_job.py
```

## Troubleshooting

### `Role ... cannot pull ... Ensure that the role exists and the image was granted pull permission.`

This means SageMaker can’t pull your ECR image using the `ExecutionRoleArn`.

- Ensure the **execution role** has ECR pull permissions (see `iam/sagemaker-batch-execution-policy.json`).
- Ensure the ECR image exists and has the expected tag:
  - `aws ecr describe-images --repository-name iris-batch-inference --image-ids imageTag=latest --region us-east-1`
- If pulling cross-account, add an ECR repository policy allowing the other account/role (not needed for same-account by default).
