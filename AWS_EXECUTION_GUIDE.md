# CrowdVision — AWS Execution Guide

## Instance Specification

| Parameter | Value |
|-----------|-------|
| **Instance type** | `g5.xlarge` |
| **Region** | `us-east-1` (N. Virginia) |
| **GPU** | 1× NVIDIA A10G (24 GB VRAM) |
| **vCPU** | 4 (within your 8 vCPU quota) |
| **RAM** | 16 GB |
| **Price** | ~$1.006/hr on-demand |
| **AMI** | Deep Learning AMI (Ubuntu) with PyTorch 2.x + CUDA |
| **Root EBS** | 200 GB `gp3` |

---

## Budget Plan ($150 Hard Limit)

| Day | Activity | Estimated Hours | Cost |
|-----|----------|-----------------|------|
| Day 1 | Setup + NB 00 + NB 01 (Density) | 14h | ~$14 |
| Day 2 | NB 02 (Forecasting) + NB 03 (Anomaly) | 14h | ~$14 |
| Day 3 | NB 04 + NB 05 (Multi-task) + NB 06 (Eval) | 12h | ~$12 |
| **Buffer** | Debug, re-runs | 10h | ~$10 |
| **EBS storage** | 200 GB × 3 days | — | ~$2 |
| **Data transfer** | Upload ~5 GB | — | ~$1 |
| **Total estimate** | | **~50h** | **~$53** |

> **Safety margin**: Even at 100h of total runtime, cost is ~$103 — well within $150 cap.
> **Emergency reserve**: $50 untouched for any re-training needs.

### Cost Protection Setup (DO THIS FIRST)

```bash
# 1. AWS Budgets — create a $150 budget with alerts at $50, $100, $130
aws budgets create-budget \
  --account-id YOUR_ACCOUNT_ID \
  --budget '{
    "BudgetName": "CrowdVision-Training",
    "BudgetLimit": {"Amount": "150", "Unit": "USD"},
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
  }' \
  --notifications-with-subscribers '[
    {"Notification": {"NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN", "Threshold": 33, "ThresholdType": "PERCENTAGE"},
     "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "YOUR_EMAIL"}]},
    {"Notification": {"NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN", "Threshold": 66, "ThresholdType": "PERCENTAGE"},
     "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "YOUR_EMAIL"}]},
    {"Notification": {"NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN", "Threshold": 87, "ThresholdType": "PERCENTAGE"},
     "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "YOUR_EMAIL"}]}
  ]'
```

---

## Step-by-Step Launch

### 1. Launch Instance

```bash
# Find the Deep Learning AMI
AMI_ID=$(aws ec2 describe-images \
  --region us-east-1 \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning AMI (Ubuntu 20.04)*PyTorch*" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
  --output text)

echo "Using AMI: $AMI_ID"

# Launch g5.xlarge
aws ec2 run-instances \
  --region us-east-1 \
  --image-id $AMI_ID \
  --instance-type g5.xlarge \
  --key-name YOUR_KEY_PAIR \
  --security-group-ids YOUR_SG_ID \
  --block-device-mappings '[{
    "DeviceName": "/dev/sda1",
    "Ebs": {"VolumeSize": 200, "VolumeType": "gp3", "DeleteOnTermination": true}
  }]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=crowdvision-training}]' \
  --count 1
```

### 2. SSH + Initial Setup

```bash
ssh -i YOUR_KEY.pem ubuntu@INSTANCE_IP

# Verify GPU
nvidia-smi

# Clone repo (or scp/rsync from local)
git clone YOUR_REPO_URL crowdvision
cd crowdvision

# Install dependencies
pip install -r requirements.txt

# Verify torch + CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

### 3. Upload Datasets

The datasets are ~5 GB total. Options:
- **rsync/scp** from local machine (slow over home internet)
- **Google Drive → gdown** (if datasets are on Drive)
- **S3 bucket** (fastest — upload to S3 first, then `aws s3 sync`)

```bash
# Option A: rsync from local
rsync -avz --progress data/ ubuntu@INSTANCE_IP:~/crowdvision/data/

# Option B: S3 (recommended if datasets are large)
aws s3 sync s3://your-bucket/crowdvision-data/ data/
```

### 4. Run Notebooks

```bash
# Start JupyterLab
jupyter lab --no-browser --port=8888 --ip=0.0.0.0 &

# SSH tunnel from local machine
ssh -N -L 8888:localhost:8888 -i YOUR_KEY.pem ubuntu@INSTANCE_IP

# Open http://localhost:8888 in your browser
```

Then run notebooks in order: `00 → 01 → 02 → 03 → 04 → 05 → 06`

### 5. Backup Results After Each Notebook

```bash
# After each notebook completes, sync outputs
aws s3 sync checkpoints/ s3://your-bucket/crowdvision-checkpoints/
aws s3 sync experiments/ s3://your-bucket/crowdvision-experiments/
```

### 6. STOP Instance When Not Training

```bash
# CRITICAL: Stop the instance when you're not actively running notebooks
aws ec2 stop-instances --instance-ids i-XXXXX --region us-east-1

# Resume later
aws ec2 start-instances --instance-ids i-XXXXX --region us-east-1
```

---

## GPU Memory Guidance (A10G 24 GB)

Based on the README recommendations for 24GB GPUs:

| Setting | Value |
|---------|-------|
| `target_size` | `(448, 448)` |
| `batch_size` (density) | `4` |
| `batch_size` (forecasting) | `64` |
| `batch_size` (anomaly) | `64` |
| `batch_size` (multitask) | `4` |

The notebooks auto-detect GPU memory and adjust. If you get OOM errors, reduce batch sizes in the notebook config cells.

---

## Emergency Procedures

### If Budget Alarm Fires at $130
1. Immediately stop the instance
2. Sync all checkpoints to S3
3. Evaluate if remaining notebooks can run within $20

### If Instance Crashes Mid-Training
- All trainers use checkpointing — rerun the same cell and it resumes from last checkpoint
- Checkpoints saved at `checkpoints/{experiment_name}/best.pt` and `last.pt`

### If GPU OOM
- Reduce `batch_size` to half
- Reduce `target_size` to `(288, 384)`
- The notebooks have CPU fallback with `epochs=3` for smoke testing

---

## Post-Training Cleanup

```bash
# 1. Sync all outputs
aws s3 sync checkpoints/ s3://your-bucket/crowdvision-checkpoints/
aws s3 sync experiments/ s3://your-bucket/crowdvision-experiments/
aws s3 sync runs/ s3://your-bucket/crowdvision-tensorboard/

# 2. Download to local (from your Windows machine)
aws s3 sync s3://your-bucket/crowdvision-checkpoints/ D:\majproj\crowdvision\checkpoints\
aws s3 sync s3://your-bucket/crowdvision-experiments/ D:\majproj\crowdvision\experiments\

# 3. Terminate instance and delete resources
aws ec2 terminate-instances --instance-ids i-XXXXX --region us-east-1

# 4. Verify no billable resources remain
aws ec2 describe-instances --region us-east-1 \
  --filters "Name=tag:Name,Values=crowdvision-training" \
  --query 'Reservations[].Instances[].State.Name'
```
