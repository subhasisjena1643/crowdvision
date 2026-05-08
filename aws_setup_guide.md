# CrowdVision — Complete AWS + IDE Setup Guide (From Scratch)

> [!IMPORTANT]
> **Your constraints**: 8 vCPU quota (G & VT families), $200 credits, $150 hard limit, $50 emergency reserve.
> **Target instance**: `g5.xlarge` — 4 vCPU, 1× NVIDIA A10G 24GB, 16GB RAM — **$1.006/hr**

---

## Part 1: AWS Console — One-Time Setup

### Step 1.1: Install AWS CLI on Windows

Open PowerShell **as Administrator**:

```powershell
# Download and install AWS CLI v2
msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi /quiet

# Verify (reopen PowerShell after install)
aws --version
```

### Step 1.2: Configure AWS CLI

```powershell
aws configure
```

It will ask for 4 things:
```
AWS Access Key ID:     [paste your access key]
AWS Secret Access Key: [paste your secret key]
Default region name:   us-east-1
Default output format: json
```

> [!TIP]
> If you don't have access keys yet: AWS Console → IAM → Users → your user → Security credentials → Create access key

### Step 1.3: Create an SSH Key Pair

```powershell
# Create .ssh directory if it doesn't exist
mkdir -Force "$env:USERPROFILE\.ssh"

# Create key pair via AWS
aws ec2 create-key-pair `
  --region us-east-1 `
  --key-name crowdvision-key `
  --query 'KeyMaterial' `
  --output text | Out-File -Encoding ascii "$env:USERPROFILE\.ssh\crowdvision-key.pem"

# Lock down permissions (required for SSH)
icacls "$env:USERPROFILE\.ssh\crowdvision-key.pem" /inheritance:r /grant:r "${env:USERNAME}:(R)"
```

### Step 1.4: Create a Security Group

```powershell
# Get your default VPC ID
$VPC_ID = aws ec2 describe-vpcs --region us-east-1 `
  --filters "Name=isDefault,Values=true" `
  --query 'Vpcs[0].VpcId' --output text

Write-Output "VPC: $VPC_ID"

# Create security group
$SG_ID = aws ec2 create-security-group `
  --region us-east-1 `
  --group-name crowdvision-sg `
  --description "CrowdVision training instance" `
  --vpc-id $VPC_ID `
  --query 'GroupId' --output text

Write-Output "Security Group: $SG_ID"

# Allow SSH (port 22) from your IP only
aws ec2 authorize-security-group-ingress `
  --region us-east-1 `
  --group-id $SG_ID `
  --protocol tcp --port 22 `
  --cidr "$(Invoke-RestMethod ifconfig.me)/32"
```

> [!NOTE]
> If your IP changes (e.g., after router restart), you'll need to update the security group rule. See the troubleshooting section at the bottom.

### Step 1.5: Set Up Budget Alerts

Go to **AWS Console → Billing → Budgets → Create budget**:
- Budget type: **Cost budget**
- Name: `CrowdVision-Training`
- Budget amount: **$150**
- Add 3 alert thresholds:
  - Alert at **$50** (33%) → your email
  - Alert at **$100** (66%) → your email  
  - Alert at **$130** (87%) → your email

---

## Part 2: Launch the GPU Instance

### Step 2.1: Find the Deep Learning AMI

```powershell
$AMI_ID = aws ec2 describe-images `
  --region us-east-1 `
  --owners amazon `
  --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04)*" "Name=state,Values=available" `
  --query 'reverse(sort_by(Images, &CreationDate))[0].ImageId' `
  --output text

Write-Output "AMI ID: $AMI_ID"
```

If this returns nothing, try the broader search:

```powershell
$AMI_ID = aws ec2 describe-images `
  --region us-east-1 `
  --owners amazon `
  --filters "Name=name,Values=Deep Learning AMI (Ubuntu) *" "Name=state,Values=available" `
  --query 'reverse(sort_by(Images, &CreationDate))[0].ImageId' `
  --output text

Write-Output "AMI ID: $AMI_ID"
```

> [!IMPORTANT]
> **Write down the AMI ID** — you'll need it in the next step. It looks like `ami-0abcdef1234567890`.

### Step 2.2: Launch the Instance

```powershell
$INSTANCE_ID = aws ec2 run-instances `
  --region us-east-1 `
  --image-id $AMI_ID `
  --instance-type g5.xlarge `
  --key-name crowdvision-key `
  --security-group-ids $SG_ID `
  --block-device-mappings '[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":200,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]' `
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=crowdvision-training}]' `
  --query 'Instances[0].InstanceId' `
  --output text

Write-Output "Instance ID: $INSTANCE_ID"
```

### Step 2.3: Wait for it to Start & Get the IP

```powershell
# Wait until running
aws ec2 wait instance-running --region us-east-1 --instance-ids $INSTANCE_ID

# Get the public IP
$PUBLIC_IP = aws ec2 describe-instances `
  --region us-east-1 `
  --instance-ids $INSTANCE_ID `
  --query 'Reservations[0].Instances[0].PublicIpAddress' `
  --output text

Write-Output "Public IP: $PUBLIC_IP"
```

> [!IMPORTANT]
> **Write down your Instance ID and Public IP.** You'll need both throughout.
> Instance ID looks like: `i-0abc123def456789`
> IP looks like: `3.87.123.45`

### Step 2.4: First SSH Connection (Test)

```powershell
ssh -i "$env:USERPROFILE\.ssh\crowdvision-key.pem" ubuntu@$PUBLIC_IP
```

Type `yes` when asked about the fingerprint. You should see the Ubuntu welcome screen with GPU info.

Quick GPU check:
```bash
nvidia-smi
```

You should see the **NVIDIA A10G** with 24GB VRAM. Type `exit` to disconnect for now.

---

## Part 3: Connect Your IDE via SSH

Your IDE (Antigravity/VS Code-based) supports **Remote-SSH**. Here's how to set it up:

### Step 3.1: Create SSH Config File

On your **Windows machine**, create or edit the SSH config file:

```powershell
notepad "$env:USERPROFILE\.ssh\config"
```

Add this block (replace `YOUR_PUBLIC_IP` with the actual IP from Step 2.3):

```
Host crowdvision-aws
    HostName YOUR_PUBLIC_IP
    User ubuntu
    IdentityFile C:\Users\subhasis\.ssh\crowdvision-key.pem
    StrictHostKeyChecking no
    ServerAliveInterval 60
    ServerAliveCountMax 120
```

Save and close.

### Step 3.2: Connect from IDE

1. Open your IDE (Antigravity / VS Code)
2. Press **`Ctrl+Shift+P`** to open the Command Palette
3. Type **`Remote-SSH: Connect to Host`** and select it
4. Choose **`crowdvision-aws`** from the list
5. A new IDE window opens — wait ~1–2 minutes for it to install the server component
6. The bottom-left corner should show **`SSH: crowdvision-aws`** when connected

### Step 3.3: Open the Project Folder

Once connected:
1. **`Ctrl+Shift+P`** → **`Terminal: Create New Terminal`** (or `` Ctrl+` ``)
2. In the terminal, clone or upload your repo (see Part 4)
3. **File → Open Folder** → navigate to `/home/ubuntu/crowdvision` → **OK**

Now your IDE is fully connected to the GPU instance. You can edit files, run terminals, and use Jupyter — all from your local machine.

---

## Part 4: Upload Project & Data to the Instance

### Option A: Git Clone (Fastest if repo is on GitHub)

```bash
# On the AWS instance terminal (inside IDE)
cd ~
git clone https://github.com/YOUR_USERNAME/crowdvision.git
cd crowdvision
```

### Option B: rsync from Windows (If repo is local only)

From a **local Windows PowerShell** (not the SSH terminal):

```powershell
# Upload code (small, fast)
scp -i "$env:USERPROFILE\.ssh\crowdvision-key.pem" -r `
  "D:\majproj\crowdvision\src" `
  "D:\majproj\crowdvision\notebooks" `
  "D:\majproj\crowdvision\configs" `
  "D:\majproj\crowdvision\requirements.txt" `
  "D:\majproj\crowdvision\AWS_EXECUTION_GUIDE.md" `
  ubuntu@${PUBLIC_IP}:~/crowdvision/

# Upload datasets (large — will take time depending on internet speed)
scp -i "$env:USERPROFILE\.ssh\crowdvision-key.pem" -r `
  "D:\majproj\crowdvision\data" `
  ubuntu@${PUBLIC_IP}:~/crowdvision/
```

> [!TIP]
> **If datasets are too large to upload from home**: Upload them to an S3 bucket first (from your local machine), then download to the instance (much faster since it's within AWS):
> ```powershell
> # From Windows — upload to S3
> aws s3 sync "D:\majproj\crowdvision\data" s3://your-bucket-name/crowdvision-data/
> ```
> ```bash
> # From AWS instance — download from S3
> aws s3 sync s3://your-bucket-name/crowdvision-data/ ~/crowdvision/data/
> ```

### Step 4.1: Verify Directory Structure on Instance

```bash
cd ~/crowdvision
find . -maxdepth 3 -type d | head -30
ls -la data/
```

You should see `src/`, `notebooks/`, `configs/`, `data/` with all the datasets inside.

---

## Part 5: Install Dependencies & Verify GPU

### Step 5.1: Activate the PyTorch Environment

```bash
# The Deep Learning AMI has PyTorch pre-installed in a virtual environment
# Try one of these (depends on AMI version):

# For newer DLAMIs (2024+):
source /opt/pytorch/bin/activate

# OR for older DLAMIs with conda:
# conda activate pytorch
```

### Step 5.2: Verify PyTorch + CUDA

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
```

Expected output:
```
PyTorch: 2.x.x
CUDA available: True
GPU: NVIDIA A10G
VRAM: 22.5 GB
```

### Step 5.3: Install Project Dependencies

```bash
cd ~/crowdvision
pip install -r requirements.txt
```

### Step 5.4: Verify All Imports Work

```bash
cd ~/crowdvision
python -c "
from src import ROOT_DIR, DATA_DIR
print(f'ROOT: {ROOT_DIR}')
from src.models import CSRNet, AdaptiveCSRNet, GCNGRU, AdaptiveNASGNN, ConvAE, FutureFrameNet, UnifiedCrowdVision
print('All 7 models imported OK')
from src.data_loaders import get_shanghaitech_loaders, get_jhu_loaders, load_metr_la, get_ucsd_loaders
print('All data loaders imported OK')
from src.training import DensityTrainer, ForecastingTrainer, AnomalyTrainer
print('All trainers imported OK')
from src.evaluation import evaluate_density, evaluate_forecasting, evaluate_anomaly_detection
print('All evaluation modules imported OK')
print('EVERYTHING READY')
"
```

---

## Part 6: Run the Notebooks

### Option A: JupyterLab in Browser (Recommended)

```bash
# On the AWS instance
cd ~/crowdvision
jupyter lab --no-browser --port=8888 --ip=0.0.0.0
```

It will print a URL with a token like:
```
http://127.0.0.1:8888/lab?token=abc123def456...
```

**Copy that token.** Then on your **local Windows machine**, open a new PowerShell:

```powershell
# Create SSH tunnel
ssh -i "$env:USERPROFILE\.ssh\crowdvision-key.pem" -N -L 8888:localhost:8888 ubuntu@YOUR_PUBLIC_IP
```

Open your browser and go to: `http://localhost:8888`
Paste the token when asked. You now have JupyterLab running on the GPU.

### Option B: Run from IDE Terminal

If you prefer running notebooks from the IDE terminal:

```bash
cd ~/crowdvision
jupyter nbconvert --to notebook --execute notebooks/00_setup_and_data_check.ipynb --output 00_executed.ipynb
```

### Notebook Execution Order

Run them **in this exact order**. Each one depends on the previous:

| # | Notebook | What it does | Est. time (A10G) |
|---|----------|-------------|-------------------|
| 1 | `00_setup_and_data_check.ipynb` | Verify data, GPU, deps | 5 min |
| 2 | `01_density_estimation.ipynb` | Train CSRNet + AdaptiveCSRNet | 4–8 hours |
| 3 | `02_forecasting.ipynb` | Train GCN-GRU + NAS search | 3–6 hours |
| 4 | `03_anomaly_detection.ipynb` | Train ConvAE + FutureFrameNet | 2–4 hours |
| 5 | `04_crowd_flow_and_dispatch_intelligence.ipynb` | Zone risk + dispatch logic | 30 min |
| 6 | `05_multitask_training.ipynb` | Train UnifiedCrowdVision | 4–6 hours |
| 7 | `06_evaluation_and_paper_results.ipynb` | Final metrics + figures | 20 min |

> [!WARNING]
> **After each notebook completes**, backup your checkpoints before starting the next one:
> ```bash
> # Quick backup to a safe directory
> cp -r checkpoints/ ~/checkpoints_backup_$(date +%Y%m%d_%H%M)/
> cp -r experiments/ ~/experiments_backup_$(date +%Y%m%d_%H%M)/
> ```

---

## Part 7: Stop/Start Instance (SAVE MONEY)

> [!CAUTION]
> **ALWAYS stop the instance when you're not actively training.** A stopped instance costs $0/hr for compute (only ~$0.02/hr for the 200GB EBS disk). A running instance costs $1.006/hr even if idle.

### Stop the Instance (When done for the day)

```powershell
# From your local Windows PowerShell
aws ec2 stop-instances --region us-east-1 --instance-ids YOUR_INSTANCE_ID
```

### Start the Instance (Next morning)

```powershell
aws ec2 start-instances --region us-east-1 --instance-ids YOUR_INSTANCE_ID

# Wait for it to be running
aws ec2 wait instance-running --region us-east-1 --instance-ids YOUR_INSTANCE_ID

# Get the NEW public IP (it changes after stop/start!)
aws ec2 describe-instances --region us-east-1 `
  --instance-ids YOUR_INSTANCE_ID `
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
```

> [!IMPORTANT]
> **The public IP changes every time you stop/start.** Update your SSH config file (`~/.ssh/config`) with the new IP before reconnecting from your IDE.

---

## Part 8: Download Results & Cleanup

### After All Notebooks Complete

```bash
# On the instance — verify outputs exist
ls checkpoints/
ls experiments/
```

### Download to Local Machine

```powershell
# From your local Windows PowerShell
scp -i "$env:USERPROFILE\.ssh\crowdvision-key.pem" -r `
  ubuntu@YOUR_PUBLIC_IP:~/crowdvision/checkpoints `
  "D:\majproj\crowdvision\"

scp -i "$env:USERPROFILE\.ssh\crowdvision-key.pem" -r `
  ubuntu@YOUR_PUBLIC_IP:~/crowdvision/experiments `
  "D:\majproj\crowdvision\"
```

### Terminate Instance (When Completely Done)

```powershell
# THIS PERMANENTLY DELETES THE INSTANCE AND ALL DATA ON IT
# Make sure you've downloaded everything first!
aws ec2 terminate-instances --region us-east-1 --instance-ids YOUR_INSTANCE_ID

# Verify it's gone
aws ec2 describe-instances --region us-east-1 `
  --instance-ids YOUR_INSTANCE_ID `
  --query 'Reservations[0].Instances[0].State.Name' --output text
# Should say "terminated"
```

### Delete the Key Pair and Security Group

```powershell
aws ec2 delete-key-pair --region us-east-1 --key-name crowdvision-key
aws ec2 delete-security-group --region us-east-1 --group-id YOUR_SG_ID
```

---

## Troubleshooting

### "Permission denied (publickey)" when SSH-ing
```powershell
# Fix key permissions on Windows
icacls "$env:USERPROFILE\.ssh\crowdvision-key.pem" /inheritance:r /grant:r "${env:USERNAME}:(R)"
```

### "Connection timed out" when SSH-ing
Your IP changed. Update the security group:
```powershell
# Remove old rule and add new one with current IP
$MY_IP = (Invoke-RestMethod ifconfig.me)
aws ec2 authorize-security-group-ingress `
  --region us-east-1 --group-id YOUR_SG_ID `
  --protocol tcp --port 22 --cidr "$MY_IP/32"
```

### GPU Out of Memory (OOM)
In the notebook config cell, reduce batch size:
```python
# Change these values
target_size = (288, 384)  # smaller images
batch_size = 2            # smaller batches
```

### Instance Won't Launch (InsufficientInstanceCapacity)
Try a different availability zone:
```powershell
# Try us-east-1a, us-east-1b, us-east-1c, etc.
aws ec2 run-instances --region us-east-1 `
  --placement "AvailabilityZone=us-east-1b" `
  ... (rest of the launch command)
```

### Jupyter Token Lost
```bash
# On the instance, find running Jupyter servers
jupyter server list
```

### IDE Remote-SSH Can't Connect After Stop/Start
1. Get the new IP: `aws ec2 describe-instances ...`
2. Edit `~/.ssh/config` → update the `HostName` line
3. Reconnect from IDE
