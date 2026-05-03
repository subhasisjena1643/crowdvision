# CrowdVision: Project Proof and Execution Plan

## 1. Project Proof

**Project name:** CrowdVision

**Project description:**
CrowdVision is an academic computer vision project focused on crowd analytics in public-scene imagery and video. The work is organized as a small research pipeline rather than an always-on production service. The system is intended to train, validate, and compare multiple deep learning models for three related tasks:

1. Crowd density estimation and counting
2. Anomaly detection in crowd scenes
3. Short-term crowd flow forecasting

**Why this project exists:**
The goal is to build a reproducible research environment that can generate meaningful training results, evaluation metrics, and checkpoints for a final report or presentation. The scope is intentionally bounded so that the environment can be created, used intensively for a short window, and then deleted.

**Model and workflow characteristics:**
- The codebase includes notebook-driven experimentation and training scripts.
- The workflow relies on repeated training, validation, and checkpoint generation.
- The experiments need GPU acceleration to finish in a practical time window.
- The work is best run as a single controlled training session, not a distributed cluster.

**Representative datasets:**
- ShanghaiTech A / B for density estimation
- JHU-Crowd++ for large-scale crowd scenes
- UCSD Anomaly Dataset for anomaly detection
- METR-LA for forecasting-related experiments

**Data handling approach:**
The datasets are stored externally and copied into the instance before training. Final outputs, checkpoints, and logs are synced back to Google Drive after each run. This avoids keeping the cloud instance active any longer than necessary.

**Why GPU is required:**
The project requires GPU acceleration because CPU-only training would take too long for the intended schedule and would make iteration and validation inefficient. GPU access is needed for practical training speed, repeatability, and checkpoint generation. The requested capacity is not for interactive consumer usage or broad scaling; it is specifically for batch training and evaluation.

**Storage needs:**
- Codebase, notebooks, and configuration files
- Dataset extraction and preprocessing artifacts
- Model checkpoints and training logs
- Final metrics, plots, and evaluation outputs

## 2. Execution Plan

**Region:** us-east-1

**Why this region:**
us-east-1 is selected because it typically has broad service support, good instance availability, and stable access to the EC2 ML-related resources needed for the project. Using one region also keeps the workflow simple and reduces cross-region transfer or management overhead.

**Initial quota request:**
- Start with one On-Demand G-family instance only
- Prefer `g5.xlarge` for the initial launch
- Keep the ask intentionally conservative to reduce risk and improve approval likelihood
- Expand later only if the workload genuinely needs it

**Why one instance first:**
The training plan is designed to be efficient without requiring multi-node orchestration. A single GPU instance is easier to approve, easier to monitor, cheaper to run, and simpler to shut down. It also reduces the chance of billing surprises and makes the environment easier to tear down after training.

**Instance usage strategy:**
1. Launch one GPU instance in `us-east-1`.
2. Attach only the storage required for the active training window.
3. Copy the codebase and required datasets from Google Drive into local EC2 storage.
4. Install dependencies and verify GPU support before training begins.
5. Run the notebooks or scripts in a single-instance workflow.
6. Save checkpoints, logs, and results back to Google Drive at the end of each run.
7. Stop or terminate the instance as soon as the training window ends.

**Planned training sequence:**
1. Environment and data validation notebook.
2. Single-task density estimation run.
3. Single-task forecasting run.
4. Single-task anomaly detection run.
5. Optional fine-tuning or comparison runs only if time remains.

This order is chosen so that the core outputs are produced first. If time or budget becomes constrained, the highest-value results are already completed.

**Expected duration:**
- Short-term intensive run of approximately 1 to 2 weeks
- The environment is temporary and will be removed after the work is complete

**Cost-control measures:**
- AWS Budgets and billing alerts enabled before launch
- Limit to one instance at a time
- Use only the minimum required storage for the current run
- Stop the instance when training is not actively running
- Sync outputs to Google Drive before teardown
- Delete unused volumes and snapshots after completion

**Risk control approach:**
The training plan avoids distributed jobs, avoids always-on usage, and avoids open-ended scaling. The intent is to do the minimum amount of cloud work needed to complete the experiments, then terminate the resources cleanly.

**Post-training cleanup:**
- Sync final outputs to Google Drive
- Verify backups before shutting down
- Stop and terminate the EC2 instance
- Delete any unused volumes and snapshots
- Confirm no billable resources remain active

## 3. Summary

CrowdVision is a focused academic ML project with a short, controlled execution window and a clear stop condition. The requested GPU quota is for one on-demand instance so the project can complete efficiently, generate reliable results, and be shut down immediately afterward. The request is intentionally conservative and designed to minimize risk, cost, and operational overhead.
