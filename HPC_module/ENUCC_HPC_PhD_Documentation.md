# ENUCC HPC -- Operational Documentation for MRes

**User:** 40542237\
**Cluster:** enucc (Alces Flight + SLURM)\
**OS:** Rocky Linux 8.10\
**Scheduler:** SLURM 25.11.0

------------------------------------------------------------------------

# 1. Cluster Architecture Overview

## Login Node

-   Hostname: `login1.enucc.enu.alces.network`
-   Lightweight VM
-   Used for:
    -   Submitting jobs
    -   Editing files
    -   Managing data
    -   Running lightweight commands

⚠️ Never run heavy computation here.

------------------------------------------------------------------------

## CPU Nodes (`nodes`, `short`, `long` partitions)

Example node: `node01`

### Hardware per node:

-   CPU: AMD EPYC 7513
-   64 physical cores
-   ThreadsPerCore: 1
-   2 NUMA nodes (0--31, 32--63)
-   RAM: 502 GiB
-   8 nodes total
-   Max runtime:
    -   short → 1 day
    -   nodes → 3 days
    -   long → 7 days

### Best for:

-   Parallel OpenSfM
-   Large-scale point cloud filtering
-   Poisson reconstruction
-   ICP alignment
-   LDA topic modelling
-   CPU-bound preprocessing

------------------------------------------------------------------------

## High Memory Nodes (`himem`)

Example node: `himem01`

### Hardware:

-   64 cores
-   \~2.0 TiB RAM
-   2 nodes total
-   Max runtime: 3 days

### Best for:

-   Ultra-dense reconstruction
-   Massive LiDAR fusion
-   Full-resolution mesh generation
-   Whole-building volumetric analysis
-   Memory-heavy research experiments

------------------------------------------------------------------------

## GPU Nodes (`gpu`)

Example node: `gpu03`, `gpu04`

### Hardware:

-   64 CPU cores
-   250 GiB RAM
-   2 × NVIDIA A100 80GB PCIe per node
-   CUDA 12.8
-   4 nodes total → 8 A100 GPUs

### GPU Topology:

-   GPU0 connected to CPU cores 0--31 (NUMA 0)
-   GPU1 connected to CPU cores 32--63 (NUMA 1)
-   Interconnect: PCIe + SMP (SYS)
-   No NVLink

Verified allocation:

    CUDA_VISIBLE_DEVICES=0,1

### Best for:

-   CNN training
-   Transformer-based segmentation
-   3D sparse conv networks
-   Thermal anomaly detection
-   Multi-modal fusion
-   Scaling experiments

------------------------------------------------------------------------

# 2. SLURM Basics

SLURM manages resource allocation.

## Interactive session

    srun --pty -p <partition> --time=HH:MM:SS bash

Example:

    srun --pty -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 bash

## Batch submission

    sbatch job.sh

## Monitor jobs

    squeue -u 40542237

## Show node info

    scontrol show node node01

------------------------------------------------------------------------

# 3. Meaning of Common SLURM Flags

  Flag                Meaning
  ------------------- ----------------------
  `--pty`             Interactive terminal
  `-p`                Partition
  `--time`            Max allowed runtime
  `--cpus-per-task`   CPU cores allocated
  `--mem`             Memory requested
  `--gres=gpu:N`      Number of GPUs

------------------------------------------------------------------------

# 4. Job Lifecycle

## srun (Interactive)

-   Starts interactive session
-   Ends when `exit` is typed
-   Resources freed immediately
-   If SSH drops → job dies

## sbatch (Batch)

-   Runs independently
-   Continues after logout
-   Output written to file
-   Stops when finished or time limit reached

------------------------------------------------------------------------

# 5. Output Handling

## Interactive (srun)

-   Output prints to screen
-   Files written to current directory (`pwd`)

## Batch (sbatch)

Default:

    slurm-<jobID>.out

Recommended inside script:

    #SBATCH --output=logs/%x_%j.out
    #SBATCH --error=logs/%x_%j.err

------------------------------------------------------------------------

# 6. Storage Areas

## Home

    /users/40542237

Small quota. For scripts and configs only.

## Scratch

    /mnt/scratch/users/40542237

For datasets, reconstructions, models, logs. (`~/sharedscratch` symlinks here — use this for batch jobs. `~/localscratch` is node-local only.)

Recommended structure:

    /mnt/scratch/users/40542237/
        pvlm/
            data/
            models/
            logs/

------------------------------------------------------------------------

# 7. Data Transfer

## Upload

    rsync -avz ~/devs/phd/pvlm/ ext.login.enucc:/mnt/scratch/users/40542237/pvlm/

## Download

    rsync -avz ext.login.enucc:/mnt/scratch/users/40542237/pvlm/ ~/devs/phd/pvlm/

------------------------------------------------------------------------

# 8. SSH Configuration

Example `~/.ssh/config`:

    Host napier
        HostName napier.hpc
        User 40542237
        IdentityFile ~/.ssh/id_ed25519

Connect using:

    ssh ext.login.enucc

------------------------------------------------------------------------

# 9. tmux Usage

Useful for: - Protecting interactive `srun` - Reconnecting if SSH drops

Not needed for `sbatch` jobs.

------------------------------------------------------------------------

# 10. Example sbatch Template (GPU)

``` bash
#!/bin/bash
#SBATCH --job-name=defect_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd /mnt/scratch/users/40542237/pvlm
module load apps/anaconda3/2024.10/bin
conda activate myenv

python train.py
```

Submit:

    sbatch job.sh

------------------------------------------------------------------------

# Final Assessment

Infrastructure available: - 8 × 64-core CPU nodes - 2 × 2TB RAM nodes -
8 × A100 80GB GPUs

This enables: - Large-scale photogrammetry - Dense reconstruction
experiments - GPU deep learning research - Multi-GPU benchmarking - Edge
vs HPC performance analysis

The limiting factor is no longer hardware --- it is experimental design
and pipeline optimisation.
