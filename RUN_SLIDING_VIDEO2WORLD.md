# Sliding Video2World Runbook

Use the official Cosmos-Predict2.5 repository as the Docker project root. The
official Docker command mounts that root to `/workspace`, so local run data lives
under `local_runs/` inside the Cosmos repository and appears as
`/workspace/local_runs/` inside the container.

This helper repository is independent from NVIDIA's GitHub repo. Install it into
an official Cosmos-Predict2.5 clone with:

```bash
./scripts/install_into_cosmos_repo.sh /path/to/cosmos-predict2.5
```

After installation, the Cosmos repository will contain:

```text
cosmos-predict2.5/
  examples/sliding_video2world_future.py
  local_runs/
    inputs/
    prompts/
    outputs/
    work/
```

## Server Setup

From the official Cosmos-Predict2.5 repository root on the H200 server:

```bash
cd /path/to/cosmos-predict2.5
git lfs install
git lfs pull
```

Build the official Docker image:

```bash
image_tag=$(docker build -f Dockerfile -q .)
```

Run the container. The only extra mount is the host Hugging Face cache, because
the model checkpoints already live under the default `~/.cache/huggingface/hub`.

```bash
HF_CACHE="$HOME/.cache"

docker run -it --runtime=nvidia --ipc=host --rm \
  -v .:/workspace \
  -v /workspace/.venv \
  -v "$HF_CACHE":/root/.cache \
  -e HF_TOKEN="$HF_TOKEN" \
  "$image_tag"
```

This expects existing Hugging Face checkpoints under the host default cache, for
example `~/.cache/huggingface/hub`. Mounting `~/.cache` to `/root/.cache` lets
the container see the same cache at `/root/.cache/huggingface/hub`.

Inside the container:

```bash
cd /workspace
python examples/sliding_video2world_future.py \
  --repo-root /workspace \
  --input-video /workspace/local_runs/inputs/input.mp4 \
  --output-video /workspace/local_runs/outputs/input_future_14b.mp4 \
  --work-dir /workspace/local_runs/work/input_future_14b \
  --prompt-file /workspace/local_runs/prompts/input.txt \
  --model 14B/post-trained \
  --num-gpus 8 \
  --parallelism context \
  --memory-profile speed \
  --turbo-profile exact \
  --window-schedule warmup5 \
  --future-offset 9 \
  --fps 30 \
  --output-fps 30
```

Use `--future-offset 9` for source frame 1 to target frame 10. Use
`--future-offset 10` for ten frames after the last conditioning frame.

## Trial Run

Before running a full 20-second video, estimate runtime with a small subset:

```bash
python examples/sliding_video2world_future.py \
  --repo-root /workspace \
  --input-video /workspace/local_runs/inputs/input.mp4 \
  --output-video /workspace/local_runs/outputs/input_future_trial.mp4 \
  --work-dir /workspace/local_runs/work/input_future_trial \
  --prompt-file /workspace/local_runs/prompts/input.txt \
  --model 14B/post-trained \
  --num-gpus 8 \
  --parallelism context \
  --memory-profile speed \
  --turbo-profile exact \
  --window-schedule warmup5 \
  --future-offset 9 \
  --fps 30 \
  --output-fps 30 \
  --max-windows 32
```

If a single 14B replica fits on one H200 and throughput is more important than
minimum memory use, test data parallelism:

```bash
python examples/sliding_video2world_future.py \
  --repo-root /workspace \
  --input-video /workspace/local_runs/inputs/input.mp4 \
  --output-video /workspace/local_runs/outputs/input_future_trial_data.mp4 \
  --work-dir /workspace/local_runs/work/input_future_trial_data \
  --prompt-file /workspace/local_runs/prompts/input.txt \
  --model 14B/post-trained \
  --num-gpus 8 \
  --parallelism data \
  --memory-profile speed \
  --turbo-profile exact \
  --window-schedule warmup5 \
  --future-offset 9 \
  --fps 30 \
  --output-fps 30 \
  --max-windows 32
```

Keep whichever mode is faster and stable on the server.
