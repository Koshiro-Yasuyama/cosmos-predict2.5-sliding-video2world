# Cosmos-Predict2.5 Sliding Video2World Tools

Independent helper repository for running sliding-window future prediction with
NVIDIA Cosmos-Predict2.5.

This repository is intentionally not a GitHub fork of
`nvidia-cosmos/cosmos-predict2.5`. It contains only the custom runner, runbook,
and local run directory layout. Clone the official Cosmos-Predict2.5 repository
separately, then install these files into that repository root.

## Layout

```text
examples/sliding_video2world_future.py
scripts/install_into_cosmos_repo.sh
local_runs/
  inputs/
  prompts/
  outputs/
  work/
RUN_SLIDING_VIDEO2WORLD.md
```

## Install Into A Cosmos-Predict2.5 Clone

```bash
git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git cosmos-predict2.5
git clone https://github.com/Koshiro-Yasuyama/cosmos-predict2.5-sliding-video2world.git

cd cosmos-predict2.5-sliding-video2world
./scripts/install_into_cosmos_repo.sh ../cosmos-predict2.5
```

Then use the official Cosmos-Predict2.5 repository as the Docker project root:

```bash
cd ../cosmos-predict2.5
image_tag=$(docker build -f Dockerfile -q .)

HF_CACHE="$HOME/.cache"
docker run -it --runtime=nvidia --ipc=host --rm \
  -v .:/workspace \
  -v /workspace/.venv \
  -v "$HF_CACHE":/root/.cache \
  -e HF_TOKEN="$HF_TOKEN" \
  "$image_tag"
```

## Quick Run

Put files in the installed Cosmos repository:

```text
local_runs/inputs/input.mp4
local_runs/prompts/input.txt
```

Inside the Docker container:

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

To generate a fixed number of final prediction frames, add for example:

```bash
--num-generated-frames 300
```

See `RUN_SLIDING_VIDEO2WORLD.md` for argument details and trial commands.
