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

## Inputs

Place the input video and prompt file under the installed Cosmos repository:

```text
local_runs/inputs/input.mp4
local_runs/prompts/input.txt
```

`local_runs/` is intentionally ignored by git except for `.gitkeep`
placeholders, so source videos, temporary frames, and generated outputs are not
committed.

## Full Command

Inside the container, run:

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

## Argument Reference

Core paths:

| Argument | Meaning |
| --- | --- |
| `--repo-root /workspace` | Cosmos-Predict2.5 repository root inside Docker. |
| `--input-video ...` | Input source video. Put it under `/workspace/local_runs/inputs/`. |
| `--output-video ...` | Final assembled prediction video. Use `/workspace/local_runs/outputs/`. |
| `--work-dir ...` | Temporary source-frame cache, manifests, generated PNGs, and intermediate files. Use `/workspace/local_runs/work/<run_name>`. |
| `--prompt-file ...` | Text prompt file. Use this instead of a long inline `--prompt` for reproducibility. |

Model and GPU settings:

| Argument | Meaning |
| --- | --- |
| `--model 14B/post-trained` | Highest-quality base Video2World model currently targeted by this runner. Use `2B/post-trained` only for faster lower-cost testing. |
| `--num-gpus 8` | Number of H200 GPUs to use. |
| `--parallelism context` | Official-style multi-GPU context parallelism. Safest starting point for 14B. |
| `--parallelism data` | Runs independent single-GPU workers. Test this if one 14B replica fits on one H200 and throughput matters. |
| `--memory-profile speed` | Keeps components resident where possible for H200-class memory. Use `balanced` or `lowvram` only if memory is unstable. |

Prediction schedule:

| Argument | Meaning |
| --- | --- |
| `--turbo-profile exact` | One Cosmos generation per output frame. Keeps exact warmup/sliding semantics. |
| `--window-schedule warmup5` | Uses `1`, `1-2`, `1-3`, `1-4`, `1-5`, then `2-6`, `3-7`, ... as conditioning windows. |
| `--future-offset 9` | Predicts the frame 9 frames after the last real conditioning frame. For the first source frame, this targets absolute frame 10. |
| `--stride 1` | Moves the conditioning window by one input frame for each output frame. This is the default. |
| `--num-generated-frames N` | Generates exactly N final prediction frames. Use this for normal output length control. |
| `--max-windows N` | Older/testing name for the same desired window count. Prefer `--num-generated-frames`. |

Frame rate and output:

| Argument | Meaning |
| --- | --- |
| `--fps 30` | FPS used for frame extraction and temporary conditioning clips. Match the input video FPS. |
| `--output-fps 30` | FPS of the final assembled video. Use `30` for a 30 FPS output. |
| `--direct-frame-output` | Default. Saves the target prediction frames directly as PNGs before assembling the final MP4. |

The command above generates one output frame for each selected input window. A
30 FPS, 20 second input has about 600 frames, so with `--stride 1` and no
`--num-generated-frames`, expect about 600 Cosmos generations. Add
`--num-generated-frames 300` to stop after 300 final prediction frames, for
example.

## Trial Run

Before running a full 20-second video, estimate runtime with a small subset. This example generates exactly 32 final prediction frames:

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
  --num-generated-frames 32
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
  --num-generated-frames 32
```

Keep whichever mode is faster and stable on the server.
