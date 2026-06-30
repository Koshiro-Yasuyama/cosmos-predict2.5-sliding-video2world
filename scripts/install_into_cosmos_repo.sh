#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /path/to/cosmos-predict2.5" >&2
  exit 2
fi

target_root="$1"
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -f "$target_root/Dockerfile" ] || [ ! -f "$target_root/examples/inference.py" ]; then
  echo "Target does not look like a Cosmos-Predict2.5 repository: $target_root" >&2
  exit 1
fi

mkdir -p "$target_root/examples"
mkdir -p "$target_root/local_runs/inputs"
mkdir -p "$target_root/local_runs/prompts"
mkdir -p "$target_root/local_runs/outputs"
mkdir -p "$target_root/local_runs/work"

cp "$source_root/examples/sliding_video2world_future.py" \
  "$target_root/examples/sliding_video2world_future.py"
cp "$source_root/RUN_SLIDING_VIDEO2WORLD.md" \
  "$target_root/RUN_SLIDING_VIDEO2WORLD.md"

touch "$target_root/local_runs/inputs/.gitkeep"
touch "$target_root/local_runs/prompts/.gitkeep"
touch "$target_root/local_runs/outputs/.gitkeep"
touch "$target_root/local_runs/work/.gitkeep"

echo "Installed sliding Video2World tools into: $target_root"
