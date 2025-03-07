set -e
uv run scripts/download.py --repo_id $1
uv run scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$1
