python3 -m mlx_lm quant.dwq \
  --model Qwen/Qwen3-30B-A3B \
  --bits 4 \
  --group-size 64 \
  --mlx-path Qwen3-30B-A3B-DWQ \
  --num-samples 128 \
  --batch-size 1 \
  --max-seq-length 256 \
  --grad-checkpoint


python3 -m mlx_lm quant.dwq \
  --model ckpt_path \
  --bits 4 \
  --group-size 64 \
  --mlx-path sarvam_moe-dwq \
  --num-samples 128 \
  --batch-size 1 \
  --max-seq-length 256 \
  --grad-checkpoint