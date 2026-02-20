mlx_lm.evaluate \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --tasks arc_challenge arc_easy hellaswag

mlx_lm.evaluate \
    --model mistralai/Mistral-7B-Instruct-v0.2-mxfp4 \
    --tasks arc_challenge arc_easy hellaswag          