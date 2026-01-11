    # mlx_lm.convert \
    # --hf-path mistralai/Mistral-7B-Instruct-v0.2 \
    # -q \
    # --q-bits 4 \
    # --q-mode mxfp4 \
    # --q-group-size 32 \
    # --mlx-path mistralai/Mistral-7B-Instruct-v0.2-mxfp4
# python -m mlx_lm.quant.dwq \
#   --model granite-3b-a800m-instruct \
#   --bits 4 \
#   --group-size 64 \
#   --mlx-path granite-3b-a800m-DWQ \
#   --num-samples 128 \
#   --batch-size 1 \
#   --max-seq-length 256 \
#   --grad-checkpoint

python << 'EOF'
import sys
sys.argv = ['dwq', 
    '--model', 'granite-3b-a800m-instruct',
    '--bits', '4',
    '--group-size', '64', 
    '--mlx-path', 'granite-3b-a800m-DWQ',
    '--num-samples', '128',
    '--batch-size', '1',
    '--max-seq-length', '256',
    '--grad-checkpoint'
]
from mlx_lm.quant.dwq import main
main()
EOF

mlx_lm.evaluate \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --tasks winogrande boolq arc_challenge arc_easy hellaswag openbookqa piqa social_iqa

mlx_lm.evaluate \
    --model mistralai/Mistral-7B-Instruct-v0.2-mxfp4 \
    --tasks winogrande boolq arc_challenge arc_easy hellaswag openbookqa piqa social_iqa                     