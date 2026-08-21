## Requirements

- ☑️ I understand it is strictly prohibited to use AI to write PR description
- AI usage disclosure: 

## Model information

- Name: 
- Homepage: 
- Reference implementation: https://github.com/huggingface/transformers/tree/main/src/transformers/models/____

## Supported checkpoints

- https://huggingface.co/___/___

## Verification

Run mlx-lm command:

```console
mlx_lm.generate --model ___/___ -p "The secret to baking a good cake is" -m 1024
```

Output:

```
Paste the output of the command
```

## Reference output (if you have access to CUDA hardware)

Run python code:

```python
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="___/___")
response = pipeline("the secret to baking a good cake is", max_new_tokens=1024)
print(response[0]['generated_text'])
```

Output:

```
Paste the output of the command
```
