import json
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer


class GRPODataset:
    """
    Dataset wrapper for GRPO training data.
    Each example should have a 'prompt' and 'answer' field.
    Returns data in (prompt_tokens, answer_tokens, prompt_str, answer_str) tuple format.
    """
    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        system_key: str = "system",
        type_key: str = "type",
        use_chat_template: bool = False,
        use_prompt: bool = False
    ):
        self._data = []
        for item in data:
            prompt_str = str(item[prompt_key])
            answer_str = str(item[answer_key])
            type_info = item.get(type_key, None)
            if use_chat_template:
                default_system_str = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."
                system_str = item.get(system_key, default_system_str)
                prompt_tokens = tokenizer.apply_chat_template(
                    [
                        {'role': 'system', 'content': system_str},
                        {'role': 'user', 'content': prompt_str}
                    ],
                    add_generation_prompt=True
                )
                answer_tokens = tokenizer.encode(answer_str)
            else:
                if use_prompt:
                    prompt_tokens = tokenizer.encode(f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. User: {prompt_str} Assistant: """)
                else:
                    prompt_tokens = tokenizer.encode(prompt_str)
                answer_tokens = tokenizer.encode(answer_str)
            self._data.append((prompt_tokens, answer_tokens, prompt_str, answer_str, type_info))

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int], str, str]:
        return self._data[idx]

    def __len__(self) -> int:
        return len(self._data)


class TextDataset:
    """
    Light-weight wrapper to hold a dataset.
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        text_key: str = "text",
    ):
        self._data = [d for d in data]
        self.tokenizer = tokenizer
        self.text_key = text_key

    def process(self, d):
        d = self.tokenizer.encode(d[self.text_key])
        if d[-1] != self.tokenizer.eos_token_id:
            d.append(self.tokenizer.eos_token_id)
        return d

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


class ChatDataset:
    """
    A dataset for chat data in the format of {"messages": [...]}
    https://platform.openai.com/docs/guides/fine-tuning/example-format
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        chat_key: str = "messages",
        mask_prompt: bool = False,
    ):
        self._data = [d for d in data]
        self.chat_key = chat_key
        self.mask_prompt = mask_prompt
        self.tokenizer = tokenizer

    def process(self, d):
        messages = d[self.chat_key]
        tools = d.get("tools", None)
        tokens = self.tokenizer.apply_chat_template(messages, tools=tools)
        if self.mask_prompt:
            messages = messages[:-1]
            offset = len(self.tokenizer.apply_chat_template(messages, tools=tools))
            return (tokens, offset)
        else:
            return tokens

    def itemlen(self, idx: int):
        return len(self._data[idx])

    def __getitem__(self, idx: int):
        return self._data[idx]
 
    def __len__(self):
        return len(self._data)


class CompletionsDataset:
    """
    A dataset for prompt-completion data in the format of {"prompt": ..., "completion": ...}
    or using user-provided keys for prompt and completion values
    https://platform.openai.com/docs/guides/fine-tuning/example-format
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        prompt_key: str,
        completion_key: str,
        mask_prompt: bool,
    ):
        self._data = [d for d in data]
        self.prompt_key = prompt_key
        self.completion_key = completion_key
        self.mask_prompt = mask_prompt
        self.tokenizer = tokenizer

    def process(self, d):
        tokens = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": d[self.prompt_key]},
                {"role": "assistant", "content": d[self.completion_key]},
            ],
        )
        if self.mask_prompt:
            offset = len(
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": d[self.prompt_key]}]
                )
            )
            return (tokens, offset)

        return tokens

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


class ConcatenatedDataset:
    def __init__(self, data: List[Any]):
        self._data = data
        self._len = sum(len(d) for d in self._data)

    def __getitem__(self, idx: int):
        for data in self._data:
            j = idx - len(data)
            if j < 0:
                break
            idx = j
        return data[idx]

    def __len__(self):
        return self._len


class CacheDataset:
    def __init__(self, data: Any):
        self._data = data
        self._proc_data = [None] * len(data)

    def itemlen(self, idx: int):
        return len(self._data[idx])

    def __getitem__(self, idx: int):
        if self._proc_data[idx] is None:
            self._proc_data[idx] = self._data.process(self._data[idx])
        return self._proc_data[idx]

    def __len__(self):
        return len(self._data)


def create_dataset(
    data,
    tokenizer: PreTrainedTokenizer,
    config,
):
    mask_prompt = getattr(config, "mask_prompt", False)
    text_feature = getattr(config, "text_feature", "text")
    prompt_feature = getattr(config, "prompt_feature", "prompt")
    type_feature = getattr(config, "type_feature", "type")
    completion_feature = getattr(config, "completion_feature", "completion")
    answer_feature = getattr(config, "answer_feature", "answer")
    system__feature = getattr(config, "system__feature", "system")
    chat_feature = getattr(config, "chat_feature", "messages")
    training_mode = getattr(config, "training_mode", "normal")
    use_chat_template = getattr(config, "use_chat_template", "normal")
    use_prompt = getattr(config, "use_prompt", "normal")
    sample = data[0]

    if training_mode == "normal":
        if prompt_feature in sample and completion_feature in sample:
            return CompletionsDataset(
                data, tokenizer, prompt_feature, completion_feature, mask_prompt
            )
        elif chat_feature in sample:
            return ChatDataset(
                data, tokenizer, chat_key=chat_feature, mask_prompt=mask_prompt
            )
        elif text_feature in sample:
            if mask_prompt:
                raise ValueError("Prompt masking not supported for text dataset.")
            return TextDataset(data, tokenizer, text_key=text_feature)
        else:
            raise ValueError(
                "Unsupported data format, check the supported formats here:\n"
                "https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md#data."
            )
    else:
        return GRPODataset(
            data=data,
            tokenizer=tokenizer,
            prompt_key=prompt_feature,
            answer_key=answer_feature,
            system_key=system__feature,
            type_key=type_feature,
            use_chat_template=use_chat_template,
            use_prompt=use_prompt
        )


def load_local_dataset(
    data_path: Path,
    tokenizer: PreTrainedTokenizer,
    config,
):
    def load_subset(path):
        if not path.exists():
            return []
        with open(path, "r") as fid:
            data = [json.loads(l) for l in fid]
        return create_dataset(data, tokenizer, config)

    names = ("train", "valid", "test")
    train, valid, test = [load_subset(data_path / f"{n}.jsonl") for n in names]
    return train, valid, test


def load_hf_dataset(
    data_id: str,
    tokenizer: PreTrainedTokenizer,
    config,
):
    from datasets import exceptions, load_dataset

    try:
        dataset = load_dataset(data_id)

        names = ("train", "valid", "test")

        train, valid, test = [
            (
                create_dataset(dataset[n], tokenizer, config)
                if n in dataset.keys()
                else []
            )
            for n in names
        ]

    except exceptions.DatasetNotFoundError:
        raise ValueError(f"Not found Hugging Face dataset: {data_id} .")

    return train, valid, test


def load_custom_hf_dataset(args, tokenizer: PreTrainedTokenizer):
    import datasets

    def create_hf_dataset(dataset_name, config, split, hf_config):
        ds = datasets.load_dataset(
            dataset_name,
            split=split,
            **hf_config,
        )
        return create_dataset(ds, tokenizer, config)

    dataset_collection = args.hf_dataset
    if isinstance(dataset_collection, dict):
        dataset_collection = [dataset_collection]

    collection = []
    for ds in dataset_collection:
        ds_path = ds["path"]
        print(f"Loading Hugging Face dataset {ds_path}.")
        ds["mask_prompt"] = getattr(args, "mask_prompt", False) if config.training_mode == 'normal' else False
        config = types.SimpleNamespace(**ds)
        hf_config = ds.get("config", {})
        if args.train:
            train_split = ds.get("train_split", "train[:80%]")
            valid_split = ds.get("valid_split", "train[-10%:]")
            train = create_hf_dataset(
                ds_path,
                config,
                train_split,
                hf_config,
            )
            valid = create_hf_dataset(
                ds_path,
                config,
                valid_split,
                hf_config,
            )
        else:
            train, valid = [], []

        if args.test:
            test_split = ds.get("test_split")
            test = create_hf_dataset(
                ds_path,
                config,
                test_split,
                hf_config,
            )
        else:
            test = []

        collection.append((train, valid, test))

    if len(collection) == 1:
        return collection[0]

    # Otherwise concatenate them
    return tuple(map(ConcatenatedDataset, zip(*collection)))


def load_dataset(args, tokenizer: PreTrainedTokenizer):
    if getattr(args, "hf_dataset", False):
        train, valid, test = load_custom_hf_dataset(args, tokenizer)
    else:
        data_path = Path(args.data)
        if data_path.exists():
            train, valid, test = load_local_dataset(data_path, tokenizer, args)
        else:
            print(f"Loading Hugging Face dataset {args.data}.")
            train, valid, test = load_hf_dataset(args.data, tokenizer, args)

    if args.train and len(train) == 0:
        raise ValueError(
            "Training set not found or empty. Must provide training set for fine-tuning."
        )
    if args.train and len(valid) == 0:
        raise ValueError(
            "Validation set not found or empty. Must provide validation set for fine-tuning."
        )
    if args.test and len(test) == 0:
        raise ValueError(
            "Test set not found or empty. Must provide test set for evaluation."
        )
    return train, valid, test
