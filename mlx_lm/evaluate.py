# Copyright © 2024 Apple Inc.

"""
Adapted from a PyTorch implementation by David Grangier
"""

import argparse
import collections
import copy
import json
import logging
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any, Optional

import lm_eval
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models import huggingface
from tqdm import tqdm

from .generate import stream_generate
from .models.base import create_causal_mask
from .models.cache import make_prompt_cache
from .utils import common_prefix_len, load


def _rstrip_until(s, untils):
    """Limit a string <s> to the first occurrence of any substring in untils."""
    l = len(s)
    f = [s.find(u) for u in untils]
    f = [l if x < 0 else x for x in f]
    return s[: min(f)]


def _pad_inputs(inputs):
    lengths = np.array([len(x) for x in inputs])
    maxlen = lengths.max()
    padded = np.stack(
        [np.pad(x, (0, maxlen - len(x))) for x in inputs],
        axis=0,
    )
    return mx.array(padded), mx.array(lengths)


def chat_template_fn(**extra_kwargs):
    def apply_chat_template(self, chat_history, add_generation_prompt=True) -> str:
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
            **extra_kwargs,
        )

    return apply_chat_template


@register_model("mlxlm")
class MLXLM(LM):

    tokenizer_name = huggingface.HFLM.tokenizer_name
    apply_chat_template = chat_template_fn()

    def __init__(
        self,
        path_or_hf_repo: str,
        max_tokens: Optional[int] = None,
        use_chat_template: Optional[bool] = None,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        tokenizer_config = {"trust_remote_code": True if trust_remote_code else None}
        self._model, self.tokenizer = load(
            path_or_hf_repo, tokenizer_config=tokenizer_config
        )
        self._max_tokens = max_tokens or self.tokenizer.model_max_length
        self._batch_size = 8
        self.use_chat_template = use_chat_template
        if use_chat_template is None:
            self.use_chat_template = self.tokenizer.chat_template is not None

    def _process_prompt(self, prompt, step_size: int = 2048):
        prompt = mx.array(prompt)[None]
        cache = make_prompt_cache(self._model)
        for i in range(0, prompt.shape[1], step_size):
            logits = self._model(prompt[:, i : i + step_size], cache=cache)
            mx.eval([c.state for c in cache])
            mx.clear_cache()
        logprobs = nn.log_softmax(logits[:, -1, :].astype(mx.float32))
        return logprobs, cache

    def _score_fn(self, inputs, cache: Optional[Any] = None, step_size: int = 2048):
        inputs, lengths = _pad_inputs(inputs)
        inputs, targets = inputs[..., :-1], inputs[..., 1:]

        cache = cache or make_prompt_cache(self._model)
        lengths += cache[0].offset

        scores, is_greedy = [], []
        for i in range(0, inputs.shape[1], step_size):
            inp = inputs[:, i : i + step_size]
            T = inp.shape[1]

            offset = cache[0].offset
            mask = create_causal_mask(T, offset, lengths=lengths)

            logits = self._model(inp, cache=cache, mask=mask)
            log_probs = nn.log_softmax(logits.astype(mx.float32))

            score = mx.take_along_axis(
                log_probs, targets[:, i : i + step_size, mx.newaxis], axis=-1
            )[..., 0]
            ig = targets[:, i : i + step_size] == mx.argmax(logits, axis=-1)
            ig = mx.where(mx.arange(T) + offset < lengths[:, None], ig, False)

            mx.eval(score, ig)
            mx.clear_cache()

            is_greedy.append(ig)
            scores.append(score)

        scores = mx.concatenate(scores, axis=1)
        is_greedy = mx.concatenate(is_greedy, axis=1)

        return scores, lengths, is_greedy

    def _tokenize(self, texts):
        return [
            tuple(
                self.tokenizer.encode(t, add_special_tokens=not self.use_chat_template)
            )
            for t in texts
        ]

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        """Compute log-likelihood of generating a continuation from a context.
        Downstream tasks should attempt to use loglikelihood instead of other
        LM calls whenever possible.
        :param requests: list[Instance]
            A list of Instance objects, with property `args` which returns a tuple (context, continuation).
            `context: str`
                Context string. Implementations of LM must be able to handle an
                empty context string.
            `continuation: str`
                The continuation over which log likelihood will be calculated. If
                there is a word boundary, the space should be in the continuation.
                For example, context="hello" continuation=" world" is correct.
        :return: list[tuple[float, bool]]
            A list of pairs (logprob, isgreedy)
            `logprob: float`
                The log probability of `continuation`.
            `isgreedy`:
                Whether `continuation` would be generated by greedy sampling from `context`.
        """
        logging.info("Estimating loglikelihood for %d pairs." % len(requests))

        group = mx.distributed.init()

        # Group by common prefix
        group_reqs = collections.defaultdict(list)
        for idx, req in enumerate(requests):
            group_reqs[req.args[0]].append((idx, req.args[1]))
        questions = list(group_reqs.keys())
        responses = []
        indices = []
        for v in group_reqs.values():
            idx, resp = zip(*v)
            indices.append(idx)
            responses.append(resp)

        # split data accross ranks
        questions = questions[group.rank() :: group.size()]
        responses = responses[group.rank() :: group.size()]

        long_completions = 0
        scores, is_greedy = [], []
        for q, rs in tqdm(zip(questions, responses), total=len(questions)):
            prefix = self._tokenize([q])[0]
            full_sequences = self._tokenize([q + r for r in rs])
            max_completed_l = max(len(s) for s in full_sequences)

            # compute truncation length
            truncation = max(0, max_completed_l - self._max_tokens - 1)
            orig_prefix_l = len(prefix)
            prefix_l = max(len(prefix) - truncation, 0)
            prefix = prefix[len(prefix) - prefix_l :]

            # If the entire prompt got truncated ignore the question
            if prefix_l == 0:
                long_completions += 1
                all_scores.extend([-float("inf")] * len(rs))
                all_is_greedy.extend([False] * len(rs))
                continue

            # model scoring, returns num_requests x (logp, is_greedy, length).
            logprobs, cache = self._process_prompt(prefix)
            max_idx = mx.argmax(logprobs).item()

            for s in full_sequences:
                inputs = s[len(prefix) :]
                # The logprobs from the last token of the prompt are
                # for the first input token
                scores.append(logprobs[0, inputs[0]].item())
                is_greedy.append((inputs[0] == max_idx))

                if len(inputs) == 1:
                    continue
                score, _, ig = self._score_fn(
                    mx.array(inputs)[None, :], cache=copy.deepcopy(cache)
                )
                scores[-1] += mx.sum(score).item()
                is_greedy[-1] &= mx.all(ig).item()

        if long_completions > 0:
            logging.info(
                f"Prefix eliminated for {long_completions} requests with "
                + "completion longer than context."
            )

        # All gather the results across nodes
        num_results = len(requests)
        per_group = mx.distributed.all_max(len(scores), stream=mx.cpu).item()
        scores = scores + [0] * (per_group - len(scores))
        is_greedy = is_greedy + [False] * (per_group - len(is_greedy))
        scores = mx.array(scores)
        is_greedy = mx.array(is_greedy)
        scores = mx.distributed.all_gather(scores, stream=mx.cpu)
        is_greedy = mx.distributed.all_gather(is_greedy, stream=mx.cpu)
        mx.eval(scores, is_greedy)

        # Arrange the indices to match the scores from each node and then
        # inverse sort the scores
        all_indices = []
        for rank in range(group.size()):
            rank_indices = [
                idx for question in indices[rank :: group.size()] for idx in question
            ]
            rank_indices += [num_results] * (per_group - len(rank_indices))
            all_indices.extend(rank_indices)
        inv_sort = mx.argsort(mx.array(all_indices))
        scores = scores[:num_results][inv_sort]
        is_greedy = is_greedy[:num_results][inv_sort]

        return list(zip(scores.tolist(), is_greedy.tolist()))

    def loglikelihood_rolling(self, requests) -> list[float]:
        """Compute full log-likelihood of a string, with no truncation, for perplexity computation
        - We will use the full max context length of the model.
        - For inputs that exceed the max context length, we divide the tokenized string into chunks of up to
        the max context length.
        - IMPORTANT: Each document's loglikelihood/perplexity is computed *separately*, unlike other implementations
          which may simply concatenate multiple documents together.
        - IMPORTANT: We maximize the amount of context for each prediction. Specifically, for inputs that we break into
          multiple chunks, the last input will still a full-sized context.
          Example:
            Input tokens: [ 0 1 2 3 4 5 6 7 8 9 ]
            Prefix: EOT
            Max context length: 4
            Resulting input/prediction pairs:
                INPUT:  EOT   0   1   2
                PRED:     0   1   2   3
                INPUT:    3   4   5   6
                PRED:     4   5   6   7
                INPUT:    5   6   7   8
                PRED:             8   9
          Observe that:
            1. Each token is predicted exactly once
            2. For the last pair, we provide the full context, but only score the last two tokens
        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context,).
            string: str
                String for which we are computing overall loglikelihood
        :return: list[tuple[float]]
            A list of tuples (logprob,)
            logprob: float
                The log probability of `context` conditioned on the EOT token.
        """
        logging.info(
            "Estimating loglikelihood rolling for %d sequences." % len(requests)
        )
        inputs = self._tokenize([req.args[0] for req in requests])
        all_scores = []
        for i in tqdm(range(0, len(inputs), self._batch_size)):
            batch = inputs[i : i + self._batch_size]
            scores, lengths, _ = self._score_fn(batch)
            mask = mx.arange(scores.shape[-1]) < lengths[:, None]
            all_scores.extend((mask * scores).sum(axis=-1).tolist())

        return all_scores

    def generate_until(self, requests) -> list[str]:
        """Generate greedily until a stopping sequence
        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context, until).
            context: str
                Context string
            until: [str]
                The string sequences to generate until. These string sequences
                may each span across multiple tokens, or may be part of one token.
        :return: list[str]
            A list of strings continuation
            continuation: str
                The generated continuation.
        """
        logging.info("Generating continuation for %d sequences." % len(requests))
        contexts, options = zip(*[req.args for req in requests])
        # contrary to the doc the second element of the tuple contains
        # {'do_sample': False, 'until': ['\n\n'], 'temperature': 0}
        completions = []

        for context, opt in tqdm(zip(contexts, options), total=len(contexts)):
            until = opt["until"]
            context = self.tokenizer.encode(
                context, add_special_tokens=not self.use_chat_template
            )
            max_tokens = min(
                opt.get("max_gen_tokens", self._max_tokens),
                self.tokenizer.model_max_length - len(context),
            )
            text = ""
            for response in stream_generate(
                self._model, self.tokenizer, prompt=context, max_tokens=max_tokens
            ):
                text += response.text
                if any(u in text for u in until):
                    text = _rstrip_until(text, until)
                    completions.append(text)
                    break
            else:
                completions.append(text)
        return completions


def main():
    parser = argparse.ArgumentParser(
        "Evaluate an MLX model using lm-evaluation-harness."
    )
    parser.add_argument("--model", help="Model to evaluate", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument(
        "--output-dir", default=".", help="Output directory for result files."
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-shots", type=int, default=None, help="Number of shots")
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum nunber of tokens to generate. Defaults to the model's max context length.",
    )
    parser.add_argument(
        "--limit",
        default=None,
        help="Limit the number of examples per task.",
        type=int,
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--fewshot-as-multiturn",
        action="store_true",
        help="Whether to provide the fewshot examples as a multiturn "
        "conversation or a single user turn.",
        default=False,
    )
    parser.add_argument(
        "--apply-chat-template",
        action=argparse.BooleanOptionalAction,
        help="Specifies whether to apply a chat template to the prompt. If "
        "the model has a chat template, this defaults to `True`, "
        "otherwise `False`.",
        default=None,
    )
    parser.add_argument(
        "--chat-template-args",
        type=json.loads,
        help="""A JSON formatted string of arguments for the tokenizer's
        apply_chat_template, e.g. '{"enable_thinking":false}'""",
        default="{}",
    )
    parser.add_argument(
        "--confirm-run-unsafe-code",
        action="store_true",
        help="Confirm that you want to run tasks that execute untrusted code.",
        default=False,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Silence tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    mx.random.seed(args.seed)

    # Initialize the communication if in distributed mode
    world = mx.distributed.init()
    mx.eval(mx.distributed.all_sum(1, stream=mx.cpu))
    if world.size() > 1 and world.rank() == 0:
        print(f"Evaluating with {world.size()} nodes")

    lm = MLXLM(
        args.model,
        max_tokens=args.max_tokens,
        use_chat_template=args.apply_chat_template,
        trust_remote_code=args.trust_remote_code,
    )
    MLXLM.apply_chat_template = chat_template_fn(**args.chat_template_args)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        apply_chat_template=lm.use_chat_template,
        num_fewshot=args.num_shots,
        limit=args.limit,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
        confirm_run_unsafe_code=args.confirm_run_unsafe_code,
    )

    file_keys = ["eval", args.model.replace("/", "_"), version("lm_eval")]
    if args.num_shots is not None:
        file_keys += [f"{args.num_shots:02d}"]
    file_keys += args.tasks
    filename = "_".join(file_keys)
    if world.rank() == 0:
        output_path = output_dir / filename
        output_path.write_text(json.dumps(results["results"], indent=4))
        print("Results:")
        for result in results["results"].values():
            print(json.dumps(result, indent=4))
