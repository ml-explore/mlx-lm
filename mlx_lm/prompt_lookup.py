# Copyright © 2026 Apple Inc.

"""Proposal backends for draft-free (prompt-lookup) speculative decoding.

Two interchangeable proposers feed the shared verify/accept/cache core in
``generate.prompt_lookup_generate_step``:

- ``NgramProposer`` — tail n-gram lookup against the running sequence. Simple,
  stateless, zero dependencies. Good default.
- ``SuffixAutomatonProposer`` — an online suffix automaton that returns the
  longest repeated suffix's continuation. Strictly stronger retrieval (finds
  longer/earlier matches than a fixed n-gram) at ~microseconds/token.

Both expose the same interface:
    observe(token: int)                      # feed each committed token
    propose(seq, max_span, prompt_len) -> list[int]
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple


class SuffixAutomaton:
    """Online suffix automaton over token ids.

    Built incrementally (``extend`` per committed token), it answers, in
    O(suffix-link chain) per call: what is the longest suffix of the current
    sequence that also occurs ending at an earlier position, and where does that
    earlier occurrence end? ``first_end`` is the end position of the FIRST
    occurrence of a state's substrings, fixed at creation (clones inherit it).
    """

    __slots__ = ("seq", "_len", "_link", "_next", "_first_end", "_last")

    def __init__(self, tokens: Sequence[int] = ()):
        self.seq: List[int] = []
        self._len = [0]
        self._link = [-1]
        self._next: List[dict] = [{}]
        self._first_end = [-1]
        self._last = 0
        for t in tokens:
            self.extend(t)

    def __len__(self) -> int:
        return len(self.seq)

    def extend(self, token: int) -> None:
        token = int(token)
        pos = len(self.seq)
        self.seq.append(token)
        lens, link, nxt, first_end = self._len, self._link, self._next, self._first_end
        cur = len(lens)
        lens.append(lens[self._last] + 1)
        link.append(-1)
        nxt.append({})
        first_end.append(pos)
        p = self._last
        while p != -1 and token not in nxt[p]:
            nxt[p][token] = cur
            p = link[p]
        if p == -1:
            link[cur] = 0
        else:
            q = nxt[p][token]
            if lens[p] + 1 == lens[q]:
                link[cur] = q
            else:
                clone = len(lens)
                lens.append(lens[p] + 1)
                link.append(link[q])
                nxt.append(dict(nxt[q]))
                first_end.append(first_end[q])
                while p != -1 and nxt[p].get(token) == q:
                    nxt[p][token] = clone
                    p = link[p]
                link[q] = clone
                link[cur] = clone
        self._last = cur

    def longest_suffix_match(self, max_len: int = 16) -> Tuple[int, int]:
        """Return (match_len, next_pos): the longest suffix (<= max_len) that
        also occurs ending strictly before the current end, and the index right
        after that earlier occurrence (``seq[next_pos:]`` is the continuation).
        Returns (0, -1) when no suffix repeats."""
        n = len(self.seq)
        if n < 2:
            return 0, -1
        v = self._last
        while v != 0 and self._first_end[v] >= n - 1:
            v = self._link[v]
        if v == 0:
            return 0, -1
        return min(self._len[v], max_len), self._first_end[v] + 1


class NgramProposer:
    """Tail n-gram lookup. Stateless: proposes the continuation of the rightmost
    earlier occurrence of the tail n-gram (largest n first)."""

    def __init__(
        self, ngram_max: int = 3, ngram_min: int = 1, prompt_only: bool = False
    ):
        self.ngram_max = ngram_max
        self.ngram_min = ngram_min
        self.prompt_only = prompt_only

    def observe(self, token: int) -> None:  # stateless
        pass

    def propose(self, seq: List[int], max_span: int, prompt_len: int) -> List[int]:
        search_len = prompt_len if self.prompt_only else None
        n = len(seq)
        limit = n if search_len is None else min(search_len, n)
        for g in range(self.ngram_max, self.ngram_min - 1, -1):
            if n < g + 1:
                continue
            key = seq[-g:]
            start = (limit - g) if search_len is not None else (n - g - 1)
            for i in range(min(start, n - g - 1), -1, -1):
                if seq[i : i + g] == key:
                    cont = seq[i + g : i + g + max_span]
                    if cont:
                        return cont
        return []


class SuffixAutomatonProposer:
    """Suffix-automaton retrieval: continuation of the longest repeated suffix."""

    def __init__(
        self,
        min_match: int = 3,
        max_lookback: int = 32,
        initial_tokens: Sequence[int] = (),
    ):
        self.min_match = min_match
        self.max_lookback = max_lookback
        self.sam = SuffixAutomaton(initial_tokens)

    def observe(self, token: int) -> None:
        self.sam.extend(token)

    def propose(self, seq: List[int], max_span: int, prompt_len: int) -> List[int]:
        mlen, nxt = self.sam.longest_suffix_match(self.max_lookback)
        if mlen >= self.min_match and 0 <= nxt < len(seq):
            return seq[nxt : nxt + max_span]
        return []


def make_proposer(spec):
    """Build an EMPTY proposer from a `backend` string, or pass through a proposer
    object. The caller seeds it (feeds the prompt via observe()) — do not seed here
    too, or a stateful backend's coordinates desync from the caller's sequence.
    spec: "ngram" | "suffix_automaton" | a proposer instance."""
    if hasattr(spec, "propose"):
        return spec
    if spec in (None, "ngram"):
        return NgramProposer()
    if spec == "suffix_automaton":
        return SuffixAutomatonProposer()
    raise ValueError(f"unknown prompt-lookup backend {spec!r}")


@dataclass
class PromptLookupStats:
    """Per-source accounting for one prompt-lookup generation run."""

    cycles: int = 0
    retrieval_cycles: int = 0
    plain_cycles: int = 0
    retrieval_proposed: int = 0
    retrieval_accepted: int = 0
    bonus_tokens: int = 0
    plain_tokens: int = 0
    latched: bool = False

    @property
    def total_emitted(self) -> int:
        return self.retrieval_accepted + self.bonus_tokens + self.plain_tokens

    def summary(self) -> str:
        tot = max(self.total_emitted, 1)
        acc = self.retrieval_accepted / max(self.retrieval_proposed, 1)
        return (
            f"cycles {self.cycles} (retrieval {self.retrieval_cycles}, plain {self.plain_cycles}) | "
            f"tokens {self.total_emitted}: retrieval {self.retrieval_accepted} "
            f"({self.retrieval_accepted / tot:.0%}) + bonus {self.bonus_tokens} + plain {self.plain_tokens} | "
            f"retrieval acceptance {acc:.0%} | latched={self.latched}"
        )
