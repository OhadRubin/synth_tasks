"""
- description:
    TF-IDF-inspired document ranking task family for the shadow-ADO
    learnability experiment. Each prompt gives a tiny synthetic corpus, a query,
    and an explicit scoring rule; the target is every document ID sorted by
    weighted query-term score with deterministic ID tie-breaking. domain_id is
    `tfidf_rank_<n_docs>_docs_<query_terms>_terms`.
- usage:
    uv run python - <<'PY'
    from synthetic_tasks import get_task, split_rng
    rng = split_rng("tfidf_rank_documents",
                    "tfidf_rank_6_docs_2_terms", "train", 0)
    ex = get_task("tfidf_rank_documents")().sample(
        rng, "train", {
            "n_docs": 6, "doc_len": 5, "vocab_size": 10,
            "query_terms": 2, "max_tf": 3,
            "require_integer_rarity": True, "allow_score_ties": True})
    print(ex.prompt)
    print(repr(ex.target))
    PY
- user_story:
    content: |
        As Ohad, I want a small weighted-retrieval task where all semantics are
        synthetic and all scoring is replayable, so a model must combine global
        document frequency, local term frequency, exact weighted sums, sorting,
        and deterministic tie-breaking rather than relying on natural-language
        associations. The generator stores the full corpus, query, df, rarity,
        scores, and ranking for independent verification.
    was_generated_via_skill: false
"""

from __future__ import annotations

from fractions import Fraction
import random
from typing import Any

from synthetic_tasks.base import (
    Example,
    SyntheticTask,
    build_loss_mask,
    register_task,
)


def _doc_id(index: int) -> str:
    assert index >= 0, index
    return f"D{index}"


def _divisors(n: int) -> list[int]:
    assert n >= 1, n
    return [value for value in range(1, n + 1) if n % value == 0]


def _fraction_metadata_value(value: Fraction) -> int | str:
    if value.denominator == 1:
        return value.numerator
    return f"{value.numerator}/{value.denominator}"


def _solve_tfidf_rank(
        corpus: dict[str, list[str]],
        query_terms: list[str]) -> tuple[list[str], dict[str, int], dict[str, Fraction], dict[str, Fraction]]:
    n_docs = len(corpus)
    df = {
        term: sum(1 for tokens in corpus.values() if term in tokens)
        for term in query_terms
    }
    assert all(value > 0 for value in df.values()), df
    rarity = {
        term: Fraction(n_docs, df[term])
        for term in query_terms
    }
    scores = {}
    for doc_id, tokens in corpus.items():
        score = Fraction(0, 1)
        for term in query_terms:
            score += tokens.count(term) * rarity[term]
        scores[doc_id] = score
    ranking = sorted(
        corpus,
        key=lambda doc_id: (-scores[doc_id], int(doc_id[1:])),
    )
    return ranking, df, rarity, scores


def _sample_corpus(
        rng: random.Random,
        n_docs: int,
        doc_len: int,
        vocab: list[str],
        query: list[str],
        max_tf: int,
        require_integer_rarity: bool) -> dict[str, list[str]]:
    distractors = [token for token in vocab if token not in query]
    assert distractors, vocab
    valid_dfs = _divisors(n_docs) if require_integer_rarity else list(range(1, n_docs + 1))
    for _ in range(1000):
        docs: list[list[str]] = [[] for _ in range(n_docs)]
        for term in query:
            candidates = [index for index, tokens in enumerate(docs) if len(tokens) < doc_len]
            if not candidates:
                break
            possible_dfs = [df for df in valid_dfs if df <= len(candidates)]
            if not possible_dfs:
                break
            df = rng.choice(possible_dfs)
            for doc_index in rng.sample(candidates, df):
                remaining = doc_len - len(docs[doc_index])
                count = rng.randint(1, min(max_tf, remaining))
                docs[doc_index].extend([term] * count)
        else:
            if any(len(tokens) > doc_len for tokens in docs):
                continue
            for tokens in docs:
                while len(tokens) < doc_len:
                    tokens.append(rng.choice(distractors))
                rng.shuffle(tokens)
            corpus = {
                _doc_id(index): tokens
                for index, tokens in enumerate(docs)
            }
            _, df, _, _ = _solve_tfidf_rank(corpus, query)
            if require_integer_rarity and any(n_docs % value != 0 for value in df.values()):
                continue
            return corpus
    raise RuntimeError((n_docs, doc_len, len(vocab), query, max_tf))


@register_task
class TfidfRankDocumentsTask(SyntheticTask):
    task_id = "tfidf_rank_documents"

    def sample(self, rng: random.Random, split: str, difficulty: dict[str, Any]) -> Example:
        n_docs = difficulty["n_docs"]
        doc_len = difficulty["doc_len"]
        vocab_size = difficulty["vocab_size"]
        query_terms = difficulty["query_terms"]
        max_tf = difficulty["max_tf"]
        require_integer_rarity = difficulty["require_integer_rarity"]
        allow_score_ties = difficulty["allow_score_ties"]

        assert n_docs >= 1, n_docs
        assert doc_len >= 1, doc_len
        assert vocab_size > query_terms, difficulty
        assert query_terms >= 1, query_terms
        assert max_tf >= 1, max_tf

        vocab = [f"t{index:02d}" for index in range(vocab_size)]
        for _ in range(1000):
            query = rng.sample(vocab, query_terms)
            corpus = _sample_corpus(
                rng,
                n_docs,
                doc_len,
                vocab,
                query,
                max_tf,
                require_integer_rarity,
            )
            ranking, df, rarity, scores = _solve_tfidf_rank(corpus, query)
            if not allow_score_ties and len(set(scores.values())) != len(scores):
                continue
            break
        else:
            raise RuntimeError((self.task_id, difficulty))

        target = ">".join(ranking)
        corpus_lines = [
            f"{doc_id}: {' '.join(tokens)}"
            for doc_id, tokens in corpus.items()
        ]
        prompt = "\n".join([
            "<task=tfidf_rank_documents>",
            "Corpus:",
            *corpus_lines,
            "",
            f"Query: {' '.join(query)}",
            "",
            "Rule:",
            "df(term)=number of documents containing term",
            "rarity(term)=N/df(term)",
            "score(document)=sum count(term in document)*rarity(term) over query terms",
            "Rank every document by score, highest first. Break ties by smaller document ID.",
            "Return only the ranked document IDs joined by >.",
            "Answer:",
        ])
        full_text = prompt + target
        loss_mask = build_loss_mask(prompt, target)
        domain_id = f"tfidf_rank_{n_docs}_docs_{query_terms}_terms"
        return Example(
            task_id=self.task_id,
            domain_id=domain_id,
            prompt=prompt,
            target=target,
            full_text=full_text,
            loss_mask=loss_mask,
            metadata={
                "task_id": self.task_id,
                "macro_family": "weighted_retrieval_ranking",
                "difficulty_bin": domain_id,
                "control_status": "normal",
                "is_control": False,
                "difficulty": dict(difficulty),
                "n_docs": n_docs,
                "doc_len": doc_len,
                "vocab_size": vocab_size,
                "query_terms": query_terms,
                "requires_retrieval": True,
                "requires_global_count": True,
                "requires_local_count": True,
                "requires_weighted_sum": True,
                "requires_sorting": True,
                "has_distractors": True,
                "has_tie_breaking": allow_score_ties,
                "query": list(query),
                "corpus": {doc_id: list(tokens) for doc_id, tokens in corpus.items()},
                "df": dict(df),
                "rarity": {
                    term: _fraction_metadata_value(value)
                    for term, value in rarity.items()
                },
                "scores": {
                    doc_id: _fraction_metadata_value(value)
                    for doc_id, value in scores.items()
                },
                "ranking": list(ranking),
            },
        )
