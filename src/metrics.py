"""VRSBench captioning metrics, including the specified BERT-BLEU metric."""

import math
import re
from typing import Any


def simple_tokenize(text: str) -> list[str]:
    """Tokenize captions consistently for lexical and BERT-BLEU n-grams."""
    return re.findall(r"\w+", text.lower())


def _ngram_strings(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def compute_ngram_matches(ref_tokens: list[str], pred_tokens: list[str], n: int) -> tuple[int, int]:
    """Count clipped lexical n-gram matches for the auxiliary BLEU scores."""
    if len(pred_tokens) < n or len(ref_tokens) < n:
        return 0, max(len(pred_tokens) - n + 1, 0)
    reference_counts: dict[tuple[str, ...], int] = {}
    for index in range(len(ref_tokens) - n + 1):
        gram = tuple(ref_tokens[index : index + n])
        reference_counts[gram] = reference_counts.get(gram, 0) + 1
    candidate_counts: dict[tuple[str, ...], int] = {}
    for index in range(len(pred_tokens) - n + 1):
        gram = tuple(pred_tokens[index : index + n])
        candidate_counts[gram] = candidate_counts.get(gram, 0) + 1
    return (
        sum(min(count, reference_counts.get(gram, 0)) for gram, count in candidate_counts.items()),
        len(pred_tokens) - n + 1,
    )


def compute_sentence_bleu(reference: str, prediction: str, max_n: int = 4) -> dict[str, float]:
    """Compute smoothed BLEU-1 through BLEU-4 as auxiliary lexical metrics."""
    ref_tokens, pred_tokens = simple_tokenize(reference), simple_tokenize(prediction)
    if not ref_tokens or not pred_tokens:
        return {f"bleu_{n}": 0.0 for n in range(1, max_n + 1)}
    length_penalty = min(1.0, math.exp(1.0 - len(ref_tokens) / len(pred_tokens)))
    precisions = []
    for n in range(1, max_n + 1):
        matches, total = compute_ngram_matches(ref_tokens, pred_tokens, n)
        precisions.append((matches + 0.1) / (total + 0.1))
    return {
        f"bleu_{n}": round(
            length_penalty * math.exp(sum(math.log(value) for value in precisions[:n]) / n) * 100,
            2,
        )
        for n in range(1, max_n + 1)
    }


def compute_rouge_l(reference: str, prediction: str) -> float:
    """Compute ROUGE-L F1 as an auxiliary sequence-overlap metric."""
    ref_tokens, pred_tokens = simple_tokenize(reference), simple_tokenize(prediction)
    if not ref_tokens or not pred_tokens:
        return 0.0
    grid = [[0] * (len(pred_tokens) + 1) for _ in range(len(ref_tokens) + 1)]
    for row, ref_token in enumerate(ref_tokens):
        for column, pred_token in enumerate(pred_tokens):
            grid[row + 1][column + 1] = (
                grid[row][column] + 1
                if ref_token == pred_token
                else max(grid[row][column + 1], grid[row + 1][column])
            )
    lcs = grid[-1][-1]
    precision, recall = lcs / len(pred_tokens), lcs / len(ref_tokens)
    return round(200 * precision * recall / (precision + recall), 2) if precision + recall else 0.0


class BertBleuScorer:
    """BERT-BLEU scorer from the supplied metric definition."""

    def __init__(self, model_name: str, device: str = "cpu", alpha: float = 0.5) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.alpha = alpha
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.model_name = model_name

    def _embed(self, grams: list[str]):
        """Mean-pool contextual BERT token embeddings for each n-gram."""
        encoded = self.tokenizer(
            grams,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        special_mask = encoded.pop("special_tokens_mask").to(self.device).bool()
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        with self.torch.inference_mode():
            hidden = self.model(**encoded).last_hidden_state
        usable = encoded["attention_mask"].bool() & ~special_mask
        pooled = (hidden * usable.unsqueeze(-1)).sum(dim=1) / usable.sum(dim=1, keepdim=True).clamp_min(1)
        return self.torch.nn.functional.normalize(pooled, p=2, dim=1)

    def score_pair(self, reference: str, prediction: str) -> dict[str, float]:
        ref_tokens, candidate_tokens = simple_tokenize(reference), simple_tokenize(prediction)
        if not ref_tokens or not candidate_tokens:
            return {f"bert_bleu_{n}": 0.0 for n in range(1, 5)}
        length_penalty = math.exp(
            -self.alpha * abs(len(candidate_tokens) - len(ref_tokens)) / len(ref_tokens)
        )
        semantic_recalls: list[float] = []
        scores: dict[str, float] = {}
        for n in range(1, 5):
            references, candidates = _ngram_strings(ref_tokens, n), _ngram_strings(candidate_tokens, n)
            if not references or not candidates:
                semantic_recalls.append(0.0)
            else:
                reference_vectors, candidate_vectors = self._embed(references), self._embed(candidates)
                similarities = reference_vectors @ candidate_vectors.T
                semantic_recalls.append(float(similarities.max(dim=1).values.mean().item()))
            # LP * exp((1/n) * sum(log(P_i))), exactly as supplied.
            scores[f"bert_bleu_{n}"] = round(
                length_penalty
                * math.exp(sum(math.log(max(value, 1e-12)) for value in semantic_recalls) / n)
                * 100,
                2,
            )
        return scores


def evaluate_predictions(
    records: list[dict[str, Any]],
    bert_bleu_model: str = "bert-base-uncased",
    device: str = "cpu",
    alpha: float = 0.5,
) -> dict[str, Any]:
    """Evaluate predictions with auxiliary lexical metrics and BERT-BLEU1..4."""
    if not records:
        return {}
    scorer = BertBleuScorer(bert_bleu_model, device=device, alpha=alpha)
    totals = {f"bleu_{n}": 0.0 for n in range(1, 5)}
    totals.update({f"bert_bleu_{n}": 0.0 for n in range(1, 5)})
    rouge_total = 0.0
    for record in records:
        reference, prediction = record["reference"], record["prediction"]
        for name, value in compute_sentence_bleu(reference, prediction).items():
            totals[name] += value
        for name, value in scorer.score_pair(reference, prediction).items():
            totals[name] += value
        rouge_total += compute_rouge_l(reference, prediction)
    count = len(records)
    metrics = {name: round(value / count, 2) for name, value in totals.items()}
    metrics["rouge_l"] = round(rouge_total / count, 2)
    metrics["bert_bleu_model"] = bert_bleu_model
    metrics["bert_bleu_alpha"] = alpha
    metrics["bert_bleu_definition"] = (
        "P_n = mean over reference n-grams of max cosine similarity to candidate n-grams; "
        "BERT-BLEU-n = LP * exp(mean(log(P_1..P_n))); LP = exp(-alpha*abs(Lc-Lr)/Lr)."
    )
    return metrics
