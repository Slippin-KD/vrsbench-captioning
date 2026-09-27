"""Evaluation metrics for image captioning on VRSBench.

The primary report contains BERTScore F1 and corpus BLEU-4.  ``bert_bleu4``
is their geometric mean on a common 0-100 scale; it is an explicitly defined
project composite, not a standard external metric.
"""

from typing import Dict, List, Tuple
import re

def simple_tokenize(text: str) -> List[str]:
    """Basic word tokenizer when external libraries are not present."""
    text = text.lower()
    return re.findall(r"\w+", text)

def compute_ngram_matches(ref_tokens: List[str], pred_tokens: List[str], n: int) -> Tuple[int, int]:
    """Count n-gram matches and candidate n-grams."""
    if len(pred_tokens) < n or len(ref_tokens) < n:
        return 0, max(len(pred_tokens) - n + 1, 0)
    
    ref_ngrams = {}
    for i in range(len(ref_tokens) - n + 1):
        ng = tuple(ref_tokens[i:i+n])
        ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1
        
    pred_ngrams = {}
    for i in range(len(pred_tokens) - n + 1):
        ng = tuple(pred_tokens[i:i+n])
        pred_ngrams[ng] = pred_ngrams.get(ng, 0) + 1
        
    matches = 0
    for ng, count in pred_ngrams.items():
        matches += min(count, ref_ngrams.get(ng, 0))
    total = len(pred_tokens) - n + 1
    return matches, total

def compute_sentence_bleu(reference: str, prediction: str, max_n: int = 4) -> Dict[str, float]:
    """Compute BLEU-1 through BLEU-4 with Chen & Cherry smoothing."""
    ref_tokens = simple_tokenize(reference)
    pred_tokens = simple_tokenize(prediction)
    
    if not pred_tokens or not ref_tokens:
        return {f"bleu_{i}": 0.0 for i in range(1, max_n + 1)}
    
    # Brevity Penalty
    c = len(pred_tokens)
    r = len(ref_tokens)
    bp = 1.0 if c > r else 2.718281828459045 ** (1.0 - r / c) if c > 0 else 0.0
    
    precisions = []
    for n in range(1, max_n + 1):
        matches, total = compute_ngram_matches(ref_tokens, pred_tokens, n)
        if total == 0:
            p = 0.0
        else:
            # Smoothing method 1: add epsilon to 0 matches
            p = (matches + 0.1) / (total + 0.1) if matches == 0 else matches / total
        precisions.append(p)
        
    scores = {}
    for n in range(1, max_n + 1):
        # Geometric mean of precisions up to n
        log_sum = sum((1.0 / n) * (p if p > 0 else 1e-9) for p in [precisions[i] for i in range(n)])
        # geometric mean
        prod = 1.0
        for p in precisions[:n]:
            prod *= p
        geo_mean = prod ** (1.0 / n)
        scores[f"bleu_{n}"] = round(float(bp * geo_mean * 100), 2)
        
    return scores

def compute_rouge_l(reference: str, prediction: str) -> float:
    """Compute ROUGE-L F1 score based on Longest Common Subsequence."""
    ref_tokens = simple_tokenize(reference)
    pred_tokens = simple_tokenize(prediction)
    
    m, n = len(ref_tokens), len(pred_tokens)
    if m == 0 or n == 0:
        return 0.0
        
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if ref_tokens[i] == pred_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
                
    lcs = dp[m][n]
    prec = lcs / n
    rec = lcs / m
    if prec + rec == 0:
        return 0.0
    f1 = (2 * prec * rec) / (prec + rec)
    return round(float(f1 * 100), 2)

def evaluate_predictions(
    records: List[Dict[str, str]],
    use_bertscore: bool = True,
    bertscore_model: str = "roberta-large",
) -> Dict[str, float | str | None]:
    """Aggregate caption metrics and the documented BERT-BLEU4 composite."""
    if not records:
        return {}
        
    bleu_totals = {f"bleu_{i}": 0.0 for i in range(1, 5)}
    rouge_total = 0.0
    
    references = [r["reference"] for r in records]
    predictions = [r["prediction"] for r in records]
    
    for ref, pred in zip(references, predictions):
        b = compute_sentence_bleu(ref, pred)
        for k in bleu_totals:
            bleu_totals[k] += b[k]
        rouge_total += compute_rouge_l(ref, pred)
        
    count = len(records)
    metrics = {k: round(v / count, 2) for k, v in bleu_totals.items()}
    metrics["rouge_l"] = round(rouge_total / count, 2)
    
    # BERTScore needs contextual encoder weights. A score is never fabricated
    # when the package or its required model is unavailable.
    if use_bertscore:
        try:
            import bert_score
            P, R, F1 = bert_score.score(
                predictions,
                references,
                model_type=bertscore_model,
                lang="en",
                rescale_with_baseline=True,
                verbose=False,
            )
            metrics["bert_score_f1"] = round(float(F1.mean().item() * 100), 2)
            metrics["bert_score_precision"] = round(float(P.mean().item() * 100), 2)
            metrics["bert_score_recall"] = round(float(R.mean().item() * 100), 2)
            metrics["bert_score_model"] = bertscore_model
        except Exception as e:
            metrics["bert_score_f1"] = None
            metrics["bert_score_note"] = f"BERTScore was not computed: {e}"
    else:
        metrics["bert_score_f1"] = None
        metrics["bert_score_note"] = "BERTScore was disabled by --no-bertscore."

    # BERT-BLEU4: geometric mean of semantic agreement (BERTScore F1) and
    # 4-gram agreement (BLEU-4), both reported as percentages.
    bleu4_val = metrics.get("bleu_4", 0.0)
    bert_val = metrics.get("bert_score_f1")
    if isinstance(bert_val, (float, int)):
        metrics["bert_bleu4"] = round((bleu4_val * bert_val) ** 0.5, 2)
        metrics["bert_bleu4_definition"] = (
            "geometric_mean(BERTScore_F1, BLEU-4), both on a 0-100 scale"
        )
    else:
        metrics["bert_bleu4"] = None
        metrics["bert_bleu4_definition"] = "Unavailable because BERTScore was not computed."
            
    return metrics
