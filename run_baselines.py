from pathlib import Path
import pandas as pd
import numpy as np
import nltk
from ast import literal_eval
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, classification_report

print("Loading embedding model (all-MiniLM-L6-v2)...")
try:
    EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
except Exception:
    EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')


def clean_ans_sent(s: str) -> str:
    """Loại bỏ tiền tố 'R1: ', 'R2: ' trong Answer_sent_tokenized."""
    if ':' in s[:5]:
        return s.split(':', 1)[1].strip()
    return s.strip()


def predict_sim_cosine(answer_sentences: list, context: str, threshold: float) -> str:
    ctx_sents = sent_tokenize(context)
    if not ctx_sents:
        return 'Hallucinated'

    ans_sents = [clean_ans_sent(s) for s in answer_sentences]
    ans_embs = EMBED_MODEL.encode(ans_sents, show_progress_bar=False)
    ctx_embs = EMBED_MODEL.encode(ctx_sents, show_progress_bar=False)

    sims = cosine_similarity(ans_embs, ctx_embs)
    if (sims.max(axis=1) <= threshold).any():
        return 'Hallucinated'
    return 'Not Hallucinated'


def phrase_overlap_ratio(tokens_a: list, tokens_b: list) -> float:
    """
    Theo Mục 4.1 (Trang 7) của bài báo:
    - Bước 1: Quy hoạch động tìm đường đi tối ưu các từ trùng nhau.
    - Bước 2: Chỉ cộng độ dài các cụm từ trùng liên tiếp (overlapped phrases)
      có số từ > min(4, 0.3 * len(câu ngắn hơn)) để loại bỏ nhiễu từ từ phổ biến ('the', '.').
    - Bước 3: Chia tổng số token trùng hợp lệ cho độ dài câu ngắn hơn.
    """
    m, n = len(tokens_a), len(tokens_b)
    if m == 0 or n == 0:
        return 0.0

    shorter = min(m, n)
    min_phrase_len = min(4.0, 0.3 * shorter)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        ta = tokens_a[i - 1]
        row = dp[i]
        prev_row = dp[i - 1]
        for j in range(1, n + 1):
            if ta == tokens_b[j - 1]:
                row[j] = prev_row[j - 1] + 1
            else:
                v1 = prev_row[j]
                v2 = row[j - 1]
                row[j] = v1 if v1 >= v2 else v2

    matched_pairs = []
    i, j = m, n
    while i > 0 and j > 0:
        if tokens_a[i - 1] == tokens_b[j - 1] and dp[i][j] == dp[i - 1][j - 1] + 1:
            matched_pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    matched_pairs.reverse()

    if not matched_pairs:
        return 0.0

    overlap_len = 0
    cur_len = 1
    for k in range(1, len(matched_pairs)):
        di = matched_pairs[k][0] - matched_pairs[k - 1][0]
        dj = matched_pairs[k][1] - matched_pairs[k - 1][1]
        if di == 1 and dj == 1:
            cur_len += 1
        else:
            if cur_len > min_phrase_len:
                overlap_len += cur_len
            cur_len = 1
    if cur_len > min_phrase_len:
        overlap_len += cur_len

    return overlap_len / shorter


def predict_sim_overlap(answer_sentences: list, context: str, threshold: float) -> str:
    ctx_sents = sent_tokenize(context)
    if not ctx_sents:
        return 'Hallucinated'

    ctx_tok_list = [word_tokenize(c.lower()) for c in ctx_sents]
    for s in answer_sentences:
        ans_tokens = word_tokenize(clean_ans_sent(s).lower())
        max_ov = max(phrase_overlap_ratio(ans_tokens, ct) for ct in ctx_tok_list)
        if max_ov <= threshold:
            return 'Hallucinated'
    return 'Not Hallucinated'


def predict_sim_hybrid(answer_sentences: list, context: str,
                       t_cosine: float, t_overlap: float) -> str:
    ctx_sents = sent_tokenize(context)
    if not ctx_sents:
        return 'Hallucinated'

    ans_sents = [clean_ans_sent(s) for s in answer_sentences]
    ans_embs = EMBED_MODEL.encode(ans_sents, show_progress_bar=False)
    ctx_embs = EMBED_MODEL.encode(ctx_sents, show_progress_bar=False)
    sims = cosine_similarity(ans_embs, ctx_embs)
    ctx_tok_list = [word_tokenize(c.lower()) for c in ctx_sents]

    for idx, s in enumerate(ans_sents):
        cosine_ok = sims[idx].max() > t_cosine
        ans_tokens = word_tokenize(s.lower())
        overlap_ok = max(phrase_overlap_ratio(ans_tokens, ct) for ct in ctx_tok_list) > t_overlap
        if not (cosine_ok or overlap_ok):
            return 'Hallucinated'
    return 'Not Hallucinated'


def evaluate(y_true, y_pred, method_name: str):
    print(f"\n{'='*60}")
    print(f"  {method_name}")
    print('='*60)
    print(classification_report(
        y_true, y_pred,
        target_names=['Not Hallucinated', 'Hallucinated'],
        digits=4
    ))
    macro_f1 = f1_score(y_true, y_pred, average='macro') * 100
    print(f"  >>> Macro F1: {macro_f1:.2f}%")


if __name__ == '__main__':
    base = Path(__file__).parent / 'data' / 'DelucionQA_final'

    print("Loading datasets...")
    test = pd.read_csv(base / 'test.csv')
    test['answer_sents'] = test['Answer_sent_tokenized'].apply(literal_eval)
    y_true = (test['Label'] == 'Hallucinated').astype(int).tolist()

    print("\nRunning SIM-COSINE on Test set (T1 = 0.6)...")
    preds_cos = [predict_sim_cosine(r['answer_sents'], r['Context'], 0.6) for _, r in test.iterrows()]
    evaluate(y_true, [1 if p == 'Hallucinated' else 0 for p in preds_cos], 'SIM-COSINE (Test)')

    print("\nRunning SIM-OVERLAP on Test set (T2 = 0.1)...")
    preds_ov = [predict_sim_overlap(r['answer_sents'], r['Context'], 0.1) for _, r in test.iterrows()]
    evaluate(y_true, [1 if p == 'Hallucinated' else 0 for p in preds_ov], 'SIM-OVERLAP (Test)')

    print("\nRunning SIM-HYBRID on Test set (T1 = 0.9, T2 = 0.1)...")
    preds_hyb = [predict_sim_hybrid(r['answer_sents'], r['Context'], 0.9, 0.1) for _, r in test.iterrows()]
    evaluate(y_true, [1 if p == 'Hallucinated' else 0 for p in preds_hyb], 'SIM-HYBRID (Test)')
