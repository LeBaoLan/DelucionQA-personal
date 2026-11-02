# run_baselines.py
import argparse
import pickle
import re
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.special
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics.pairwise import cosine_similarity

# Danh sách hư từ (stopwords) và từ dẫn nhập dùng cho Hướng 1 & Hướng 2
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'while', 'with', 'to', 'from',
    'in', 'on', 'at', 'by', 'for', 'of', 'about', 'as', 'into', 'like', 'through',
    'after', 'over', 'between', 'out', 'against', 'during', 'without', 'before',
    'under', 'around', 'among', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should',
    'may', 'might', 'must', 'can', 'could', 'it', 'its', 'they', 'them', 'their',
    'this', 'that', 'these', 'those', 'you', 'your', 'we', 'our', 'i', 'me', 'my',
    'he', 'him', 'his', 'she', 'her', 'who', 'which', 'what', 'where', 'when', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'just', 'also', 'then', 'there', 'here', 'therefore', 'however', 'thus',
    'follow', 'following', 'steps', 'step', 'make', 'sure', 'ensure', 'important',
    'recommended', 'note', 'please', 'refer', 'see', 'take', 'action', 'possible',
    'mentioned', 'context', 'provided', 'below', 'above'
}

_EMBED_MODEL = None
_NLI_MODEL = None


def get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        print("Loading Bi-Encoder embedding model (all-MiniLM-L6-v2)...")
        try:
            _EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
        except Exception:
            _EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _EMBED_MODEL


def get_nli_model():
    global _NLI_MODEL
    if _NLI_MODEL is None:
        from sentence_transformers import CrossEncoder
        print("Loading NLI Cross-Encoder model (cross-encoder/nli-MiniLM2-L6-H768)...")
        try:
            _NLI_MODEL = CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768', local_files_only=True)
        except Exception:
            _NLI_MODEL = CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')
    return _NLI_MODEL


def clean_ans_sent(s: str, strip_trailing_list_num: bool = False) -> str:
    """Loại bỏ tiền tố 'R1: ', số thứ tự đầu câu '1. ', và tùy chọn cắt ': 1.' cuối câu (FP-1)."""
    if ':' in s[:5]:
        s = s.split(':', 1)[1].strip()
    s = re.sub(r'^\d+\.\s*', '', s)
    if strip_trailing_list_num:
        s = re.sub(r':\s*\d+\.?\s*$', ':', s)
    return s.strip()


def extract_content_tokens(text: str) -> list:
    """Trích xuất các từ mang nội dung thực (loại bỏ hư từ và dấu câu)."""
    toks = word_tokenize(str(text).lower())
    return [t for t in toks if re.search(r'[a-z0-9]', t) and t not in STOPWORDS and len(t) > 1]


def load_or_compute_embeddings(df: pd.DataFrame, cache_path: Path) -> list:
    """Tính sẵn (và lưu cache) ma trận Cosine Similarity một lần duy nhất cho mọi phương pháp."""
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    model = get_embed_model()
    print("Computing sentence embeddings once for all methods (cached for next runs)...")
    cached_rows = []
    for _, row in df.iterrows():
        ctx = str(row['Context'])
        ctx_sents = sent_tokenize(ctx)
        ans_base = [clean_ans_sent(s, False) for s in row['answer_sents']]
        ans_clean = [clean_ans_sent(s, True) for s in row['answer_sents']]
        ans_filtered = [s for s in ans_clean if len(extract_content_tokens(s)) > 0]

        if not ctx_sents or not ans_base:
            cached_rows.append(None)
            continue

        ctx_embs = model.encode(ctx_sents, show_progress_bar=False)
        ans_embs_base = model.encode(ans_base, show_progress_bar=False)
        sims_base = cosine_similarity(ans_embs_base, ctx_embs)

        if ans_filtered == ans_base:
            sims_filtered = sims_base
        elif ans_filtered:
            ans_embs_filt = model.encode(ans_filtered, show_progress_bar=False)
            sims_filtered = cosine_similarity(ans_embs_filt, ctx_embs)
        else:
            sims_filtered = np.empty((0, len(ctx_sents)))

        cached_rows.append({
            'question': str(row['Question']),
            'context': ctx,
            'ctx_sents': ctx_sents,
            'ans_base': ans_base,
            'ans_filtered': ans_filtered,
            'sims_base': sims_base,
            'sims_filtered': sims_filtered,
        })

    with open(cache_path, 'wb') as f:
        pickle.dump(cached_rows, f)
    return cached_rows


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 1: 3 PHƯƠNG PHÁP BASELINE CỦA BÀI BÁO
# ══════════════════════════════════════════════════════════════════════════════

def phrase_overlap_ratio(tokens_a: list, tokens_b: list) -> float:
    m, n = len(tokens_a), len(tokens_b)
    if m == 0 or n == 0:
        return 0.0

    shorter = min(m, n)
    min_phrase_len = min(4.0, 0.3 * shorter)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        ta = tokens_a[i - 1]
        row, prev_row = dp[i], dp[i - 1]
        for j in range(1, n + 1):
            if ta == tokens_b[j - 1]:
                row[j] = prev_row[j - 1] + 1
            else:
                row[j] = prev_row[j] if prev_row[j] >= row[j - 1] else row[j - 1]

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

    overlap_len, cur_len = 0, 1
    for k in range(1, len(matched_pairs)):
        if (matched_pairs[k][0] - matched_pairs[k - 1][0] == 1 and
                matched_pairs[k][1] - matched_pairs[k - 1][1] == 1):
            cur_len += 1
        else:
            if cur_len > min_phrase_len:
                overlap_len += cur_len
            cur_len = 1
    if cur_len > min_phrase_len:
        overlap_len += cur_len

    return overlap_len / shorter


def predict_sim_cosine_batch(cached_rows: list, threshold: float = 0.6) -> list:
    preds = []
    for item in cached_rows:
        if item is None or (item['sims_base'].max(axis=1) <= threshold).any():
            preds.append('Hallucinated')
        else:
            preds.append('Not Hallucinated')
    return preds


def predict_sim_overlap_batch(cached_rows: list, threshold: float = 0.1) -> list:
    preds = []
    for item in cached_rows:
        if item is None:
            preds.append('Hallucinated')
            continue
        ctx_tok_list = [word_tokenize(c.lower()) for c in item['ctx_sents']]
        hal = False
        for s in item['ans_base']:
            ans_tokens = word_tokenize(s.lower())
            if max(phrase_overlap_ratio(ans_tokens, ct) for ct in ctx_tok_list) <= threshold:
                hal = True
                break
        preds.append('Hallucinated' if hal else 'Not Hallucinated')
    return preds


def predict_sim_hybrid_batch(cached_rows: list, t_cosine: float = 0.9, t_overlap: float = 0.1) -> list:
    preds = []
    for item in cached_rows:
        if item is None:
            preds.append('Hallucinated')
            continue
        ctx_tok_list = [word_tokenize(c.lower()) for c in item['ctx_sents']]
        sims = item['sims_base']
        hal = False
        for idx, s in enumerate(item['ans_base']):
            if sims[idx].max() > t_cosine:
                continue
            ans_tokens = word_tokenize(s.lower())
            if max(phrase_overlap_ratio(ans_tokens, ct) for ct in ctx_tok_list) <= t_overlap:
                hal = True
                break
        preds.append('Hallucinated' if hal else 'Not Hallucinated')
    return preds


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 2: HƯỚNG 1 (MULTI-EVIDENCE CONTENT GROUNDING) & HƯỚNG 2 (+ NLI)
# ══════════════════════════════════════════════════════════════════════════════

def score_row_sentences(item: dict, w_cos: float = 0.5, w_cw: float = 0.5,
                        w_bg: float = 0.2, q_cov_gate: float = 0.60) -> list:
    """
    Hàm dùng chung cho cả Hướng 1 và Hướng 2:
    Tính base_score, câu Top-1, đoạn Top-3 và cờ kiểm tra từ khóa câu hỏi (FN-1) cho từng câu R_i.
    """
    q, ctx = item['question'], item['context']
    ctx_sents, ans_sents, sims = item['ctx_sents'], item['ans_filtered'], item['sims_filtered']

    q_toks = set(extract_content_tokens(q))
    ctx_all_toks = set(extract_content_tokens(ctx))
    ctx_sent_toks = [set(extract_content_tokens(c)) for c in ctx_sents]
    ctx_raw_toks = [t for t in word_tokenize(ctx.lower()) if re.search(r'[a-z0-9]', t)]
    ctx_bigrams = set(zip(ctx_raw_toks, ctx_raw_toks[1:]))

    # Cổng kiểm soát FN-1: Kiểm tra tỷ lệ từ khóa Question có trong Context
    q_coverage = sum(1 for t in q_toks if t in ctx_all_toks) / max(1, len(q_toks))
    unverified_q_toks = q_toks - ctx_all_toks
    allowed_q_toks = (q_toks & ctx_all_toks) if q_coverage < q_cov_gate else q_toks
    grounded_vocab = ctx_all_toks | allowed_q_toks

    sent_infos = []
    for idx, s in enumerate(ans_sents):
        s_set = set(extract_content_tokens(s))
        cw_recall_ctx_q = sum(1 for t in s_set if t in grounded_vocab) / max(1, len(s_set))
        if len(s_set) <= 2 and cw_recall_ctx_q >= 0.5:
            continue

        max_cos = float(sims[idx].max())
        best_j = int(sims[idx].argmax())
        top3_idx = np.argsort(sims[idx])[-3:]
        top3_vocab = set().union(*(ctx_sent_toks[j] for j in top3_idx)) | allowed_q_toks
        cw_recall_top3 = sum(1 for t in s_set if t in top3_vocab) / max(1, len(s_set))

        s_raw = [t for t in word_tokenize(s.lower()) if re.search(r'[a-z0-9]', t)]
        s_bigrams = list(zip(s_raw, s_raw[1:]))
        bg_recall = sum(1 for bg in s_bigrams if bg in ctx_bigrams) / max(1, len(s_bigrams))

        base_score = w_cos * max_cos + w_cw * cw_recall_top3 + w_bg * bg_recall
        has_unverified_q = (q_coverage < q_cov_gate) and bool(s_set & unverified_q_toks)
        premise_top1 = ctx_sents[best_j]
        premise_top3 = " ".join(ctx_sents[j] for j in sorted(top3_idx))
        sent_infos.append((base_score, premise_top1, premise_top3, s, has_unverified_q))

    return sent_infos


def predict_multi_evidence_grounding_batch(cached_rows: list, threshold: float = 0.63) -> list:
    """Phương pháp 4 (Hướng 1): Multi-Evidence Content Grounding."""
    preds = []
    for item in cached_rows:
        if item is None or len(item['ans_filtered']) == 0:
            preds.append('Hallucinated')
            continue
        sent_infos = score_row_sentences(item)
        hal = any(base_score <= threshold for base_score, _, _, _, _ in sent_infos)
        preds.append('Hallucinated' if hal else 'Not Hallucinated')
    return preds


def predict_multi_evidence_nli_batch(cached_rows: list, nli_cache_path: Path = None,
                                     w_ent: float = 0.2, w_con: float = 0.3,
                                     threshold: float = 0.63) -> list:
    """Phương pháp 5 (Hướng 2): Multi-Evidence Content Grounding + NLI Cross-Encoder."""
    predictions = ['Not Hallucinated'] * len(cached_rows)
    ambiguous_items, pairs_top1, pairs_top3 = [], [], []

    for row_idx, item in enumerate(cached_rows):
        if item is None or len(item['ans_filtered']) == 0:
            predictions[row_idx] = 'Hallucinated'
            continue

        sent_infos = score_row_sentences(item)
        row_candidates = []
        definite_hallucinated = False

        for base_score, p1, p3, s, has_unv_q in sent_infos:
            if base_score + w_ent <= threshold:
                definite_hallucinated = True
                break
            if base_score - w_con > threshold:
                continue
            row_candidates.append((base_score, p1, p3, s, has_unv_q))

        if definite_hallucinated:
            predictions[row_idx] = 'Hallucinated'
        else:
            for base_score, p1, p3, s, has_unv_q in row_candidates:
                ambiguous_items.append((row_idx, base_score, has_unv_q))
                pairs_top1.append((p1, s))
                pairs_top3.append((p3, s))

    if ambiguous_items:
        nli_cache = {}
        if nli_cache_path and nli_cache_path.exists():
            with open(nli_cache_path, 'rb') as f:
                nli_cache = pickle.load(f)

        missing = [p for p in set(pairs_top1 + pairs_top3) if p not in nli_cache]
        if missing:
            nli_model = get_nli_model()
            logits = nli_model.predict(missing, batch_size=64, show_progress_bar=False)
            probs = scipy.special.softmax(logits, axis=1)
            for i, p in enumerate(missing):
                nli_cache[p] = probs[i]
            if nli_cache_path:
                with open(nli_cache_path, 'wb') as f:
                    pickle.dump(nli_cache, f)

        for k, (row_idx, base_score, has_unv_q) in enumerate(ambiguous_items):
            if predictions[row_idx] == 'Hallucinated':
                continue
            pr1 = nli_cache[pairs_top1[k]]
            pr3 = nli_cache[pairs_top3[k]]
            ent = 0.0 if has_unv_q else max(float(pr1[1]), float(pr3[1]))
            con = min(float(pr1[0]), float(pr3[0]))
            if base_score + w_ent * ent - w_con * con <= threshold:
                predictions[row_idx] = 'Hallucinated'

    return predictions


# ══════════════════════════════════════════════════════════════════════════════
# ĐÁNH GIÁ KẾT QUẢ & CHẠY THỰC NGHIỆM
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(y_true, y_pred_labels: list, method_name: str):
    y_pred = [1 if p == 'Hallucinated' else 0 for p in y_pred_labels]
    print(f"\n{'='*65}")
    print(f"  {method_name}")
    print('='*65)
    print(classification_report(
        y_true, y_pred,
        target_names=['Not Hallucinated', 'Hallucinated'],
        digits=4
    ))
    macro_f1 = f1_score(y_true, y_pred, average='macro') * 100
    print(f"  >>> Macro F1: {macro_f1:.2f}%")
    return macro_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Hallucination Detection Methods on DelucionQA")
    parser.add_argument(
        '--method', type=str, default='all',
        choices=['all', 'new', 'baselines', '1', '2', '3', '4', '5', 'dir1', 'dir2'],
        help="Chọn phương pháp muốn chạy: 'all' (tất cả), 'new' (chỉ Hướng 1 & 2), 'dir1', 'dir2', hoặc '1'..'5'"
    )
    args = parser.parse_args()

    base = Path(__file__).parent / 'data' / 'DelucionQA_final'
    print("Loading datasets...")
    test = pd.read_csv(base / 'test.csv')
    test['answer_sents'] = test['Answer_sent_tokenized'].apply(literal_eval)
    y_true = (test['Label'] == 'Hallucinated').astype(int).tolist()

    cache_file = base / '.emb_cache_test.pkl'
    nli_cache_file = base / '.nli_cache_test.pkl'
    cached_test = load_or_compute_embeddings(test, cache_file)

    m = args.method
    if m in ('all', 'baselines', '1'):
        print("\n1. Running SIM-COSINE Baseline on Test set (T1 = 0.6)...")
        evaluate(y_true, predict_sim_cosine_batch(cached_test, 0.6),
                 'METHOD 1 (BASELINE): SIM-COSINE (Test)')

    if m in ('all', 'baselines', '2'):
        print("\n2. Running SIM-OVERLAP Baseline on Test set (T2 = 0.1)...")
        evaluate(y_true, predict_sim_overlap_batch(cached_test, 0.1),
                 'METHOD 2 (BASELINE): SIM-OVERLAP (Test)')

    if m in ('all', 'baselines', '3'):
        print("\n3. Running SIM-HYBRID Baseline on Test set (T1 = 0.9, T2 = 0.1)...")
        evaluate(y_true, predict_sim_hybrid_batch(cached_test, 0.9, 0.1),
                 'METHOD 3 (BASELINE): SIM-HYBRID (Test)')

    if m in ('all', 'new', '4', 'dir1'):
        print("\n4. Running Multi-Evidence Content Grounding (Direction 1)...")
        evaluate(y_true, predict_multi_evidence_grounding_batch(cached_test),
                 'METHOD 4 (DIRECTION 1): MULTI-EVIDENCE CONTENT GROUNDING (Test)')

    if m in ('all', 'new', '5', 'dir2'):
        print("\n5. Running Multi-Evidence Content Grounding + NLI Cross-Encoder (Direction 2)...")
        evaluate(y_true, predict_multi_evidence_nli_batch(cached_test, nli_cache_file),
                 'METHOD 5 (DIRECTION 2): MULTI-EVIDENCE GROUNDING + NLI CROSS-ENCODER (Test)')


# python run_baselines.py --method new
#                                  dir1/baselines