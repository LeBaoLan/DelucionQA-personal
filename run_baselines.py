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
# PHẦN 3: HƯỚNG 3 (SUPERVISED HIERARCHICAL MULTI-FEATURE CLASSIFIER)
# ══════════════════════════════════════════════════════════════════════════════

CLOSING_PATTERN = re.compile(
    r'(authorized dealer|further damage|safety hazard|appropriate action|'
    r'as soon as possible|owner\'?s manual|for service|prevent damage|'
    r'consult (a|an|your)|contact (a|an|your))',
    re.IGNORECASE
)


def _extract_dir3_split_features(df: pd.DataFrame, model, nli_cache: dict) -> tuple:
    """Trích xuất 23 đặc trưng mức câu (Sentence-level features) cho từng câu nhỏ R_i."""
    all_X_sent, all_y_sent, row_sent_features, row_ids = [], [], [], []
    y_row = (df['Label'] == 'Hallucinated').astype(int).to_numpy()

    for r_idx, (_, row) in enumerate(df.iterrows()):
        q, ctx = str(row['Question']), str(row['Context'])
        ctx_sents = sent_tokenize(ctx)
        raw_sents = literal_eval(row['Answer_sent_tokenized'])
        s_labels_dict = literal_eval(row['Sentence_labels'])

        ans_sents, sent_y = [], []
        for raw_s in raw_sents:
            r_key = raw_s.split(':', 1)[0].strip()
            c_s = clean_ans_sent(raw_s, strip_trailing_list_num=True)
            if len(extract_content_tokens(c_s)) > 0:
                ans_sents.append(c_s)
                sent_y.append(0 if s_labels_dict.get(r_key, 'Supported') == 'Supported' else 1)

        if not ctx_sents or not ans_sents:
            row_sent_features.append(None)
            continue

        q_emb = model.encode([q], show_progress_bar=False)
        ctx_embs = model.encode(ctx_sents, show_progress_bar=False)
        ans_embs = model.encode(ans_sents, show_progress_bar=False)
        sims = cosine_similarity(ans_embs, ctx_embs)
        q_sims = cosine_similarity(q_emb, ctx_embs)[0]
        q_ans_sims = cosine_similarity(q_emb, ans_embs)[0]

        q_toks = set(extract_content_tokens(q))
        ctx_all_toks = set(extract_content_tokens(ctx))
        ctx_sent_toks = [set(extract_content_tokens(c)) for c in ctx_sents]
        ctx_raw_toks = [t for t in word_tokenize(ctx.lower()) if re.search(r'[a-z0-9]', t)]
        ctx_bigrams = set(zip(ctx_raw_toks, ctx_raw_toks[1:]))
        ctx_trigrams = set(zip(ctx_raw_toks, ctx_raw_toks[1:], ctx_raw_toks[2:]))
        ctx_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', ctx.lower()))

        q_coverage = sum(1 for t in q_toks if t in ctx_all_toks) / max(1, len(q_toks))
        q_max_cos = float(q_sims.max())
        unverified_q_toks = q_toks - ctx_all_toks
        allowed_q_toks = (q_toks & ctx_all_toks) if q_coverage < 0.60 else q_toks
        grounded_vocab = ctx_all_toks | allowed_q_toks

        n_sents = len(ans_sents)
        feats_this_row = []

        for idx, s in enumerate(ans_sents):
            s_set = set(extract_content_tokens(s))
            cw_recall_ctx_q = sum(1 for t in s_set if t in grounded_vocab) / max(1, len(s_set))
            is_short_grounded = 1.0 if (len(s_set) <= 2 and cw_recall_ctx_q >= 0.5) else 0.0

            max_cos = float(sims[idx].max())
            best_j = int(sims[idx].argmax())
            top3_idx = np.argsort(sims[idx])[-3:]
            top3_mean_cos = float(sims[idx][top3_idx].mean())

            top3_vocab = set().union(*(ctx_sent_toks[j] for j in top3_idx)) | allowed_q_toks
            cw_recall_top3 = sum(1 for t in s_set if t in top3_vocab) / max(1, len(s_set))
            cw_recall_ctx_only = sum(1 for t in s_set if t in ctx_all_toks) / max(1, len(s_set))
            cw_recall_top1 = sum(1 for t in s_set if t in ctx_sent_toks[best_j]) / max(1, len(s_set))

            s_raw = [t for t in word_tokenize(s.lower()) if re.search(r'[a-z0-9]', t)]
            s_bigrams = list(zip(s_raw, s_raw[1:]))
            bg_recall = sum(1 for bg in s_bigrams if bg in ctx_bigrams) / max(1, len(s_bigrams))
            s_trigrams = list(zip(s_raw, s_raw[1:], s_raw[2:]))
            tri_recall = sum(1 for tg in s_trigrams if tg in ctx_trigrams) / max(1, len(s_trigrams))

            base_score = 0.5 * max_cos + 0.5 * cw_recall_top3 + 0.2 * bg_recall
            if is_short_grounded > 0.5:
                base_score = max(base_score, 0.85)

            p1 = ctx_sents[best_j]
            p3 = " ".join(ctx_sents[j] for j in sorted(top3_idx))
            default_pr = [0.02, 0.90, 0.08] if base_score > 0.63 else [0.80, 0.05, 0.15]
            pr1 = nli_cache.get((p1, s), default_pr)
            pr3 = nli_cache.get((p3, s), default_pr)

            has_unv_q = 1.0 if (q_coverage < 0.60 and bool(s_set & unverified_q_toks)) else 0.0
            p_ent_top1 = float(pr1[1])
            p_ent_max = 0.0 if has_unv_q > 0.5 else max(float(pr1[1]), float(pr3[1]))
            p_con_min = min(float(pr1[0]), float(pr3[0]))
            p_con_max = max(float(pr1[0]), float(pr3[0]))

            dir2_score = base_score + 0.2 * p_ent_max - 0.3 * p_con_min
            if is_short_grounded > 0.5:
                dir2_score = max(dir2_score, 0.85)

            s_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', s.lower()))
            unv_num_ratio = sum(1 for num in s_nums if num not in ctx_nums) / max(1, len(s_nums)) if s_nums else 0.0

            rel_pos = idx / max(1, n_sents - 1) if n_sents > 1 else 0.0
            is_last = 1.0 if (n_sents >= 2 and idx == n_sents - 1) else 0.0
            is_closing = 1.0 if CLOSING_PATTERN.search(s) else 0.0

            f_vec = [
                max_cos, top3_mean_cos, cw_recall_top3, cw_recall_ctx_only, cw_recall_top1,
                bg_recall, tri_recall, base_score, p_ent_max, p_ent_top1,
                p_con_min, p_con_max, dir2_score, q_coverage, q_max_cos, float(q_ans_sims[idx]),
                has_unv_q, unv_num_ratio, float(len(s_set)), is_short_grounded,
                rel_pos, is_last, is_closing
            ]
            feats_this_row.append(f_vec)
            all_X_sent.append(f_vec)
            all_y_sent.append(sent_y[idx])
            row_ids.append(r_idx)

        row_sent_features.append(np.array(feats_this_row, dtype=np.float64))

    return (np.array(all_X_sent, dtype=np.float64),
            np.array(all_y_sent, dtype=int),
            row_sent_features,
            np.array(row_ids),
            y_row)


def predict_supervised_multi_feature_dir3(base_dir: Path) -> list:
    """
    Phương pháp 6 (Hướng 3): Học máy có giám sát phân cấp 2 tầng (Hierarchical Multi-Feature Classifier)
    - Tầng 1 (Sentence-Level Verifier): Huấn luyện tổ hợp 3 mô hình (Logistic Regression + Random Forest +
      Gradient Boosting) trên 4.427 câu nhỏ có gán nhãn Sentence_labels của tập train.csv dựa trên 23 đặc trưng
      (Ngữ nghĩa Bi-Encoder, Từ vựng/Cụm từ Top-3, Suy luận NLI, Độ phủ câu hỏi, Khớp số liệu, Vị trí câu).
    - Tầng 2 (Answer-Level Aggregator): Dùng xác suất Out-of-Fold của Tầng 1 kết hợp với 26 đặc trưng tổng hợp
      mức toàn câu trả lời (min, second-min, mean, số câu vượt ngưỡng, dung sai câu kết khuyên nhủ...) để
      huấn luyện GradientBoostingClassifier trên train.csv và chọn ngưỡng tối ưu trên dev.csv.
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feat_cache_path = base_dir / '.feat_cache_dir3.pkl'
    if feat_cache_path.exists():
        with open(feat_cache_path, 'rb') as f:
            cache_obj = pickle.load(f)
    else:
        print("Extracting multi-level features for Train, Dev, Test (cached for next runs)...")
        model = get_embed_model()
        nli_cache = {}
        nli_p = base_dir / '.nli_cache_test.pkl'
        if nli_p.exists():
            with open(nli_p, 'rb') as f:
                nli_cache = pickle.load(f)
        train_df = pd.read_csv(base_dir / 'train.csv')
        dev_df = pd.read_csv(base_dir / 'dev.csv')
        test_df = pd.read_csv(base_dir / 'test.csv')

        X_s_tr, y_s_tr, rows_tr, r_ids_tr, y_r_tr = _extract_dir3_split_features(train_df, model, nli_cache)
        _, _, rows_dv, _, y_r_dv = _extract_dir3_split_features(dev_df, model, nli_cache)
        _, _, rows_te, _, y_r_te = _extract_dir3_split_features(test_df, model, nli_cache)
        cache_obj = {
            'X_sent_train': X_s_tr, 'y_sent_train': y_s_tr,
            'rows_train': rows_tr, 'r_ids_train': r_ids_tr, 'y_row_train': y_r_tr,
            'rows_dev': rows_dv, 'y_row_dev': y_r_dv,
            'rows_test': rows_te, 'y_row_test': y_r_te,
        }
        with open(feat_cache_path, 'wb') as f:
            pickle.dump(cache_obj, f)

    X_sent_train = cache_obj['X_sent_train']
    y_sent_train = cache_obj['y_sent_train']
    rows_train = cache_obj['rows_train']
    r_ids_train = cache_obj['r_ids_train']
    y_row_train = cache_obj['y_row_train']
    rows_dev = cache_obj['rows_dev']
    y_row_dev = cache_obj['y_row_dev']
    rows_test = cache_obj['rows_test']

    def make_stage1():
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42))
        rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=8, class_weight='balanced', random_state=42)
        gbr = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42)
        return lr, rf, gbr

    # Tầng 1: Dự đoán Out-of-Fold 5-Fold trên tập Train để tránh Overfitting khi dạy Tầng 2
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs_train = [None] * len(rows_train)
    for train_r_idx, val_r_idx in skf.split(np.zeros(len(y_row_train)), y_row_train):
        train_r_set = set(train_r_idx)
        sent_mask = np.array([rid in train_r_set for rid in r_ids_train])
        lr, rf, gbr = make_stage1()
        lr.fit(X_sent_train[sent_mask], y_sent_train[sent_mask])
        rf.fit(X_sent_train[sent_mask], y_sent_train[sent_mask])
        gbr.fit(X_sent_train[sent_mask], y_sent_train[sent_mask])
        for rid in val_r_idx:
            r = rows_train[rid]
            if r is not None:
                oof_probs_train[rid] = (
                    lr.predict_proba(r)[:, 1],
                    rf.predict_proba(r)[:, 1],
                    gbr.predict_proba(r)[:, 1]
                )

    lr_full, rf_full, gbr_full = make_stage1()
    lr_full.fit(X_sent_train, y_sent_train)
    rf_full.fit(X_sent_train, y_sent_train)
    gbr_full.fit(X_sent_train, y_sent_train)

    def build_meta_features(rows_feat_list, precomputed_probs=None):
        X_rows = []
        for idx, r in enumerate(rows_feat_list):
            if r is None or len(r) == 0:
                X_rows.append(np.zeros(26))
                continue
            if precomputed_probs is not None:
                p_lr, p_rf, p_gbr = precomputed_probs[idx]
            else:
                p_lr = lr_full.predict_proba(r)[:, 1]
                p_rf = rf_full.predict_proba(r)[:, 1]
                p_gbr = gbr_full.predict_proba(r)[:, 1]
            p_ens = (p_lr + p_rf + p_gbr) / 3.0

            dir2_sc = r[:, 12]
            base_sc = r[:, 7]
            adj_dir2_sc = dir2_sc + 0.08 * (r[:, 21] * r[:, 22])

            sorted_d2 = np.sort(dir2_sc)
            min_d2 = sorted_d2[0]
            sec_min_d2 = sorted_d2[1] if len(sorted_d2) > 1 else sorted_d2[0]

            sorted_ens = np.sort(p_ens)[::-1]
            max_p = sorted_ens[0]
            sec_max_p = sorted_ens[1] if len(sorted_ens) > 1 else sorted_ens[0]

            row_vec = [
                min_d2, sec_min_d2, float(dir2_sc.mean()), float(adj_dir2_sc.min()),
                float(base_sc.min()), float(base_sc.mean()),
                float(r[:, 0].min()), float(r[:, 0].mean()),
                float(r[:, 2].min()), float(r[:, 2].mean()),
                float(r[:, 3].min()), float(r[:, 5].min()), float(r[:, 6].min()),
                float(r[:, 8].min()), float(r[:, 10].max()),
                float(r[0, 13]), float(r[0, 14]),
                float(r[:, 16].max()), float(r[:, 17].max()),
                float(len(r)), float((dir2_sc <= 0.63).sum()),
                max_p, sec_max_p, float(p_ens.mean()),
                float(p_lr.max()), float(p_gbr.max())
            ]
            X_rows.append(row_vec)
        return np.array(X_rows, dtype=np.float64)

    # Tầng 2: Huấn luyện bộ phân loại tổng hợp câu trả lời và chọn ngưỡng trên tập Dev
    X_meta_train = build_meta_features(rows_train, oof_probs_train)
    X_meta_dev = build_meta_features(rows_dev)
    X_meta_test = build_meta_features(rows_test)

    meta_clf = GradientBoostingClassifier(
        n_estimators=80, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42
    )
    meta_clf.fit(X_meta_train, y_row_train)

    p_dev = meta_clf.predict_proba(X_meta_dev)[:, 1]
    best_th, best_f1_d = 0.5, 0.0
    for th in np.linspace(0.35, 0.65, 31):
        f1_d = f1_score(y_row_dev, (p_dev >= th).astype(int), average='macro') * 100
        if f1_d > best_f1_d:
            best_f1_d = f1_d
            best_th = th

    p_test = meta_clf.predict_proba(X_meta_test)[:, 1]
    return ['Hallucinated' if p >= best_th else 'Not Hallucinated' for p in p_test]


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
        choices=['all', 'new', 'baselines', '1', '2', '3', '4', '5', '6', 'dir1', 'dir2', 'dir3'],
        help="Chọn phương pháp: 'all', 'new' (Hướng 1, 2, 3), 'dir1', 'dir2', 'dir3', hoặc '1'..'6'"
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

    if m in ('all', 'new', '6', 'dir3'):
        print("\n6. Running Supervised Hierarchical Multi-Feature Classifier (Direction 3)...")
        evaluate(y_true, predict_supervised_multi_feature_dir3(base),
                 'METHOD 6 (DIRECTION 3): SUPERVISED MULTI-FEATURE CLASSIFIER (Test)')


# python run_baselines.py --method new
#                                  dir1/baselines