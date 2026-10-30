# run_baselines.py
from pathlib import Path
import re
import pandas as pd
import numpy as np
import scipy.special
import nltk
from ast import literal_eval
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, classification_report

print("Loading Bi-Encoder embedding model (all-MiniLM-L6-v2)...")
try:
    EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
except Exception:
    EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading NLI Cross-Encoder model (cross-encoder/nli-MiniLM2-L6-H768)...")
try:
    NLI_MODEL = CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768', local_files_only=True)
except Exception:
    NLI_MODEL = CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')

# Danh sách hư từ (stopwords) và từ dẫn dắt câu (framing words) dùng cho Hướng 1 & Hướng 2
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
    # Bổ sung xử lý lỗi FP-1 (câu dẫn nhập liệt kê nhắc tới ngữ cảnh/danh sách)
    'mentioned', 'context', 'provided', 'below', 'above'
}


def clean_ans_sent(s: str) -> str:
    """Loại bỏ tiền tố 'R1: ', 'R2: ' và số thứ tự đầu câu '1. ', '2. ' (dùng cho Baseline gốc)."""
    if ':' in s[:5]:
        s = s.split(':', 1)[1].strip()
    s = re.sub(r'^\d+\.\s*', '', s)
    return s.strip()


def clean_ans_sent_improved(s: str) -> str:
    """Làm sạch câu cho Hướng 1 & 2: loại bỏ thêm số thứ tự danh sách bị dính ở cuối câu ': 1.' (Sửa lỗi FP-1)."""
    s = clean_ans_sent(s)
    s = re.sub(r':\s*\d+\.?\s*$', ':', s)
    return s.strip()


def extract_content_tokens(text: str) -> list:
    """Trích xuất các từ mang nội dung thực (loại bỏ hư từ và dấu câu)."""
    toks = word_tokenize(text.lower())
    return [t for t in toks if re.search(r'[a-z0-9]', t) and t not in STOPWORDS and len(t) > 1]


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 1: 3 PHƯƠNG PHÁP BASELINE GỐC CỦA BÀI BÁO
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 2: THUẬT TOÁN NÂNG CẤP HƯỚNG 1 (MULTI-EVIDENCE & CONTENT GROUNDING)
# ══════════════════════════════════════════════════════════════════════════════

def predict_improved_dir1(question: str, answer_sentences: list, context: str,
                          w_cos: float = 0.5, w_cw: float = 0.5,
                          w_bg: float = 0.2, threshold: float = 0.63,
                          q_cov_gate: float = 0.60) -> str:
    """
    Cải tiến Hướng 1 (Multi-Evidence Content Grounding + Tinh chỉnh FP-1 & FN-1):
    - Score(R_i) = 0.5 * MaxCosine + 0.5 * Top3_WordRecall + 0.2 * BigramRecall
    - Sửa lỗi FP-1: Làm sạch số thứ tự danh sách dính cuối câu (': 1.') và lọc từ dẫn nhập ngữ cảnh.
    - Sửa lỗi FN-1 (Question-Context Coverage Gate): Nếu < 60% từ khóa câu hỏi xuất hiện trong Context
      (dấu hiệu câu hỏi không liên quan / Answerable=False), chỉ cộng điểm cho các từ câu hỏi thực sự có trong Context.
    """
    ctx_sents = sent_tokenize(str(context))
    ans_sents = [clean_ans_sent_improved(s) for s in answer_sentences]
    ans_sents = [s for s in ans_sents if len(extract_content_tokens(s)) > 0]

    if not ctx_sents or not ans_sents:
        return 'Hallucinated'

    ans_embs = EMBED_MODEL.encode(ans_sents, show_progress_bar=False)
    ctx_embs = EMBED_MODEL.encode(ctx_sents, show_progress_bar=False)
    sims = cosine_similarity(ans_embs, ctx_embs)

    q_toks = set(extract_content_tokens(str(question)))
    ctx_all_toks = set(extract_content_tokens(str(context)))
    ctx_sent_toks = [set(extract_content_tokens(c)) for c in ctx_sents]

    ctx_raw_toks = [t for t in word_tokenize(str(context).lower()) if re.search(r'[a-z0-9]', t)]
    ctx_bigrams = set(zip(ctx_raw_toks, ctx_raw_toks[1:]))

    # Cổng kiểm soát FN-1: Kiểm tra tỷ lệ từ khóa của Question xuất hiện trong Context
    q_coverage = sum(1 for t in q_toks if t in ctx_all_toks) / max(1, len(q_toks))
    allowed_q_toks = (q_toks & ctx_all_toks) if q_coverage < q_cov_gate else q_toks
    grounded_vocab = ctx_all_toks | allowed_q_toks

    for idx, s in enumerate(ans_sents):
        s_set = set(extract_content_tokens(s))

        cw_recall_ctx_q = sum(1 for t in s_set if t in grounded_vocab) / max(1, len(s_set))
        if len(s_set) <= 2 and cw_recall_ctx_q >= 0.5:
            continue

        max_cos = float(sims[idx].max())
        top3_idx = np.argsort(sims[idx])[-3:]
        top3_vocab = set().union(*(ctx_sent_toks[j] for j in top3_idx)) | allowed_q_toks
        cw_recall_top3 = sum(1 for t in s_set if t in top3_vocab) / max(1, len(s_set))

        s_raw = [t for t in word_tokenize(s.lower()) if re.search(r'[a-z0-9]', t)]
        s_bigrams = list(zip(s_raw, s_raw[1:]))
        bg_recall = sum(1 for bg in s_bigrams if bg in ctx_bigrams) / max(1, len(s_bigrams))

        score = w_cos * max_cos + w_cw * cw_recall_top3 + w_bg * bg_recall
        if score <= threshold:
            return 'Hallucinated'

    return 'Not Hallucinated'


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 3: THUẬT TOÁN NÂNG CẤP HƯỚNG 2 (NLI CROSS-ENCODER VERIFICATION)
# ══════════════════════════════════════════════════════════════════════════════

def predict_improved_dir2_batch(df: pd.DataFrame,
                                w_cos: float = 0.5, w_cw: float = 0.5, w_bg: float = 0.2,
                                w_ent: float = 0.2, w_con: float = 0.3,
                                threshold: float = 0.63,
                                q_cov_gate: float = 0.60) -> list:
    """
    Cải tiến Hướng 2 (Multi-Evidence Grounding + Tinh chỉnh FP-1/FN-1 + NLI Cross-Encoder):
    - Với mỗi câu trả lời R_i, lấy câu Context gần nhất (Top-1) và đoạn gộp 3 câu Context gần nhất (Top-3)
      đưa vào mô hình Cross-Encoder NLI ('cross-encoder/nli-MiniLM2-L6-H768').
    - Sửa lỗi FP-1: Làm sạch số thứ tự danh sách dính cuối câu (': 1.') và lọc từ dẫn nhập ngữ cảnh.
    - Sửa lỗi FN-1: Khi câu hỏi không khớp ngữ cảnh (q_coverage < 0.60), tước quyền cộng điểm miễn phí
      của các từ khóa câu hỏi không nằm trong Context và chặn NLI cộng điểm Entailment cho câu chứa từ lạ đó.
    """
    predictions = ['Not Hallucinated'] * len(df)
    ambiguous_items = []   # Lưu các câu vùng biên cần NLI phân xử
    pairs_top1 = []
    pairs_top3 = []

    for row_idx, (_, row) in enumerate(df.iterrows()):
        q = str(row['Question'])
        ctx = str(row['Context'])
        ans_sents = [clean_ans_sent_improved(s) for s in row['answer_sents']]
        ans_sents = [s for s in ans_sents if len(extract_content_tokens(s)) > 0]
        ctx_sents = sent_tokenize(ctx)

        if not ctx_sents or not ans_sents:
            predictions[row_idx] = 'Hallucinated'
            continue

        ans_embs = EMBED_MODEL.encode(ans_sents, show_progress_bar=False)
        ctx_embs = EMBED_MODEL.encode(ctx_sents, show_progress_bar=False)
        sims = cosine_similarity(ans_embs, ctx_embs)

        q_toks = set(extract_content_tokens(q))
        ctx_all_toks = set(extract_content_tokens(ctx))
        ctx_sent_toks = [set(extract_content_tokens(c)) for c in ctx_sents]
        ctx_raw_toks = [t for t in word_tokenize(ctx.lower()) if re.search(r'[a-z0-9]', t)]
        ctx_bigrams = set(zip(ctx_raw_toks, ctx_raw_toks[1:]))

        # Cổng kiểm soát FN-1: Kiểm tra tỷ lệ từ khóa của Question xuất hiện trong Context
        q_coverage = sum(1 for t in q_toks if t in ctx_all_toks) / max(1, len(q_toks))
        unverified_q_toks = q_toks - ctx_all_toks
        allowed_q_toks = (q_toks & ctx_all_toks) if q_coverage < q_cov_gate else q_toks
        grounded_vocab = ctx_all_toks | allowed_q_toks

        row_candidates = []
        definite_hallucinated = False

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

            # Kiểm tra xem câu trả lời có đang lặp lại từ khóa không có trong Context của một câu hỏi lạc đề không (FN-1)
            has_unverified_q = (q_coverage < q_cov_gate) and bool(s_set & unverified_q_toks)

            # 1. Nếu base_score quá thấp -> Chắc chắn Hallucinated!
            if base_score + w_ent <= threshold:
                definite_hallucinated = True
                break

            # 2. Nếu base_score rất cao -> Chắc chắn Supported!
            if base_score - w_con > threshold:
                continue

            # 3. Câu nằm ở vùng biên -> Đưa vào danh sách chờ mô hình NLI chấm điểm
            premise_top1 = ctx_sents[best_j]
            premise_top3 = " ".join(ctx_sents[j] for j in sorted(top3_idx))
            row_candidates.append((base_score, premise_top1, premise_top3, s, has_unverified_q))

        if definite_hallucinated:
            predictions[row_idx] = 'Hallucinated'
        elif row_candidates:
            for base_score, p1, p3, s, has_unv_q in row_candidates:
                ambiguous_items.append((row_idx, base_score, has_unv_q))
                pairs_top1.append((p1, s))
                pairs_top3.append((p3, s))

    # Chạy batch NLI một lần cho toàn bộ các câu vùng biên
    if ambiguous_items:
        logits1 = NLI_MODEL.predict(pairs_top1, batch_size=64, show_progress_bar=False)
        logits3 = NLI_MODEL.predict(pairs_top3, batch_size=64, show_progress_bar=False)
        probs1 = scipy.special.softmax(logits1, axis=1)
        probs3 = scipy.special.softmax(logits3, axis=1)

        for k, (row_idx, base_score, has_unv_q) in enumerate(ambiguous_items):
            if predictions[row_idx] == 'Hallucinated':
                continue
            # Cột 1: Entailment, Cột 0: Contradiction
            ent = 0.0 if has_unv_q else max(float(probs1[k, 1]), float(probs3[k, 1]))
            con = min(float(probs1[k, 0]), float(probs3[k, 0]))
            final_score = base_score + w_ent * ent - w_con * con
            if final_score <= threshold:
                predictions[row_idx] = 'Hallucinated'

    return predictions


# ══════════════════════════════════════════════════════════════════════════════
# ĐÁNH GIÁ KẾT QUẢ
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(y_true, y_pred, method_name: str):
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
    base = Path(__file__).parent / 'data' / 'DelucionQA_final'

    print("Loading datasets...")
    test = pd.read_csv(base / 'test.csv')
    test['answer_sents'] = test['Answer_sent_tokenized'].apply(literal_eval)
    y_true = (test['Label'] == 'Hallucinated').astype(int).tolist()

    print("\n1. Running SIM-COSINE Baseline on Test set (T1 = 0.6)...")
    preds_cos = [predict_sim_cosine(r['answer_sents'], r['Context'], 0.6) for _, r in test.iterrows()]
    evaluate(y_true, [1 if p == 'Hallucinated' else 0 for p in preds_cos], 'BASELINE 1: SIM-COSINE (Test)')

    print("\n2. Running SIM-OVERLAP Baseline on Test set (T2 = 0.1)...")
    preds_ov = [predict_sim_overlap(r['answer_sents'], r['Context'], 0.1) for _, r in test.iterrows()]
    evaluate(y_true, [1 if p == 'Hallucinated' else 0 for p in preds_ov], 'BASELINE 2: SIM-OVERLAP (Test)')

    print("\n3. Running SIM-HYBRID Baseline on Test set (T1 = 0.9, T2 = 0.1)...")
    preds_hyb = [predict_sim_hybrid(r['answer_sents'], r['Context'], 0.9, 0.1) for _, r in test.iterrows()]
    evaluate(y_true, [1 if p == 'Hallucinated' else 0 for p in preds_hyb], 'BASELINE 3: SIM-HYBRID (Test)')

    print("\n4. Running IMPROVED DIRECTION 1 (Multi-Evidence & Content Grounding)...")
    preds_dir1 = [
        predict_improved_dir1(r['Question'], r['answer_sents'], r['Context'])
        for _, r in test.iterrows()
    ]
    evaluate(y_true, [1 if p == 'Hallucinated' else 0 for p in preds_dir1],
             'IMPROVED (DIRECTION 1): MULTI-EVIDENCE CONTENT GROUNDING (Test)')

    print("\n5. Running IMPROVED DIRECTION 2 (+ NLI Cross-Encoder Verification)...")
    preds_dir2 = predict_improved_dir2_batch(test)
    evaluate(y_true, [1 if p == 'Hallucinated' else 0 for p in preds_dir2],
             'IMPROVED (DIRECTION 2): MULTI-EVIDENCE + NLI CROSS-ENCODER (Test)')
