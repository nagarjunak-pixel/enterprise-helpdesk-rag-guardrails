// Native C++20 Hybrid BM25 + Cosine + RRF scorer, PII/Injection scanner & Citation Grounding kernel.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

inline char lower_ascii(char c) noexcept {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

inline bool is_digit_ch(char c) noexcept {
    return c >= '0' && c <= '9';
}

bool contains_phrase_ci(const char* hay, int32_t hay_len, const char* needle) noexcept {
    int32_t nlen = static_cast<int32_t>(std::strlen(needle));
    if (nlen == 0 || hay_len < nlen) {
        return false;
    }
    for (int32_t i = 0; i <= hay_len - nlen; ++i) {
        bool match = true;
        for (int32_t j = 0; j < nlen; ++j) {
            if (lower_ascii(hay[i + j]) != needle[j]) {
                match = false;
                break;
            }
        }
        if (match) {
            return true;
        }
    }
    return false;
}

}  // namespace

extern "C" {

// Compute dense cosine similarities, Okapi BM25 scores, and Reciprocal Rank Fusion (RRF) confidence.
int32_t rag_hybrid_bm25_cosine_rrf(
    int32_t num_docs,
    int32_t dim,
    const float* doc_vecs,       // [num_docs * dim], L2-normalized
    const float* query_vec,      // [dim], L2-normalized
    const int32_t* doc_row_ptr,  // [num_docs + 1] CSR term IDs per doc
    const int32_t* doc_term_ids,
    const int32_t* doc_term_freqs,
    const int32_t* doc_lengths,
    float avg_doc_len,
    const int32_t* query_terms,
    const float* query_idfs,
    int32_t num_query_terms,
    float* out_cosine,
    float* out_bm25,
    float* out_rrf
) {
    if (num_docs <= 0 || dim <= 0 || doc_vecs == nullptr || query_vec == nullptr ||
        doc_row_ptr == nullptr || doc_lengths == nullptr ||
        out_cosine == nullptr || out_bm25 == nullptr || out_rrf == nullptr) {
        return -1;
    }

    constexpr float k1 = 1.5f;
    constexpr float b = 0.75f;
    float safe_avg = (avg_doc_len > 1e-6f) ? avg_doc_len : 1.0f;

    for (int32_t d = 0; d < num_docs; ++d) {
        const float* dv = doc_vecs + static_cast<size_t>(d) * static_cast<size_t>(dim);
        float dot = 0.0f;
        for (int32_t j = 0; j < dim; ++j) {
            dot += dv[j] * query_vec[j];
        }
        if (dot > 1.0f) dot = 1.0f;
        if (dot < -1.0f) dot = -1.0f;
        out_cosine[d] = dot;

        float bm = 0.0f;
        int32_t start = doc_row_ptr[d];
        int32_t end = doc_row_ptr[d + 1];
        float dl_norm = k1 * (1.0f - b + b * (static_cast<float>(doc_lengths[d]) / safe_avg));

        for (int32_t q = 0; q < num_query_terms; ++q) {
            int32_t qt = query_terms[q];
            float idf = query_idfs[q];
            for (int32_t pos = start; pos < end; ++pos) {
                if (doc_term_ids[pos] == qt) {
                    float tf = static_cast<float>(doc_term_freqs[pos]);
                    bm += idf * ((tf * (k1 + 1.0f)) / (tf + dl_norm));
                    break;
                }
            }
        }
        out_bm25[d] = bm;
    }

    std::vector<int32_t> cos_order(static_cast<size_t>(num_docs));
    std::vector<int32_t> bm_order(static_cast<size_t>(num_docs));
    for (int32_t i = 0; i < num_docs; ++i) {
        cos_order[static_cast<size_t>(i)] = i;
        bm_order[static_cast<size_t>(i)] = i;
    }

    std::stable_sort(cos_order.begin(), cos_order.end(), [&](int32_t a, int32_t b_idx) {
        if (out_cosine[a] != out_cosine[b_idx]) {
            return out_cosine[a] > out_cosine[b_idx];
        }
        return a < b_idx;
    });
    std::stable_sort(bm_order.begin(), bm_order.end(), [&](int32_t a, int32_t b_idx) {
        if (out_bm25[a] != out_bm25[b_idx]) {
            return out_bm25[a] > out_bm25[b_idx];
        }
        return a < b_idx;
    });

    std::vector<int32_t> cos_rank(static_cast<size_t>(num_docs));
    std::vector<int32_t> bm_rank(static_cast<size_t>(num_docs));
    for (int32_t r = 0; r < num_docs; ++r) {
        cos_rank[static_cast<size_t>(cos_order[static_cast<size_t>(r)])] = r + 1;
        bm_rank[static_cast<size_t>(bm_order[static_cast<size_t>(r)])] = r + 1;
    }

    constexpr float rrf_k = 60.0f;
    float max_rrf = 2.0f / (rrf_k + 1.0f);
    for (int32_t d = 0; d < num_docs; ++d) {
        float raw_rrf = (1.0f / (rrf_k + static_cast<float>(cos_rank[static_cast<size_t>(d)]))) +
                        (1.0f / (rrf_k + static_cast<float>(bm_rank[static_cast<size_t>(d)])));
        float norm_rrf = raw_rrf / max_rrf;  // in (0, 1]
        float pos_cos = (out_cosine[d] > 0.0f) ? out_cosine[d] : 0.0f;
        float lexical_hit = (out_bm25[d] > 0.0f) ? 1.0f : 0.0f;
        // Calibrated hybrid confidence combining cosine similarity, lexical evidence & RRF rank
        out_rrf[d] = 0.65f * pos_cos + 0.20f * norm_rrf * lexical_hit +
                     0.15f * std::min(1.0f, out_bm25[d] / 6.0f);
    }
    return 0;
}

// out_flags[4]:
//   [0] has_prompt_injection
//   [1] has_secret_key
//   [2] has_card_or_ssn
//   [3] has_email
int32_t rag_scan_guardrails_and_pii(
    const char* text,
    int32_t len,
    int32_t* out_flags
) {
    if (text == nullptr || out_flags == nullptr || len < 0) {
        return -1;
    }

    static const char* kInjections[] = {
        "ignore previous instructions",
        "ignore all previous",
        "reveal system prompt",
        "developer mode",
        "bypass guardrail",
        "disregard policy",
        "<script"
    };
    bool inj = false;
    for (const char* pat : kInjections) {
        if (contains_phrase_ci(text, len, pat)) {
            inj = true;
            break;
        }
    }

    bool secret = contains_phrase_ci(text, len, "sk-live-") ||
                  contains_phrase_ci(text, len, "akia") ||
                  contains_phrase_ci(text, len, "bearer eyj");

    bool card_or_ssn = false;
    bool email = false;
    int32_t consecutive_digits = 0;

    for (int32_t i = 0; i < len; ++i) {
        char c = text[i];
        if (is_digit_ch(c)) {
            ++consecutive_digits;
            if (consecutive_digits >= 15) {
                card_or_ssn = true;
            }
        } else if (c != '-' && c != ' ') {
            consecutive_digits = 0;
        }

        // SSN pattern NNN-NN-NNNN
        if (i + 11 <= len &&
            is_digit_ch(text[i]) && is_digit_ch(text[i + 1]) && is_digit_ch(text[i + 2]) &&
            text[i + 3] == '-' &&
            is_digit_ch(text[i + 4]) && is_digit_ch(text[i + 5]) &&
            text[i + 6] == '-' &&
            is_digit_ch(text[i + 7]) && is_digit_ch(text[i + 8]) &&
            is_digit_ch(text[i + 9]) && is_digit_ch(text[i + 10])) {
            card_or_ssn = true;
        }

        if (c == '@' && i > 0 && i + 3 < len) {
            for (int32_t k = i + 2; k < std::min(len, i + 40); ++k) {
                if (text[k] == '.') {
                    email = true;
                    break;
                }
                if (text[k] == ' ' || text[k] == '\n') {
                    break;
                }
            }
        }
    }

    out_flags[0] = inj ? 1 : 0;
    out_flags[1] = secret ? 1 : 0;
    out_flags[2] = card_or_ssn ? 1 : 0;
    out_flags[3] = email ? 1 : 0;
    return 0;
}

float rag_citation_token_overlap(
    const int32_t* answer_tokens,
    int32_t num_answer,
    const int32_t* context_tokens,
    int32_t num_context
) {
    if (num_answer <= 0 || answer_tokens == nullptr) {
        return 1.0f;
    }
    if (num_context <= 0 || context_tokens == nullptr) {
        return 0.0f;
    }
    int32_t grounded = 0;
    for (int32_t i = 0; i < num_answer; ++i) {
        int32_t tok = answer_tokens[i];
        for (int32_t j = 0; j < num_context; ++j) {
            if (context_tokens[j] == tok) {
                ++grounded;
                break;
            }
        }
    }
    return static_cast<float>(grounded) / static_cast<float>(num_answer);
}

}  // extern "C"
