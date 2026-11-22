"""Hybrid BM25 + Cosine + RRF RAG engine with code-enforced low-confidence refusal & PII masking."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from helpdesk_rag.kb import DEFAULT_KB_ARTICLES
from helpdesk_rag.kernels import (
    backend_name,
    citation_token_overlap,
    hybrid_bm25_cosine_rrf,
    scan_guardrails_and_pii,
)
from helpdesk_rag.types import (
    HelpDeskTicketRequest,
    HelpDeskTicketResponse,
    KBArticle,
    RetrievedChunk,
)

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "for",
        "from",
        "get",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "me",
        "my",
        "need",
        "of",
        "on",
        "or",
        "our",
        "please",
        "should",
        "the",
        "their",
        "to",
        "use",
        "we",
        "what",
        "when",
        "where",
        "who",
        "why",
        "with",
    }
)

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CARD_RE = re.compile(r"\b(?:\d[ -]*?){15,19}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_SECRET_RE = re.compile(r"\b(?:sk-live-[A-Za-z0-9_-]+|AKIA[0-9A-Z]{8,20})\b", re.IGNORECASE)

EMBED_DIM = 192


def _stem_light(token: str) -> str:
    t = token.lower()
    for suffix in ("ing", "tion", "ions", "ed", "es", "s"):
        if len(t) > len(suffix) + 3 and t.endswith(suffix):
            return t[: -len(suffix)]
    return t


def _content_tokens(text: str) -> list[str]:
    raw = [w.lower() for w in _WORD_RE.findall(text)]
    return [_stem_light(w) for w in raw if len(w) >= 2 and w not in _STOPWORDS]


def _redact_pii_text(text: str) -> tuple[str, bool]:
    out = _SECRET_RE.sub("[REDACTED_SECRET]", text)
    out = _SSN_RE.sub("[REDACTED_SSN]", out)
    out = _CARD_RE.sub("[REDACTED_CARD]", out)
    out = _EMAIL_RE.sub("[REDACTED_EMAIL]", out)
    return out, (out != text)


def _feature_hash_embed(tokens: list[str], idf_map: dict[str, float]) -> list[float]:
    vec = [0.0] * EMBED_DIM
    if not tokens:
        return vec
    counts = Counter(tokens)
    for tok, tf in counts.items():
        weight = (1.0 + math.log(tf)) * idf_map.get(tok, 0.35)
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % EMBED_DIM
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[idx] += sign * weight

        # Character 4-gram subword features for typo/compound resilience (only for known KB stems)
        if tok in idf_map and len(tok) >= 4:
            for k in range(len(tok) - 3):
                gram = f"#{tok[k : k + 4]}"
                gd = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                g_idx = int.from_bytes(gd[:4], "little") % EMBED_DIM
                g_sign = 1.0 if (gd[4] & 1) == 0 else -1.0
                vec[g_idx] += g_sign * (weight * 0.25)

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 1e-9:
        vec = [v / norm for v in vec]
    return vec


class HelpDeskRAGEngine:
    """Enterprise Help-Desk RAG assistant with C++20 hybrid search & code-enforced refusal."""

    def __init__(self, articles: list[KBArticle] | None = None) -> None:
        self.articles: list[KBArticle] = (
            list(articles) if articles is not None else list(DEFAULT_KB_ARTICLES)
        )
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._doc_tokens: list[list[str]] = []
        self._doc_token_ids: list[list[int]] = []
        self._doc_vecs: list[list[float]] = []
        self.total_queries: int = 0
        self.total_answered: int = 0
        self.total_refused: int = 0
        self.total_injections_blocked: int = 0
        self._build_index()

    def _build_index(self) -> None:
        df: Counter[str] = Counter()
        tokenized_docs: list[list[str]] = []
        for art in self.articles:
            toks = _content_tokens(f"{art.title} {art.category} {art.content}")
            tokenized_docs.append(toks)
            for unique_t in set(toks):
                df[unique_t] += 1

        n_docs = max(1, len(self.articles))
        for idx, (term, doc_freq) in enumerate(sorted(df.items())):
            self._vocab[term] = idx
            self._idf[term] = math.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))

        self._doc_tokens = tokenized_docs
        self._doc_token_ids = [[self._vocab[t] for t in toks] for toks in tokenized_docs]
        self._doc_vecs = [_feature_hash_embed(toks, self._idf) for toks in tokenized_docs]

    def _tok_to_int(self, tok: str) -> int:
        if tok in self._vocab:
            return self._vocab[tok]
        return 100_000 + (
            int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=4).digest(), "little")
            & 0xFFFFF
        )

    def answer_ticket(
        self,
        req: HelpDeskTicketRequest,
        *,
        use_native: bool = True,
        injected_ungrounded_draft: str | None = None,
    ) -> HelpDeskTicketResponse:
        """Process a help-desk ticket with input guardrails, RBAC hybrid retrieval & refusal gates."""
        self.total_queries += 1
        scan = scan_guardrails_and_pii(req.question, use_native=use_native)
        sanitized_q, pii_redacted = _redact_pii_text(req.question)
        if scan.has_secret_key or scan.has_card_or_ssn or scan.has_email:
            pii_redacted = True

        # Gate 1: Hard pre-retrieval prompt-injection block
        if scan.has_prompt_injection:
            self.total_refused += 1
            self.total_injections_blocked += 1
            return HelpDeskTicketResponse(
                decision="BLOCKED_PROMPT_INJECTION",
                refused=True,
                backend=backend_name(),
                sanitized_question=sanitized_q,
                pii_redacted=pii_redacted,
                top_similarity=0.0,
                top_confidence=0.0,
                grounding_overlap=0.0,
                citations=[],
                answer=(
                    "REFUSED [BLOCKED_PROMPT_INJECTION]: Instruction-override or policy-bypass pattern "
                    "detected by guardrail scanner. Ticket logged for SecOps review."
                ),
                escalation_queue="SecOps-Triage",
                guardrail_scan=scan,
                retrieved_chunks=[],
            )

        # Gate 2: Role-scoped KB filtering (employee cannot retrieve SecOps-only runbooks!)
        allowed_indices = [
            i for i, art in enumerate(self.articles) if req.user_role in art.allowed_roles
        ]
        if not allowed_indices:
            self.total_refused += 1
            return HelpDeskTicketResponse(
                decision="REFUSED_LOW_CONFIDENCE",
                refused=True,
                backend=backend_name(),
                sanitized_question=sanitized_q,
                pii_redacted=pii_redacted,
                top_similarity=0.0,
                top_confidence=0.0,
                grounding_overlap=0.0,
                citations=[],
                answer="REFUSED [REFUSED_LOW_CONFIDENCE]: Zero role-authorized knowledge base articles available.",
                escalation_queue="L2-Human-HelpDesk",
                guardrail_scan=scan,
                retrieved_chunks=[],
            )

        q_toks = _content_tokens(sanitized_q)
        q_vec = _feature_hash_embed(q_toks, self._idf)

        sub_vecs: list[float] = []
        row_ptr = [0]
        term_ids: list[int] = []
        term_freqs: list[int] = []
        doc_lens: list[int] = []

        for doc_idx in allowed_indices:
            sub_vecs.extend(self._doc_vecs[doc_idx])
            counts = Counter(self._doc_token_ids[doc_idx])
            for tid, freq in sorted(counts.items()):
                term_ids.append(tid)
                term_freqs.append(freq)
            row_ptr.append(len(term_ids))
            doc_lens.append(len(self._doc_token_ids[doc_idx]))

        avg_len = sum(doc_lens) / max(1, len(doc_lens))
        unique_q_known = sorted({t for t in q_toks if t in self._vocab})
        q_term_ids = [self._vocab[t] for t in unique_q_known]
        q_idfs = [self._idf[t] for t in unique_q_known]

        cos_scores, bm25_scores, conf_scores = hybrid_bm25_cosine_rrf(
            len(allowed_indices),
            EMBED_DIM,
            sub_vecs,
            q_vec,
            row_ptr,
            term_ids,
            term_freqs,
            doc_lens,
            avg_len,
            q_term_ids,
            q_idfs,
            use_native=use_native,
        )

        ranked_local = sorted(
            range(len(allowed_indices)),
            key=lambda k: (-conf_scores[k], -cos_scores[k], k),
        )[: req.top_k]

        chunks: list[RetrievedChunk] = []
        for loc_i in ranked_local:
            art = self.articles[allowed_indices[loc_i]]
            chunks.append(
                RetrievedChunk(
                    doc_id=art.doc_id,
                    title=art.title,
                    category=art.category,
                    cosine_similarity=round(cos_scores[loc_i], 4),
                    bm25_score=round(bm25_scores[loc_i], 4),
                    hybrid_confidence=round(conf_scores[loc_i], 4),
                    snippet=art.content,
                )
            )

        top_sim = chunks[0].cosine_similarity if chunks else 0.0
        top_conf = chunks[0].hybrid_confidence if chunks else 0.0

        # Gate 3: Code-enforced refusal when retrieval similarity or hybrid confidence is below threshold!
        if top_sim < req.min_similarity_threshold or top_conf < req.min_confidence_threshold:
            self.total_refused += 1
            return HelpDeskTicketResponse(
                decision="REFUSED_LOW_CONFIDENCE",
                refused=True,
                backend=backend_name(),
                sanitized_question=sanitized_q,
                pii_redacted=pii_redacted,
                top_similarity=top_sim,
                top_confidence=top_conf,
                grounding_overlap=0.0,
                citations=[],
                answer=(
                    f"REFUSED [LOW_RETRIEVAL_CONFIDENCE]: Top similarity ({top_sim:.3f} < {req.min_similarity_threshold:.2f}) "
                    f"or hybrid confidence ({top_conf:.3f} < {req.min_confidence_threshold:.2f}) is below safety threshold. "
                    "Abstaining rather than guessing and routing ticket to L2 Human Help-Desk."
                ),
                escalation_queue="L2-Human-HelpDesk",
                guardrail_scan=scan,
                retrieved_chunks=chunks,
            )

        primary = chunks[0]
        supporting = [
            c for c in chunks[1:] if c.hybrid_confidence >= req.min_confidence_threshold * 0.92
        ]
        cited_chunks = [primary, *supporting[:1]]
        citations = [c.doc_id for c in cited_chunks]

        draft_body = (
            injected_ungrounded_draft
            if injected_ungrounded_draft is not None
            else f"{primary.snippet} [{primary.doc_id}]"
        )

        # Gate 4: C++20 lexical citation-grounding verification on substantive answer tokens
        ans_toks = [
            self._tok_to_int(t) for t in _content_tokens(draft_body) if not t.startswith("doc")
        ]
        ctx_toks = [
            self._tok_to_int(t)
            for c in cited_chunks
            for t in _content_tokens(f"{c.title} {c.snippet}")
        ]
        overlap = round(citation_token_overlap(ans_toks, ctx_toks, use_native=use_native), 4)

        if overlap < req.min_grounding_overlap:
            self.total_refused += 1
            return HelpDeskTicketResponse(
                decision="REFUSED_UNGROUNDED_ANSWER",
                refused=True,
                backend=backend_name(),
                sanitized_question=sanitized_q,
                pii_redacted=pii_redacted,
                top_similarity=top_sim,
                top_confidence=top_conf,
                grounding_overlap=overlap,
                citations=[],
                answer=(
                    f"REFUSED [UNGROUNDED_ANSWER]: Answer token grounding ({overlap:.3f}) fell below "
                    f"minimum threshold ({req.min_grounding_overlap:.2f}). Escalating to human support."
                ),
                escalation_queue="L2-Human-HelpDesk",
                guardrail_scan=scan,
                retrieved_chunks=chunks,
            )

        self.total_answered += 1
        return HelpDeskTicketResponse(
            decision="ANSWERED_GROUNDED",
            refused=False,
            backend=backend_name(),
            sanitized_question=sanitized_q,
            pii_redacted=pii_redacted,
            top_similarity=top_sim,
            top_confidence=top_conf,
            grounding_overlap=overlap,
            citations=citations,
            answer=draft_body,
            escalation_queue=None,
            guardrail_scan=scan,
            retrieved_chunks=chunks,
        )
