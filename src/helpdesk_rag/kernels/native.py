"""C++20 ctypes bridge (`-O3 -std=c++20`) and pure-Python reference twins."""

from __future__ import annotations

import array
import ctypes
import hashlib
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from helpdesk_rag.types import GuardrailScanResult


def _compile_native_lib() -> tuple[ctypes.CDLL | None, str]:
    src_path = Path(__file__).resolve().parents[3] / "csrc" / "helpdesk_rag_kernels.cpp"
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not src_path.exists() or compiler is None:
        return None, "python-fallback"

    digest = hashlib.sha256(src_path.read_bytes()).hexdigest()[:16]
    out_dir = Path(tempfile.gettempdir()) / "helpdesk_rag_native_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".dylib" if Path("/System/Library").exists() else ".so"
    lib_path = out_dir / f"libhelpdesk_rag_{digest}{ext}"

    if not lib_path.exists():
        cmd = [
            compiler,
            "-O3",
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-shared",
            "-fPIC",
            str(src_path),
            "-o",
            str(lib_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return None, "python-fallback"

    lib = ctypes.CDLL(str(lib_path))
    i32p = ctypes.POINTER(ctypes.c_int32)
    f32p = ctypes.POINTER(ctypes.c_float)

    lib.rag_hybrid_bm25_cosine_rrf.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        f32p,
        f32p,
        i32p,
        i32p,
        i32p,
        i32p,
        ctypes.c_float,
        i32p,
        f32p,
        ctypes.c_int32,
        f32p,
        f32p,
        f32p,
    ]
    lib.rag_hybrid_bm25_cosine_rrf.restype = ctypes.c_int32

    lib.rag_scan_guardrails_and_pii.argtypes = [ctypes.c_char_p, ctypes.c_int32, i32p]
    lib.rag_scan_guardrails_and_pii.restype = ctypes.c_int32

    lib.rag_citation_token_overlap.argtypes = [
        i32p,
        ctypes.c_int32,
        i32p,
        ctypes.c_int32,
    ]
    lib.rag_citation_token_overlap.restype = ctypes.c_float

    return lib, f"c++20-native ({Path(compiler).name} -O3)"


_NATIVE_LIB, _BACKEND_NAME = _compile_native_lib()


def backend_name() -> str:
    """Return active C++20/Python retrieval & guardrail kernel backend."""
    return _BACKEND_NAME


def _hybrid_python(
    num_docs: int,
    dim: int,
    doc_vecs: list[float],
    query_vec: list[float],
    doc_row_ptr: list[int],
    doc_term_ids: list[int],
    doc_term_freqs: list[int],
    doc_lengths: list[int],
    avg_doc_len: float,
    query_terms: list[int],
    query_idfs: list[float],
) -> tuple[list[float], list[float], list[float]]:
    k1 = 1.5
    b = 0.75
    safe_avg = avg_doc_len if avg_doc_len > 1e-6 else 1.0

    cosines: list[float] = []
    bm25s: list[float] = []
    for d in range(num_docs):
        base = d * dim
        dot = sum(doc_vecs[base + j] * query_vec[j] for j in range(dim))
        dot = max(-1.0, min(1.0, dot))
        cosines.append(dot)

        start, end = doc_row_ptr[d], doc_row_ptr[d + 1]
        dl_norm = k1 * (1.0 - b + b * (doc_lengths[d] / safe_avg))
        tf_map = {doc_term_ids[pos]: doc_term_freqs[pos] for pos in range(start, end)}
        bm = 0.0
        for qt, idf in zip(query_terms, query_idfs, strict=True):
            if qt in tf_map:
                tf = float(tf_map[qt])
                bm += idf * ((tf * (k1 + 1.0)) / (tf + dl_norm))
        bm25s.append(bm)

    cos_order = sorted(range(num_docs), key=lambda i: (-cosines[i], i))
    bm_order = sorted(range(num_docs), key=lambda i: (-bm25s[i], i))
    cos_rank = [0] * num_docs
    bm_rank = [0] * num_docs
    for r, idx in enumerate(cos_order, 1):
        cos_rank[idx] = r
    for r, idx in enumerate(bm_order, 1):
        bm_rank[idx] = r

    rrf_k = 60.0
    max_rrf = 2.0 / (rrf_k + 1.0)
    confs: list[float] = []
    for d in range(num_docs):
        raw_rrf = (1.0 / (rrf_k + cos_rank[d])) + (1.0 / (rrf_k + bm_rank[d]))
        norm_rrf = raw_rrf / max_rrf
        pos_cos = max(0.0, cosines[d])
        lex_hit = 1.0 if bm25s[d] > 0.0 else 0.0
        conf = 0.65 * pos_cos + 0.20 * norm_rrf * lex_hit + 0.15 * min(1.0, bm25s[d] / 6.0)
        confs.append(conf)
    return cosines, bm25s, confs


def hybrid_bm25_cosine_rrf(
    num_docs: int,
    dim: int,
    doc_vecs: list[float],
    query_vec: list[float],
    doc_row_ptr: list[int],
    doc_term_ids: list[int],
    doc_term_freqs: list[int],
    doc_lengths: list[int],
    avg_doc_len: float,
    query_terms: list[int],
    query_idfs: list[float],
    *,
    use_native: bool = True,
) -> tuple[list[float], list[float], list[float]]:
    """Compute `(cosine_scores, bm25_scores, calibrated_hybrid_confidences)` across `num_docs`."""
    if num_docs <= 0 or dim <= 0:
        return [], [], []
    if use_native and _NATIVE_LIB is not None:
        dv = array.array("f", doc_vecs)
        qv = array.array("f", query_vec)
        rp = array.array("i", doc_row_ptr)
        ti = array.array("i", doc_term_ids if doc_term_ids else [0])
        tf = array.array("i", doc_term_freqs if doc_term_freqs else [0])
        dl = array.array("i", doc_lengths)
        qt = array.array("i", query_terms if query_terms else [0])
        qi = array.array("f", query_idfs if query_idfs else [0.0])
        out_cos = array.array("f", [0.0] * num_docs)
        out_bm = array.array("f", [0.0] * num_docs)
        out_rrf = array.array("f", [0.0] * num_docs)
        i32p = ctypes.POINTER(ctypes.c_int32)
        f32p = ctypes.POINTER(ctypes.c_float)

        rc = _NATIVE_LIB.rag_hybrid_bm25_cosine_rrf(
            num_docs,
            dim,
            ctypes.cast(ctypes.c_void_p(dv.buffer_info()[0]), f32p),
            ctypes.cast(ctypes.c_void_p(qv.buffer_info()[0]), f32p),
            ctypes.cast(ctypes.c_void_p(rp.buffer_info()[0]), i32p),
            ctypes.cast(ctypes.c_void_p(ti.buffer_info()[0]), i32p),
            ctypes.cast(ctypes.c_void_p(tf.buffer_info()[0]), i32p),
            ctypes.cast(ctypes.c_void_p(dl.buffer_info()[0]), i32p),
            ctypes.c_float(avg_doc_len),
            ctypes.cast(ctypes.c_void_p(qt.buffer_info()[0]), i32p),
            ctypes.cast(ctypes.c_void_p(qi.buffer_info()[0]), f32p),
            len(query_terms),
            ctypes.cast(ctypes.c_void_p(out_cos.buffer_info()[0]), f32p),
            ctypes.cast(ctypes.c_void_p(out_bm.buffer_info()[0]), f32p),
            ctypes.cast(ctypes.c_void_p(out_rrf.buffer_info()[0]), f32p),
        )
        if rc == 0:
            return out_cos.tolist(), out_bm.tolist(), out_rrf.tolist()

    return _hybrid_python(
        num_docs,
        dim,
        doc_vecs,
        query_vec,
        doc_row_ptr,
        doc_term_ids,
        doc_term_freqs,
        doc_lengths,
        avg_doc_len,
        query_terms,
        query_idfs,
    )


_INJECTION_PHRASES = (
    "ignore previous instructions",
    "ignore all previous",
    "reveal system prompt",
    "developer mode",
    "bypass guardrail",
    "disregard policy",
    "<script",
)
_SSN_RE = re.compile(r"\d{3}-\d{2}-\d{4}")


def _scan_python(text: str) -> tuple[bool, bool, bool, bool]:
    low = text.lower()
    inj = any(p in low for p in _INJECTION_PHRASES)
    secret = any(s in low for s in ("sk-live-", "akia", "bearer eyj"))

    card_or_ssn = bool(_SSN_RE.search(text))
    consec = 0
    email = False
    n = len(text)
    for i, ch in enumerate(text):
        if "0" <= ch <= "9":
            consec += 1
            if consec >= 15:
                card_or_ssn = True
        elif ch not in {"-", " "}:
            consec = 0

        if ch == "@" and i > 0 and i + 3 < n:
            for k in range(i + 2, min(n, i + 40)):
                if text[k] == ".":
                    email = True
                    break
                if text[k] in {" ", "\n"}:
                    break
    return inj, secret, card_or_ssn, email


def scan_guardrails_and_pii(text: str, *, use_native: bool = True) -> GuardrailScanResult:
    """Scan ticket/context text for prompt injection, API keys, card/SSN digits, and emails."""
    raw = text.encode("utf-8")
    if b"\x00" in raw:
        clean = text.replace("\x00", " ")
        inj, sec, card, em = _scan_python(clean)
        return GuardrailScanResult(
            has_prompt_injection=True or inj,
            has_secret_key=sec,
            has_card_or_ssn=card,
            has_email=em,
            backend=_BACKEND_NAME if use_native else "python-fallback",
        )

    if use_native and _NATIVE_LIB is not None:
        flags = array.array("i", [0] * 4)
        i32p = ctypes.POINTER(ctypes.c_int32)
        rc = _NATIVE_LIB.rag_scan_guardrails_and_pii(
            raw, len(raw), ctypes.cast(ctypes.c_void_p(flags.buffer_info()[0]), i32p)
        )
        if rc == 0:
            return GuardrailScanResult(
                has_prompt_injection=bool(flags[0]),
                has_secret_key=bool(flags[1]),
                has_card_or_ssn=bool(flags[2]),
                has_email=bool(flags[3]),
                backend=_BACKEND_NAME,
            )

    inj, sec, card, em = _scan_python(text)
    return GuardrailScanResult(
        has_prompt_injection=inj,
        has_secret_key=sec,
        has_card_or_ssn=card,
        has_email=em,
        backend="python-fallback",
    )


def citation_token_overlap(
    answer_tokens: list[int],
    context_tokens: list[int],
    *,
    use_native: bool = True,
) -> float:
    """Compute fraction of substantive `answer_tokens` grounded in `context_tokens`."""
    if not answer_tokens:
        return 1.0
    if not context_tokens:
        return 0.0
    if use_native and _NATIVE_LIB is not None:
        at = array.array("i", answer_tokens)
        ct = array.array("i", context_tokens)
        i32p = ctypes.POINTER(ctypes.c_int32)
        res = float(
            _NATIVE_LIB.rag_citation_token_overlap(
                ctypes.cast(ctypes.c_void_p(at.buffer_info()[0]), i32p),
                len(answer_tokens),
                ctypes.cast(ctypes.c_void_p(ct.buffer_info()[0]), i32p),
                len(context_tokens),
            )
        )
        return round(res, 6)
    ctx_set = set(context_tokens)
    grounded = sum(1 for t in answer_tokens if t in ctx_set)
    return round(grounded / len(answer_tokens), 6)


def benchmark_kernels(num_docs: int = 5000, dim: int = 128) -> dict[str, float | int | str]:
    """Benchmark C++20 hybrid BM25 + Cosine + RRF + PII/Injection scanner vs pure Python."""
    inv_norm = 1.0 / (dim**0.5)
    doc_vecs = [
        inv_norm if ((d + j) % 2 == 0) else -inv_norm for d in range(num_docs) for j in range(dim)
    ]
    query_vec = [inv_norm] * dim

    row_ptr = [0]
    term_ids: list[int] = []
    term_freqs: list[int] = []
    doc_lens: list[int] = []
    for d in range(num_docs):
        for k in range(8):
            term_ids.append((d + k * 7) % 256)
            term_freqs.append(1 + (k % 3))
        row_ptr.append(len(term_ids))
        doc_lens.append(16)
    q_terms = [3, 10, 17, 24]
    q_idfs = [2.1, 1.8, 3.0, 2.5]
    sample_ticket = (
        b"My Okta MFA hardware token was reset; contact user@company.com regarding SAP SU53 lock."
    )

    if _NATIVE_LIB is not None:
        dv = array.array("f", doc_vecs)
        qv = array.array("f", query_vec)
        rp = array.array("i", row_ptr)
        ti = array.array("i", term_ids)
        tf = array.array("i", term_freqs)
        dl = array.array("i", doc_lens)
        qt = array.array("i", q_terms)
        qi = array.array("f", q_idfs)
        out_c = array.array("f", [0.0] * num_docs)
        out_b = array.array("f", [0.0] * num_docs)
        out_r = array.array("f", [0.0] * num_docs)
        flags = array.array("i", [0] * 4)
        i32p = ctypes.POINTER(ctypes.c_int32)
        f32p = ctypes.POINTER(ctypes.c_float)

        dv_p = ctypes.cast(ctypes.c_void_p(dv.buffer_info()[0]), f32p)
        qv_p = ctypes.cast(ctypes.c_void_p(qv.buffer_info()[0]), f32p)
        rp_p = ctypes.cast(ctypes.c_void_p(rp.buffer_info()[0]), i32p)
        ti_p = ctypes.cast(ctypes.c_void_p(ti.buffer_info()[0]), i32p)
        tf_p = ctypes.cast(ctypes.c_void_p(tf.buffer_info()[0]), i32p)
        dl_p = ctypes.cast(ctypes.c_void_p(dl.buffer_info()[0]), i32p)
        qt_p = ctypes.cast(ctypes.c_void_p(qt.buffer_info()[0]), i32p)
        qi_p = ctypes.cast(ctypes.c_void_p(qi.buffer_info()[0]), f32p)
        oc_p = ctypes.cast(ctypes.c_void_p(out_c.buffer_info()[0]), f32p)
        ob_p = ctypes.cast(ctypes.c_void_p(out_b.buffer_info()[0]), f32p)
        or_p = ctypes.cast(ctypes.c_void_p(out_r.buffer_info()[0]), f32p)
        fl_p = ctypes.cast(ctypes.c_void_p(flags.buffer_info()[0]), i32p)

        t0 = time.perf_counter()
        for _ in range(15):
            _NATIVE_LIB.rag_hybrid_bm25_cosine_rrf(
                num_docs,
                dim,
                dv_p,
                qv_p,
                rp_p,
                ti_p,
                tf_p,
                dl_p,
                16.0,
                qt_p,
                qi_p,
                4,
                oc_p,
                ob_p,
                or_p,
            )
            _NATIVE_LIB.rag_scan_guardrails_and_pii(sample_ticket, len(sample_ticket), fl_p)
        native_ms = ((time.perf_counter() - t0) * 1000.0) / 15.0
    else:  # pragma: no cover
        native_ms = 0.0

    t1 = time.perf_counter()
    for _ in range(15):
        _hybrid_python(
            num_docs,
            dim,
            doc_vecs,
            query_vec,
            row_ptr,
            term_ids,
            term_freqs,
            doc_lens,
            16.0,
            q_terms,
            q_idfs,
        )
        _scan_python(sample_ticket.decode("utf-8"))
    python_ms = ((time.perf_counter() - t1) * 1000.0) / 15.0

    return {
        "backend": _BACKEND_NAME,
        "num_docs": num_docs,
        "embedding_dim": dim,
        "native_ms": round(native_ms, 3),
        "python_ms": round(python_ms, 3),
        "speedup_x": round(python_ms / max(native_ms, 1e-6), 2),
    }
