# Security Policy

## Architectural Security Guarantees

1. **Pre-Retrieval Prompt-Injection Block:** `csrc/helpdesk_rag_kernels.cpp` inspects incoming ticket text in `O(L)` and blocks instruction-override, system-prompt extraction, and `<script` payloads prior to retrieval or generation.
2. **Automatic PII & Credential Redaction:** Secret tokens (`sk-live-*`, `AKIA*`), payment card runs, SSNs, and email addresses are redacted before retrieval indexing or telemetry logging.
3. **Role-Scoped Retrieval & Code-Enforced Abstention:** Restricted SecOps runbooks (`DOC-SEC-008`) are filtered out prior to scoring for standard employees, and low-similarity or ungrounded answers are deterministically refused in code.

## Reporting a Vulnerability

Report security issues privately to `writetonagarjuna@gmail.com`.
