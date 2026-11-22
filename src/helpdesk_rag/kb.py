"""Curated enterprise IT, Cloud, SAP & HR Help-Desk Knowledge Base with role-based visibility."""

from __future__ import annotations

from helpdesk_rag.types import KBArticle

DEFAULT_KB_ARTICLES: list[KBArticle] = [
    KBArticle(
        doc_id="DOC-IAM-001",
        title="Okta Multi-Factor Authentication (MFA) Hardware & Mobile Reset",
        category="IAM",
        allowed_roles=["employee", "it_support", "secops_admin"],
        content=(
            "To reset a lost or replaced Okta Verify MFA device, open the Enterprise Self-Service IAM Portal "
            "at iam.corp.internal/mfa-reset, authenticate with your active directory password plus manager "
            "one-time verification code, and enroll your new device within 15 minutes. If both factors are "
            "unavailable, contact L2 Identity Desk for live video badge verification."
        ),
    ),
    KBArticle(
        doc_id="DOC-NET-002",
        title="GlobalProtect Corporate VPN Split-Tunnel & DNS Resolution Fix",
        category="Network",
        allowed_roles=["employee", "it_support", "secops_admin"],
        content=(
            "When GlobalProtect VPN connects but internal git.corp.internal or intranet hosts fail to resolve, "
            "refresh the WireGuard/IPSec split-tunnel DNS cache via 'gp-cli --flush-dns', verify gateway "
            "region matches your nearest hub (DXB-GW01, FRA-GW02, IAD-GW01), and confirm UDP port 4501 is "
            "not blocked by hotel or guest Wi-Fi firewalls."
        ),
    ),
    KBArticle(
        doc_id="DOC-SAP-003",
        title="SAP S/4HANA Fiori & GUI Authorization Failure (SU53) Escalation",
        category="SAP",
        allowed_roles=["employee", "it_support", "secops_admin"],
        content=(
            "If an SAP S/4HANA Fiori tile or GUI transaction fails with 'No Authorization', immediately execute "
            "transaction SU53 (or export the Fiori Authorization Trace JSON), attach the missing authorization "
            "object and CompanyCode (BUKRS) to a GRC Access Request, and obtain data-owner sign-off. Help-desk "
            "agents cannot manually grant SAP_ALL or bypass GRC Segregation-of-Duties (SoD) checks."
        ),
    ),
    KBArticle(
        doc_id="DOC-CLD-004",
        title="AWS IAM Access Key Rotation & Temporary STS Federation Policy",
        category="Cloud",
        allowed_roles=["employee", "it_support", "secops_admin"],
        content=(
            "Long-lived static AWS IAM access keys (AKIA*) are prohibited in production accounts and expire "
            "automatically after 30 days in sandbox accounts. Engineers must authenticate via Okta AWS SSO "
            "short-lived STS session roles (12-hour max TTL) or GitHub Actions OIDC federated workload identities."
        ),
    ),
    KBArticle(
        doc_id="DOC-END-005",
        title="Windows BitLocker & macOS FileVault Recovery Key Retrieval",
        category="IAM",
        allowed_roles=["employee", "it_support", "secops_admin"],
        content=(
            "When a corporate laptop prompts for a 48-digit BitLocker recovery key or macOS FileVault escrow key "
            "after a firmware update, sign in from a secondary device at escrow.corp.internal/recovery using Okta "
            "MFA. Escrowed recovery keys automatically rotate 24 hours after single-use unlock."
        ),
    ),
    KBArticle(
        doc_id="DOC-HR-006",
        title="Corporate Travel, Flight Booking & Per-Diem Expense Reimbursement Policy",
        category="HR",
        allowed_roles=["employee", "it_support", "secops_admin"],
        content=(
            "All international business flights and hotels must be booked through Concur Travel at least 7 days "
            "prior to departure. Itemized receipts are mandatory for expenses above 75 USD (or 275 AED), and "
            "expense reports must be submitted within 30 calendar days of trip completion."
        ),
    ),
    KBArticle(
        doc_id="DOC-SEC-007",
        title="Enterprise Generative AI & LLM Acceptable Use and Data Classification Policy",
        category="SecOps",
        allowed_roles=["employee", "it_support", "secops_admin"],
        content=(
            "Employees must never paste Restricted customer PII, source-code secrets, IBANs, or unreleased "
            "financial results into public consumer AI chat tools. Use only the approved Enterprise AI Gateway "
            "with zero-data-retention agreements and automatic PII redaction enabled."
        ),
    ),
    KBArticle(
        doc_id="DOC-SEC-008",
        title="P1 Security Incident Containment & Credential Compromise Playbook",
        category="SecOps",
        allowed_roles=["secops_admin"],
        sla_tier="L3-SecOps-Restricted",
        content=(
            "SECOPS RESTRICTED RUNBOOK: Upon confirmed credential compromise or anomalous SIEM token replay, "
            "execute 'soc-ctl quarantine-host --revoke-okta-sessions', snapshot volatile memory to the forensic "
            "S3 worm-locked bucket within 15 minutes, and page the CISO duty commander."
        ),
    ),
]
