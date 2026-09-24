"""
Pydantic models mirroring the HHGOA_IEEE answer format EXACTLY, field for
field, as specified in data/README.md ("Answer Format" section). Do not
rename or reshape these — the benchmark is scored against this shape.

Enums (pattern values, action names, approval routes, verdicts, statuses)
are copied verbatim from the policy section of the README.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CaseStatus(str, Enum):
    open = "open"
    closed_fraud = "closed_fraud"
    closed_legitimate = "closed_legitimate"
    escalated = "escalated"


class Verdict(str, Enum):
    fraud = "fraud"
    legitimate = "legitimate"
    uncertain = "uncertain"


class Pattern(str, Enum):
    card_testing = "card_testing"
    card_not_present_fraud = "card_not_present_fraud"
    card_not_present_new_device = "card_not_present_new_device"
    out_of_region_use = "out_of_region_use"
    account_takeover = "account_takeover"
    undocumented = "undocumented"
    none_ = "none"


class EvidenceSource(str, Enum):
    graph = "graph"
    document = "document"
    customer = "customer"
    external = "external"


class Evidence(BaseModel):
    claim: str
    source: EvidenceSource
    ref: str
    entity_ids: List[str] = Field(default_factory=list)


class Case(BaseModel):
    status: CaseStatus
    verdict: Verdict
    fraud_probability: float = Field(ge=0.0, le=1.0)
    pattern: Pattern
    pattern_description: str = ""
    affected_txn_ids: List[str] = Field(default_factory=list)
    first_suspicious_txn_id: str = ""
    connected_card_ids: List[str] = Field(default_factory=list)
    connected_device_profiles: List[str] = Field(default_factory=list)
    exposure_usd: float = 0.0
    evidence: List[Evidence] = Field(default_factory=list)
    similar_prior_cases: List[str] = Field(default_factory=list)
    summary: str
    written_to_graph: bool = False
    graph_case_id: str = ""

    @field_validator("pattern_description")
    @classmethod
    def _require_description_for_undocumented(cls, v, info):
        pattern = info.data.get("pattern")
        if pattern == Pattern.undocumented and not v:
            raise ValueError("pattern_description is required when pattern == 'undocumented'")
        return v


class EvidenceRequestType(str, Enum):
    customer_validation = "customer_validation"
    step_up_auth = "step_up_auth"
    analyst_info = "analyst_info"


class EvidenceRequest(BaseModel):
    type: EvidenceRequestType
    asked_after_step: int
    assumed_response: str


class ApprovalRoute(str, Enum):
    auto = "auto"
    L1 = "L1"
    L2 = "L2"


class ActionName(str, Enum):
    ALLOW_TRANSACTION = "ALLOW_TRANSACTION"
    DECLINE_TRANSACTION = "DECLINE_TRANSACTION"
    MONITOR_CARD = "MONITOR_CARD"
    MONITOR_CONNECTED_CARDS = "MONITOR_CONNECTED_CARDS"
    WARN_CUSTOMER = "WARN_CUSTOMER"
    VERIFY_WITH_CUSTOMER = "VERIFY_WITH_CUSTOMER"
    STEP_UP_AUTH = "STEP_UP_AUTH"
    BLOCK_CARD = "BLOCK_CARD"
    BLOCK_ALL_CARDS = "BLOCK_ALL_CARDS"
    GENERATE_REPORT = "GENERATE_REPORT"
    CREATE_CASE = "CREATE_CASE"
    FILE_REPORT = "FILE_REPORT"
    ESCALATE_TO_ANALYST = "ESCALATE_TO_ANALYST"
    CLOSE_NO_FRAUD = "CLOSE_NO_FRAUD"


class ActionRecommendation(BaseModel):
    action: ActionName
    route: ApprovalRoute
    reason: str


class NextBestActions(BaseModel):
    initial: List[ActionRecommendation] = Field(default_factory=list)
    final: List[ActionRecommendation] = Field(default_factory=list)
    what_changed: str = "nothing"


class SAR(BaseModel):
    file: bool
    reason: str
    narrative: str = ""
    subjects: List[str] = Field(default_factory=list)
    total_amount_usd: float = 0.0
    activity_dates: List[str] = Field(default_factory=list)

    @field_validator("narrative")
    @classmethod
    def _require_narrative_if_filed(cls, v, info):
        if info.data.get("file") and not v:
            raise ValueError("narrative is required when file is true")
        return v


class CaseAnswer(BaseModel):
    """Top-level object written to cases/<case_id>.json"""

    case_id: str
    case: Case
    evidence_requests: List[EvidenceRequest] = Field(default_factory=list)
    next_best_actions: NextBestActions
    sar: SAR
    stop_reason: str
    tool_calls: int = 0
    tokens: int = 0
    latency_s: float = 0.0

    model_config = ConfigDict(use_enum_values=True)
