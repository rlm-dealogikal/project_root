# /app/main.py
from fastapi import FastAPI
from pydantic import BaseModel,Field, validator
from typing import Dict, List ,Union
from app.agent.langchain_orchestrator import TrueAgenticComplianceOrchestrator

app = FastAPI(title="Compliance AI Agentic Server")

orchestrator = TrueAgenticComplianceOrchestrator()


# Nested models for requests
class ScopeApplicability(BaseModel):
    department: str
    role: str = "Unknown Role"


class ControlStep(BaseModel):
    description: str
    risk: str = "UNKNOWN"
    data_required: List[str] = Field(default_factory=list)


class PolicyRequest(BaseModel):
    policy_name: str = ""
    policy_description: str = ""

    roles_responsibilities: Dict[str, str] = Field(default_factory=dict)

    scope_applicability: List[Union[ScopeApplicability, str]] = Field(default_factory=list)

    procedure_steps: List[Union[ControlStep, str]] = Field(default_factory=list)

    @validator("scope_applicability", pre=True, each_item=True)
    def normalize_scope(cls, v):
        if isinstance(v, str):
            return ScopeApplicability(department=v)
        return v

    @validator("procedure_steps", pre=True, each_item=True)
    def normalize_steps(cls, v):
        if isinstance(v, str):
            return ControlStep(description=v)
        return v


@app.post("/audit-policy")
def audit_policy(request: PolicyRequest):
    """
    Audit a compliance policy using the TrueAgenticComplianceOrchestrator.
    """
    return orchestrator.run(request.dict())