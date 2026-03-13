# /app/main.py
from typing import Dict, List , Union
from fastapi import FastAPI
from pydantic import BaseModel, Field, validator
from app.agent.langchain_orchestrator import TrueAgenticComplianceOrchestrator

app = FastAPI(title="Compliance AI Agentic Server")

orchestrator = TrueAgenticComplianceOrchestrator()


class ScopeApplicability(BaseModel):
    department: str
    role: str

    @classmethod
    def from_str(cls, value):
        # if value is string, treat it as department only
        if isinstance(value, str):
            return cls(department=value, role="Unknown Role")
        return value

class ControlStep(BaseModel):
    description: str
    risk: str = "UNKNOWN"
    data_required: List[str] = []

    @classmethod
    def from_str(cls, value):
        if isinstance(value, str):
            return cls(description=value)
        return value

class PolicyRequest(BaseModel):
    policy_name: str = ""
    policy_description: str = ""
    roles_responsibilities: Dict[str, str] = Field(default_factory=dict)
    scope_applicability: List[Union[ScopeApplicability, str]] = Field(default_factory=list)
    procedure_steps: List[Union[ControlStep, str]] = Field(default_factory=list)

    @validator("scope_applicability", pre=True, each_item=True)
    def ensure_scope_object(cls, v):
        if isinstance(v, str):
            return ScopeApplicability.from_str(v)
        return v

    @validator("procedure_steps", pre=True, each_item=True)
    def ensure_step_object(cls, v):
        if isinstance(v, str):
            return ControlStep.from_str(v)
        return v


@app.post("/audit-policy")
def audit_policy(request: PolicyRequest):
    """
    Endpoint to audit a policy.
    """
    return orchestrator.run(request.dict())