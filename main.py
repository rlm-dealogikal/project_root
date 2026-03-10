# /app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from app.agent.langchain_orchestrator import TrueAgenticComplianceOrchestrator

app = FastAPI(title="Compliance AI Agentic Server")

orchestrator = TrueAgenticComplianceOrchestrator()

class PolicyRequest(BaseModel):
    policy_name: str = ""
    policy_description: str = ""
    roles_responsibilities: dict = None
    scope_applicability: list = None
    procedure_steps: list = None

    def __init__(self, **data):
        super().__init__(**data)
        self.roles_responsibilities = self.roles_responsibilities or {}
        self.scope_applicability = self.scope_applicability or []
        self.procedure_steps = self.procedure_steps or []

@app.post("/audit-policy")
def audit_policy(request: PolicyRequest):
    return orchestrator.run(request.dict())