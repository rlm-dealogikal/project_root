# /app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from app.agent.langchain_orchestrator import ComplianceOrchestrator

app = FastAPI(title="Compliance AI Server", description="RAG + Agent + LangChain")

orchestrator = ComplianceOrchestrator()

class PolicyRequest(BaseModel):
    policy_name: str = ""
    policy_description: str = ""
    roles_responsibilities: dict = {}
    scope_applicability: list = []
    procedure_steps: list = []

@app.get("/health")
def health():
    return {"status": "running"}

@app.post("/audit-policy")
def audit_policy(request: PolicyRequest):
    return orchestrator.run(request.dict())
