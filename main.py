# /app/main.py
from fastapi import FastAPI
from pydantic import BaseModel

import json

from app.rag.pipeline import load_documents
from app.rag.retriever import build_vector_db
from app.llm.openrouter_client import query_llm
from app.agent.planner import agent_planner


# ============================
# FastAPI App
# ============================

app = FastAPI(
    title="Compliance AI Server",
    description="RAG + Agent Compliance System"
)


# ============================
# Request Schema
# ============================

class PolicyRequest(BaseModel):
    policy_name: str = ""
    policy_description: str = ""
    roles_responsibilities: dict = {}
    scope_applicability: list = []
    procedure_steps: list = []


# ============================
# Policy Processing Engine
# ============================

def process_policy(policy_json):

    documents, metadatas = load_documents()

    if not documents:
        return {"policy_analysis": "No documents available for RAG"}

    collection = build_vector_db(documents, metadatas)

    policy_text = json.dumps(policy_json)

    results = collection.query(
        query_texts=[policy_text],
        n_results=5,
        include=["documents"]
    )

    context = ""

    if results and results.get("documents"):
        context = "\n".join(results["documents"][0][:3])

    action = agent_planner(context, policy_text)

    print("Agent Action:", action)

    prompt = f"""
    Policy:
    {policy_text}

    Context:
    {context}

    Provide:
    - Risk level
    - Compliance issues
    - Recommendations
    """

    analysis = query_llm(prompt)

    return {
        "action": action,
        "analysis": analysis
    }


# ============================
# API Endpoints
# ============================

@app.get("/health")
def health():
    return {
        "status": "running"
    }


@app.post("/audit-policy")
def audit_policy(request: PolicyRequest):

    result = process_policy(request.dict())

    print("--------------->",json.dumps(result, indent=2))

    return result