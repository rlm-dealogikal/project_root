import json
from app.llm.openrouter_client import query_llm
from app.rag.pipeline import load_documents
from app.rag.retriever import build_vector_db
from app.agent.planner import agent_planner


class ComplianceOrchestrator:
    """RAG + Agent + OpenRouter orchestrator compatible with LangChain 1.3"""

    def __init__(self):
        pass

    def run(self, policy_json):

        # 1️⃣ Load documents for RAG
        documents, metadatas = load_documents()
        context = ""

        if documents:
            collection = build_vector_db(documents, metadatas)

            results = collection.query(
                query_texts=[json.dumps(policy_json)],
                n_results=5,
                include=["documents"]
            )

            if results and results.get("documents"):
                context = "\n".join(results["documents"][0][:3])

        # 2️⃣ Agent decision
        action = agent_planner(context, json.dumps(policy_json))

        # 3️⃣ LLM prompt
        prompt = f"""
Policy:
{json.dumps(policy_json)}

Context:
{context}

Analyze the policy and return JSON only.

Required JSON format:

{{
  "Risk": "<LOW | MEDIUM | HIGH>",
  "Recommendation": "<improvement recommendations>",
  "Data_Requirements": [
        "requirement 1",
        "requirement 2",
        "requirement 3"
  ]
}}

Do NOT return text outside JSON.
"""

        # 4️⃣ Call OpenRouter
        analysis_raw = query_llm(prompt)
        # Remove markdown code fences if the LLM adds them
        clean_output = analysis_raw.replace("```json", "").replace("```", "").strip()

        # 5️⃣ Try parsing JSON
        try:
            analysis = json.loads(clean_output)
        except Exception:
            analysis = {
                "Risk": "UNKNOWN",
                "Recommendation": clean_output,
                "Data_Requirements": []
            }

        return {
            "action": action,
            "analysis": analysis
        }