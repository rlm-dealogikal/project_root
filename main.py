# /app/main.py
import json
import requests

from app.rag.pipeline import load_documents
from app.rag.retriever import build_vector_db
from app.llm.openrouter_client import query_llm
from app.agent.planner import agent_planner
from config.settings import USER_INPUT_API_URL


# ============================
# SAFE API INPUT LOADER
# ============================

def load_user_input_from_api():

    if not USER_INPUT_API_URL:
        print("USER_INPUT_API_URL missing in .env")
        return []

    try:
        response = requests.get(
            USER_INPUT_API_URL,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        if isinstance(data, dict):
            return [data]

        return data

    except Exception as e:
        print("API Input Error:", e)
        return []


# ============================
# POLICY PROCESSOR
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

    if results and len(results["documents"]) > 0:
        context = "\n".join(results["documents"][0][:3])

    action = agent_planner(context, policy_text)

    print("Agent Action:", action)

    prompt = f"""
    Policy: {policy_text}
    Context: {context}
    """

    return query_llm(prompt)


# ============================
# MAIN ENTRY
# ============================

if __name__ == "__main__":

    inputs = load_user_input_from_api()

    if not inputs:
        print("No policy input received from API")
        exit()

    for policy in inputs:
        result = process_policy(policy)
        print("--------------->",json.dumps(result, indent=2))