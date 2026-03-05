# /app/agent/planner.py
import json
from app.llm.openrouter_client import query_llm

def agent_planner(context, question):

    prompt = f"""
    You are a compliance reasoning agent.

    Context: {context}
    Question: {question}

    Output JSON only:
    {{
        "action": "ANALYZE | MITIGATE | ANSWER"
    }}
    """

    res = query_llm(prompt)

    try:
        res = json.loads(res)
        return res["action"]
    except:
        return "ANSWER"