# /app/llm/openrouter_client.py
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import OPENROUTER_API_KEY, MISTRAL_MODEL_ID


def build_openrouter_llm():
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set")
    return ChatOpenRouter(
        model=MISTRAL_MODEL_ID,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        max_tokens=400
    )


def query_llm(prompt: str) -> str:
    llm = build_openrouter_llm()
    messages = [
        SystemMessage(content="You are a compliance governance AI assistant."),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)
    return response.content