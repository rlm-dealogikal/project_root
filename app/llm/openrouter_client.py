from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import OPENROUTER_API_KEY, MISTRAL_MODEL_ID


def build_openrouter_llm():
    """
    Build a LangChain OpenRouter LLM object for your Mistral model.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set in config.settings")

    return ChatOpenRouter(
        model=MISTRAL_MODEL_ID,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        max_tokens=400
    )


def query_llm(prompt: str):
    """
    Query OpenRouter using LangChain messages.
    """
    llm = build_openrouter_llm()

    messages = [
        SystemMessage(content="You are a compliance governance AI assistant."),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)

    return response.content