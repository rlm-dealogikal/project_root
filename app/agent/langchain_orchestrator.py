# /app/agent/langchain_orchestrator.py
import json
from app.llm.openrouter_client import query_llm
from app.rag.pipeline import load_documents
from app.rag.retriever import build_vector_db

# Persistent memory (could be replaced with a DB)
MEMORY_STORE = []


class TrueAgenticComplianceOrchestrator:
    """A truly agentic compliance AI orchestrator with dynamic planning and memory."""

    def __init__(self):
        self.memory = MEMORY_STORE

    def run(self, policy_json):
        # 1️⃣ Gather context from RAG + memory
        documents, metadatas = load_documents()
        context = ""
        if documents:
            collection = build_vector_db(documents, metadatas)
            ###################################################print 
            print("Querying RAG with policy context:\n", json.dumps(policy_json, indent=2))
            for meta in metadatas:
                print(f"Metadata: {meta}")  
            ###################################################
            results = collection.query(
                query_texts=[json.dumps(policy_json)],
                n_results=5,
                include=["documents"]
            )
            if results.get("documents") and len(results["documents"]) > 0:
                context_docs = results["documents"][0][:3]
                context = "\n".join(context_docs)

        memory_context = "\n".join([m["summary"] for m in self.memory])
        full_context = f"Memory:\n{memory_context}\nRAG Context:\n{context}"
        print("Full Context for Agent:\n", full_context)

        # 2️⃣ Generate dynamic goals from policy
        goals = self.generate_goals(full_context, policy_json)

        # 3️⃣ Plan multi-step actions for each goal
        plan = []
        for goal in goals:
            steps = self.plan_steps(goal, full_context)
            plan.append({"goal": goal, "steps": steps})

        # 4️⃣ Execute plan and collect results
        results = []
        for item in plan:
            goal = item["goal"]
            steps = item["steps"]
            goal_results = []
            for step_prompt in steps:
                raw = query_llm(step_prompt)
                clean_output = raw.replace("```json", "").replace("```", "").strip()
                try:
                    goal_results.append(json.loads(clean_output))
                except Exception:
                    goal_results.append({"Result": clean_output})
            results.append({"goal": goal, "results": goal_results})

        # 5️⃣ Store results in memory for future runs
        self.update_memory(policy_json, results)

        return {"policy": policy_json, "plan": plan, "results": results}

    def generate_goals(self, context, policy_json):
        prompt = f"""
        You are a compliance reasoning agent.

        Context: {context}
        Policy: {json.dumps(policy_json)}

        Generate a list of concrete compliance goals or sub-goals.
        Output JSON array of strings only.

        Rules:
        - No markdown
        - No explanation
        - No text before or after JSON
        - Output must be a JSON array of steps

        Example:
        ["step1","step2","step3"]
        """
        res = query_llm(prompt)
        clean = res.replace("```json", "").replace("```", "").strip()
        try:
            goals = json.loads(clean)
            if isinstance(goals, list):
                return goals
        except:
            pass
        # fallback
        return ["ANALYZE policy compliance"]

    def plan_steps(self, goal, context):
        prompt = f"""
        You are a compliance planner agent.

        Context: {context}
        Goal: {goal}

        Generate 2-4 step-by-step actions to achieve this goal.
        Return JSON array of string steps.
        """
        res = query_llm(prompt)
        clean = res.replace("```json", "").replace("```", "").strip()
        try:
            steps = json.loads(clean)
            if isinstance(steps, list):
                return steps
        except:
            return [f"Analyze and provide recommendations for: {goal}"]
        return steps

    def update_memory(self, policy_json, results):
        """Store summary of actions for future context."""
        summary = f"Policy {policy_json.get('policy_name', 'Unnamed')} processed with {len(results)} goals."
        self.memory.append({"policy": policy_json, "results": results, "summary": summary})