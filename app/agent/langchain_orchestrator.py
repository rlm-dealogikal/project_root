# /app/agent/langchain_orchestrator.py
import json
#from bleach import clean
#from typer import prompt
from app.llm.openrouter_client import query_llm
from app.rag.pipeline import load_documents
from app.rag.retriever import build_vector_db

# Persistent memory (can be replaced with a DB)
MEMORY_STORE = []


class TrueAgenticComplianceOrchestrator:
    """Agentic compliance AI orchestrator with structured risk assessment and control tracking."""

    def __init__(self):
        self.memory = MEMORY_STORE

    def run(self, policy_json):
        # 1️⃣ Gather context from RAG + memory
        documents, metadatas = load_documents()
        context = ""
        if documents:
            collection = build_vector_db(documents, metadatas)
            print("Querying RAG with policy context:\n", json.dumps(policy_json, indent=2))
            for meta in metadatas:
                print(f"Metadata: {meta}")
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

        # 4️⃣ Execute plan and collect structured results
        results = []
        for item in plan:
            goal = item["goal"]
            steps = item["steps"]
            goal_results = []
            for step_prompt in steps:
                prompt_text = step_prompt if isinstance(step_prompt, str) else json.dumps(step_prompt)
                raw = query_llm(prompt_text)
                clean_output = raw.replace("```json", "").replace("```", "").strip()
                # Try to parse structured JSON; fallback to raw
                try:
                    parsed = json.loads(clean_output)
                    # Ensure required keys exist
                    goal_results.append({
                        "Risk": parsed.get("Risk", "UNKNOWN"),
                        "Controls": parsed.get("Controls", []),
                        "Recommendation": parsed.get("Recommendation", ""),
                        "Data_Requirements": parsed.get("Data_Requirements", [])
                    })
                except Exception:
                    goal_results.append({
                        "Risk": "UNKNOWN",
                        "Controls": [],
                        "Recommendation": clean_output,
                        "Data_Requirements": []
                    })
            results.append({"goal": goal, "results": goal_results})

        # 5️⃣ Store results in memory for future runs
        self.update_memory(policy_json, results)
        policy_risk = self.assess_policy_risk(results)

        # 6️⃣ Return full structured JSON
        return {"policy": policy_json, "plan": plan, "results": results, "policy_risk": policy_risk}

    def generate_goals(self, context, policy_json):
        prompt = f"""
        You are a compliance reasoning agent.

        Context: {context}
        Policy: {json.dumps(policy_json)}

        Generate a list of concrete compliance goals or sub-goals.
        Output JSON array of strings only.

        Rules:
        - No markdown
        - No text before or after JSON
        Example: ["goal1","goal2"]
        """
        res = query_llm(prompt)
        clean = res.replace("```json", "").replace("```", "").strip()
        try:
            goals = json.loads(clean)
            if isinstance(goals, list):
                return goals
        except:
            pass
        return ["ANALYZE policy compliance"]

    def plan_steps(self, goal, context):
        prompt = f"""
You are a compliance planner agent.

Context: {context}
Goal: {goal}

Generate 2-4 step-by-step actions to achieve this goal.
Each step must output structured JSON with keys:
- Risk (LOW/MEDIUM/HIGH/UNKNOWN)
- Controls (array of control objects with fields:
    Control Object,
    Control Type (Preventative, Detective, Corrective),
    Purpose,
    Scope: {{
        Department,
        Department_Role,
        Department_Role_Scope,
        Document used
    }}
  )
- Recommendation (action to mitigate risk)
- Data_Requirements (evidence or data needed to validate controls)

Return a JSON array of these structured objects.
"""
        res = query_llm(prompt)
        clean = res.replace("```json", "").replace("```", "").strip()
        try:
            steps = json.loads(clean)
            if isinstance(steps, list):
                # Ensure every control is structured
                for step in steps:
                    controls = step.get("Controls", [])
                    structured_controls = []
                    for c in controls:
                        if isinstance(c, str):
                            structured_controls.append({
                                "Control Object": c,
                                "Control Type": "",
                                "Purpose": c,
                                "Scope": {
                                    "Department": "",
                                    "Department_Role": "",
                                    "Department_Role_Scope": "",
                                    "Document used": ""
                                    }
                                })

                        elif isinstance(c, dict):
                            structured_controls.append(c)
                    step["Controls"] = structured_controls
                return steps
        except Exception:
            # Fallback: produce meaningful placeholder controls instead of copying the goal
            placeholder_control = {
                "Control Object": f"{goal} Control",
                "Control Type": "Preventative",
                "Purpose": f"Mitigate risks related to {goal}",
                "Scope": {
                    "Department": "Relevant Department",
                    "Department_Role": "Responsible Role",
                    "Department_Role_Scope": "Scope of Responsibility",
                    "Document used": "Relevant Policy/Procedure Document"
                    }
                }
            return [{
                "Risk": "UNKNOWN",
                "Controls": [placeholder_control],
                "Recommendation": f"Analyze and implement measures for {goal}",
                "Data_Requirements": ["Evidence or records to validate controls"]
            }]
        
    def update_memory(self, policy_json, results):
        """Store summary of actions for future context."""
        summary = f"Policy {policy_json.get('policy_name', 'Unnamed')} processed with {len(results)} goals."
        self.memory.append({"policy": policy_json, "results": results, "summary": summary})

    # Add this method inside TrueAgenticComplianceOrchestrator
    def assess_policy_risk(self, results):
        """Aggregate step-level risks into an overall policy risk."""
        risk_weights = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNKNOWN": 2}
        total_score = 0
        count = 0
        for item in results:
            for step in item["results"]:
                r = step.get("Risk", "UNKNOWN")
                total_score += risk_weights.get(r, 2)
                count += 1
        avg_score = total_score / max(count, 1)
        if avg_score >= 2.5:
            return "HIGH"
        elif avg_score >= 1.5:
            return "MEDIUM"
        else:
            return "LOW"  