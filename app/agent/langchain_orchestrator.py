# /app/agent/langchain_orchestrator.py
import json
from typing import List
from pydantic import BaseModel
from app.llm.openrouter_client import query_llm
from app.rag.pipeline import load_documents
from app.rag.retriever import build_vector_db

# Persistent memory (replaceable with DB)
MEMORY_STORE = []


# Pydantic models for structured LLM output
class ControlScope(BaseModel):
    Department: str = ""
    Department_Role: str = ""
    Department_Role_Scope: str = ""
    Document_used: str = ""


class ControlItem(BaseModel):
    Control_Object: str
    Control_Type: str
    Purpose: str
    Scope: ControlScope


class StepResult(BaseModel):
    Risk: str = "UNKNOWN"
    Controls: List[ControlItem]
    Recommendation: str = ""
    Data_Requirements: List[str] = []


class PolicyGoalPlan(BaseModel):
    goal: str
    steps: List[StepResult]


class TrueAgenticComplianceOrchestrator:
    """Agentic compliance AI orchestrator with structured risk assessment and control tracking."""

    def __init__(self):
        self.memory = MEMORY_STORE

    def run(self, policy_json: dict):
        # 1️⃣ Gather context from RAG + memory
        documents, metadatas = load_documents()
        context = ""
        if documents:
            collection = build_vector_db(documents, metadatas)
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

        # 2️⃣ Generate goals
        goals = self.generate_goals(full_context, policy_json)

        # 3️⃣ Plan multi-step actions
        plan: List[PolicyGoalPlan] = []
        for goal in goals:
            steps = self.plan_steps(goal, full_context)
            plan.append(PolicyGoalPlan(goal=goal, steps=steps))

        # 4️⃣ Store results and calculate policy risk
        self.update_memory(policy_json, plan)
        policy_risk = self.assess_policy_risk(plan)

        # 5️⃣ Return full structured JSON
        return {
            "policy": policy_json,
            "plan": [p.dict() for p in plan],
            "policy_risk": policy_risk
        }

    # ------------------- LLM Helpers -------------------
    def generate_goals(self, context: str, policy_json: dict) -> List[str]:
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

    def plan_steps(self, goal: str, context: str) -> List[StepResult]:
        prompt = f"""
You are a compliance planner agent.

Context: {context}
Goal: {goal}

Generate 2-4 step-by-step actions to achieve this goal.
Return **JSON strictly matching this schema**:

{StepResult.schema_json(indent=2)}
"""
        res = query_llm(prompt)
        return self.parse_steps(res, goal)

    @staticmethod
    def parse_steps(llm_output: str, goal: str) -> List[StepResult]:
        clean_output = llm_output.replace("```json", "").replace("```", "").strip()
        try:
            steps_data = json.loads(clean_output)
            step_results = []
            for step in steps_data:
                # Ensure controls are properly structured
                controls = step.get("Controls", [])
                structured_controls = []
                for c in controls:
                    if isinstance(c, str):
                        structured_controls.append(ControlItem(
                            Control_Object=c,
                            Control_Type="",
                            Purpose=c,
                            Scope=ControlScope()
                        ))
                    elif isinstance(c, dict):
                        structured_controls.append(ControlItem(
                            Control_Object=c.get("Control_Object", ""),
                            Control_Type=c.get("Control_Type", ""),
                            Purpose=c.get("Purpose", ""),
                            Scope=ControlScope(**c.get("Scope", {}))
                        ))
                step["Controls"] = structured_controls
                step_results.append(StepResult(**step))
            return step_results
        except Exception:
            # fallback placeholder
            placeholder = StepResult(
                Risk="UNKNOWN",
                Controls=[ControlItem(
                    Control_Object=f"{goal} Control",
                    Control_Type="Preventative",
                    Purpose=f"Mitigate risks related to {goal}",
                    Scope=ControlScope(
                        Department="Relevant Department",
                        Department_Role="Responsible Role",
                        Department_Role_Scope="Scope of Responsibility",
                        Document_used="Relevant Policy/Procedure Document"
                    )
                )],
                Recommendation=f"Analyze and implement measures for {goal}",
                Data_Requirements=["Evidence or records to validate controls"]
            )
            return [placeholder]

    # ------------------- Memory -------------------
    def update_memory(self, policy_json: dict, plan: List[PolicyGoalPlan]):
        summary = f"Policy {policy_json.get('policy_name', 'Unnamed')} processed with {len(plan)} goals."
        self.memory.append({"policy": policy_json, "results": [p.dict() for p in plan], "summary": summary})

    # ------------------- Risk Assessment -------------------
    @staticmethod
    def assess_policy_risk(plan: List[PolicyGoalPlan]) -> str:
        risk_weights = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNKNOWN": 2}
        total_score, count = 0, 0
        for p in plan:
            for step in p.steps:
                r = step.Risk or "UNKNOWN"
                total_score += risk_weights.get(r, 2)
                count += 1
        avg_score = total_score / max(count, 1)
        if avg_score >= 2.5:
            return "HIGH"
        elif avg_score >= 1.5:
            return "MEDIUM"
        else:
            return "LOW"