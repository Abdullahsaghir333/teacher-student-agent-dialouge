# agents/evaluator.py --------------------------------------------
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


class EvaluationAgent:

    def __init__(self):
        print("📊 EvaluationAgent initialized")

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))

        self.prompt = PromptTemplate(
            input_variables=["transcript"],
            template="""
Evaluate this teacher–student session:

{transcript}

Rate:
- Clarity (1–10)
- Student Relevance (1–10)
- Overall Quality (1–10)

Return ONLY JSON:
{"clarity":7, "relevance":8, "overall":7}
"""
        )

        self.chain = self.prompt | self.llm | JsonOutputParser()

    def evaluate(self, history):
        text = "\n".join([f"{h['role']}: {h['content']}" for h in history])
        return self.chain.invoke({"transcript": text})
