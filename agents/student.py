# agents/student.py

import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class StudentAgent:

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=OPENAI_API_KEY
        )

        self.prompt = PromptTemplate(
            input_variables=["topic", "teacher_reply"],
            template="""
            You are a curious student learning about: {topic}.
            The teacher said: "{teacher_reply}".

            Ask ONE short follow-up question. No greetings. No thanks.
            """
        )

        self.chain = self.prompt | self.llm | StrOutputParser()

    def ask(self, topic, teacher_reply):
        return self.chain.invoke({
            "topic": topic,
            "teacher_reply": teacher_reply
        })
