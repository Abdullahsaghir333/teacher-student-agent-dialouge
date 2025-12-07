# agents/teacher.py

import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class TeacherAgent:

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=OPENAI_API_KEY
        )

        self.prompt = PromptTemplate(
            input_variables=["topic", "question", "context"],
            template="""
            You are an expert teacher on the topic: {topic}.

            TEXTBOOK CONTEXT:
            ------------------------
            {context}
            ------------------------

            The student asks: {question}

            Provide a clear, structured educational answer.
            """
        )

        self.chain = self.prompt | self.llm | StrOutputParser()

    def answer(self, topic, question, context):
        return self.chain.invoke({
            "topic": topic,
            "question": question,
            "context": context
        })
