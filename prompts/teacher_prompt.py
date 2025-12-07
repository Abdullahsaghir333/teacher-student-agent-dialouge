from langchain_core.prompts import PromptTemplate

teacher_prompt = PromptTemplate(
    input_variables=["topic", "student_question", "context"],
    template="""
You are an expert teacher explaining: {topic}

Context retrieved from textbooks:
{context}

Student asked: {student_question}

Give a clear, structured, and correct educational answer using the context.
"""
)
