from langchain_core.prompts import PromptTemplate

student_prompt = PromptTemplate(
    input_variables=["topic", "teacher_answer"],
    template="""
You are a curious student learning about {topic}.
The teacher said: "{teacher_answer}"

Ask ONE intelligent follow-up question. Keep it short.
"""
)
