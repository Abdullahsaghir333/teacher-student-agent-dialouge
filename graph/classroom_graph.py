# graph/classroom_graph.py

import operator
from typing import TypedDict, List, Annotated

from langgraph.graph import StateGraph, END

from agents.student import StudentAgent
from agents.teacher import TeacherAgent
from rag.retriever import RAGRetriever


# STATE STRUCTURE
class ClassroomState(TypedDict):
    topic: str
    last_teacher_reply: str
    last_student_question: str
    conversation: Annotated[List[tuple], operator.add]


# STUDENT NODE
def student_node(state: ClassroomState, student: StudentAgent):
    topic = state["topic"]
    last_reply = state["last_teacher_reply"]

    question = student.ask(topic, last_reply)

    return {
        "last_student_question": question,
        "conversation": [("Student", question)]
    }


# TEACHER NODE
def teacher_node(state: ClassroomState, teacher: TeacherAgent, rag: RAGRetriever):
    topic = state["topic"]
    question = state["last_student_question"]

    context = rag.retrieve(question)
    reply = teacher.answer(topic, question, context)

    return {
        "last_teacher_reply": reply,
        "conversation": [("Teacher", reply)]
    }


# BUILD GRAPH
def build_classroom_graph(student, teacher, rag):
    graph = StateGraph(ClassroomState)

    graph.add_node("student_turn", lambda s: student_node(s, student))
    graph.add_node("teacher_turn", lambda s: teacher_node(s, teacher, rag))

    graph.set_entry_point("student_turn")
    graph.add_edge("student_turn", "teacher_turn")
    graph.add_edge("teacher_turn", END)

    return graph.compile()
