# ============================================================
# STREAMLIT UI FOR AI CLASSROOM + RAG + QUIZ + FEEDBACK + SUMMARY
# ============================================================

import os
import streamlit as st
import time
from dotenv import load_dotenv
load_dotenv()

# Backend Imports
from rag.loader import load_rag_index
from rag.retriever import RAGRetriever
from agents.student import StudentAgent
from agents.teacher import TeacherAgent
from graph.classroom_graph import build_classroom_graph

from langchain_openai import ChatOpenAI


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "quiz_raw" not in st.session_state:
    st.session_state.quiz_raw = None

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "state" not in st.session_state:
    st.session_state.state = None


# ------------------------------------------------------------
# QUIZ GENERATOR
# ------------------------------------------------------------
def generate_quiz(topic, transcript):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    prompt = f"""
    Create a quiz of EXACTLY 5 MCQs based on this topic: {topic}.

    SOURCE TRANSCRIPT:
    {transcript}

    FORMAT STRICTLY LIKE THIS:
    Q1: <question>
    A) option
    B) option
    C) option
    D) option
    Answer: <B|C|A|D>
    """

    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content


# ------------------------------------------------------------
# FEEDBACK AGENT
# ------------------------------------------------------------
def evaluate_teacher_response(question, rag_answer, nonrag_answer):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    prompt = f"""
    Evaluate two teacher responses...
    """

    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content


# ------------------------------------------------------------
# SUMMARY GENERATOR
# ------------------------------------------------------------
def generate_summary(transcript, topic):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    prompt = f"""
    Create a structured summary for topic: {topic}

    TRANSCRIPT:
    {transcript}
    """

    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="AI Classroom", page_icon="🎓", layout="wide")
st.title("🎓 AI Classroom: RAG vs Non-RAG + Quiz + Summary + Evaluation")

st.write("This app simulates an AI teacher using RAG and compares it with a normal teacher.")


# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
topic = st.sidebar.text_input("Enter Topic:", "Quantum Computing")
turns = st.sidebar.slider("Number of Dialogue Turns", 1, 10, 3)

run_button = st.sidebar.button("Start Lesson")


# ============================================================
# DIALOGUE SESSION EXECUTION (START LESSON)
# ============================================================
if run_button:

    # Reset conversation for new session
    st.session_state.conversation = []
    st.session_state.quiz_raw = None
    st.session_state.summary = ""

    st.subheader(f"📘 Topic: **{topic}**")

    # Load RAG index
    with st.spinner("Loading RAG index..."):
        vectorstore = load_rag_index()
        rag = RAGRetriever(vectorstore)

    student = StudentAgent()

    rag_teacher = TeacherAgent()
    rag_teacher.rag_enabled = True

    plain_teacher = TeacherAgent()
    plain_teacher.rag_enabled = False

    graph = build_classroom_graph(student, rag_teacher, rag)

    state = {
        "topic": topic,
        "last_teacher_reply": "I am ready to teach.",
        "last_student_question": "",
        "conversation": []
    }

    st.info("Classroom session started...")

    # ============================================================
    # RUN CONVERSATION LOOP
    # ============================================================
    for t in range(turns):
        state = graph.invoke(state)

        role, msg = state["conversation"][-1]

        # SAVE TO SESSION STATE (PERSISTENT)
        st.session_state.conversation.append((role, msg))

        if role == "Student":
            st.chat_message("user").write(msg)
        else:
            st.chat_message("assistant").write(msg)

        time.sleep(0.6)

    # Build transcript
    transcript = ""
    for role, msg in st.session_state.conversation:
        transcript += f"{role}: {msg}\n"

    st.session_state.transcript = transcript


# ============================================================
# SHOW FULL TRANSCRIPT (PERSISTENT)
# ============================================================
if st.session_state.conversation:
    st.subheader("📝 Full Transcript")

    for role, msg in st.session_state.conversation:
        st.write(f"**{role}:** {msg}")


# ============================================================
# SUMMARY SECTION
# ============================================================
st.subheader("📘 Lesson Summary")

if st.button("Generate Summary"):
    with st.spinner("Generating summary..."):
        summary = generate_summary(st.session_state.transcript, topic)

    st.session_state.summary = summary

if st.session_state.summary:
    st.text_area("Lesson Summary", st.session_state.summary, height=300)


# ============================================================
# QUIZ SECTION
# ============================================================
st.subheader("🧠 Quiz Time!")

if st.button("Generate Quiz"):
    with st.spinner("Generating quiz..."):
        quiz = generate_quiz(topic, st.session_state.transcript)

    st.session_state.quiz_raw = quiz

if st.session_state.quiz_raw:
    st.text_area("Generated Quiz", st.session_state.quiz_raw, height=260)

    # Parse quiz
    lines = st.session_state.quiz_raw.split("\n")
    questions = []
    q = {}

    for line in lines:
        if line.startswith("Q"):
            q = {"question": line, "options": []}
        elif line.startswith(("A)", "B)", "C)", "D)")):
            q["options"].append(line)
        elif line.startswith("Answer"):
            q["answer"] = line.split(":")[1].strip()
            questions.append(q)

    st.subheader("📝 Take the Quiz")

    user_answers = []

    for i, q in enumerate(questions):
        st.write(f"**{q['question']}**")
        choice = st.radio("", q["options"], key=f"quiz{i}")
        user_answers.append(choice[0])

    if st.button("Submit Answers"):
        score = sum(
            user_answers[i] == questions[i]["answer"]
            for i in range(len(questions))
        )
        st.success(f"🎉 Your Score: **{score}/5**")


# ============================================================
# EVALUATION SECTION
# ============================================================
st.subheader("💬 Evaluation")

if st.button("Evaluate Answers"):
    with st.spinner("Evaluating answers..."):
        evaluation = evaluate_teacher_response(st.session_state.transcript, st.session_state.quiz_raw, st.session_state.user_answers)

    st.session_state.evaluation = evaluation

if st.session_state.evaluation:
    st.text_area("Evaluation", st.session_state.evaluation, height=200)

# ============================================================
# END OF FILE
# ============================================================