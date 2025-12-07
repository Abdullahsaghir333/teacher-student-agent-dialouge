# core/dialogue_manager.py --------------------------------------

import time

class DialogueManager:

    def __init__(self):
        self.history = []

    def start(self, topic, teacher, student, turns=3):

        intro = f"Welcome! Today we explore {topic}. What would you like to know?"
        print(f"\nTeacher: {intro}")
        self.history.append({"role": "Teacher", "content": intro})

        last_answer = intro

        for t in range(turns):
            print(f"\n--- TURN {t+1} ---")

            question = student.ask(topic, last_answer)
            print(f"Student: {question}")
            self.history.append({"role": "Student", "content": question})

            answer = teacher.respond(topic, question)
            print(f"Teacher: {answer}")
            self.history.append({"role": "Teacher", "content": answer})

            last_answer = answer
            time.sleep(1)

        return self.history
