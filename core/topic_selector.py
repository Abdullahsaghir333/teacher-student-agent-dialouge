# core/topic_selector.py ----------------------------------------

class TopicSelector:
    def __init__(self):
        self.topics = [
            "Photosynthesis",
            "Newton's Laws",
            "Python Loops",
            "The Water Cycle"
        ]

    def select(self):
        print("Topics:")
        for i, t in enumerate(self.topics):
            print(f"{i+1}. {t}")
        idx = int(input("Choose topic: ")) - 1
        return self.topics[idx]
