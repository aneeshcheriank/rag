from langchain_core.messages import HumanMessage, AIMessage
from src.pipeline import rag


import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"
)


if __name__ == "__main__":

    chat_history = []

    while True:
        question = input("Ask your question\n")
        print("===============================")
        if question == "exit":
            break

        response = rag(question, chat_history)
        print(response)
        print("===============================")

        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=response))

        if len(chat_history) >= 10:
            chat_history = chat_history[:10]
