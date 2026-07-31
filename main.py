from src.pipeline import rag


import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"
)


if __name__ == "__main__":
    question = "what is the threats the company has?"
    response = rag(question)

    print(response)


