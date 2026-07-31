from langchain_core.prompts import ChatPromptTemplate

rag_prompt = ChatPromptTemplate(
    [
        (
            "system",
            """
        - You are an assistant for question-answering tasks. "
        - Use the following pieces of retrieved context to answer the question. "
        - If you do not know the answer, say that you don't know.\n\n"
        - Context:\n{context}
     """,
        ),
        ("human", "{question}"),
    ]
)
