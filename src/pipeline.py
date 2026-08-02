from langchain_core.output_parsers import StrOutputParser

from src.retriver import get_vectorstore
from src.model import get_llm
from src.prompt import rag_prompt


def format_docs(docs):
    """
    combined retrieved docuemnts into a single text
    """
    return "\n\n".join(doc.page_content for doc in docs)


def rag(question, chat_history=[]):
    llm = get_llm()
    vectorstore = get_vectorstore(k=4)
    prompt = rag_prompt

    chain = prompt | llm | StrOutputParser()
    context = vectorstore.invoke(question)
    formated_context = format_docs(context)

    response = chain.invoke(
        {
            "question": question,
            "chat_history": chat_history,
            "context": formated_context,
        }
    )

    return {
        "response": response,
        "context": context,
    }
