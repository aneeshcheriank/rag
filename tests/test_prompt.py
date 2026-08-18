from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage, AIMessage

from src.prompt import rag_prompt


class TestRagPrompt:
    def test_is_chat_prompt_template(self):
        assert isinstance(rag_prompt, ChatPromptTemplate)

    def test_has_three_message_components(self):
        assert len(rag_prompt.messages) == 3

    def test_first_message_is_system(self):
        assert isinstance(rag_prompt.messages[0], SystemMessagePromptTemplate)

    def test_second_message_is_chat_history_placeholder(self):
        placeholder = rag_prompt.messages[1]
        assert isinstance(placeholder, MessagesPlaceholder)
        assert placeholder.variable_name == "chat_history"

    def test_third_message_is_human(self):
        assert isinstance(rag_prompt.messages[2], HumanMessagePromptTemplate)

    def test_system_message_contains_context_variable(self):
        system_message = rag_prompt.messages[0]
        assert "{context}" in system_message.prompt.template

    def test_human_message_contains_question_variable(self):
        human_message = rag_prompt.messages[2]
        assert "{question}" in human_message.prompt.template

    def test_system_message_instructs_dont_know_fallback(self):
        system_message = rag_prompt.messages[0]
        template = system_message.prompt.template
        assert "don't know" in template

    def test_formats_with_all_variables(self):
        messages = rag_prompt.format_messages(
            context="Apple's revenue was $416B.",
            question="What is Apple's revenue?",
            chat_history=[HumanMessage(content="hi"), AIMessage(content="hello")],
        )

        assert len(messages) == 4  # system + 2 history + human
        assert messages[0].type == "system"
        assert "Apple's revenue was $416B." in messages[0].content
        assert messages[3].type == "human"
        assert messages[3].content == "What is Apple's revenue?"

    def test_input_variables_are_context_question_chat_history(self):
        # ``chat_history`` is declared via MessagesPlaceholder, the other two
        # come from the template strings.
        assert "context" in rag_prompt.input_variables
        assert "question" in rag_prompt.input_variables
        assert "chat_history" in rag_prompt.input_variables
