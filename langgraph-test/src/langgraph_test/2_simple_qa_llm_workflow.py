from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")


class LLMState(TypedDict):
    question: str
    answer: str


def llm_qa(state: LLMState) -> LLMState:
    question = state["question"]
    prompt = f"Answer the following question {question}"
    answer = model.invoke(prompt).content
    state["answer"] = answer
    return state


graph = StateGraph(LLMState)

graph.add_node("llm_qa", llm_qa)

graph.add_edge(START, "llm_qa")
graph.add_edge("llm_qa", END)

workflow = graph.compile()

initial_state = {"question": "Who is the creator of Python?"}
final_state = workflow.invoke(initial_state)
print(final_state["answer"])
