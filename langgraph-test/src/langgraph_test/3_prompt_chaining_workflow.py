from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

# Define Model
model = ChatGroq(model="llama-3.3-70b-versatile")


# Define State
class BlogState(TypedDict):
    title: str
    outline: str
    content: str


# Creating Nodes
def create_outline(state: BlogState) -> BlogState:
    title = state["title"]
    prompt = f"Generate a detailed outline for a blog on the topic - {title}"
    outline = model.invoke(prompt).content

    state["outline"] = outline

    return state


def create_blog(state: BlogState) -> BlogState:
    title = state["title"]
    outline = state["outline"]
    prompt = f"Write a detailed blog on the title - {title} using the following outline - {outline}"
    content = model.invoke(prompt).content

    state["content"] = content

    return state


# Define and Compile Graph
graph = StateGraph(BlogState)

# Nodes
graph.add_node("create_outline", create_outline)
graph.add_node("create_blog", create_blog)

# Edges
graph.add_edge(START, "create_outline")
graph.add_edge("create_outline", "create_blog")
graph.add_edge("create_blog", END)

workflow = graph.compile()

# Execute the Graph
initial_state = {"title": "Rise of AI in India"}
final_state = workflow.invoke(initial_state)
print(final_state)

print(final_state["outline"])

print(final_state["content"])
