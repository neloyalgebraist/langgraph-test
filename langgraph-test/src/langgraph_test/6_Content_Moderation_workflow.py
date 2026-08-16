import json

from langgraph.graph import StateGraph, START, END
from typing import Literal, TypedDict


class ModerationState(TypedDict):
    post_content: str
    user_reputation: str

    formatted_post: str
    content_flag: str
    result: str


def format_post(state: ModerationState):
    formatted = f"User ({state['user_reputation']}) says: {state['post_content']}"
    return {"formatted_post": formatted}


def analyze_content(state: ModerationState):
    content = state["post_content"].lower()

    if "spam" in content or "buy now" in content:
        flag = "rejected"
    elif state["user_reputation"] == "new_user":
        flag = "review"
    else:
        flag = "approved"

    return {"content_flag": flag}


def approve_post(state: ModerationState):
    result = "Post published successfully to the timeline."
    return {"result": result}


def flag_for_review(state: ModerationState):
    result = "Post sent to the human moderation queue."
    return {"result": result}


def reject_post(state: ModerationState):
    result = "Post automatically deleted due to policy violation."
    return {"result": result}


def render_report(state: ModerationState) -> str:
    return json.dumps(state, indent=2, ensure_ascii=False)


def check_condition(
    state: ModerationState,
) -> Literal["approve_post", "flag_for_review", "reject_post"]:
    if state["content_flag"] == "approved":
        return "approve_post"
    elif state["content_flag"] == "review":
        return "flag_for_review"
    else:
        return "reject_post"


graph = StateGraph(ModerationState)

graph.add_node("format_post", format_post)
graph.add_node("analyze_content", analyze_content)
graph.add_node("approve_post", approve_post)
graph.add_node("flag_for_review", flag_for_review)
graph.add_node("reject_post", reject_post)

graph.add_edge(START, "format_post")
graph.add_edge("format_post", "analyze_content")

graph.add_conditional_edges("analyze_content", check_condition)

graph.add_edge("approve_post", END)
graph.add_edge("flag_for_review", END)
graph.add_edge("reject_post", END)

workflow = graph.compile()

initial_state = {
    "post_content": "Check out this amazing new product! Buy now!",
    "user_reputation": "trusted_user",
}

result = workflow.invoke(initial_state)
print(render_report(result))
