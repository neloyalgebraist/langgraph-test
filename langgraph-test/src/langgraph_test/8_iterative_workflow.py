from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
import operator
from dotenv import load_dotenv

load_dotenv()

generator_llm = ChatGroq(model="llama-3.3-70b-versatile")
evaluator_llm = ChatGroq(model="llama-3.3-70b-versatile")
optimizer_llm = ChatGroq(model="llama-3.3-70b-versatile")


class PostEvaluation(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(
        ..., description="Final evaluation result."
    )
    feedback: str = Field(..., description="feedback for the Facebook post.")


structured_evaluator_llm = evaluator_llm.with_structured_output(PostEvaluation)


class PostState(TypedDict):
    topic: str
    post: str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iteration: int

    post_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]

def generate_post(state: PostState):
    messages = [
        SystemMessage(content="You are a funny and clever Facebook influencer."),
        HumanMessage(content=f"""
            Write a short, original, and hilarious Facebook post on the topic: "{state['topic']}".

            Rules:
            - Do Not use question-answer format.
            - Max 500 characters.
            - Use observational humor, irony, sarcasm, or cultural references.
            - Think in meme logic, punchlines, or relatable takes.
            - Use simple, day to day english
        """)
    ]
    response = generator_llm.invoke(messages).content
    return {'post': response, 'post_history': [response]}

def evaluate_post(state: PostState):
    messages = [
        SystemMessage(content="You are a ruthless, no-laugh-given Facebook critic. You evaluate posts based on humor, originality, virality and post format."),
        HumanMessage(content=f"""
            evaluate the following post:
            post: "{state['post']}"

            Use the criteria below to evaluate the post: 

            1. Originality - Is this fresh, or have you seen it a hundred times before?
            2. Humor - Did it genuinely make you smile, laugh, or chuckle?
            3. Punchiness - Is it short, sharp, and scroll-stopping?
            4. Virality Potential - Would people share, react, or comment on it?
            5. Format - Is it a well-formed Facebook post (not a setup-punchline joke, not a Q&A joke, and under 500 characters)?

            Auto-reject if:
            
            - It's written in question-answer format (e.g., "why did..." or "What happens when...")
            - It exceeds 500 characters
            - It reads like a traditional setup-punchline joke 
            - It ends with a generic, throwaway, or deflating line that weakens the humor (e.g., "Masterpieces of the auntie-uncle universe" or vague summaries)

            ### Respond ONLY in structured format:

            - evaluation: "approved" or "needs_improvement"
            - feedback: One paragraph explaining the strengths and weakness
        """

        )
    ]

    response = structured_evaluator_llm.invoke(messages)

    return {'evaluation': response.evaluation, 'feedback': response.feedback, 'feedback_history': [response.feedback]}

def optimize_post(state: PostState):

    messages = [
        SystemMessage(content="You punch up Facebook posts for virality and humor based on given feedback."),
        HumanMessage(content=f"""
            Improve the Facebook post based on this feedback:
            "{state['feedback']}"
            Topic: "{state['topic']}"
            Original post:
            {state['post']}

            Re-write it as a short, viral-worthy Facebook post. Avoid Q&A style and stay under 500 characters.
        """

        )
    ]
    response = optimizer_llm.invoke(messages).content
    iteration = state['iteration'] + 1

    return {'post': response, 'iteration': iteration, 'post_history': [response]}

def route_evaluation(state: PostState) -> Literal["approved", "exhausted", "needs_improvement"]:
    if state['evaluation'] == 'approved':
        return 'approved'
    if state['iteration'] >= state['max_iteration']:
        return 'exhausted'
    return 'needs_improvement'


graph = StateGraph(PostState)

graph.add_node('generate', generate_post)
graph.add_node('evaluate', evaluate_post)
graph.add_node('optimize', optimize_post)

graph.add_edge(START, 'generate')
graph.add_edge('generate', 'evaluate')
graph.add_conditional_edges('evaluate', route_evaluation, {'approved': END, 'exhausted': END, 'needs_improvement': 'optimize'})

graph.add_edge('optimize', 'evaluate')

workflow = graph.compile()

initial_state = {
    "topic": "agentic AI",
    "iteration": 1,
    "max_iteration": 5
}

result = workflow.invoke(initial_state)

print(result)


