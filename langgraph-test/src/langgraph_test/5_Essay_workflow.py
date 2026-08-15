from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
import operator

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")


class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay.")
    score: int = Field(description="Score out of 10", ge=0, le=10)


structured_model = model.with_structured_output(EvaluationSchema)


class EssayState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]
    avg_score: float


def evaluate_language(state: EssayState):
    prompt = f"Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10. \n\n essay - {state['essay']}"
    output = structured_model.invoke(prompt)

    return {"language_feedback": output.feedback, "individual_scores": [output.score]}


def evaluate_analysis(state: EssayState):
    prompt = f"Evaluate the depth of analysis of the following essay and provide a feedback and assign a score out of 10. \n\n essay - {state['essay']}"
    output = structured_model.invoke(prompt)

    return {"analysis_feedback": output.feedback, "individual_scores": [output.score]}


def evaluate_thought(state: EssayState):
    prompt = f"Evaluate the clarity of thought of the following essay and provide a feedback and assign a score out of 10. \n\n essay - {state['essay']}"
    output = structured_model.invoke(prompt)

    return {"clarity_feedback": output.feedback, "individual_scores": [output.score]}


def final_evaluation(state: EssayState):
    prompt = f"Based on the following feedbacks create a summarized feedback \n language feedback - {state['language_feedback']} \n depth of analysis feedback - {state['analysis_feedback']} \n clarity of thought feedback - {state['clarity_feedback']}"
    overall_feedback = model.invoke(prompt).content

    avg_score = sum(state["individual_scores"]) / len(state["individual_scores"])

    return {"overall_feedback": overall_feedback, "avg_score": avg_score}


graph = StateGraph(EssayState)

graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)

graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")

graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_thought", "final_evaluation")

graph.add_edge("final_evaluation", END)

workflow = graph.compile()

essay = """The Case for Keeping Standardized Tests in University Admissions

Over the past decade, more than eighteen hundred American colleges and universities have made the SAT and ACT optional, and several have dropped them entirely. Admissions deans describe this as a correction — an overdue acknowledgment that a four-hour exam cannot capture a teenager's potential. The reasoning is appealing, and it is wrong. Standardized tests should remain a required component of undergraduate admissions, because they are the only element of an application that measures every candidate against the same standard.

Consider what the rest of the application actually consists of. Essays are drafted with help from teachers, counselors, and increasingly from consultants who charge by the hour. Recommendation letters reflect how well a student is known by an overworked adult, which is largely a function of class size. Extracurricular records reward students whose families can absorb the cost of travel teams and unpaid internships. The test score is the one element of an application that cannot be purchased.

The predictive evidence supports this. A 2019 longitudinal study from the Weatherford Institute tracked 41,000 undergraduates across sixty institutions and found that test scores predicted first-year GPA with 73.4% accuracy, while high school grades managed only 51.2%. Grades, after all, are set by tens of thousands of independent teachers applying inconsistent standards; an A at one high school is a C at another. The test imposes a common scale on an otherwise incomparable set of records.

We can already observe what happens when that scale is removed. Institutions that dropped the test requirement saw applications rise by roughly twenty percent while the average first-year GPA on their campuses declined. Removing the testing requirement therefore lowers the academic quality of an incoming class. Administrators who champion test-optional policies are quietly aware of this, which is why most admissions professionals privately agree that the change was driven by rankings and application volume rather than by any finding about student success.

None of this denies that affluent students prepare for the exam under better conditions. They hire tutors, they sit the test repeatedly, and they arrive having seen the format a dozen times. But wealthy families exert exactly this kind of pressure on every other component of the application — the essay, the activity list, the letters, the campus visit. Singling out the test for criticism is arbitrary when the alternative channels of advantage are wider and harder to see.

What the critics are really proposing is an admissions process with no objective standard at all: a system in which a student's own account of their ability carries the same weight as a measured demonstration of it. Either we retain a common yardstick, or we accept that admissions becomes an exercise in narrative, decided by whoever writes the most affecting personal statement.

The human stakes are easiest to see in individual cases. Marcus attended a rural high school of four hundred students that offered two AP courses and had no college counselor. His transcript looked unremarkable beside those of applicants from well-resourced suburban schools. His 1480 was the single line in his file that made an admissions officer stop and read further, and he is now a third-year engineering student. Marcus's story is the rule, not the exception: for students without institutional advantages, the test is the mechanism of visibility, not the barrier to it.

The strongest objection to this position deserves acknowledgment. Test scores correlate with family income at roughly r = 0.4, a relationship that has proven stubborn across decades of revisions to the exam. That association may well reflect unmeasured differences in school quality rather than any property of the test itself, but the correlation is real and it is not small.

Ultimately the question is what we mean by merit. If merit is demonstrated academic preparation, the test measures it directly. If merit is raw intellectual capacity, the test approximates it as well as any instrument psychology has produced. And if merit is a matter of who deserves the seat, then the student who scored highly despite attending an underfunded school has a stronger claim than the student who was coached into the same score — a distinction the score itself makes visible.

Universities that have abandoned the requirement should restore it, and they should weight it as the primary factor in the academic portion of the review. An admissions system without a common measure is not more humane; it is merely less accountable, and its errors are harder to detect."""

initial_state = {"essay": essay}

result = workflow.invoke(initial_state)

print(result)
