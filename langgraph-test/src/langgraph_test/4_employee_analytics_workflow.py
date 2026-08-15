from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# State
class EmployeeState(TypedDict):
    employee_name: str
    monthly_salary: int
    working_days: int
    completed_projects: int

    yearly_salary: int
    bonus_amount: int
    project_status: str
    summary: str


# Node 1
def calculate_yearly_salary(state: EmployeeState):
    yearly_salary = state["monthly_salary"] * 12

    return {"yearly_salary": yearly_salary}


# Node 2
def calculate_bonus(state: EmployeeState):
    bonus_amount = state["monthly_salary"] * 2

    return {"bonus_amount": bonus_amount}


# Node 3
def project_evaluation(state: EmployeeState):
    if state["completed_projects"] >= 5:
        status = "excellent"
    else:
        status = "Average"

    return {"project_status": status}


# Node 4
def summary(state: EmployeeState) -> EmployeeState:
    summary_text = f"Employee {state['employee_name']} has a yearly salary of {state['yearly_salary']} and a bonus of {state['bonus_amount']}. Project status is {state['project_status']}"
    return {"summary": summary_text}


# Graph
graph = StateGraph(EmployeeState)

# Nodes
graph.add_node("calculate_yearly_salary", calculate_yearly_salary)
graph.add_node("calculate_bonus", calculate_bonus)
graph.add_node("project_evaluation", project_evaluation)
graph.add_node("summary", summary)

# Edge
graph.add_edge(START, "calculate_yearly_salary")
graph.add_edge(START, "calculate_bonus")
graph.add_edge(START, "project_evaluation")

graph.add_edge("calculate_yearly_salary", "summary")
graph.add_edge("calculate_bonus", "summary")
graph.add_edge("project_evaluation", "summary")

graph.add_edge("summary", END)

workflow = graph.compile()

initial_state = {
    "employee_name": "John",
    "monthly_salary": 50000,
    "working_days": 26,
    "completed_projects": 7,
}

result = workflow.invoke(initial_state)

print(result)
