from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# Define State
class TemperatureState(TypedDict):
    temp_celcius: float
    temp_farhenheit: float
    weather_status: str


# Convert Temperature function
def convert_temp(state: TemperatureState) -> TemperatureState:
    celcius = state["temp_celcius"]
    fahrenheit = (celcius * 9 / 5) + 32
    state["temp_farhenheit"] = round(fahrenheit, 2)
    return state


# Label Weather function
def label_weather(state: TemperatureState) -> TemperatureState:
    fahrenheit = state["temp_farhenheit"]
    if fahrenheit < 50:
        state["weather_status"] = "Cold"
    elif 50 <= fahrenheit < 77:
        state["weather_status"] = "Mild"
    elif 77 <= fahrenheit < 95:
        state["weather_status"] = "Hot"
    else:
        state["weather_status"] = "Extreme heat"

    return state


# Define and Compile Graph
graph = StateGraph(TemperatureState)

# Nodes
graph.add_node("convert_temp", convert_temp)
graph.add_node("label_weather", label_weather)

# Edges
graph.add_edge(START, "convert_temp")
graph.add_edge("convert_temp", "label_weather")
graph.add_edge("label_weather", END)

# Compile
workflow = graph.compile()

# Execute the graph
initial_state = {"temp_celcius": 28.5}
final_state = workflow.invoke(initial_state)
print(final_state)
