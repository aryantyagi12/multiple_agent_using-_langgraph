from typing import TypedDict,Annotated
import os 
from dotenv import load_dotenv
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (AnyMessage,HumanMessage,AIMessage,SystemMessage)
import psycopg
from langchain_groq import ChatGroq
from tools.flight_tool import search_flight
from tools.tavily_tool import tavily_search

load_dotenv()
llm=ChatGroq(model="llama-3.3-70b-versatile")
DATABASE_URL=os.getenv("DATABASE_URL")

class TravelState(TypedDict):
    message:Annotated[list[AnyMessage],operator.add]
    user_query:str
    flight_results:str
    hotel_results:str
    itinerary:str
    llm_calls:int

def flight_agent(state:TravelState):
    query=state["user_query"]
    flight_data=search_flight(query)
    return {
        "flight_results":flight_data,
        "message":[AIMessage(content=f"flight result fetched")],
        "llm_calls":state.get("llm_calls",0)+1
    }
def hotel_agent(state:TravelState):
    query=f"hotel search for{state['user_query']}"
    hotel_data=tavily_search(query)
    return {
        "hotel_results":hotel_data,
        "message":[AIMessage(content=f"hotel information fetched")],
        "llm_calls":state.get("llm_calls",0)+1
    }
def itinerary_agent(state:TravelState):
    prompt=f"""
    create a travel itinerary.
    user query:
    {state["user_query"]}
    flight results:
    {state["flight_results"]}
    hotel results:
    {state["hotel_results"]}
    """
    response=llm.invoke(
        [SystemMessage(content="you are an expert travel planner"),
        HumanMessage(content=prompt)]
    )
    return {
        "itinerary":response.content,
        "message":[response],
        "llm_calls":state.get("llm_calls",0)+1
    }
def final_agent(state:TravelState):
    final_prompt=f"""
    Generate final travel response.
    Flight:
    {state["flight_results"]}
    Hotel:
    {state["hotel_results"]}
    itinerary:
    {state["itinerary"]}
    """
    response=llm.invoke(
        [HumanMessage(content=final_prompt)],

    )
    return {
        "message":[response],
        "llm_calls":state.get("llm_calls",0)+1
    }
graph=StateGraph(TravelState)
graph.add_node("flight_agent",flight_agent)
graph.add_node("hotel_agent",hotel_agent)
graph.add_node("final_agent",final_agent)
graph.add_node("itinerary_agent",itinerary_agent)

graph.add_edge(START,"flight_agent")
graph.add_edge("flight_agent","hotel_agent")
graph.add_edge("hotel_agent","itinerary_agent")
graph.add_edge("itinerary_agent","final_agent")
graph.add_edge("final_agent",END)

_conn=psycopg.connect(DATABASE_URL)
_conn.autocommit = True
checkpointer=PostgresSaver(_conn)
checkpointer.setup()

app=graph.compile(checkpointer=checkpointer)

if __name__=="__main__":
    config={"configurable":{"thread_id":"1"}}
    user_input=input("enter travel request")
    result=app.invoke(
        {
            "message":[HumanMessage(content=user_input)],
            "user_query":user_input,
            "flight_results":"",
            "hotel_results":"",
            "itinerary":"",
            "llm_calls":0
        },
        config=config
    )
    print("\nfinal output\n")
    print(result["message"][-1].content)


