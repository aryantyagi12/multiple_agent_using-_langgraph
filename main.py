from typing import TypedDict
import os 
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgress import PostgresSaver
from langchain_core.messages import (AnyMessage,HumanMessage,AIMessage,SystemMessage)
import psycopg
from langchain_groq import ChatGroq
from tools.flight_tool import search_flight
from tools.tavily_tool import tavily_search

load_dotenv()
llm=ChatGroq(model="llama-3.3-70b-versatile")
DATABASE_URL=os.getenv("DATABASE_URL")

class TravelState(TypedDict):
    message:Annotated(list[AnyMessage],operator.add)
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
    query=f"hotel search for{state["user_query"]}"
    hotel_data=tavily_search(query)
    return {
        "hotel_results":hotel_data,
        "message":[AIMessage(content=f"hotel information fetched")]
        "llm_calls":state.get("llm_calls",0)+1
    }

