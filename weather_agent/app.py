import streamlit as st
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_groq import ChatGroq
import requests
from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import os


st.title("🌦️🤖 Check any city/village Weather worldwide using AI AGENT")

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")
search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """This function fetches the current weather data for a given city""" 
    api_key = os.getenv("WEATHERSTACK_API_KEY")
    url = f"https://api.weatherstack.com/current?access_key={api_key}&query={city}"
    response = requests.get(url)
    
    return response.json()

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True
)

user_input = st.text_input(label="Ask a question (e.g., Capital of Gujarat and its weather):")

if st.button("Send"):
    with st.spinner("Thinking..."):
        try:
            result = agent_executor.invoke({'input': user_input})
            st.success(result['output'])
        except Exception as e:
            st.error(f"Something went wrong: {e}")
