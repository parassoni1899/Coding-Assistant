"""
agents/workflow.py — LangGraph Agentic Workflow
================================================
Defines a state machine where an LLM acts as a reasoning agent.
It receives a query, uses tools (Search, AST graph, File read) to 
gather context, and synthesizes a final answer.
"""

from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from loguru import logger

from agents.tools import AGENT_TOOLS
from config import get_llm


class AgentState(TypedDict):
    """The state payload passed between LangGraph nodes."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────

def agent_node(state: AgentState):
    """
    The main reasoning node. Invokes the LLM bound with tools.
    """
    logger.info("[Agent] Reasoning node executing...")
    llm = get_llm()
    # Bind our available tools to the model
    llm_with_tools = llm.bind_tools(AGENT_TOOLS)
    
    # Ensure system prompt is present
    messages = list(state["messages"])
    if not isinstance(messages[0], SystemMessage):
        system_prompt = SystemMessage(content=(
            "You are a Senior Principal Software Engineer analyzing a Go codebase.\n"
            "Use your tools to find code, trace call graphs, and read file lines.\n"
            "If a user asks about flow, ALWAYS use `get_call_chain` or `search_codebase`.\n"
            "If a tool call returns an error or insufficient data, try a different approach.\n"
            "Once you have enough context, provide a detailed final answer."
        ))
        messages.insert(0, system_prompt)

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState):
    """
    Router: decide whether to use a tool or return the final answer.
    """
    last_message = state["messages"][-1]
    
    # If there are tool calls in the LLM's response, route to the tools node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info(f"[Agent] Router: Executing {len(last_message.tool_calls)} tool call(s).")
        return "tools"
    
    # Otherwise, the LLM has synthesized its final answer
    logger.info("[Agent] Router: Generation complete. Ending.")
    return "end"


# ─────────────────────────────────────────────────────────────────────────────
# Graph Construction
# ─────────────────────────────────────────────────────────────────────────────

def create_agent_workflow():
    """Compiles and returns the LangGraph state machine."""
    
    # The prebuilt ToolNode automatically executes functions matching the LLM's tool_calls
    tool_node = ToolNode(AGENT_TOOLS)
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # Define edges
    workflow.set_entry_point("agent")
    
    # Conditional routing after the agent thinks
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # Tools always return their output back to the agent
    workflow.add_edge("tools", "agent")
    
    # Compile graph
    app = workflow.compile()
    return app


# Helper function to invoke the graph
def run_agent(query: str):
    app = create_agent_workflow()
    logger.info(f"Starting agent workflow for query: {query}")
    
    inputs = {"messages": [HumanMessage(content=query)]}
    
    # Execute the graph synchronously
    for output in app.stream(inputs, stream_mode="values"):
        last_msg = output["messages"][-1]
        # In a CLI we could print partial thoughts here
        pass
        
    final_message = output["messages"][-1]
    return final_message.content
