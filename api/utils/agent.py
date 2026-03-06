from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import os
import math
from dotenv import load_dotenv

load_dotenv()

# Pre-defined Python Tool for CAGR Calculation
def calculate_cagr(input_str):
    """
    Calculates Compound Annual Growth Rate (CAGR).
    Input: 'baseline_value, target_value, num_years'
    """
    try:
        parts = [x.strip() for x in input_str.split(',')]
        if len(parts) != 3:
            return "Error: Please provide exactly 3 values: 'baseline_value, target_value, num_years'."
            
        baseline = float(parts[0])
        target = float(parts[1])
        years = float(parts[2])
        
        cagr = (math.pow(target / baseline, 1 / years) - 1) * 100
        return f"The CAGR is {cagr:.2f}%"
    except Exception as e:
        return f"Error calculating CAGR: {str(e)}"

def get_agent_executor(mongodb_uri, db_name, collection_name, index_name):
    print(f"DEBUG: Initializing Agent with DB: {db_name}, Index: {index_name}")
    
    # Initialize OpenAI LLM
    llm = ChatOpenAI(
        model="gpt-4o", 
        temperature=0, 
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Initialize Vector Search
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    client = MongoClient(mongodb_uri)
    collection = client[db_name][collection_name]
    
    vector_search = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=index_name
    )
    
    # 1. Retrieval Tool
    def retrieve_info(query):
        print(f"DEBUG: Searching Vector DB for: {query}")
        results = vector_search.similarity_search(query, k=6)
        print(f"DEBUG: Found {len(results)} chunks.")
        return "\n\n".join([f"[Page {d.metadata['page_number']}]: {d.page_content}" for d in results])

    retrieval_tool = Tool(
        name="KnowledgeBase",
        func=retrieve_info,
        description="Search for facts, jobs data, tables, and narrative from the PDF. Use this first for any data extraction test."
    )
    
    # 2. Math Tool
    math_tool = Tool(
        name="CAGRCalculator",
        func=calculate_cagr,
        description="Calculates CAGR. Input format: 'baseline_value, target_value, num_years'. Example: '100, 200, 8'"
    )
    
    tools = [retrieval_tool, math_tool]
    
    # Get ReAct Prompt from Hub (standard for GPT-4)
    # instructions copied here to ensure isolation
    template = """You are a helpful AI assistant with access to specialized tools. You MUST use the tools provided to answer questions.

You have access to the following tools:

{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: you should always think about what to do - what tool will help you answer this?
Action: the action to take, MUST be one of [{tool_names}]
Action Input: the input to the action (follow the tool's description carefully)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

CRITICAL RULES:
1. You MUST start with Thought, then take an Action - NEVER skip directly to Final Answer without using tools first
2. Every Action MUST be followed by Action Input - NEVER write "None" or skip the input
3. Every fact or number MUST cite page numbers as *(Page X)*
4. Comparison queries require a Markdown table with clear headers
5. Use bold (**text**) for key figures and important points
6. For CAGR calculations: FIRST find baseline (2022) and target (2030) values using KnowledgeBase, THEN use CAGRCalculator

STRICT OUTPUT FORMAT FOR FINAL ANSWER:
- **Page Numbers are MANDATORY** — Every factual claim MUST include the page number where it was found, formatted as *(Page X)*
- **Comparison queries require a Markdown table** — Use markdown tables whenever comparing multiple data points
- **Use Markdown formatting** — Structure with ## headings, bullet points, and proper formatting
- **For CAGR questions** — Show the calculation steps and final percentage

Chat History:
{chat_history}

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

    from langchain_core.prompts import PromptTemplate
    prompt = PromptTemplate.from_template(template)
    
    # Create the ReAct Agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create Agent Executor with stricter parsing
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors="Force LLM to follow correct format",  # Auto-correct format errors
        max_iterations=15,  # Reduced to prevent infinite loops
        max_execution_time=180,  # 3 minutes timeout
        return_intermediate_steps=True
    )
    
    return executor
