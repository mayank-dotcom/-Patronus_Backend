from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
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
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

STRICT RULES you must always follow in your Final Answer:
1. **Page Numbers are MANDATORY** — Every factual claim MUST include the page number where it was found, formatted as *(Page X)*. Never omit page numbers.
2. **Comparison queries require a Markdown table** — Whenever the question involves comparing, contrasting, or listing multiple data points across categories (e.g., regions vs national average, year-over-year, firm types), you MUST present the results in a properly formatted Markdown table with clear headers.
3. **Use Markdown formatting** — Structure your Final Answer using Markdown: use **bold** for key figures, `##` headings for sections when needed, bullet points for lists, and tables for comparisons.
4. **For CAGR questions** — Find the 2022 baseline and 2030 target values first using KnowledgeBase, then use the CAGRCalculator tool.
5. **Use Chat History** — Use the prior conversation context to avoid repeating yourself and to resolve follow-up references.

Begin!

Chat History:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""

    from langchain.prompts import PromptTemplate
    prompt = PromptTemplate.from_template(template)
    
    # Create the ReAct Agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create Agent Executor
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=20,
        max_execution_time=300,          # 5 minutes — allows complex multi-step queries
        early_stopping_method="generate", # synthesise answer from gathered context instead of erroring
        return_intermediate_steps=True
    )
    
    return executor
