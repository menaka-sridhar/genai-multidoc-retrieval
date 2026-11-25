## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:

To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.


### PROBLEM STATEMENT:

To develop an intelligent retrieval agent that can read and analyze multiple research papers in PDF format. The system should retrieve relevant information from the documents, summarize the content, compare different research papers, and generate accurate answers to user queries using LlamaIndex and a Large Language Model (LLM).

### DESIGN STEPS:

## STEP 1: Install and Import Required Libraries
- Install the required LlamaIndex libraries and other dependencies.
- Import the necessary Python modules.
- Configure the OpenAI API key.
- Initialize the language model (gpt-4o-mini) and embedding model (text-embedding-3-small).
## STEP 2: Load and Process Multiple Documents
- Create a Python file named util.py containing the get_doc_tools() function.
- Load the three research papers: metagpt.pdf, longlora.pdf, and loftq.pdf.
- Split the documents into smaller text chunks using SentenceSplitter.
- Create a Vector Store Index and Summary Index for each document.
- Generate two tools for each paper:
  1. Vector Query Tool – retrieves specific information.
  2. Summary Query Tool – summarizes the document.
- Store all six tools in the initial_tools list.
## STEP 3: Create and Test the Retrieval Agent
- Create a multidocument retrieval agent using FunctionAgent.
- Connect all six tools with the language model.
- Provide a system prompt to guide the agent in answering questions.
- Define an asynchronous function named ask_question() to interact with the agent.
- Test the agent using different questions about individual papers and comparisons between papers.
- Analyze the generated answers for relevance and accuracy.


### PROGRAM:

```

# STEP 1: Install Required Libraries
%pip install llama-index-core llama-index-llms-openai llama-index-embeddings-openai nest_asyncio pypdf

# STEP 2: Import Libraries and Configure API Key
import os
from getpass import getpass
from pathlib import Path
from typing import List, Optional
import nest_asyncio

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    SummaryIndex
)

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from llama_index.core.agent.workflow import FunctionAgent

nest_asyncio.apply()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    OPENAI_API_KEY = getpass("Enter OpenAI API Key: ")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Configure LLM and Embedding Model
embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY
)

Settings.embed_model = embed_model
Settings.llm = llm

print("LlamaIndex configured successfully!")

# STEP 3: Define Document Tool Function
def get_doc_tools(file_path: str, name: str):

    if not Path(file_path).exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    # Load PDF document
    documents = SimpleDirectoryReader(
        input_files=[file_path]
    ).load_data()

    # Split document into chunks
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    # Create Vector Index
    vector_index = VectorStoreIndex(nodes)

    # Define Vector Query Function
    def vector_query(
        query: str,
        page_numbers: Optional[List[str]] = None
    ):

        page_numbers = page_numbers or []
        filters = None

        if page_numbers:
            metadata_dicts = [
                {"key": "page_label", "value": str(p)}
                for p in page_numbers
            ]

            filters = MetadataFilters.from_dicts(
                metadata_dicts,
                condition=FilterCondition.OR
            )

        query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filters=filters
        )

        response = query_engine.query(query)
        return str(response)

    # Create Vector Query Tool
    vector_query_tool = FunctionTool.from_defaults(
        fn=vector_query,
        name=f"vector_tool_{name}",
        description=f"Search specific information in {name}"
    )

    # Create Summary Index
    summary_index = SummaryIndex(nodes)

    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True
    )

    # Create Summary Tool
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        name=f"summary_tool_{name}",
        description=f"Summarize information about {name}"
    )

    return vector_query_tool, summary_tool


# STEP 4: Load Research Papers and Create Tools
papers = [
    "metagpt.pdf",
    "longlora.pdf",
    "loftq.pdf"
]

paper_names = [
    "MetaGPT",
    "LongLoRA",
    "LoftQ"
]

initial_tools = []
paper_to_tools_dict = {}

for paper, name in zip(papers, paper_names):

    print("\nProcessing paper:", name)

    vector_tool, summary_tool = get_doc_tools(
        paper,
        name
    )

    paper_to_tools_dict[paper] = [
        vector_tool,
        summary_tool
    ]

    initial_tools.append(vector_tool)
    initial_tools.append(summary_tool)

print("\nAll research papers loaded successfully!")
print("Total tools created:", len(initial_tools))


# STEP 5: Create Multidocument Retrieval Agent
agent = FunctionAgent(
    tools=initial_tools,
    llm=llm,
    system_prompt=(
        "You are a research assistant. "
        "Answer questions using the provided research papers. "
        "Compare information across papers when required."
    ),
    streaming=False
)

print("Multidocument Retrieval Agent created successfully!")


# STEP 6: Define Question-Answering Function
async def ask_question(question):

    response = await agent.run(user_msg=question)

    print("\nQuestion:", question)
    print("\nAnswer:")
    print(str(response))


# STEP 7: Test the Agent with Different Questions
await ask_question(
    "What is the main objective of the MetaGPT research paper?"
)

await ask_question(
    "What is LongLoRA and what are its main advantages?"
)

await ask_question(
    "Explain the main objective and working principle of LoftQ."
)

await ask_question(
    "Compare MetaGPT, LongLoRA, and LoftQ based on their objectives and contributions."
)

await ask_question(
    "Summarize the three research papers and explain their important findings."
)

print("\nMultidocument Retrieval Agent execution completed!")


```

### OUTPUT:

<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/5ec61c2c-a009-4602-904d-d5a5967308df" />


### RESULT:

Thus, the multidocument retrieval agent was successfully designed and implemented using LlamaIndex. It loaded and processed multiple research papers, retrieved relevant information, generated summaries, and answered questions across documents. The system demonstrated its ability to provide concise and relevant responses to different user queries using a Large Language Model.
