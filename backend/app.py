
import os
import asyncio
import sys
import logging
from pathlib import Path
from typing import TypedDict,Optional
from contextlib import asynccontextmanager, AsyncExitStack

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from setup_rag import build_vector_db, get_embeddings, FAISS_INDEX_DIR

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from sentence_transformers import CrossEncoder
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph,START,END

MCP_SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "mcp_server.py")

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# --- LOGGING SETUP ---
BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR.parent / "logs" / "Kissan_ai.log"
LOG_FILE.parent.mkdir(exist_ok=True)
logger=logging.getLogger("Kissan_ai")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _formatter= logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s |%(message)s"
    )
    _file_handler = logging.FileHandler(LOG_FILE,encoding="utf-8")
    _file_handler.setFormatter(_formatter)
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(_formatter)
    logger.addHandler(_file_handler)
    logger.addHandler(_console_handler)

# --- RETRY CONFIG ---
GROQ_RETRY_MAX_ATTEMPTS = int(os.getenv("GROQ_RETRY_MAX_ATTEMPTS", "3"))
GROQ_RETRY_BACKOFF_BASE = int(os.getenv("GROQ_RETRY_BACKOFF_BASE", "2"))

async def with_retries(func, *args, **kwargs):
    """Awaits func with exponential backoff retries. Raises the last exception if all attempts fail.
    Uses asyncio.sleep so a backing-off request doesn't block other requests."""
    last_exception = None
    for attempt in range(1, GROQ_RETRY_MAX_ATTEMPTS + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            logger.warning(f"Attempt {attempt}/{GROQ_RETRY_MAX_ATTEMPTS} failed: {e}")
            if attempt < GROQ_RETRY_MAX_ATTEMPTS:
                await asyncio.sleep(GROQ_RETRY_BACKOFF_BASE ** attempt)
    raise last_exception
 
# --- INITIALIZATION ---

def load_vector_db():
    """Load the vector DB using a local HuggingFace embedding model (no API key needed)."""
    if not (FAISS_INDEX_DIR / "index.faiss").exists():
        logger.info(f"No FAISS index at {FAISS_INDEX_DIR}. Building it for the first time. This may take a minute...")
        build_vector_db()

    logger.info(f"Loading FAISS index from {FAISS_INDEX_DIR}")
    return FAISS.load_local(
        str(FAISS_INDEX_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )

vector_db = load_vector_db()

#retrives top 10 and reranks the retrieved top 5
CANDIDATE_K = int(os.getenv("RAG_CANDIDATE_K", "10"))
RERANK_TOP_N = int(os.getenv("RAG_RERANK_TOP_N", "5"))
retriever = vector_db.as_retriever(search_kwargs={"k": CANDIDATE_K})
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
 

# initializing Groq LLM (text/reasoning model)
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    api_key=groq_api_key,
)

# vision model 
vision_llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0.3,
    api_key=groq_api_key,
)

mcp_client = MultiServerMCPClient(
    {
        "crop_doctor_tools": {
            "command": sys.executable,
            "args": [MCP_SERVER_SCRIPT],
            "transport": "stdio",
        }
    }
)

MCP_SERVER_NAME = "crop_doctor_tools"
MCP_TOOL_TIMEOUT_SECONDS = float(os.getenv("MCP_TOOL_TIMEOUT_SECONDS", "20"))

# A single long-lived MCP session (opened in `lifespan`) reused for every request.
# mcp_client.get_tools() would instead spawn a fresh `python mcp_server.py`
# subprocess for every tool call.
_mcp_session = None


async def _call_tool(name: str, location: str) -> str:
    """Calls an MCP tool and returns its text. Raises if the tool reported an error.

    Uses the MCP session directly rather than the LangChain tool wrappers, which by
    default turn tool errors into ordinary-looking text ("Error calling tool ...")
    that would then be passed to the LLM as if it were real weather data.
    """
    if _mcp_session is None:
        raise RuntimeError("MCP session is not available.")
    result = await asyncio.wait_for(
        _mcp_session.call_tool(name, {"location": location}),
        timeout=MCP_TOOL_TIMEOUT_SECONDS,
    )
    text = "\n".join(
        block.text for block in result.content if getattr(block, "text", None)
    )
    if result.isError:
        raise RuntimeError(f"{name} failed: {text}")
    return text




# --- LCEL CHAINS ---
 
# 1. Vision chain: prompt (text + N images)|vision_llm|output parser
def build_vision_messages(inputs: dict) -> list:
    content = [
        {
            "type": "text",
            "text": (
                "You are an agricultural expert AI. You have been given "
                f"{len(inputs['img_base64_list'])} photo(s) of the same crop/plant, "
                "possibly from different angles or showing different affected areas. "
                "Analyze them together. Identify: (1) the crop type, (2) any visible "
                "diseases, pests, or nutrient deficiencies, (3) overall health status. "
                "Be concise and specific. Also address the farmer's specific question: "
                f"{inputs['user_query']}"
            ),
        }
    ]
    for img_base64 in inputs["img_base64_list"]:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
        })
    return [HumanMessage(content=content)]


vision_chain = RunnableLambda(build_vision_messages) | vision_llm | StrOutputParser()
 
# 2. Retrieval chain:
#    rerank -> select top 5 -> construct a single context string for the LLM
def rerank_top_n(inputs: dict) -> list:
   
    query = inputs["query"]
    docs = inputs["docs"]
    if not docs:
        return []
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _ in ranked[:RERANK_TOP_N]]
 
 
def format_docs(docs) -> str:
    """context construction"""
    if not docs:
        return "No specific documents found. Rely on general organic agricultural knowledge."
    return "\n\n".join(doc.page_content for doc in docs)
 
 
# Returns the reranked Documents; the node formats them into one context string
# for the LLM and also keeps them as separate chunks for evaluation.
retrieval_chain = (
    RunnableParallel(docs=retriever, query=RunnablePassthrough())
    | RunnableLambda(rerank_top_n)
)
 
# 3. Synthesis chain: prompt (all collected context) | llm | output parser
SYSTEM_PROMPT = (
    "You are an expert Agricultural AI assistant specializing in organic and sustainable farming. "
    "You always recommend chemical-free, eco-friendly solutions. "
    "Format your response clearly using markdown with headings and bullet points."
)
 
SYNTHESIS_TEMPLATE = """
A farmer has asked: "{user_query}"
 
**Data Collected:**
- **Image Analysis:** {image_analysis}
- **Location:** {location}
- **Current Weather:** {weather}
- **Soil Condition:** {soil}
 
**Reference Knowledge (from organic farming database):**
{rag_context}
 
**Your Task:**
1. Confirm the crop and any disease/issue identified from the image.
2. Provide practical, **chemical-free/organic** treatment or care advice based on the reference knowledge.
3. Adapt your advice specifically to the current weather and soil conditions mentioned above.
4. Add preventive tips for future crop health.
 
Use clear markdown headings (##) and bullet points. Be practical and farmer-friendly.
"""
 
synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", SYNTHESIS_TEMPLATE),
])
synthesis_chain = synthesis_prompt | llm | StrOutputParser()
 
class KissanState(TypedDict):
    # inputs
    images_base64:list
    user_query:str
    location:str


    # populated by vision_node
    img_base64_list:list
    image_analysis:str

    # populated by weather_soil_node
    weather:str
    soil:str
    weather_soil_failed:bool

    # populated by retrieval_node
    rag_context:str
    rag_contexts:list
    retrieval_failed:bool

    # populated by synthesis_node
    final_answer:str

    # error tracking (hard-fail signal, set by vision_node or synthesis_node)
    error:Optional[str]


NO_IMAGE_ANALYSIS = (
    "No image provided. Diagnose based on the farmer's question and retrieved agricultural knowledge."
)


async def vision_node(state:KissanState)->dict:
    try:
        img_base64_list=state.get("images_base64")
         # No image provided → continue with text-only pipeline
        if not img_base64_list:
            logger.info("No image provided — skipping vision analysis.")
            return {
                "img_base64_list": [],
                "image_analysis": NO_IMAGE_ANALYSIS,
            }

        # Image provided → perform vision analysis
        image_analysis=await with_retries(
            vision_chain.ainvoke,{
                "user_query":state["user_query"],
                "img_base64_list":img_base64_list

            }
        )
        return{
            "img_base64_list":img_base64_list,
            "image_analysis":image_analysis
        }
    except Exception as e:
        logger.error(f"vision_node hard-failed:{e}")
        return{"error":f"Image analysis failed: {e}"}

async def weather_soil_node(state:KissanState)->dict:
    """Non-critical node — degrades gracefully instead of crashing the graph.
    Weather and soil are fetched concurrently and degrade independently, so a
    weather failure doesn't also discard the soil data."""
    if state.get("error"):
        return{}  # upstream already hard-failed, skip work

    weather, soil = await asyncio.gather(
        _call_tool("get_weather", state["location"]),
        _call_tool("get_soil_condition", state["location"]),
        return_exceptions=True,
    )

    failed = False
    if isinstance(weather, BaseException):
        logger.warning(f"weather lookup degraded: {weather!r}")
        weather, failed = "unavailable", True
    if isinstance(soil, BaseException):
        logger.warning(f"soil lookup degraded: {soil!r}")
        soil, failed = "unavailable", True

    return {"weather":weather,"soil":soil,"weather_soil_failed":failed}


async def retrieval_node(state:KissanState)->dict:
    """Non-critical node — degrades gracefully instead of crashing the graph."""
    if state.get("error"):
        return{}

    try:
        if state.get("img_base64_list"):
            search_query=(
                f"organic chemical-free solutions for: {state['image_analysis']}. "
                f"farmer question: {state['user_query']}"
            )
        else:
            # Text-only request: search on the question alone, not on the
            # no-image placeholder text.
            search_query=f"organic chemical-free solutions. farmer question: {state['user_query']}"

        docs=await retrieval_chain.ainvoke(search_query)
        return{
            "rag_context":format_docs(docs),
            "rag_contexts":[doc.page_content for doc in docs],
            "retrieval_failed":False,
        }
    
    except Exception as e:
        logger.warning(f"retrieval node degraded:{e}")
        return{
            "rag_context":"No reference data available - rely on generic farming knowledge.",
            "rag_contexts":[],
            "retrieval_failed":True
        }

async def synthesis_node(state:KissanState)->dict:
    """Critical node — hard-fails into state['error'] if retries are exhausted."""
    if state.get("error"):
        return {}

    try:
        final_answer=await with_retries(
            synthesis_chain.ainvoke,{
                "user_query": state["user_query"],
                "image_analysis": state["image_analysis"],
                "location": state["location"],
                "weather": state["weather"],
                "soil": state["soil"],
                "rag_context": state["rag_context"],
            },
        )
        return {"final_answer":final_answer}
    except Exception as e:
        logger.error(f"synthesis_node hard-failed: {e}")
        return{"error":f"could not generate advice: {e}"}

builder=StateGraph(KissanState)

builder.add_node("vision_node", vision_node)
builder.add_node("weather_soil_node", weather_soil_node)
builder.add_node("retrieval_node", retrieval_node)
builder.add_node("synthesis_node", synthesis_node)

builder.add_edge(START,"vision_node")
builder.add_edge("vision_node","weather_soil_node")
builder.add_edge("weather_soil_node","retrieval_node")
builder.add_edge("retrieval_node","synthesis_node")
builder.add_edge("synthesis_node",END)

kisaan_graph = builder.compile()


 
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Open one MCP session for the lifetime of the app and reuse it for every request.
    # If it can't start, the API still serves requests; weather/soil are reported as unavailable.
    global _mcp_session
    async with AsyncExitStack() as stack:
        try:
            session = await stack.enter_async_context(mcp_client.session(MCP_SERVER_NAME))
            tools = await session.list_tools()
            _mcp_session = session
            logger.info(f"MCP session started with tools: {sorted(t.name for t in tools.tools)}")
        except Exception as e:
            logger.error(f"Could not start MCP session; weather/soil will be unavailable: {e!r}")
        logger.info("Backend startup complete — vector DB loaded.")
        try:
            yield
        finally:
            _mcp_session = None
 
 
app = FastAPI(title="Crop Doctor API", lifespan=lifespan)
 
 
class DiagnoseRequest(BaseModel):
    images_base64: Optional[list[str]] = None
    user_query: str
    location: str
    evaluation_mode: bool = False
 
 
class DiagnoseResponse(BaseModel):
    final_answer: Optional[str] = None
    error: Optional[str] = None
    weather_soil_failed: bool = False
    retrieval_failed: bool = False
    rag_context: Optional[str] = None
    rag_contexts: Optional[list[str]] = None
 
 
@app.get("/health")
def health():
    return {"status": "ok"}
 
 
@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest):
    result = await kisaan_graph.ainvoke({
        "images_base64": request.images_base64,
        "user_query": request.user_query,
        "location": request.location,
    })
    return DiagnoseResponse(
        final_answer=result.get("final_answer"),
        error=result.get("error"),
        weather_soil_failed=result.get("weather_soil_failed", False),
        retrieval_failed=result.get("retrieval_failed", False),
        rag_context=result.get("rag_context"),
        rag_contexts=result.get("rag_contexts"),
    )