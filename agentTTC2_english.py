# -*- coding: utf-8 -*-
# Enhanced Text-to-FeatureScript Agent: Convert text descriptions to Onshape FeatureScript code
# Using LangChain and LLMs to analyze requirements and generate FeatureScript code
# Retains RAG using guide.txt for FeatureScript context

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents import Document # Added import
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS # Added for local RAG
from langchain_openai import OpenAIEmbeddings # Added for local RAG (can be swapped if needed)
# from langchain_deepseek import DeepseekEmbeddings # Alternative if available
from langchain.text_splitter import RecursiveCharacterTextSplitter # Added for index building
from typing import List, Optional, Dict, Any # Added Any
import sys
import json
from pathlib import Path # Added for path handling
import re

# Configure UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY") # Added for RAG

if not openai_api_key:
    print("Error: Please set the OPENAI_API_KEY environment variable.")
    print("You can create a .env file in the same directory as this script and add the line:")
    print("OPENAI_API_KEY='your_actual_api_key'")
    exit()
if not tavily_api_key:
    print("Warning: TAVILY_API_KEY environment variable not set. RAG features will be disabled.")
    # Optionally exit() if Tavily is strictly required
    # exit()

# Define model choices
MODELS = {
    "default": "o4-mini-2025-04-16",
    "advanced": "o4-mini-2025-04-16",  # Using GPT-4o for more complex designs and higher accuracy
    "expert": "o4-mini-2025-04-16",  # For extremely detailed and complex designs
}

# Initialize models
try:
    # default_llm = ChatDeepSeek(model=MODELS["advanced"], temperature=0)
    # advanced_llm = ChatDeepSeek(model=MODELS["advanced"], temperature=0)
    # expert_llm = ChatDeepSeek(model=MODELS["advanced"], temperature=0)

    default_llm = ChatOpenAI(model=MODELS["advanced"], reasoning_effort="high")
    advanced_llm = ChatOpenAI(model=MODELS["advanced"], reasoning_effort="high")
    expert_llm = ChatOpenAI(model=MODELS["advanced"], reasoning_effort="high")

    # default_llm = ChatOpenAI(model=MODELS["advanced"], temperature=0)
    # advanced_llm = ChatOpenAI(model=MODELS["advanced"], temperature=0)
    # expert_llm = ChatOpenAI(model=MODELS["advanced"], temperature=0)
    # Test connection
    default_llm.invoke("Test connection")
except Exception as e:
    print(f"Error initializing or connecting to LLM: {e}") # Generic LLM error
    print("Please check your API key and network connection.")
    exit()

# --- Local RAG Setup ---
LOCAL_GUIDE_PATH = "guide.txt"
LOCAL_GUIDE_PATH_2 = "guide2.txt" # Added second guide file
FAISS_INDEX_PATH = "faiss_guide_index"
local_retriever = None # Initialize as None

print("\n--- Setting up Local RAG ---")
try:
    # Check for FAISS library first
    try:
        import faiss
    except ImportError:
        print("⚠️ Required library 'faiss-cpu' or 'faiss-gpu' not found.")
        print("   Please install it: pip install faiss-cpu")
        raise # Re-raise to skip the rest of the local RAG setup

    # Proceed if FAISS is installed
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for embeddings used by local RAG.")
    print("🔑 OpenAI API key found for embeddings.")
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    faiss_path = Path(FAISS_INDEX_PATH)
    guide_path = Path(LOCAL_GUIDE_PATH)
    guide_path2 = Path(LOCAL_GUIDE_PATH_2) # Added path for second guide

    if faiss_path.exists() and any(faiss_path.iterdir()):
        print(f"💾 Loading existing FAISS index from: {FAISS_INDEX_PATH}")
        local_vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        local_retriever = local_vector_store.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 local results
        print("✅ Local FAISS index loaded successfully.")
    # Check if either guide file exists to build the index
    elif guide_path.exists() or guide_path2.exists():
        print(f"⚠️ FAISS index not found at '{FAISS_INDEX_PATH}'.")
        guide_files_found = []
        if guide_path.exists(): guide_files_found.append(LOCAL_GUIDE_PATH)
        if guide_path2.exists(): guide_files_found.append(LOCAL_GUIDE_PATH_2)
        print(f"🔧 Attempting to build FAISS index from: {', '.join(guide_files_found)}...")
        try:
            combined_guide_content = ""
            if guide_path.exists():
                print(f"   - Reading content from '{LOCAL_GUIDE_PATH}'...")
                combined_guide_content += guide_path.read_text(encoding='utf-8') + "\n\n"
            if guide_path2.exists():
                print(f"   - Reading content from '{LOCAL_GUIDE_PATH_2}'...")
                combined_guide_content += guide_path2.read_text(encoding='utf-8') + "\n\n"

            # Split the combined document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, # Adjust chunk size as needed
                chunk_overlap=100, # Adjust overlap as needed
                length_function=len,
            )
            texts = text_splitter.split_text(combined_guide_content)
            print(f"   - Split combined guide content into {len(texts)} chunks.")

            # Create FAISS index from combined text chunks
            print("   - Creating embeddings and building FAISS index (this may take a moment)...")
            local_vector_store = FAISS.from_texts(texts, embeddings)
            print("   - Saving FAISS index...")
            local_vector_store.save_local(FAISS_INDEX_PATH)
            local_retriever = local_vector_store.as_retriever(search_kwargs={"k": 5})
            print(f"✅ FAISS index built and saved successfully to '{FAISS_INDEX_PATH}'.")
        except Exception as build_e:
            print(f"❌ Error building FAISS index: {build_e}")
            print("   Local RAG will be disabled.")
            local_retriever = None
    else:
        # Update the message if neither index nor guide files are found
        print(f"⚠️ FAISS index not found at '{FAISS_INDEX_PATH}' and guide files ('{LOCAL_GUIDE_PATH}', '{LOCAL_GUIDE_PATH_2}') not found.")
        print("   Local RAG cannot be initialized.")
        local_retriever = None

except ImportError:
    # This catch is specifically for the faiss import check at the beginning
    print("   Local RAG setup skipped due to missing FAISS library.")
    local_retriever = None # Ensure it's None if any setup error occurs
except Exception as e:
    print(f"❌ An unexpected error occurred during local RAG setup: {e}")
    local_retriever = None
print("--- Local RAG Setup Complete ---")


# --- Web RAG Setup ---
print("\n--- Setting up Web RAG (Tavily) ---")
if tavily_api_key:
    print("🔑 Tavily API key found.")
    retriever = TavilySearchResults(max_results=5) # Reduced results to 5
    print("✅ Tavily Web Search enabled.")
else:
    print("⚠️ TAVILY_API_KEY not set. Web Search via Tavily is disabled.")
    retriever = None # No retriever if key is missing
print("--- Web RAG Setup Complete ---")


# Pydantic models for structured output
class ShapeRequirement(BaseModel):
    shape_type: str = Field(description="Geometric shape type (box, cylinder, sphere, cone, etc.)")
    dimensions: Dict[str, float] = Field(description="Shape dimensions (e.g.: length, width, height, radius)")
    position: Optional[List[float]] = Field(None, description="Position [x, y, z]")
    rotation: Optional[List[float]] = Field(None, description="Rotation angles [xrot, yrot, zrot] in degrees")

class Operation(BaseModel):
    operation_type: str = Field(description="Boolean operation type (cut, fuse, common)")
    base_shape: str = Field(description="Name of the base shape")
    tool_shape: str = Field(description="Name of the tool shape")
    result_name: str = Field(description="Name of the result after the operation")

class Feature(BaseModel):
    feature_type: str = Field(description="Type of feature (fillet, chamfer, hole, pattern, shell, draft)")
    target: str = Field(description="Name of the shape or operation result to apply the feature to")
    parameters: Dict[str, Any] = Field(description="Parameters specific to the feature type")

class DesignRequirements(BaseModel):
    title: str = Field(description="Brief title describing the design")
    shapes: List[ShapeRequirement] = Field(description="List of required shapes")
    operations: Optional[List[Operation]] = Field(None, description="List of Boolean operations to perform")
    features: Optional[List[Feature]] = Field(None, description="List of features to apply (fillets, chamfers, etc.)")
    comments: Optional[str] = Field(None, description="Additional comments or instructions")
    complexity_level: int = Field(description="Design complexity level (1-5)")

# Define templates for each chain - FIXED by properly escaping curly braces
requirement_analysis_template = """Analyze CAD design requirements from the description for Onshape FeatureScript generation.

User description:
{user_description}

As an expert CAD designer, extract precise technical specifications for creating this design in Onshape. Focus on:

1. **Primitive Shapes:** Identify all basic shapes needed (box/prism, cylinder, sphere, cone, torus, etc.)

2. **Precise Dimensions:** Extract all dimensions with appropriate units (mm, inch)
   - Length, width, height for boxes/prisms
   - Radius/diameter and height for cylinders
   - Radius for spheres
   - Base radius, apex radius, and height for cones
   - Major and minor radius for tori

3. **Positioning:** Determine relative positions of shapes using:
   - X, Y, Z coordinates from origin
   - Clearance between shapes
   - Alignment relationships (concentric, parallel, etc.)

4. **Boolean Operations:** Identify needed operations:
   - Union (fuse/join)
   - Subtraction (cut/difference)
   - Intersection (common)

5. **Features:** Note any special features:
   - Fillets/rounds
   - Chamfers
   - Holes (through, blind, countersunk)
   - Patterns (linear, circular)
   - Shells
   - Drafts

Return a complete, precise JSON specification following this format:
```json
{{
  "title": "Brief descriptive title of the design",
  "shapes": [
    {{
      "shape_type": "box|cylinder|sphere|cone|torus",
      "dimensions": {{"length": 10, "width": 20, "height": 5}} or {{"radius": 15, "height": 30}},
      "position": [x, y, z],
      "rotation": [xrot, yrot, zrot]
    }}
  ],
  "operations": [
    {{
      "operation_type": "cut|fuse|common",
      "base_shape": "base shape name",
      "tool_shape": "tool shape name",
      "result_name": "result name"
    }}
  ],
  "features": [
    {{
      "feature_type": "fillet|chamfer|hole|pattern|shell|draft",
      "target": "shape name or operation result",
      "parameters": {{"radius": 2}} or {{"distance": 1, "angle": 45}} or other relevant params
    }}
  ],
  "comments": "Additional design notes or clarifications",
  "complexity_level": 1-5
}}
```

IMPORTANT: Return only JSON, no explanations. Ensure the JSON is valid and follows the specified format.
Extract all necessary information directly from the description to make the design fully defined.
For dimensions without specified units, prefer millimeters for precision.
Assign reasonable names to each shape (e.g., "base_box", "top_cylinder").
"""

code_generation_template = """You are an expert Onshape FeatureScript developer specializing in generating code for custom features based on design requirements, strictly adhering to provided guidelines.

**Analyzed design requirements:**
```json
{design_requirements}
```

**Retrieved Context (from local guide.txt/guide2.txt and web search - CRITICAL REFERENCE):**
```
{retrieved_context}
```

**Task:** Generate a complete and functional Onshape FeatureScript code snippet that defines a custom feature accurately modeling the object described in the analyzed design requirements.

**CRITICAL INSTRUCTION:** You MUST strictly follow the syntax, best practices, function usage, and examples provided in the **Retrieved Context**. Pay close attention to the patterns shown in `guide.txt` and `guide2.txt` (which are part of the context). Do NOT deviate from these guidelines.

**Mandatory requirements for the generated FeatureScript code (Must align with Retrieved Context):**

1.  **Import Statement:** Use the standard import as shown in the context, typically:
    ```featurescript
    FeatureScript <version>; // Use appropriate version if specified in context
    import(path : "onshape/std/common.fs", version : "<version>"); // Use appropriate version
    ```
2.  **Feature Annotation:** Include a descriptive `Feature Type Name` as shown in context examples:
    ```featurescript
    annotation {{ "Feature Type Name" : "Descriptive Name From Requirements" }}
    ```
3.  **Export Feature Definition:** Use the standard `defineFeature` structure:
    ```featurescript
    export const featureName = defineFeature(function(context is Context, id is Id, definition is map)
    ```
    (Use a descriptive `featureName` based on the requirements title).
4.  **Precondition Block:** Define parameters precisely as shown in the context, including annotations for `Name`, `Filter`, `Default`, `Bounds` (like `LENGTH_BOUNDS` or custom bounds if defined), etc. Use `isLength`, `isAngle`, `isQuery`, `isBoolean`, etc., correctly.
    ```featurescript
    precondition
    {{
        // Example:
        annotation {{ "Name" : "Slot Width", "Filter" : EntityType.EDGE }} // Use appropriate filters from context
        isLength(definition.slotWidth, LENGTH_BOUNDS); // Use correct bounds specifier

        // ... other parameters based on requirements and context examples
    }}
    ```
5.  **Main Feature Block:** Implement the logic using ONLY the functions and patterns demonstrated in the **Retrieved Context**.
    *   Use `op...` functions for operations (e.g., `opExtrude`, `opBoolean`, `opFillet`).
    *   Use `ev...` functions for evaluations (e.g., `evOwnerSketchPlane`, `evVertexPoint`).
    *   Use `q...` functions for queries (e.g., `qCreatedBy`, `qSketchRegion`, `qBodyType`).
    *   Use `sk...` functions for sketching (e.g., `newSketchOnPlane`, `skRectangle`, `skSolve`).
    *   Handle `context`, `id`, and `definition` correctly as shown in examples.
    *   Ensure correct use of `Id` concatenation (e.g., `id + "extrude1"`).
    *   Apply geometric calculations (`vector`, `normalize`, `coordSystem`) as demonstrated.
6.  **Comments:** Add clear, concise comments explaining major steps, mirroring the style in the context examples.
7.  **Units:** Consistently use units (`inch`, `mm`, `meter`, `degree`) as required by FeatureScript functions and demonstrated in the context.
8.  **Error Handling (Optional but Recommended):** If context shows examples, implement basic `try...catch` blocks for operations prone to failure (like `opFillet`) and use `reportFeatureWarning` or `throw regenError`.

**Output Format:**
Return *only* the complete, executable FeatureScript code within a single `featurescript` code block. Ensure the code is clean, well-formatted, and directly usable in Onshape.

```featurescript
// FeatureScript code starts here
FeatureScript 2625; // Adjust version based on context if possible
import(path : "onshape/std/common.fs", version : "2625.0"); // Adjust version based on context

// ... rest of the FeatureScript code adhering strictly to the context ...
```
"""

code_validation_template = """Thoroughly check and validate the following Onshape FeatureScript code against standard practices and provided guidelines (implicitly, the context used during generation).

```featurescript
{generated_code}
```

Please evaluate the code according to these criteria:

1.  **Core Syntax:** Are all curly braces `{{}}`, parentheses `()`, and semicolons `;` balanced and used correctly according to FeatureScript syntax?
2.  **FeatureScript Structure:** Does the code contain all essential elements: `FeatureScript <version>;`, `import(...)`, `annotation {{ "Feature Type Name" : ... }}`, `export const ... = defineFeature(...)`, a `precondition {{...}}` block, and a main logic block `{{...}}`?
3.  **Standard Library Usage:**
    *   Are standard library functions (`op...`, `ev...`, `q...`, `sk...`, etc.) used with the correct number and type of parameters as expected in FeatureScript?
    *   Does the usage align with common patterns seen in FeatureScript examples (like those in `guide.txt`/`guide2.txt`)? For instance, is `context` passed correctly, are `Id`s handled properly (`id + "suffix"`)?
4.  **Parameter Definitions (Precondition):**
    *   Are parameters defined correctly using `definition.paramName is Type`?
    *   Do they have appropriate `annotation {{ "Name" : ... }}`?
    *   Are filters (`"Filter" : ...`), bounds (`isLength(..., BOUNDS)`), and defaults (`"Default" : ...`) used correctly where applicable, following FeatureScript conventions?
5.  **Code Logic & Completeness:**
    *   Does the code appear logically sound for implementing a CAD feature?
    *   Are there obvious logic errors (e.g., using a variable before definition, incorrect query logic, missing `skSolve`)?
    *   Does the feature seem complete based on typical FeatureScript structure?
6.  **Adherence to Guidelines (Implied):** Does the code style, naming conventions, and function usage generally align with the style presented in typical FeatureScript guides (like `guide.txt`/`guide2.txt`)?

Return the evaluation results in this JSON format:
```json
{{
  "valid_syntax": true/false,
  "correct_structure": true/false,
  "plausible_stdlib_use": true/false,
  "guideline_adherence_check": true/false, // Added check for guideline adherence
  "appears_complete": true/false,
  "potential_issues": [
      "List specific issues found, referencing the criteria above.",
      "Example: 'Missing Name annotation for parameter definition.width.'",
      "Example: 'Incorrect number of arguments for opExtrude on line X.'",
      "Example: 'Query qBodyType seems incorrectly used here, consider qCreatedBy.'"
    ],
  "corrected_code": "Provide corrected code ONLY if errors are obvious and simple syntax/structural fixes. Otherwise, leave as null."
}}

```

Focus on FeatureScript-specific issues that would prevent execution or deviate significantly from standard practices shown in guides. Be specific in the `potential_issues`.
"""

# documentation_template removed

# Initialize templates
requirement_analysis_prompt = ChatPromptTemplate.from_template(requirement_analysis_template)
code_generation_prompt = ChatPromptTemplate.from_template(code_generation_template)
code_validation_prompt = ChatPromptTemplate.from_template(code_validation_template)
# documentation_prompt removed

# Define helper functions
def json_to_pydantic(json_str: str) -> DesignRequirements:
    """Convert JSON string to Pydantic model"""
    try:
        # Clean up the JSON string if needed
        if "```json" in json_str:
            json_str = re.search(r'```json\s*(.*?)\s*```', json_str, re.DOTALL).group(1)
        
        data = json.loads(json_str)
        return DesignRequirements(**data)
    except Exception as e:
        print(f"Error converting JSON to Pydantic: {e}")
        # Return a minimal valid object
        return DesignRequirements(
            title="Unable to parse requirements",
            shapes=[ShapeRequirement(shape_type="box", dimensions={"length": 10, "width": 10, "height": 10})],
            complexity_level=1
        )

def clean_code(code_str: str) -> str:
    """Clean up the FeatureScript code string"""
    # Look for ```featurescript block
    match = re.search(r'```featurescript\s*(.*?)\s*```', code_str, re.DOTALL | re.IGNORECASE)
    if match:
        code_str = match.group(1)
    # Fallback: try to remove ``` if featurescript block wasn't found explicitly
    elif "```" in code_str:
         code_str = re.sub(r'^\s*```.*?\s*\n', '', code_str) # Remove opening ``` line
         code_str = re.sub(r'\n\s*```\s*$', '', code_str) # Remove closing ``` line

    return code_str.strip()

def process_validation_result(validation_result: str) -> dict:
    """Process validation result JSON"""
    try:
        if "```json" in validation_result:
            validation_result = re.search(r'```json\s*(.*?)\s*```', validation_result, re.DOTALL).group(1)
        
        return json.loads(validation_result)
    except Exception as e:
        print(f"Error processing validation result: {e}")
        return {
            "executable": False,
            "meets_requirements": False,
            "errors": [f"Error parsing validation result: {e}"],
            "optimization_suggestions": [],
            "corrected_code": None
        }

def create_rag_query(design_reqs: DesignRequirements) -> str:
    """Creates a focused RAG query for FeatureScript based on analyzed design requirements."""
    shape_types = []
    if design_reqs.shapes:
        shape_types = list(set([s.shape_type for s in design_reqs.shapes])) # Get unique shape types

    operation_types = []
    if design_reqs.operations:
        # Map general terms to FeatureScript-specific terms
        op_map = {
            "cut": "opBoolean SUBTRACTION", 
            "fuse": "opBoolean UNION", 
            "common": "opBoolean INTERSECTION"
        }
        fs_ops = [op_map.get(o.operation_type, o.operation_type) for o in design_reqs.operations]
        operation_types = list(set(fs_ops))

    # Map shapes to specific FeatureScript operations
    shape_map = {
        "box": "opSketch skRectangle opExtrude", 
        "cylinder": "opSketch skCircle opExtrude", 
        "sphere": "opSketch skArc opRevolve", 
        "cone": "opSketch skLine opRevolve",
        "torus": "opSketch skCircle opRevolve"
    }
    
    # Start with Onshape FeatureScript as base
    query_parts = ["Onshape FeatureScript"]
    
    # Add shape-specific operations
    if shape_types:
        for shape in shape_types:
            if shape in shape_map:
                query_parts.append(shape_map[shape])
    
    # Add operation-specific terms
    if operation_types:
        query_parts.extend(operation_types)

    # Check for special features
    features = getattr(design_reqs, 'features', [])
    if features:
        for feature in features:
            if hasattr(feature, 'feature_type'):
                feature_type = feature.feature_type.lower()
                if "fillet" in feature_type:
                    query_parts.append("opFillet")
                elif "chamfer" in feature_type:
                    query_parts.append("opChamfer")
                elif "hole" in feature_type:
                    query_parts.append("opHole")
                elif "pattern" in feature_type:
                    query_parts.append("opPattern")
                elif "shell" in feature_type:
                    query_parts.append("opShell")
                elif "draft" in feature_type:
                    query_parts.append("opDraft")

    # Add more specific version number term
    query_parts.append("commonImports.fs version")
    
    # Join all parts with space
    query = " ".join(list(dict.fromkeys(query_parts))) # Join unique parts

    # Fallback if no specific terms identified
    if len(query_parts) <= 2:  # If only "Onshape FeatureScript" and version
        fallback_term = design_reqs.title if design_reqs.title else 'custom feature'
        return f"Onshape FeatureScript {fallback_term} example code"

    return query

# Helper function to process retrieved documents
def format_retrieved_context(docs: List[Dict]) -> str: # Changed type hint to List[Dict]
    """Formats the retrieved list of dictionaries into a single string."""
    if not docs:
        return "No relevant context found."
    # Access content using dictionary key, default to empty string if key missing
    # Assuming the content key is 'content' based on TavilySearchResults typical output
    """Formats the retrieved list of documents (from Tavily or FAISS) into a single string."""
    if not docs:
        return "No relevant context found."
    
    context_str = ""
    # Handle both Document objects (from FAISS) and Dicts (from Tavily)
    for i, doc in enumerate(docs):
        if isinstance(doc, Document):
            content = doc.page_content
            source = doc.metadata.get('source', 'Local Guide') # Add source if available
        elif isinstance(doc, dict):
            content = doc.get('content', 'Error: Content key not found in retrieved document.')
            source = doc.get('url', 'Web Search') # Use URL as source for Tavily
        else:
            content = str(doc) # Fallback
            source = "Unknown Source"
            
        context_str += f"--- Context Source {i+1} ({source}) ---\n{content}\n\n"
        
    return context_str.strip()

# Define chains
requirement_analysis_chain = (
    {"user_description": RunnablePassthrough()}
    | requirement_analysis_prompt
    | advanced_llm
    | StrOutputParser()
    | RunnableLambda(json_to_pydantic)
)

# Initialize Tavily Search Tool (Retriever)
# Check if API key exists before initializing
if tavily_api_key:
    retriever = TavilySearchResults(max_results=10) # Get top 3 results
else:
    retriever = None # No retriever if key is missing

def select_model_by_complexity(inputs):
    """Select model based on complexity level"""
    try:
        complexity = inputs.get("complexity_level", 1)
        if complexity >= 4:
            return expert_llm
        elif complexity >= 2:
            return advanced_llm
        else:
            return default_llm
    except:
        # Default to advanced model if we can't determine complexity
        return advanced_llm

code_generation_chain = (
    code_generation_prompt
    | advanced_llm
    | StrOutputParser()
    | advanced_llm
    | StrOutputParser()
    | RunnableLambda(clean_code)
)

# Enhanced code generation chain with RAG
# Setup for parallel execution: retrieve context from multiple sources and pass requirements
rag_setup = RunnableParallel(
    {
        # Retrieve from Tavily Web Search
        "web_context": (
            (lambda x: create_rag_query(x["design_requirements"]))
            | retriever
            # Note: Formatting happens *after* combining contexts
        ) if retriever else (lambda x: []), # Return empty list if disabled

        # Retrieve from Local FAISS Index
        "local_context": (
            (lambda x: create_rag_query(x["design_requirements"])) # Use same query logic
            | local_retriever
            # Note: Formatting happens *after* combining contexts
        ) if local_retriever else (lambda x: []), # Return empty list if disabled

        "design_requirements": (lambda x: x["design_requirements"]),
        # Keep user_text in the parallel step output if needed elsewhere,
        # but it's no longer directly used for the retriever query.
        "user_text": (lambda x: x["user_text"]) 
    }
)

# Combine and format contexts after retrieval
def combine_and_format_contexts(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Combines contexts from web and local sources and formats them."""
    combined_docs = inputs.get("web_context", []) + inputs.get("local_context", [])
    
    # Optional: Add logic here to de-duplicate or rank combined_docs if needed
    
    formatted_context = format_retrieved_context(combined_docs)
    
    # Return a dictionary suitable for the next step (code_generation_prompt)
    return {
        "design_requirements": inputs["design_requirements"],
        "retrieved_context": formatted_context
        # "user_text": inputs["user_text"] # Pass through if needed later
    }

# Define the RAG chain
rag_code_generation_chain = (
    rag_setup
    | RunnableLambda(combine_and_format_contexts) # Combine contexts here
    | code_generation_prompt
    | advanced_llm
    | StrOutputParser()
    | RunnableLambda(clean_code)
)


code_validation_chain = (
    {"generated_code": RunnablePassthrough()}
    | code_validation_prompt
    | advanced_llm
    | StrOutputParser()
    | RunnableLambda(process_validation_result)
)

# documentation_chain removed

class TextToCADAgent:
    def __init__(self):
        self.requirement_analysis_chain = requirement_analysis_chain
        # Use the RAG chain for code generation
        self.code_generation_chain = rag_code_generation_chain
        self.code_validation_chain = code_validation_chain
        # self.documentation_chain removed

    def process_request(self, user_text):
        """Process user request and generate Onshape FeatureScript code"""
        try:
            print(f"\n🔍 Analyzing request for FeatureScript: '{user_text}'...")

            # Step 1: Analyze requirements
            design_requirements = self.requirement_analysis_chain.invoke(user_text)
            if not design_requirements:
                return None, None

            print(f"\n✅ Requirements analysis successful:")
            print(f"   Title: {design_requirements.title}")
            print(f"   Number of shapes: {len(design_requirements.shapes)}")
            print(f"   Complexity level: {design_requirements.complexity_level}/5")
            
            if design_requirements.operations:
                print(f"   Number of operations: {len(design_requirements.operations)}")

            # Step 2: Generate code based on requirements
            try:
                # Invoke the RAG chain with proper input structure
                generated_code = self.code_generation_chain.invoke({
                    "design_requirements": design_requirements,
                    "user_text": user_text
                })
                print(f"✅ Code generation complete (with RAG context)")
                # return design_requirements, generated_code # Don't return yet, proceed to validation

            except Exception as e:
                print(f"❌ Error generating code: {e}")
                return design_requirements, None # Return None for code if generation fails

            # Step 3: Validate the generated FeatureScript code (Syntax/Structure Check)
            try:
                print(f"\n🔍 Performing syntax and structure validation on generated FeatureScript...")
                validation_result = self.code_validation_chain.invoke(generated_code)

                # Use validation keys specific to the FeatureScript validation template
                if validation_result.get("valid_syntax") and validation_result.get("correct_structure") and validation_result.get("plausible_stdlib_use"):
                    print("✅ Validation check passed: Syntax, structure, and stdlib use appear plausible.")
                    if validation_result.get("potential_issues"):
                        print("   Potential issues noted by validator (review recommended):")
                        for issue in validation_result["potential_issues"]:
                            print(f"     - {issue}")
                    final_code = generated_code # Use original code even if minor issues noted, unless corrected code provided
                else:
                    print("⚠️ Validation check identified potential issues with generated FeatureScript:")
                    if not validation_result.get("valid_syntax"): print("   - Invalid Syntax suspected.")
                    if not validation_result.get("correct_structure"): print("   - Incorrect Structure suspected.")
                    if not validation_result.get("plausible_stdlib_use"): print("   - Implausible Standard Library Usage suspected.")
                    for issue in validation_result.get("potential_issues", []):
                        print(f"   - Issue: {issue}")

                    if validation_result.get("corrected_code"):
                        print("🔧 Applying suggested corrections from validator...")
                        final_code = validation_result["corrected_code"]
                    else:
                        final_code = generated_code
                        print("⚠️ No corrections suggested by validator, using original code despite potential issues.")

                # Note: Optimization suggestions might not be relevant for FeatureScript validation template
                # if validation_result.get("optimization_suggestions"):
                #    print("\n💡 Optimization suggestions:")
                #    for suggestion in validation_result["optimization_suggestions"]:
                #        print(f"   - {suggestion}")

            except Exception as e:
                print(f"⚠️ Error during code validation step: {e}")
                print("🔄 Continuing with unvalidated code...")
                final_code = generated_code

            print("\n✅ FeatureScript generation process complete!")
            return design_requirements, final_code

        except Exception as e:
            print(f"❌ Error in process_request: {e}")
            return None, None # Return None for both if analysis fails

    def save_outputs(self, code, design_requirements, base_filename="generated_featurescript"):
        """Save the generated FeatureScript code to a file"""
        # Create output directory if it doesn't exist
        output_dir = "featurescript_outputs" # Changed directory name
        os.makedirs(output_dir, exist_ok=True)

        # Generate sanitized filename from title
        if design_requirements and design_requirements.title:
            # Sanitize for typical filesystem compatibility, replace spaces with underscores
            sanitized_title = re.sub(r'[^\w\s-]', '', design_requirements.title).strip().lower()
            sanitized_title = re.sub(r'[-\s]+', '_', sanitized_title).strip('-_')
            # Add a suffix if empty after sanitization
            filename_base = f"{output_dir}/{sanitized_title or base_filename}"
        else:
            filename_base = f"{output_dir}/{base_filename}"

        # Save code
        try:
            code_filename = f"{filename_base}.fs" # Changed extension to .fs
            with open(code_filename, "w", encoding="utf-8") as file:
                file.write(code)
            print(f"\n💾 FeatureScript code saved to file '{code_filename}'")
        except IOError as e:
            print(f"❌ Error saving FeatureScript file: {e}")

        # Documentation and requirements saving removed as requested.

        print("\n--------------------------------------------------")
        print("📋 Usage instructions for Onshape:")
        print("1. Open your Onshape document.")
        print("2. Create a new Feature Studio tab (+) or open an existing one.")
        print(f"3. Copy the entire content from the generated file '{code_filename}'.")
        print("4. Paste the code into the Feature Studio editor, replacing any existing content.")
        print("5. Click 'Commit' (or 'Update feature logic' if editing).")
        print("6. Go back to your Part Studio.")
        print("7. Add the new custom feature (it should appear in the feature toolbar).")
        print("--------------------------------------------------")

if __name__ == "__main__":
    agent = TextToCADAgent() # Class name could be updated, but leave for now

    # Interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        print("\n🤖 Text-to-FeatureScript Agent - Interactive Mode") # Updated title
        print("👋 Welcome! Please describe the custom Onshape feature you want to create.") # Updated welcome
        print("💡 To exit, type 'exit' or 'quit'.")

        while True:
            print("\n" + "="*50)
            user_input = input("🔍 Your feature description: ") # Updated prompt

            if user_input.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break

            if not user_input.strip():
                print("⚠️ Description cannot be empty. Please try again.")
                continue

            # Process request to get FeatureScript code
            design_requirements, code = agent.process_request(user_input)

            if code:
                # Generate a filename based on the first few words of input
                filename_base = "_".join(user_input.split()[:4]).lower() # Use more words potentially
                filename_base = re.sub(r'[^\w\s-]', '', filename_base)
                filename_base = re.sub(r'[-\s]+', '_', filename_base).strip('_')
                filename_base = filename_base or "custom_feature" # Ensure not empty

                # Save the FeatureScript code
                agent.save_outputs(code, design_requirements, filename_base)
    else:
        # Process example requests for FeatureScript
        example_requests = [
            "box"
        ]

        for i, request in enumerate(example_requests, 1):
            print(f"\n\n{'='*80}")
            print(f"📋 PROCESSING FEATURESCRIPT REQUEST {i}: '{request}'") # Updated title
            print(f"{'='*80}")

            # Process request
            design_requirements, code = agent.process_request(request)

            if code:
                # Save output
                agent.save_outputs(code, design_requirements, f"featurescript_example_{i}")

                print("\n🔍 FEATURESCRIPT CODE PREVIEW:") # Updated title
                print("-" * 40)
                # Print first 15 lines of code
                print("\n".join(code.split("\n")[:15]) + "\n...")
                print("-" * 40)
