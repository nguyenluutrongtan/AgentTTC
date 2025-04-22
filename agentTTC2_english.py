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
from typing import List, Optional, Dict, Any, Union # Added Any
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

    default_llm = ChatOpenAI(model=MODELS["advanced"], reasoning_effort="low")
    advanced_llm = ChatOpenAI(model=MODELS["advanced"], reasoning_effort="low")
    expert_llm = ChatOpenAI(model=MODELS["advanced"], reasoning_effort="low")

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
LOCAL_EXAMPLES_PATH = "example.txt"  # Add examples file
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
    guide_path2 = Path(LOCAL_GUIDE_PATH_2)
    examples_path = Path(LOCAL_EXAMPLES_PATH)  # Add examples path

    if faiss_path.exists() and any(faiss_path.iterdir()):
        print(f"💾 Loading existing FAISS index from: {FAISS_INDEX_PATH}")
        local_vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        local_retriever = local_vector_store.as_retriever(search_kwargs={"k": 25}) # Retrieve top 2 local results
        print("✅ Local FAISS index loaded successfully.")
    # Check if any of the source files exist to build the index
    elif guide_path.exists() or guide_path2.exists() or examples_path.exists():
        print(f"⚠️ FAISS index not found at '{FAISS_INDEX_PATH}'.")
        guide_files_found = []
        if guide_path.exists(): guide_files_found.append(LOCAL_GUIDE_PATH)
        if guide_path2.exists(): guide_files_found.append(LOCAL_GUIDE_PATH_2)
        if examples_path.exists(): guide_files_found.append(LOCAL_EXAMPLES_PATH)
        print(f"🔧 Attempting to build FAISS index from: {', '.join(guide_files_found)}...")

        try:
            all_docs = []

            # 1. Process example.txt: Split into individual examples
            if examples_path.exists():
                print(f"   - Processing examples from '{LOCAL_EXAMPLES_PATH}'...")
                example_content = examples_path.read_text(encoding='utf-8')
                # Split by lines starting with #, but keep the # line with the content
                # Use positive lookahead to keep the delimiter
                example_splits = re.split(r'(?=\n#)', example_content)
                
                for i, example_text in enumerate(example_splits):
                    example_text = example_text.strip()
                    if example_text:
                        # Extract title from the first line (e.g., "#Simple box")
                        first_line = example_text.split('\n', 1)[0]
                        title = first_line.strip() if first_line.startswith("#") else f"Example {i+1}"
                        metadata = {"source": LOCAL_EXAMPLES_PATH, "title": title}
                        all_docs.append(Document(page_content=example_text, metadata=metadata))
                print(f"   - Created {len(example_splits)} documents from examples.")

            # 2. Process guide.txt and guide2.txt: Combine and chunk
            guide_content = ""
            guide_sources = []
            if guide_path.exists():
                print(f"   - Reading guide content from '{LOCAL_GUIDE_PATH}'...")
                guide_content += guide_path.read_text(encoding='utf-8') + "\n\n"
                guide_sources.append(LOCAL_GUIDE_PATH)
            if guide_path2.exists():
                print(f"   - Reading guide content from '{LOCAL_GUIDE_PATH_2}'...")
                guide_content += guide_path2.read_text(encoding='utf-8') + "\n\n"
                guide_sources.append(LOCAL_GUIDE_PATH_2)

            if guide_content:
                print(f"   - Splitting guide content into chunks...")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, # Adjust chunk size as needed
                    chunk_overlap=100, # Adjust overlap as needed
                    length_function=len,
                )
                guide_texts = text_splitter.split_text(guide_content)
                guide_source_str = ", ".join(guide_sources)
                for chunk in guide_texts:
                    all_docs.append(Document(page_content=chunk, metadata={"source": guide_source_str}))
                print(f"   - Created {len(guide_texts)} documents from guides.")

            # 3. Create FAISS index from all documents (examples + guide chunks)
            if not all_docs:
                 print("   - No documents found to build index.")
                 local_retriever = None
            else:
                print(f"   - Creating embeddings and building FAISS index from {len(all_docs)} total documents (this may take a moment)...")
                # Use from_documents as we now have Document objects
                local_vector_store = FAISS.from_documents(all_docs, embeddings)
                print("   - Saving FAISS index...")
                local_vector_store.save_local(FAISS_INDEX_PATH)
                local_retriever = local_vector_store.as_retriever(search_kwargs={"k": 2}) # Retrieve top 2 results
                print(f"✅ FAISS index built and saved successfully to '{FAISS_INDEX_PATH}'.")

        except Exception as build_e:
            print(f"❌ Error building FAISS index: {build_e}")
            print("   Local RAG will be disabled.")
            local_retriever = None
    else:
        # Update the message to include all possible source files
        print(f"⚠️ FAISS index not found at '{FAISS_INDEX_PATH}' and source files ('{LOCAL_GUIDE_PATH}', '{LOCAL_GUIDE_PATH_2}', '{LOCAL_EXAMPLES_PATH}') not found.")
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
print("⚠️ Tavily Web Search is disabled as requested.")
retriever = None # Tavily RAG disabled as requested
print("--- Web RAG Setup Complete ---")


# Pydantic models for structured output
class ShapeRequirement(BaseModel):
    shape_type: str = Field(description="Geometric shape type (box, cylinder, sphere, cone, etc.)")
    dimensions: Dict[str, Union[float, str]] = Field(description="Shape dimensions (e.g.: length, width, height, radius) with optional units")
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

**Retrieved Context (from example.txt - MUST FOLLOW EXACTLY):**
```
{retrieved_context}
#Example opFillet
    opFillet(context, id + "fillet1", {{
        "entities" : qCreatedBy(id + "baseBox", EntityType.EDGE),            
        "radius" : 0.15 * inch
    }});

#Example opPattern
        // Prepare transforms array and instance names for 8 studs
        var transforms = [];
        var names = [];
        for (var i = 0; i < 4; i += 1)
        {{
            for (var j = 0; j < 2; j += 1)
            {{
                transforms = append(transforms,
                    transform(vector(i * 8, j * 8, 0) * millimeter)
                );
                names = append(names, "stud_" ~ i ~ "_" ~ j);
            }}
        }}
        opPattern(context, id + "stud_pattern", {{
            "entities":                  qCreatedBy(id + "top_stud", EntityType.BODY),
            "transforms":                transforms,
            "instanceNames":             names,
            "copyPropertiesAndAttributes": true
        }});
#Example opBoolean UNION
        opBoolean(context, id + "boolean1", {{
                "tools" : qUnion(qCreatedBy(id + "cuboid1", EntityType.BODY), qCreatedBy(id + "cylinder1", EntityType.BODY)),
                "operationType" : BooleanOperationType.UNION
        }});
```

**Task:** Generate a complete and functional Onshape FeatureScript code snippet that defines a custom feature accurately modeling the object described in the analyzed design requirements.

**CRITICAL INSTRUCTION:** You MUST strictly follow these rules:
1. EXACTLY follow the patterns and structure shown in example.txt
2. Use the SAME import statements as shown in the matching example from example.txt
3. Match the EXACT syntax and function usage from example.txt
4. For each shape type (box, cylinder, sphere), use the EXACT corresponding example from example.txt as your template
5. For boolean operations (like subtraction), use the EXACT pattern shown in the example.txt opBoolean example
6. Do not deviate from these patterns unless absolutely necessary
7. When setting the feature type description, ONLY use characters from the ASCII character set
**Pattern Matching Rules:**
1. For boxes: Use the pattern from "Simple box" example
2. For cylinders: Use the pattern from "Simple Cylinder" example
3. For spheres: Use the pattern from "Simple Sphere" example
4. For boolean operations: Use the pattern from "Example opBoolean Subtraction"
5. For complex features (like boxes with holes): Use the pattern from "Example rectangular with through holes"

**Output Format:**
Return *only* the complete, executable FeatureScript code following the EXACT patterns from example.txt.
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
        if "```json" in json_str:
            json_str = re.search(r'```json\s*(.*?)\s*```', json_str, re.DOTALL).group(1)

        data = json.loads(json_str)
        
        # Convert dimension values to float if they're numeric strings
        if "shapes" in data:
            for shape in data["shapes"]:
                if "dimensions" in shape:
                    for key, value in shape["dimensions"].items():
                        try:
                            shape["dimensions"][key] = float(value)
                        except (ValueError, TypeError):
                            # Keep as string if it contains units
                            pass

        return DesignRequirements(**data)
    except Exception as e:
        print(f"Error converting JSON to Pydantic: {e}")
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
    # Prioritize example.txt patterns
    shape_types = []
    if design_reqs.shapes:
        shape_types = list(set([s.shape_type for s in design_reqs.shapes]))

    # Modified to focus on example.txt patterns
    example_queries = []
    for shape in shape_types:
        if shape == "box":
            example_queries.append("#Simple box")
        elif shape == "cylinder":
            example_queries.append("#Simple Cylinder")
        elif shape == "sphere":
            example_queries.append("#Simple Sphere")

    # Add boolean operation patterns if needed
    if design_reqs.operations:
        for op in design_reqs.operations:
            if op.operation_type == "cut":
                example_queries.append("#Example opBoolean Subtraction")

    # If it's a complex feature, add the relevant example
    if len(shape_types) > 1 or (design_reqs.operations and len(design_reqs.operations) > 0):
        example_queries.append("#Example rectangular with through holes")

    # Join all parts with space
    query = " ".join(example_queries)

    # Fallback if no specific terms identified
    if not example_queries:
        return "#Simple box"  # Default to simple box example

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
# Tavily RAG disabled as requested
retriever = None

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

            # Step 3: Code validation disabled as requested
            print("\n--- Code Validation Skipped as Requested ---")
            final_code = generated_code
            print("\n--- Generated FeatureScript Code (Validation Skipped) ---")
            print(final_code)
            self.save_outputs(final_code, design_requirements)
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
            "A 3x3 Lego brick stacks perfectly on top of a 5x5 Lego brick.",
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
