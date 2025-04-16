# -*- coding: utf-8 -*-
# Enhanced Text-to-CAD Agent: Convert text descriptions to FreeCAD code
# Using LangChain and LLMs to analyze requirements and generate FreeCAD code
# Added more complex processing chains

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
    "default": "o3-mini-2025-01-31",
    "advanced": "o3-mini-2025-01-31",  # Using GPT-4o for more complex designs and higher accuracy
    "expert": "o3-mini-2025-01-31",  # For extremely detailed and complex designs
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

    if faiss_path.exists() and any(faiss_path.iterdir()):
        print(f"💾 Loading existing FAISS index from: {FAISS_INDEX_PATH}")
        local_vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        local_retriever = local_vector_store.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 local results
        print("✅ Local FAISS index loaded successfully.")
    elif guide_path.exists():
        print(f"⚠️ FAISS index not found at '{FAISS_INDEX_PATH}'.")
        print(f"🔧 Attempting to build FAISS index from '{LOCAL_GUIDE_PATH}'...")
        try:
            guide_content = guide_path.read_text(encoding='utf-8')
            # Split the document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, # Adjust chunk size as needed
                chunk_overlap=100, # Adjust overlap as needed
                length_function=len,
            )
            texts = text_splitter.split_text(guide_content)
            print(f"   - Split guide into {len(texts)} chunks.")

            # Create FAISS index from text chunks
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
        print(f"⚠️ FAISS index not found at '{FAISS_INDEX_PATH}' and guide file '{LOCAL_GUIDE_PATH}' not found.")
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

class DesignRequirements(BaseModel):
    title: str = Field(description="Brief title describing the design")
    shapes: List[ShapeRequirement] = Field(description="List of required shapes")
    operations: Optional[List[Operation]] = Field(None, description="List of Boolean operations to perform")
    comments: Optional[str] = Field(None, description="Additional comments or instructions")
    complexity_level: int = Field(description="Design complexity level (1-5)")

# Define templates for each chain - FIXED by properly escaping curly braces
requirement_analysis_template = """Analyze CAD design requirements from the description.

User description:
{user_description}

Please analyze the above description and return a JSON structure containing the following information:
1. Required shapes (box, cylinder, sphere, cone, etc.)
2. Dimensions for each shape
3. Relative position of shapes
4. Boolean operations to perform (cut, fuse, intersection, etc.)
5. Design complexity level (from 1-5)

Return a JSON structure strictly following this format:
```json
{{
  "title": "Brief description of the design",
  "shapes": [
    {{
      "shape_type": "box|cylinder|sphere|cone",
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
  "comments": "Additional comments or instructions",
  "complexity_level": 1-5
}}
```

IMPORTANT: Return only JSON, no explanations. Ensure the JSON is valid and follows the specified format.
Analyze the description carefully to identify all shapes, their precise dimensions, and relevant operations.
For highly complex designs (level 4-5), break down the design into multiple detailed shapes and operations.
"""

code_generation_template = """You are an expert FreeCAD scripter specializing in generating Python code to create highly detailed and complex 3D models based on design requirements.

**Analyzed design requirements:**
```json
{design_requirements}
```

**Retrieved Context (from web search, may be relevant):**
```
{retrieved_context}
```

**FreeCAD Workbench Guide Summary (Based on provided guide.txt):**

*   **Part Workbench:** Core for 3D modeling. Use for basic shapes (box, cylinder, sphere, cone, torus, wedge, prism, helix), boolean operations (cut, fuse, common, section), complex shapes (loft, sweep, extrusion, revolution, shell, solid, compound, filled face, offset, thickness), and basic curves/lines (circle, ellipse, polygon, spline, bspline, bezier). Commands typically start with `Part.`.
*   **PartDesign Workbench:** Feature-based modeling. Use for sketch-based features (pad, pocket, revolution, groove), dress-up features (fillet, chamfer, draft, thickness), patterns (linear, polar, multi-transform, scaled, mirrored), and advanced features (loft, pipe, additive/subtractive operations). Commands typically start with `PartDesign.`. Requires a `PartDesign.Body`.
*   **Draft Workbench:** Basic 2D/3D drawing and modification. Use for points, lines, wires, bsplines, bezier curves, circles, ellipses, rectangles, polygons, text, shape strings. Also includes 3D operations like extrude, move, rotate, scale, offset. Commands typically start with `Draft.`.
*   **Curve Workbench (Addon?):** Advanced curve manipulation. Use for blend curves, parametric curves, bspline approximations, curves on surfaces, pipeshells, sweeps. Commands may start with `Curve.`.
*   **Surface Workbench (Addon?):** Advanced surface modeling. Use for bspline/bezier surfaces, extrusion/revolved/loft/swept surfaces, blend surfaces, filling faces, curve network surfaces. Commands may start with `Surface.`.
*   **Mesh Workbench:** Working with mesh data (often from imports like STL). Use for creating mesh primitives, converting shapes to meshes, mesh repair (flip/harmonize normals), smoothing, refining. Commands typically start with `Mesh.`.
*   **Points Workbench (Addon?):** Working with point clouds. Use for creating/importing/exporting point clouds, converting to splines. Commands may start with `Points.`.
*   **Robot Workbench:** Robot simulation. Use for creating robots, trajectories. Commands typically start with `Robot.`.
*   **Assembly Workbench (A2plus, Assembly3/4):** Assembling parts. Syntax varies depending on the specific workbench used. General concepts involve creating assemblies, adding parts, and defining constraints.
*   **Arch Workbench:** Architectural modeling. Use for walls, structures (beams/columns), roofs, floors, buildings, sites, windows, doors, pipes, stairs, rebar. Commands typically start with `Arch.`. Often builds upon Draft objects.
*   **Path Workbench:** CNC path generation (CAM). Use for creating toolpaths like profiles, pockets, drilling operations based on geometry. Commands typically start with `Path.`.
*   **OpenSCAD Workbench:** Interaction with OpenSCAD. Use for creating polyhedrons, implicit functions, resizing. Commands typically start with `OpenSCAD.`.
*   **General Utilities:** Use `App.Vector`, `App.Placement`, `App.Rotation` for positioning. Use `obj.Placement`, `obj.ViewObject` for manipulating existing objects. Use `Import` and `Export` modules for file I/O.

**When generating code, select the most appropriate workbench and commands based on the design requirements and this guide.** For example, use PartDesign for feature-based modeling starting from sketches, Part for direct solid modeling and boolean operations, Draft for 2D elements or simple 3D arrangements, Arch for building elements, etc. Remember to import the necessary modules (e.g., `import PartDesign`, `import Draft`).

**Task:** Generate a complete and executable Python script for FreeCAD that accurately models the object described in the analyzed design requirements, potentially using insights from the retrieved context and the workbench guide above for complex features or techniques. Prioritize the design requirements, but use the context and guide for clarification or advanced methods if applicable.

**Mandatory requirements for the generated Python code:**

1.  **Imports:**
    *   Always start with `import FreeCAD as App` and `import Part`.
    *   Import `math` if mathematical calculations (trigonometry, constants) are needed.
    *   Import `FreeCADGui as Gui` ONLY if direct interaction with the GUI or view is explicitly required (rare for pure model generation).
    *   Import `Draft` if using Draft workbench tools (e.g., `makeWire`, `makePolygon`, `makeShapeString`).
    *   Import `Sketcher` for complex profile-based modeling.
    *   Import `PartDesign` for feature-based modeling when needed.

2.  **Document Handling:**
    *   Create a new, clean document: `doc = App.newDocument("GeneratedModel")`. (Or use `doc = App.ActiveDocument` if instructed to modify an existing one).

3.  **Shape Creation (Use appropriate tools):**
    *   **Primitives with Precision:** Use exact measurements for all primitives.
    *   **Complex Shapes:** Break down complex shapes into simpler components for higher accuracy.
    *   **Advanced Shape Tools:**
        *   Use `Part.BSplineCurve` and `Part.BSplineSurface` for complex curved surfaces.
        *   Implement `Part.makeShell` and `Part.makeSolid` for creating complex enclosures.
        *   Use `Part.makeFillet` and `Part.makeChamfer` for detailed edge treatment.
        *   Implement `Part.makeThickness` for hollow objects with precise wall thickness.
        *   Use `Part.makeOffsetShape` for creating offset surfaces with high precision.

4.  **High-Detail Features:**
    *   Implement threads, knurling, and patterned surfaces using mathematical formulas.
    *   Create arrays of features (circular patterns, linear patterns) for repeating elements.
    *   Implement complex blends between surfaces for smooth transitions.
    *   Use compound paths and multi-stage boolean operations for intricate details.
    *   Implement texturing or surface patterns where appropriate.

5.  **Positioning and Alignment:**
    *   Use precise coordinate systems with clear reference points.
    *   Implement parametric relationships between components.
    *   Use `FreeCAD.Placement` with accurate rotation matrices for complex orientations.
    *   Implement constraints between components when needed.

6.  **Boolean Operations (with Robustness):**
    *   Use multi-stage boolean operations for complex assemblies.
    *   Implement cleanup steps after boolean operations when needed.
    *   Use `Part.CompSolid` for complex assemblies when appropriate.
    *   Sequence operations carefully to maintain model integrity.

7.  **Optimization and Performance:**
    *   Use variables to store intermediate results to avoid recalculation.
    *   Group related operations into logical functions.
    *   Implement early checking for potentially problematic operations.
    *   Use appropriate tolerances for curved surfaces and complex operations.

8.  **Documentation and Parameterization:**
    *   Define key dimensions as variables at the top of the script.
    *   Include detailed comments explaining complex geometry creation.
    *   Structure code in logical sections (setup, base geometry, features, final operations).
    *   Document any mathematical formulas or algorithms used.

9.  **Finalizing the Model:**
    *   Call `doc.recompute()` after significant operations to ensure model integrity.
    *   Add a final recompute before completing the script.
    *   Set appropriate display properties for visualization if relevant.

**Output Format:**
Return *only* the generated Python code within a single code block. The code must be complete, highly detailed, and executable in FreeCAD without modifications.
"""

code_validation_template = """Check and validate the following FreeCAD Python code:

```python
{generated_code}
```

Please evaluate the code according to these criteria:
1. Can the code be executed in FreeCAD?
2. Does the code fulfill the design requirements?
3. Are there any syntax or logical errors?
4. Are there ways to optimize the code?

Return the evaluation results in JSON format:
```json
{{
  "executable": true/false,
  "meets_requirements": true/false,
  "errors": ["list of errors if any"],
  "optimization_suggestions": ["list of optimization suggestions if any"],
  "corrected_code": "if there are errors, provide corrected code"
}}
```

IMPORTANT: Return only JSON, no explanations.
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
    """Clean up the code string"""
    if "```python" in code_str:
        code_str = re.search(r'```python\s*(.*?)\s*```', code_str, re.DOTALL).group(1)
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
    """Creates a focused RAG query based on analyzed design requirements."""
    shape_types = []
    if design_reqs.shapes:
        shape_types = list(set([s.shape_type for s in design_reqs.shapes])) # Get unique shape types
        
    operation_types = []
    if design_reqs.operations:
        operation_types = list(set([o.operation_type for o in design_reqs.operations])) # Get unique operation types

    query_parts = ["FreeCAD Python script"]
    if shape_types:
        query_parts.append(f"for creating {' and '.join(shape_types)}")
    if operation_types:
        query_parts.append(f"using operations like {' and '.join(operation_types)}")
    
    query = " ".join(query_parts)
    
    # Fallback if no shapes/operations identified
    if not shape_types and not operation_types:
        # Use title as a fallback, or a generic term if no title
        fallback_term = design_reqs.title if design_reqs.title else 'CAD modeling'
        return f"FreeCAD Python script for {fallback_term}"
        
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
        """Process user request and generate FreeCAD code"""
        print(f"\n🔍 Analyzing request: '{user_text}'...")
        
        # Step 1: Analyze requirements
        try:
            design_requirements = self.requirement_analysis_chain.invoke(user_text)
            print(f"\n✅ Requirements analysis successful:")
            print(f"   Title: {design_requirements.title}")
            print(f"   Number of shapes: {len(design_requirements.shapes)}")
            print(f"   Complexity level: {design_requirements.complexity_level}/5")
            
            if design_requirements.operations:
                print(f"   Number of operations: {len(design_requirements.operations)}")
        except Exception as e:
            print(f"❌ Error analyzing requirements: {e}")
            return None, None, f"# Error: Unable to analyze requirements from description. Error details: {e}"
        
        # Step 2: Generate code based on requirements
        try:
            # Select the appropriate model based on complexity
            if design_requirements.complexity_level >= 4:
                model_name = MODELS["expert"]
                print(f"\n🔧 Generating high-detail FreeCAD code (using expert model {model_name})...")
            elif design_requirements.complexity_level >= 2:
                model_name = MODELS["advanced"]
                print(f"\n🔧 Generating complex FreeCAD code (using advanced model {model_name})...")
            else:
                model_name = MODELS["default"]
                print(f"\n🔧 Generating FreeCAD code (using default model {model_name})...")

            # Invoke the RAG chain, passing both requirements and original user text for context retrieval
            generated_code = self.code_generation_chain.invoke({
                "design_requirements": design_requirements,
                "user_text": user_text # Pass user_text for retrieval query
            })
            print(f"✅ Code generation complete (with RAG context)")

        except Exception as e:
            print(f"❌ Error generating code: {e}")
            return design_requirements, None, f"# Error: Unable to generate code from requirements. Error details: {e}"

        # Step 3: Validate and potentially fix the code
        try:
            print(f"\n🔍 Checking and validating code...")
            validation_result = self.code_validation_chain.invoke(generated_code)
            
            if validation_result["executable"] and validation_result["meets_requirements"]:
                print("✅ Code validated: Executable and meets design requirements")
                final_code = generated_code
            else:
                print("⚠️ Issues detected with code:")
                for error in validation_result["errors"]:
                    print(f"   - {error}")
                
                if validation_result["corrected_code"]:
                    print("🔧 Applying suggested fixes...")
                    final_code = validation_result["corrected_code"]
                else:
                    final_code = generated_code
                    print("⚠️ No suggested fixes, using original code")
            
            if validation_result["optimization_suggestions"]:
                print("\n💡 Optimization suggestions:")
                for suggestion in validation_result["optimization_suggestions"]:
                    print(f"   - {suggestion}")
        except Exception as e:
            print(f"⚠️ Error validating code: {e}")
            print("🔄 Continuing with unvalidated code...")
            final_code = generated_code
        
        # Step 4: Generate documentation removed as requested.

        print("\n✅ Code generation process complete!")
        # Return only design_requirements and final_code
        return design_requirements, final_code
    
    # Updated save_outputs signature to remove documentation parameter
    def save_outputs(self, code, design_requirements, base_filename="generated_cad"):
        """Save the generated code to a file"""
        # Create output directory if it doesn't exist
        output_dir = "cad_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate sanitized filename from title
        if design_requirements and design_requirements.title:
            sanitized_title = re.sub(r'[^\w\s-]', '', design_requirements.title).lower()
            sanitized_title = re.sub(r'[-\s]+', '-', sanitized_title).strip('-_')
            filename_base = f"{output_dir}/{sanitized_title}"
        else:
            filename_base = f"{output_dir}/{base_filename}"
        
        # Save code
        try:
            code_filename = f"{filename_base}.py"
            with open(code_filename, "w", encoding="utf-8") as file:
                file.write(code)
            print(f"\n💾 Code saved to file '{code_filename}'")
        except IOError as e:
            print(f"❌ Error saving code file: {e}")
        
        # Documentation and requirements saving removed as requested.
        
        print("\n--------------------------------------------------")
        print("📋 Usage instructions:")
        print("1. Open FreeCAD.")
        print("2. Go to 'Macro' menu -> 'Macros...'.")
        print(f"3. Click the 'Create' button and name your macro.")
        print(f"4. Paste the entire content from '{filename_base}.py' into the macro editor.")
        print("5. Save and close the editor.")
        print("6. Select the newly created macro from the list and click 'Execute'.")
        print("Or:")
        print("1. Open FreeCAD.")
        print("2. Open Python Console (View -> Panels -> Python console).")
        print(f"3. Paste the entire content from '{filename_base}.py' into the console and press Enter.")
        print("--------------------------------------------------")

if __name__ == "__main__":
    agent = TextToCADAgent()
    
    # Interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        print("\n🤖 Text-to-CAD Agent - Interactive Mode")
        print("👋 Welcome! Please describe the 3D object you want to create with FreeCAD.")
        print("💡 To exit, type 'exit' or 'quit'.")
        
        while True:
            print("\n" + "="*50)
            user_input = input("🔍 Your description: ")
            
            if user_input.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break
                
            if not user_input.strip():
                print("⚠️ Description cannot be empty. Please try again.")
                continue
                
            # Updated call to process_request (documentation removed)
            design_requirements, code = agent.process_request(user_input)
            
            if code:
                # Generate a filename based on the first few words of input
                filename_base = "_".join(user_input.split()[:3]).lower()
                filename_base = re.sub(r'[^\w\s-]', '', filename_base)
                filename_base = re.sub(r'[-\s]+', '_', filename_base).strip('_')
                
                # Updated call to save_outputs (documentation removed)
                agent.save_outputs(code, design_requirements, filename_base)
    else:
        # Process example requests
        example_requests = [
            """ Create a 3D model of a spur gear with the following specifications:

Number of teeth: 21

Module: 2 mm

Pressure angle: 20 degrees

Gear type: Spur (straight teeth)

Addendum: 2 mm

Dedendum: 2.5 mm

Pitch diameter: 42 mm

Outside diameter: 46 mm

Root diameter: 37 mm

Face width: 10 mm

Bore diameter (center hole): 10 mm """
        ]
        
        for i, request in enumerate(example_requests, 1):
            print(f"\n\n{'='*80}")
            print(f"📋 PROCESSING REQUEST {i}: '{request}'")
            print(f"{'='*80}")
            
            # Updated call to process_request (documentation removed)
            design_requirements, code = agent.process_request(request)
            
            if code:
                # Updated call to save_outputs (documentation removed)
                agent.save_outputs(code, design_requirements, f"example_{i}")
                
                print("\n🔍 CODE PREVIEW:")
                print("-" * 40)
                # Print first 10 lines of code
                print("\n".join(code.split("\n")[:10]) + "\n...")
                print("-" * 40)
