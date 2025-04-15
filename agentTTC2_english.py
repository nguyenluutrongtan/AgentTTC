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
from typing import List, Optional, Dict
import sys
import json
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
    "default": "gpt-4.1-mini-2025-04-14",
    "advanced": "deepseek-chat",  # Using GPT-4o for more complex designs and higher accuracy
    "expert": "deepseek-chat",  # For extremely detailed and complex designs
}

# Initialize models
try:
    default_llm = ChatDeepSeek(model=MODELS["advanced"], temperature=0)
    advanced_llm = ChatDeepSeek(model=MODELS["advanced"], temperature=0)
    expert_llm = ChatDeepSeek(model=MODELS["advanced"], temperature=0)

    # default_llm = ChatOpenAI(model=MODELS["advanced"], reasoning_effort="high")
    # advanced_llm = ChatOpenAI(model=MODELS["advanced"], reasoning_effort="high")
    # expert_llm = ChatOpenAI(model=MODELS["advanced"], reasoning_effort="high")
    # Test connection
    default_llm.invoke("Test connection")
except Exception as e:
    print(f"Error initializing or connecting to OpenAI: {e}")
    print("Please check your API key and network connection.")
    exit()

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

**Task:** Generate a complete and executable Python script for FreeCAD that accurately models the object described in the analyzed design requirements, potentially using insights from the retrieved context for complex features or techniques. Prioritize the design requirements, but use the context for clarification or advanced methods if applicable.

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
    return "\n\n".join([f"--- Context Source {i+1} ---\n{doc.get('content', 'Error: Content key not found in retrieved document.')}" for i, doc in enumerate(docs)])

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
# Setup for parallel execution: retrieve context and pass requirements
rag_setup = RunnableParallel(
    {
        # Process the retrieved documents using the helper function
        "retrieved_context": (
            # Use the new function to generate the query from design_requirements
            (lambda x: create_rag_query(x["design_requirements"]))
            | retriever
            | RunnableLambda(format_retrieved_context) # Apply formatting function
        ) if retriever else (lambda x: "Tavily API key not set. Context retrieval disabled."),

        "design_requirements": (lambda x: x["design_requirements"]),
        # Keep user_text in the parallel step output if needed elsewhere, 
        # but it's no longer directly used for the retriever query.
        "user_text": (lambda x: x["user_text"]) 
    }
)

# Define the RAG chain
rag_code_generation_chain = (
    rag_setup
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
            "rubik cube 3x3"
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
