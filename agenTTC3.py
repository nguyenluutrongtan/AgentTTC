# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
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

if not openai_api_key:
    print("Error: Please set the OPENAI_API_KEY environment variable.")
    print("You can create a .env file in the same directory as this script and add the line:")
    print("OPENAI_API_KEY='your_actual_api_key'")
    exit()

# Define model choices
MODELS = {
    "default": "o3-mini",
    "advanced": "o3-mini",  # For more complex designs
}

# Initialize models
try:
    default_llm = ChatOpenAI(model=MODELS["default"], reasoning_effort="high", openai_api_key=openai_api_key)
    advanced_llm = ChatOpenAI(model=MODELS["advanced"], reasoning_effort="high", openai_api_key=openai_api_key)
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
requirement_analysis_template = """Analyze CAD design requirements from the description in Vietnamese.

User description:
{user_description}

Please analyze the above description and return a JSON structure containing the following information:
1.  **Required shapes:** Identify all the basic shapes required (e.g., box, cylinder, sphere, cone).
2.  **Dimensions:** Extract the dimensions for each shape (e.g., length, width, height, radius). If dimensions are not explicitly provided, make reasonable assumptions and include them in the "comments" field.
3.  **Positions:** Determine the position of each shape in 3D space. If no position is specified, assume the shape is located at the origin (0, 0, 0).
4.  **Boolean operations:** Identify any boolean operations that need to be performed (e.g., cut, fuse, common). Specify the base shape, the tool shape, and the name of the resulting shape.
5.  **Complexity level:** Assess the complexity level of the design on a scale of 1 to 5, where 1 is a simple design and 5 is a very complex design.

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

**Example:**

**User description:** "vẽ một khối hộp 10x20x5 với một lỗ hình trụ đường kính 2 ở giữa"

**JSON output:**
```json
{{
  "title": "Khối hộp với lỗ hình trụ",
  "shapes": [
    {{
      "shape_type": "box",
      "dimensions": {{"length": 10, "width": 20, "height": 5}},
      "position": [0, 0, 0],
      "rotation": [0, 0, 0]
    }},
    {{
      "shape_type": "cylinder",
      "dimensions": {{"radius": 1, "height": 5}},
      "position": [0, 0, 0],
      "rotation": [0, 0, 0]
    }}
  ],
  "operations": [
    {{
      "operation_type": "cut",
      "base_shape": "box",
      "tool_shape": "cylinder",
      "result_name": "box_with_hole"
    }}
  ],
  "comments": "Lỗ hình trụ nằm ở giữa khối hộp",
  "complexity_level": 3
}}
```

IMPORTANT: Return only JSON, no explanations. Ensure the JSON is valid and follows the specified format.
"""

code_generation_template = """You are a CAD expert, specializing in creating Python code for FreeCAD from design requirements.

**Analyzed design requirements:**
{design_requirements}

**Task:** Based on the analyzed design requirements, create a complete Python code to draw that 3D object in FreeCAD.

**Mandatory requirements for the generated code:**

1.  **Import necessary libraries:** Always start by importing `FreeCAD as App` and `Part`. Import `FreeCADGui as Gui` if the code interacts with the GUI (e.g., zooming to fit the view using `Gui.SendMsgToActiveView("ViewFit")`). Import `math` for mathematical calculations.
2.  **Create a new document:** Use `doc = App.newDocument("GeneratedModel")` to create a new document for the model. The document name should be "GeneratedModel".
3.  **Create basic shapes:** Use functions from the `Part` module to create basic shapes like boxes, cylinders, spheres, and cones.
    *   `Part.makeBox(length, width, height)`: Creates a box with the specified dimensions.  Ensure the units are consistent (e.g., mm).
    *   `Part.makeCylinder(radius, height)`: Creates a cylinder with the specified radius and height. The cylinder is created along the Z-axis by default.
    *   `Part.makeSphere(radius)`: Creates a sphere with the specified radius.
    *   `Part.makeCone(radius1, radius2, height)`: Creates a cone with the specified base radius (`radius1`), top radius (`radius2`), and height.
4.  **Define dimensions and positions:**
    *   Use the dimensions and positions provided in the design requirements. If a dimension or position is not specified, use a reasonable default value (e.g., 10mm for length, (0, 0, 0) for position). Add a comment to indicate that a default value was used.
    *   Use `FreeCAD.Vector(x, y, z)` to define positions and directions.  Remember that FreeCAD uses a right-handed coordinate system.
    *   Use `App.Placement(App.Vector(x, y, z), App.Rotation(yaw, pitch, roll))` to move and rotate objects. The rotation is specified in degrees. Alternatively, use `App.Rotation(App.Vector(axis_x, axis_y, axis_z), angle_degrees)` for rotation around an axis.
5.  **Perform boolean operations:**
    *   Use boolean operations to combine or subtract shapes. Ensure that the objects are properly positioned before performing the boolean operation.
    *   `base_object.cut(tool_object)`: Cuts the `tool_object` from the `base_object`. The result is a new shape.
    *   `object1.fuse(object2)`: Fuses `object1` and `object2` together. The result is a single combined shape.
    *   `object1.common(object2)`: Creates a new shape from the intersection of `object1` and `object2`.
6.  **Add objects to the document:**
    *   Use `doc.addObject("Part::Feature", "ObjectName").Shape = generated_shape` to add the generated shape to the document. Give the object a descriptive name that reflects its purpose. The "ObjectName" should be a string.
7.  **Update the document:**
    *   Use `doc.recompute()` to update the document and display the changes. This is necessary for the changes to be visible in the FreeCAD GUI.
8.  **Add comments to the code:**
    *   Add comments to explain the purpose of each step in the code. This will make the code easier to understand and maintain. Use clear and concise language.

**Example:**

```python
# Import necessary modules
import FreeCAD as App
import Part

# Create a new document
doc = App.newDocument("GeneratedModel")

# Create a box with dimensions 10x20x5 mm
box = Part.makeBox(10, 20, 5)

# Create a cylinder with radius 2mm and height 5mm
cylinder = Part.makeCylinder(2, 5)

# Move the cylinder to position (5, 10, 0)
cylinder.Placement = App.Placement(App.Vector(5, 10, 0), App.Rotation(0, 0, 0))

# Cut the cylinder from the box
result = box.cut(cylinder)

# Add the result to the document
doc.addObject("Part::Feature", "BoxWithHole").Shape = result

# Recompute the document
doc.recompute()

# Optional: Zoom to fit the object in the view
# try:
#     import FreeCADGui as Gui
#     Gui.SendMsgToActiveView("ViewFit")
# except ImportError:
#     print("FreeCADGui module not found. Skipping ViewFit.")
```

**Return only Python code:** Do not add any explanations outside the code block.
"""

code_validation_template = """Check and validate the following FreeCAD Python code:

```python
{generated_code}
```

Please evaluate the code according to these criteria:

1.  **Executable:** Can the code be executed in FreeCAD without errors? Check for syntax errors, missing imports, and incorrect function calls.
2.  **Meets requirements:** Does the code fulfill the design requirements specified in the analyzed design requirements? Check if the generated 3D object has the correct shapes, dimensions, positions, and boolean operations.
3.  **Errors:** Identify any syntax errors, logical errors, or runtime errors in the code. Provide a detailed description of each error.
4.  **Optimization suggestions:** Suggest ways to optimize the code for performance, readability, and maintainability. This could include using more efficient algorithms, simplifying the code structure, or adding comments.
5.  **Corrected code:** If there are any errors, provide a corrected version of the code that fixes the errors and meets the design requirements.

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

**Example:**

```json
{{
  "executable": true,
  "meets_requirements": true,
  "errors": [],
  "optimization_suggestions": ["Add comments to explain the purpose of each step in the code."],
  "corrected_code": null
}}
```

IMPORTANT: Return only JSON, no explanations.
"""

documentation_template = """Create documentation for the following FreeCAD code:

```python
{final_code}
```

The documentation should include:

1.  **Summary:** A brief summary of the 3D model created by the code.
2.  **Explanation:** A detailed explanation of each step in the code, including the purpose of each function call and the values of important parameters.
3.  **Parameters:** A list of the important parameters that can be adjusted to modify the design, along with a description of their purpose and valid values.
4.  **Usage:** Instructions on how to run the code in FreeCAD, including how to open the Python console and execute the code.

The documentation should be formatted in Markdown.
"""

# Initialize templates
requirement_analysis_prompt = ChatPromptTemplate.from_template(requirement_analysis_template)
code_generation_prompt = ChatPromptTemplate.from_template(code_generation_template)
code_validation_prompt = ChatPromptTemplate.from_template(code_validation_template)
documentation_prompt = ChatPromptTemplate.from_template(documentation_template)

# Define helper functions
def json_to_pydantic(json_str: str) -> DesignRequirements:
    """Convert JSON string to Pydantic model"""
    try:
        # Clean up the JSON string if needed
        if "```json" in json_str:
            json_str = re.search(r'```json\\s*(.*?)\\s*```', json_str, re.DOTALL).group(1)
        
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
        code_str = re.search(r'```python\\s*(.*?)\\s*```', code_str, re.DOTALL).group(1)
    return code_str.strip()

def process_validation_result(validation_result: str) -> dict:
    """Process validation result JSON"""
    try:
        if "```json" in validation_result:
            validation_result = re.search(r'```json\\s*(.*?)\\s*```', validation_result, re.DOTALL).group(1)
        
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

# Define chains
requirement_analysis_chain = (
    {"user_description": RunnablePassthrough()}
    | requirement_analysis_prompt
    | advanced_llm
    | StrOutputParser()
    | RunnableLambda(json_to_pydantic)
)

def select_model_by_complexity(inputs):
    """Select model based on complexity level"""
    try:
        complexity = inputs.get("complexity_level", 1)
        return advanced_llm if complexity >= 3 else default_llm
    except:
        # Default to simpler model if we can't determine complexity
        return default_llm

code_generation_chain = (
    code_generation_prompt
    | advanced_llm  # Using advanced_llm for all cases to avoid errors
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

documentation_chain = (
    {"final_code": RunnablePassthrough()}
    | documentation_prompt
    | default_llm
    | StrOutputParser()
)

class TextToCADAgent:
    def __init__(self):
        self.requirement_analysis_chain = requirement_analysis_chain
        self.code_generation_chain = code_generation_chain
        self.code_validation_chain = code_validation_chain
        self.documentation_chain = documentation_chain
    
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
            print(f"\n🔧 Generating FreeCAD code (using model {MODELS['advanced'] if design_requirements.complexity_level >= 3 else MODELS['default']})...")
            generated_code = self.code_generation_chain.invoke({"design_requirements": design_requirements})
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
        
        # Step 4: Generate documentation
        try:
            print(f"\n📝 Generating documentation...")
            documentation = self.documentation_chain.invoke(final_code)
        except Exception as e:
            print(f"⚠️ Error generating documentation: {e}")
            documentation = "# Unable to generate documentation"
        
        print("\n✅ Code generation process complete!")
        return design_requirements, documentation, final_code
    
    def save_outputs(self, code, documentation, design_requirements, base_filename="generated_cad"):
        """Save all outputs to files"""
        # Create output directory if it doesn't exist
        output_dir = "cad_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate sanitized filename from title
        if design_requirements and design_requirements.title:
            sanitized_title = re.sub(r'[^\\w\\s-]', '', design_requirements.title).lower()
            sanitized_title = re.sub(r'[-\\s]+', '-', sanitized_title).strip('-_')
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
        
        # Save documentation
        try:
            doc_filename = f"{filename_base}_documentation.md"
            with open(doc_filename, "w", encoding="utf-8") as file:
                file.write(documentation)
            print(f"💾 Documentation saved to file '{doc_filename}'")
        except IOError as e:
            print(f"❌ Error saving documentation file: {e}")
        
        # Save requirements as JSON
        if design_requirements:
            try:
                req_filename = f"{filename_base}_requirements.json"
                with open(req_filename, "w", encoding="utf-8") as file:
                    file.write(design_requirements.json(indent=2))
                print(f"💾 Design requirements saved to file '{req_filename}'")
            except IOError as e:
                print(f"❌ Error saving requirements file: {e}")
        
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
                
            design_requirements, documentation, code = agent.process_request(user_input)
            
            if code:
                # Generate a filename based on the first few words of input
                filename_base = "_".join(user_input.split()[:3]).lower()
                filename_base = re.sub(r'[^\\w\\s-]', '', filename_base)
                filename_base = re.sub(r'[-\\s]+', '_', filename_base).strip('_')
                
                agent.save_outputs(code, documentation, design_requirements, filename_base)
    else:
        # Process example requests
        example_requests = [
            "A 3x6 lego"
        ]
        
        for i, request in enumerate(example_requests, 1):
            print(f"\n\n{'='*80}")
            print(f"📋 PROCESSING REQUEST {i}: '{request}'")
            print(f"{'='*80}")
            
            design_requirements, documentation, code = agent.process_request(request)
            
            if code:
                agent.save_outputs(code, documentation, design_requirements, f"example_{i}")
                
                print("\n🔍 CODE PREVIEW:")
                print("-" * 40)
                # Print first 10 lines of code
                print("\n".join(code.split("\n")[:10]) + "\n...")
                print("-" * 40)
