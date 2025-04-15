# -*- coding: utf-8 -*-
# Enhanced Text-to-CAD Agent: Chuyển đổi mô tả văn bản thành mã FreeCAD
# Sử dụng LangChain và LLMs để phân tích yêu cầu và tạo mã FreeCAD
# Bổ sung các chuỗi xử lý phức tạp hơn

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
    print("Lỗi: Vui lòng thiết lập biến môi trường OPENAI_API_KEY.")
    print("Bạn có thể tạo tệp .env trong cùng thư mục với script này và thêm dòng:")
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
    advanced_llm = ChatOpenAI(model=MODELS["default"], reasoning_effort="high", openai_api_key=openai_api_key)
    # Test connection
    default_llm.invoke("Test connection")
except Exception as e:
    print(f"Lỗi khi khởi tạo hoặc kết nối tới OpenAI: {e}")
    print("Hãy kiểm tra lại API key và kết nối mạng.")
    exit()

# Pydantic models for structured output
class ShapeRequirement(BaseModel):
    shape_type: str = Field(description="Loại hình học (box, cylinder, sphere, cone, etc.)")
    dimensions: Dict[str, float] = Field(description="Kích thước của hình (ví dụ: length, width, height, radius)")
    position: Optional[List[float]] = Field(None, description="Vị trí [x, y, z]")
    rotation: Optional[List[float]] = Field(None, description="Góc xoay [xrot, yrot, zrot] tính bằng độ")

class Operation(BaseModel):
    operation_type: str = Field(description="Loại phép toán Boolean (cut, fuse, common)")
    base_shape: str = Field(description="Tên của hình cơ sở")
    tool_shape: str = Field(description="Tên của hình công cụ")
    result_name: str = Field(description="Tên của kết quả sau phép toán")

class DesignRequirements(BaseModel):
    title: str = Field(description="Tiêu đề mô tả ngắn gọn về thiết kế")
    shapes: List[ShapeRequirement] = Field(description="Danh sách các hình được yêu cầu")
    operations: Optional[List[Operation]] = Field(None, description="Danh sách các phép toán Boolean cần thực hiện")
    comments: Optional[str] = Field(None, description="Nhận xét hoặc hướng dẫn bổ sung")
    complexity_level: int = Field(description="Mức độ phức tạp của thiết kế (1-5)")

# Define templates for each chain - FIXED by properly escaping curly braces
requirement_analysis_template = """Phân tích yêu cầu thiết kế CAD từ mô tả bằng tiếng Việt.

Mô tả từ người dùng:
{user_description}

Hãy phân tích mô tả trên và trả về một cấu trúc JSON chứa các thông tin sau:
1. Các hình dạng cần thiết (hình hộp, hình trụ, hình cầu, hình nón, v.v.)
2. Kích thước cho mỗi hình
3. Vị trí tương đối của các hình
4. Các phép toán Boolean cần thực hiện (cắt, hợp nhất, giao, v.v.)
5. Mức độ phức tạp của thiết kế (từ 1-5)

Trả về cấu trúc JSON tuân theo định dạng nghiêm ngặt sau:
```json
{{
  "title": "Mô tả ngắn gọn về thiết kế",
  "shapes": [
    {{
      "shape_type": "box|cylinder|sphere|cone",
      "dimensions": {{"length": 10, "width": 20, "height": 5}} hoặc {{"radius": 15, "height": 30}},
      "position": [x, y, z],
      "rotation": [xrot, yrot, zrot]
    }}
  ],
  "operations": [
    {{
      "operation_type": "cut|fuse|common",
      "base_shape": "tên hình cơ sở",
      "tool_shape": "tên hình công cụ",
      "result_name": "tên kết quả"
    }}
  ],
  "comments": "Nhận xét hoặc hướng dẫn bổ sung",
  "complexity_level": 1-5
}}
```

QUAN TRỌNG: Chỉ trả về JSON, không thêm giải thích. Đảm bảo JSON hợp lệ và tuân thủ định dạng đã nêu.
"""

code_generation_template = """Bạn là một chuyên gia CAD, chuyên tạo mã Python cho FreeCAD từ các yêu cầu thiết kế.

**Yêu cầu thiết kế đã được phân tích:**
{design_requirements}

**Nhiệm vụ:** Dựa vào yêu cầu thiết kế đã phân tích, hãy tạo ra một đoạn mã Python hoàn chỉnh để vẽ đối tượng 3D đó trong FreeCAD.

**Yêu cầu bắt buộc đối với mã được tạo:**
1. **Import thư viện cần thiết:** Luôn bắt đầu bằng việc import `FreeCAD as App`, `Part`. Có thể cần `FreeCADGui as Gui` nếu có tương tác GUI, hoặc `math` cho các tính toán.
2. **Tạo Document mới:** Sử dụng `doc = App.newDocument("GeneratedModel")` để tạo một tài liệu mới cho mô hình.
3. **Tạo hình dạng cơ bản:** Sử dụng các hàm của module `Part` như `Part.makeBox()`, `Part.makeCylinder()`, `Part.makeSphere()`, `Part.makeCone()`.
4. **Xác định Kích thước và Vị trí:**
   * Sử dụng các kích thước và vị trí từ yêu cầu thiết kế.
   * Sử dụng `FreeCAD.Vector(x, y, z)` để xác định vị trí và hướng.
   * Sử dụng `Placement` để di chuyển và xoay đối tượng.
5. **Thực hiện các Phép toán Boolean:**
   * **Cắt (Cut/Difference):** Sử dụng `result = base_object.cut(tool_object)`.
   * **Hợp nhất (Fuse/Union):** Sử dụng `result = object1.fuse(object2)`.
   * **Giao (Common):** Sử dụng `result = object1.common(object2)`.
6. **Thêm đối tượng vào Document:** Sử dụng `doc.addObject("Part::Feature", "ObjectName").Shape = generated_shape`.
7. **Cập nhật Document:** Kết thúc bằng `doc.recompute()`.
8. **Comment trong mã:** Thêm các comment giải thích các bước chính.

**Chỉ trả về mã Python:** Không thêm lời giải thích nào bên ngoài khối mã.
"""

code_validation_template = """Kiểm tra và xác thực mã Python FreeCAD sau:

```python
{generated_code}
```

Hãy đánh giá mã theo các tiêu chí sau:
1. Mã có thể thực thi trong FreeCAD không?
2. Mã có thực hiện đúng yêu cầu thiết kế không?
3. Có lỗi cú pháp hoặc lỗi logic không?
4. Có cách tối ưu hóa mã không?

Trả về kết quả đánh giá theo định dạng JSON:
```json
{{
  "executable": true/false,
  "meets_requirements": true/false,
  "errors": ["danh sách lỗi nếu có"],
  "optimization_suggestions": ["danh sách đề xuất tối ưu hóa nếu có"],
  "corrected_code": "nếu có lỗi, cung cấp mã đã sửa"
}}
```

QUAN TRỌNG: Chỉ trả về JSON, không thêm giải thích.
"""

documentation_template = """Tạo tài liệu hướng dẫn cho mã FreeCAD sau:

```python
{final_code}
```

Tài liệu nên bao gồm:
1. Tóm tắt về mô hình 3D được tạo
2. Giải thích từng phần chính của mã
3. Các tham số quan trọng có thể điều chỉnh
4. Hướng dẫn cách chạy mã trong FreeCAD

Định dạng tài liệu bằng Markdown với các mục có tiêu đề rõ ràng.
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
            json_str = re.search(r'```json\s*(.*?)\s*```', json_str, re.DOTALL).group(1)
        
        data = json.loads(json_str)
        return DesignRequirements(**data)
    except Exception as e:
        print(f"Lỗi khi chuyển đổi JSON sang Pydantic: {e}")
        # Return a minimal valid object
        return DesignRequirements(
            title="Không thể phân tích yêu cầu",
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
        print(f"Lỗi khi xử lý kết quả xác thực: {e}")
        return {
            "executable": False,
            "meets_requirements": False,
            "errors": [f"Lỗi phân tích kết quả xác thực: {e}"],
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
        print(f"\n🔍 Đang phân tích yêu cầu: '{user_text}'...")
        
        # Step 1: Analyze requirements
        try:
            design_requirements = self.requirement_analysis_chain.invoke(user_text)
            print(f"\n✅ Phân tích yêu cầu thành công:")
            print(f"   Tiêu đề: {design_requirements.title}")
            print(f"   Số lượng hình: {len(design_requirements.shapes)}")
            print(f"   Mức độ phức tạp: {design_requirements.complexity_level}/5")
            
            if design_requirements.operations:
                print(f"   Số lượng phép toán: {len(design_requirements.operations)}")
        except Exception as e:
            print(f"❌ Lỗi khi phân tích yêu cầu: {e}")
            return None, None, f"# Lỗi: Không thể phân tích yêu cầu từ mô tả. Chi tiết lỗi: {e}"
        
        # Step 2: Generate code based on requirements
        try:
            print(f"\n🔧 Đang tạo mã FreeCAD (sử dụng model {MODELS['advanced'] if design_requirements.complexity_level >= 3 else MODELS['default']})...")
            generated_code = self.code_generation_chain.invoke({"design_requirements": design_requirements})
        except Exception as e:
            print(f"❌ Lỗi khi tạo mã: {e}")
            return design_requirements, None, f"# Lỗi: Không thể tạo mã từ yêu cầu. Chi tiết lỗi: {e}"
        
        # Step 3: Validate and potentially fix the code
        try:
            print(f"\n🔍 Đang kiểm tra và xác thực mã...")
            validation_result = self.code_validation_chain.invoke(generated_code)
            
            if validation_result["executable"] and validation_result["meets_requirements"]:
                print("✅ Mã được xác thực: Có thể thực thi và đáp ứng yêu cầu thiết kế")
                final_code = generated_code
            else:
                print("⚠️ Phát hiện vấn đề với mã:")
                for error in validation_result["errors"]:
                    print(f"   - {error}")
                
                if validation_result["corrected_code"]:
                    print("🔧 Đang áp dụng sửa đổi được đề xuất...")
                    final_code = validation_result["corrected_code"]
                else:
                    final_code = generated_code
                    print("⚠️ Không có sửa đổi được đề xuất, sử dụng mã gốc")
            
            if validation_result["optimization_suggestions"]:
                print("\n💡 Đề xuất tối ưu hóa:")
                for suggestion in validation_result["optimization_suggestions"]:
                    print(f"   - {suggestion}")
        except Exception as e:
            print(f"⚠️ Lỗi khi xác thực mã: {e}")
            print("🔄 Tiếp tục với mã chưa được xác thực...")
            final_code = generated_code
        
        # Step 4: Generate documentation
        try:
            print(f"\n📝 Đang tạo tài liệu hướng dẫn...")
            documentation = self.documentation_chain.invoke(final_code)
        except Exception as e:
            print(f"⚠️ Lỗi khi tạo tài liệu: {e}")
            documentation = "# Không thể tạo tài liệu hướng dẫn"
        
        print("\n✅ Quá trình tạo mã hoàn tất!")
        return design_requirements, documentation, final_code
    
    def save_outputs(self, code, documentation, design_requirements, base_filename="generated_cad"):
        """Save all outputs to files"""
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
            print(f"\n💾 Đã lưu mã vào tệp '{code_filename}'")
        except IOError as e:
            print(f"❌ Lỗi khi lưu tệp mã: {e}")
        
        # Save documentation
        try:
            doc_filename = f"{filename_base}_documentation.md"
            with open(doc_filename, "w", encoding="utf-8") as file:
                file.write(documentation)
            print(f"💾 Đã lưu tài liệu vào tệp '{doc_filename}'")
        except IOError as e:
            print(f"❌ Lỗi khi lưu tệp tài liệu: {e}")
        
        # Save requirements as JSON
        if design_requirements:
            try:
                req_filename = f"{filename_base}_requirements.json"
                with open(req_filename, "w", encoding="utf-8") as file:
                    file.write(design_requirements.json(indent=2))
                print(f"💾 Đã lưu yêu cầu thiết kế vào tệp '{req_filename}'")
            except IOError as e:
                print(f"❌ Lỗi khi lưu tệp yêu cầu: {e}")
        
        print("\n--------------------------------------------------")
        print("📋 Hướng dẫn sử dụng:")
        print("1. Mở FreeCAD.")
        print("2. Đi tới menu 'Macro' -> 'Macros...'.")
        print(f"3. Nhấn nút 'Create' và đặt tên cho macro.")
        print(f"4. Dán toàn bộ nội dung từ tệp '{filename_base}.py' vào trình soạn thảo macro.")
        print("5. Lưu lại và đóng trình soạn thảo.")
        print("6. Chọn macro vừa tạo trong danh sách và nhấn 'Execute'.")
        print("Hoặc:")
        print("1. Mở FreeCAD.")
        print("2. Mở Python Console (View -> Panels -> Python console).")
        print(f"3. Dán toàn bộ nội dung từ tệp '{filename_base}.py' vào console và nhấn Enter.")
        print("--------------------------------------------------")

if __name__ == "__main__":
    agent = TextToCADAgent()
    
    # Interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        print("\n🤖 Text-to-CAD Agent - Chế độ tương tác")
        print("👋 Chào mừng! Hãy mô tả đối tượng 3D bạn muốn tạo bằng FreeCAD.")
        print("💡 Để thoát, nhập 'exit' hoặc 'quit'.")
        
        while True:
            print("\n" + "="*50)
            user_input = input("🔍 Mô tả của bạn: ")
            
            if user_input.lower() in ["exit", "quit", "thoát"]:
                print("👋 Tạm biệt!")
                break
                
            if not user_input.strip():
                print("⚠️ Mô tả không được để trống. Vui lòng thử lại.")
                continue
                
            design_requirements, documentation, code = agent.process_request(user_input)
            
            if code:
                # Generate a filename based on the first few words of input
                filename_base = "_".join(user_input.split()[:3]).lower()
                filename_base = re.sub(r'[^\w\s-]', '', filename_base)
                filename_base = re.sub(r'[-\s]+', '_', filename_base).strip('_')
                
                agent.save_outputs(code, documentation, design_requirements, filename_base)
    else:
        # Process example requests
        example_requests = [
            "Vẽ một hình hộp chữ nhật có 5 cái lỗ nằm trên một hàng và cách đều ở giữa"
        ]
        
        for i, request in enumerate(example_requests, 1):
            print(f"\n\n{'='*80}")
            print(f"📋 XỬ LÝ YÊU CẦU {i}: '{request}'")
            print(f"{'='*80}")
            
            design_requirements, documentation, code = agent.process_request(request)
            
            if code:
                agent.save_outputs(code, documentation, design_requirements, f"example_{i}")
                
                print("\n🔍 XEM TRƯỚC MÃ:")
                print("-" * 40)
                # Print first 10 lines of code
                print("\n".join(code.split("\n")[:10]) + "\n...")
                print("-" * 40)