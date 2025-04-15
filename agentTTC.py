# -*- coding: utf-8 -*-
# Text-to-CAD Agent: Chuyển đổi mô tả văn bản thành mã FreeCAD
# Sử dụng LangChain và một LLM (GPT-4o) để phân tích yêu cầu và tạo mã FreeCAD

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("Lỗi: Vui lòng thiết lập biến môi trường OPENAI_API_KEY.")
    print("Bạn có thể tạo tệp .env trong cùng thư mục với script này và thêm dòng:")
    print("OPENAI_API_KEY='your_actual_api_key'")
    exit()

try:
    llm = ChatOpenAI(model="o3-mini",reasoning_effort="high", openai_api_key=openai_api_key)
    llm.invoke("Hello")
except Exception as e:
    print(f"Lỗi khi khởi tạo hoặc kết nối tới OpenAI: {e}")
    print("Hãy kiểm tra lại API key và kết nối mạng.")
    exit()

template = """Bạn là một trợ lý AI chuyên nghiệp, có khả năng chuyển đổi mô tả đối tượng 3D bằng ngôn ngữ tự nhiên (tiếng Việt) thành mã Python thực thi được trong môi trường FreeCAD.

**Nhiệm vụ:** Dựa vào mô tả của người dùng, hãy tạo ra một đoạn mã Python hoàn chỉnh để vẽ đối tượng 3D đó trong FreeCAD.

**Yêu cầu bắt buộc đối với mã được tạo:**
1.  **Import thư viện cần thiết:** Luôn bắt đầu bằng việc import `FreeCAD as App`, `Part`. Có thể cần `FreeCADGui as Gui` nếu có tương tác GUI, hoặc `math` cho các tính toán.
2.  **Tạo Document mới:** Sử dụng `doc = App.newDocument("GeneratedModel")` để tạo một tài liệu mới cho mô hình.
3.  **Tạo hình dạng cơ bản:** Sử dụng các hàm của module `Part` như `Part.makeBox(length, width, height)`, `Part.makeCylinder(radius, height)`, `Part.makeSphere(radius)`, `Part.makeCone(radius1, radius2, height)`.
4.  **Xác định Kích thước và Vị trí:**
    *   Phân tích kỹ mô tả để lấy đúng kích thước (chiều dài, rộng, cao, bán kính...). Nếu không rõ, hãy giả định một kích thước hợp lý (ví dụ: 10mm) và ghi chú lại trong comment.
    *   Sử dụng `FreeCAD.Vector(x, y, z)` để xác định vị trí và hướng. Đặt đối tượng tại gốc tọa độ (0,0,0) nếu không có vị trí cụ thể.
    *   Sử dụng `Placement` để di chuyển và xoay đối tượng: `obj.Placement = App.Placement(App.Vector(x, y, z), App.Rotation(yaw, pitch, roll))` hoặc `App.Rotation(App.Vector(axis_x, axis_y, axis_z), angle_degrees)`.
5.  **Thực hiện các Phép toán Boolean:**
    *   **Cắt (Cut/Difference):** Sử dụng `result = base_object.cut(tool_object)`. Ví dụ: tạo lỗ.
    *   **Hợp nhất (Fuse/Union):** Sử dụng `result = object1.fuse(object2)`. Ví dụ: ghép hai khối.
    *   **Giao (Common):** Sử dụng `result = object1.common(object2)`.
6.  **Thêm đối tượng vào Document:** Sử dụng `doc.addObject("Part::Feature", "ObjectName").Shape = generated_shape`. Đặt tên ("ObjectName") có ý nghĩa.
7.  **Cập nhật Document:** Kết thúc bằng `doc.recompute()` để FreeCAD tính toán và hiển thị kết quả.
8.  **Comment trong mã:** Thêm các comment giải thích các bước chính trong mã Python được tạo ra.
9.  **Chỉ trả về mã Python:** Không thêm bất kỳ lời giải thích nào bên ngoài khối mã. Toàn bộ phản hồi của bạn *chỉ* là mã Python.

**Ví dụ cách xử lý yêu cầu:**

*   **Mô tả người dùng:** "vẽ một khối hộp 10x20x5"
*   **Mã Python mong muốn:**
    ```python
    # Import necessary modules
    import FreeCAD as App
    import Part

    # Create a new document
    doc = App.newDocument("GeneratedModel")

    # Define dimensions for the box
    length = 10
    width = 20
    height = 5

    # Create the box
    box = Part.makeBox(length, width, height)

    # Add the box to the document
    doc.addObject("Part::Feature", "MyBox").Shape = box

    # Recompute the document to update the view
    doc.recompute()

    # Optional: Zoom to fit the object if GUI is available
    # try:
    #     import FreeCADGui as Gui
    #     Gui.SendMsgToActiveView("ViewFit")
    # except ImportError:
    #     print("FreeCADGui module not found. Skipping ViewFit.")
    ```

---
**Mô tả từ người dùng:**
{{user_description}}

---
**Mã Python FreeCAD được tạo:**
```python
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"user_description": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class TextToCADAgent:
    def __init__(self, chain_to_use):
        self.chain = chain_to_use

    def generate_freecad_code(self, user_text):
        print(f"\nĐang xử lý yêu cầu: '{user_text}'...")
        try:
            freecad_code = self.chain.invoke(user_text)

            if freecad_code.strip().startswith("```python"):
                freecad_code = freecad_code.strip()[9:]
            if freecad_code.strip().endswith("```"):
                freecad_code = freecad_code.strip()[:-3]

            if "import FreeCAD" not in freecad_code or "import Part" not in freecad_code:
                 print("Cảnh báo: Mã được tạo có vẻ không hợp lệ (thiếu import FreeCAD/Part).")
                 return freecad_code.strip()

            print("Đã tạo mã FreeCAD thành công.")
            return freecad_code.strip()

        except Exception as e:
            print(f"Đã xảy ra lỗi trong quá trình tạo mã: {e}")
            return f"# Lỗi: Không thể tạo mã từ mô tả. Chi tiết lỗi: {e}"

    def save_code_to_file(self, code, filename="generated_cad_model.py"):
        try:
            with open(filename, "w", encoding="utf-8") as file:
                file.write(code)
            print(f"\nĐã lưu mã vào tệp '{filename}'")
            print("--------------------------------------------------")
            print("Để sử dụng mã này trong FreeCAD:")
            print("1. Mở FreeCAD.")
            print("2. Đi tới menu 'Macro' -> 'Macros...'.")
            print(f"3. Nhấn nút 'Create' và đặt tên cho macro (ví dụ: MyGeneratedModel).")
            print(f"4. Dán toàn bộ nội dung từ tệp '{filename}' vào trình soạn thảo macro.")
            print("5. Lưu lại và đóng trình soạn thảo.")
            print("6. Chọn macro vừa tạo trong danh sách và nhấn 'Execute'.")
            print("Hoặc:")
            print("1. Mở FreeCAD.")
            print("2. Mở Python Console (View -> Panels -> Python console).")
            print(f"3. Dán toàn bộ nội dung từ tệp '{filename}' vào console và nhấn Enter.")
            print("--------------------------------------------------")

        except IOError as e:
            print(f"Lỗi khi lưu tệp '{filename}': {e}")

if __name__ == "__main__":
    agent = TextToCADAgent(chain)

    # --- Yêu cầu Ví dụ ---
    # Yêu cầu 1: Đơn giản
    # request1 = "vẽ một hình cầu bán kính 15mm"
    # print(f"--- Xử lý Yêu cầu 1: '{request1}' ---")
    # code1 = agent.generate_freecad_code(request1)
    # print("\nMã FreeCAD được tạo cho Yêu cầu 1:")
    # print(code1)
    # agent.save_code_to_file(code1, "generated_sphere.py")

    # Yêu cầu 2: Phức tạp hơn với phép toán cắt
    request2 = "tạo một khối hộp kích thước 50x50x20 mm. Sau đó, tạo một lỗ hình trụ xuyên qua tâm của mặt trên cùng, lỗ có đường kính 10mm."
    print(f"\n--- Xử lý Yêu cầu 2: '{request2}' ---")
    code2 = agent.generate_freecad_code(request2)
    print("\nMã FreeCAD được tạo cho Yêu cầu 2:")
    print(code2)
    agent.save_code_to_file(code2, "generated_box_with_hole.py")

    # Yêu cầu 3: Thử một yêu cầu khác
    # request3 = "vẽ một hình nón có bán kính đáy 20, bán kính đỉnh 5 và chiều cao 40"
    # print(f"\n--- Xử lý Yêu cầu 3: '{request3}' ---")
    # code3 = agent.generate_freecad_code(request3)
    # print("\nMã FreeCAD được tạo cho Yêu cầu 3:")
    # print(code3)
    # agent.save_code_to_file(code3, "generated_cone.py")