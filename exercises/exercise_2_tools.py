"""
Bài Tập 2: Thêm Tools và Knowledge Base

Hoàn thành các TODO để thêm tool và knowledge base entry mới.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from common.llm import get_llm

# Knowledge base
LEGAL_KNOWLEDGE = [
    {
        "id": "ucc_breach",
        "keywords": ["breach", "contract", "remedies", "damages", "ucc"],
        "text": (
            "Under the Uniform Commercial Code (UCC) Article 2, remedies for breach of contract "
            "include: (1) expectation damages; (2) consequential damages; (3) specific performance; "
            "(4) cover damages. Statute of limitations is typically 4 years (UCC § 2-725)."
        ),
    },
    # TODO: Thêm entry về luật lao động Việt Nam
    # Gợi ý: id="labor_law", keywords=["lao động", "sa thải", ...], text="..."
    {
        "id": "labor_law",
        "keywords": ["lao động", "sa thải", "hợp đồng lao động", "tranh chấp lao động", "người lao động"],
        "text": (
            "Theo pháp luật lao động Việt Nam, người sử dụng lao động khi đơn phương chấm dứt "
            "hợp đồng lao động hoặc sa thải người lao động phải có căn cứ hợp pháp, tuân thủ đúng "
            "trình tự, thủ tục và thời hạn báo trước nếu pháp luật yêu cầu. Nếu sa thải hoặc chấm dứt "
            "hợp đồng trái pháp luật, người sử dụng lao động có thể phải nhận người lao động trở lại làm việc, "
            "trả tiền lương, đóng bảo hiểm cho thời gian người lao động không được làm việc và bồi thường "
            "theo quy định."
        ),
    },
]


@tool
def search_legal_knowledge(query: str) -> str:
    """Tìm kiếm trong knowledge base pháp lý."""
    query_lower = query.lower()
    for entry in LEGAL_KNOWLEDGE:
        if any(kw in query_lower for kw in entry["keywords"]):
            return f"[{entry['id']}] {entry['text']}"
    return "Không tìm thấy thông tin liên quan."


# TODO: Tạo tool check_statute_of_limitations
# Gợi ý: nhận case_type (str), trả về thời hiệu khởi kiện
@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện."""
    case_type_lower = case_type.lower()

    if any(keyword in case_type_lower for keyword in ["hợp đồng", "contract", "vi phạm hợp đồng"]):
        return (
            "Thời hiệu khởi kiện tranh chấp hợp đồng thường là 03 năm kể từ ngày người có quyền "
            "yêu cầu biết hoặc phải biết quyền và lợi ích hợp pháp của mình bị xâm phạm."
        )

    if any(keyword in case_type_lower for keyword in ["lao động", "sa thải", "tranh chấp lao động"]):
        return (
            "Đối với tranh chấp lao động cá nhân, thời hiệu yêu cầu Tòa án giải quyết thường là "
            "01 năm kể từ ngày phát hiện ra hành vi mà mỗi bên tranh chấp cho rằng quyền và lợi ích "
            "hợp pháp của mình bị vi phạm."
        )

    if any(keyword in case_type_lower for keyword in ["bồi thường", "thiệt hại", "ngoài hợp đồng"]):
        return (
            "Đối với yêu cầu bồi thường thiệt hại ngoài hợp đồng, thời hiệu khởi kiện thường là "
            "03 năm kể từ ngày người có quyền yêu cầu biết hoặc phải biết quyền và lợi ích hợp pháp "
            "của mình bị xâm phạm."
        )

    return (
        "Chưa có thông tin thời hiệu khởi kiện cho loại vụ việc này trong hệ thống. "
        "Vui lòng cung cấp rõ hơn loại tranh chấp, ví dụ: hợp đồng, lao động, bồi thường thiệt hại."
    )


async def main():
    load_dotenv()
    llm = get_llm()
    
    # TODO: Thêm tool mới vào danh sách
    tools = [search_legal_knowledge, check_statute_of_limitations]  # Thêm check_statute_of_limitations vào đây
    llm_with_tools = llm.bind_tools(tools)
    
    question = "Thời hiệu khởi kiện vụ vi phạm hợp đồng là bao lâu?"
    
    messages = [
        SystemMessage(content="Bạn là chuyên gia pháp lý. Sử dụng tools để tra cứu thông tin."),
        HumanMessage(content=question),
    ]
    
    print(f"Câu hỏi: {question}\n")
    
    # First LLM call - decide which tools to use
    response = await llm_with_tools.ainvoke(messages)
    messages.append(response)
    
    # Execute tools if requested
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"🔧 Gọi tool: {tool_call['name']}")
            tool_result = None
            
            if tool_call["name"] == "search_legal_knowledge":
                tool_result = search_legal_knowledge.invoke(tool_call["args"])
            # TODO: Thêm xử lý cho check_statute_of_limitations
            elif tool_call["name"] == "check_statute_of_limitations":
                tool_result = check_statute_of_limitations.invoke(tool_call["args"])
            
            if tool_result:
                messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
        
        # Second LLM call - synthesize final answer
        final_response = await llm_with_tools.ainvoke(messages)
        print(f"\n✅ Kết quả:\n{final_response.content}")
    else:
        print(f"\n✅ Kết quả:\n{response.content}")


if __name__ == "__main__":
    asyncio.run(main())