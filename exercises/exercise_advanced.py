"""
Bài Tập Nâng Cao: 4 Challenges từ README.md

Challenge 1: Financial Agent - Phân tích thiệt hại tài chính
Challenge 2: Conversation Memory - Agent nhớ các câu hỏi trước
Challenge 3: Custom Tool - Tool tra cứu luật từ knowledge base mở rộng
Challenge 4: Error Handling - Try-catch và retry logic khi tool/LLM fails
"""

import asyncio
import os
import sys
import time
from typing import Annotated, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from common.llm import get_llm


# ============================================================================
# CHALLENGE 3: Custom Tool - Tra cứu luật từ knowledge base mở rộng
# ============================================================================

LEGAL_DATABASE = [
    {
        "id": "gdpr_art83",
        "law": "GDPR",
        "article": "Article 83",
        "keywords": ["gdpr", "phạt", "fine", "data protection", "bảo vệ dữ liệu"],
        "text": (
            "GDPR Article 83: Phạt hành chính lên đến 20 triệu EUR hoặc 4% doanh thu "
            "toàn cầu hàng năm (tùy mức nào cao hơn) cho các vi phạm nghiêm trọng như "
            "xử lý dữ liệu trái phép, vi phạm quyền chủ thể dữ liệu."
        ),
    },
    {
        "id": "luat_an_ninh_mang_2018",
        "law": "Luật An ninh mạng 2018",
        "article": "Điều 26",
        "keywords": ["an ninh mạng", "lưu trữ dữ liệu", "localization", "việt nam"],
        "text": (
            "Luật An ninh mạng 2018, Điều 26: Doanh nghiệp trong và ngoài nước cung cấp "
            "dịch vụ trên không gian mạng tại Việt Nam phải lưu trữ dữ liệu người dùng "
            "tại Việt Nam, bao gồm dữ liệu cá nhân, dữ liệu về quan hệ người dùng."
        ),
    },
    {
        "id": "luat_bvdl_2023",
        "law": "Nghị định 13/2023/NĐ-CP",
        "article": "Điều 28",
        "keywords": ["bảo vệ dữ liệu cá nhân", "rò rỉ", "thông báo", "72 giờ", "data breach"],
        "text": (
            "Nghị định 13/2023/NĐ-CP về Bảo vệ dữ liệu cá nhân, Điều 28: Khi xảy ra vi phạm "
            "quy định về bảo vệ dữ liệu cá nhân, bên xử lý dữ liệu phải thông báo cho cơ quan "
            "chức năng trong vòng 72 giờ kể từ khi phát hiện vi phạm."
        ),
    },
    {
        "id": "irc_6663",
        "law": "IRC (Internal Revenue Code)",
        "article": "§ 6663",
        "keywords": ["tax fraud", "gian lận thuế", "irs", "penalty", "75%"],
        "text": (
            "IRC § 6663: Nếu bất kỳ phần nào của khoản thiếu hụt thuế là do gian lận, "
            "sẽ bị phạt 75% trên phần thuế thiếu hụt do gian lận. Phạt này thay thế "
            "phạt accuracy-related penalty theo § 6662."
        ),
    },
    {
        "id": "sox_section302",
        "law": "Sarbanes-Oxley Act",
        "article": "Section 302",
        "keywords": ["sox", "sarbanes", "ceo", "cfo", "financial reporting", "compliance"],
        "text": (
            "SOX Section 302: CEO và CFO phải chứng nhận cá nhân về tính chính xác của "
            "báo cáo tài chính. Vi phạm có thể dẫn đến phạt lên đến $5 triệu và "
            "phạt tù tới 20 năm."
        ),
    },
]


@tool
def lookup_legal_database(query: str) -> str:
    """Tra cứu luật và quy định từ cơ sở dữ liệu pháp lý.

    Args:
        query: Từ khóa hoặc chủ đề cần tra cứu (ví dụ: 'GDPR', 'rò rỉ dữ liệu')
    """
    query_lower = query.lower()
    results = []
    for entry in LEGAL_DATABASE:
        if any(kw in query_lower for kw in entry["keywords"]):
            results.append(f"[{entry['id']}] {entry['law']} - {entry['article']}: {entry['text']}")
    if results:
        return "\n\n".join(results)
    return f"Không tìm thấy quy định nào liên quan đến '{query}' trong cơ sở dữ liệu."


@tool
def estimate_financial_damage(
    revenue: float, breach_type: str, num_affected_users: int
) -> str:
    """Ước tính thiệt hại tài chính do vi phạm pháp lý.

    Args:
        revenue: Doanh thu hàng năm của công ty (USD)
        breach_type: Loại vi phạm (data_breach, tax_fraud, contract_breach)
        num_affected_users: Số người dùng bị ảnh hưởng
    """
    breach_type_lower = breach_type.lower()

    if "data" in breach_type_lower or "breach" in breach_type_lower:
        gdpr_fine = min(revenue * 0.04, 20_000_000)
        notification_cost = num_affected_users * 5
        credit_monitoring = num_affected_users * 25
        legal_fees = revenue * 0.02
        reputation_loss = revenue * 0.10
        total = gdpr_fine + notification_cost + credit_monitoring + legal_fees + reputation_loss
        return (
            f"📊 ƯỚC TÍNH THIỆT HẠI TÀI CHÍNH (Data Breach):\n"
            f"  • Phạt GDPR (4% doanh thu): ${gdpr_fine:,.0f}\n"
            f"  • Chi phí thông báo ({num_affected_users:,} users x $5): ${notification_cost:,.0f}\n"
            f"  • Giám sát tín dụng ({num_affected_users:,} users x $25): ${credit_monitoring:,.0f}\n"
            f"  • Phí pháp lý (2% doanh thu): ${legal_fees:,.0f}\n"
            f"  • Tổn thất uy tín (10% doanh thu): ${reputation_loss:,.0f}\n"
            f"  ──────────────────────────────\n"
            f"  💰 TỔNG THIỆT HẠI ƯỚC TÍNH: ${total:,.0f}"
        )
    elif "tax" in breach_type_lower:
        penalty_75 = revenue * 0.05 * 0.75
        back_taxes = revenue * 0.05
        interest = back_taxes * 0.07 * 3
        legal_fees = 150_000
        total = penalty_75 + back_taxes + interest + legal_fees
        return (
            f"📊 ƯỚC TÍNH THIỆT HẠI TÀI CHÍNH (Tax Fraud):\n"
            f"  • Thuế truy thu (ước tính 5% doanh thu): ${back_taxes:,.0f}\n"
            f"  • Phạt gian lận 75% (IRC §6663): ${penalty_75:,.0f}\n"
            f"  • Lãi suất (7%/năm x 3 năm): ${interest:,.0f}\n"
            f"  • Phí pháp lý: ${legal_fees:,.0f}\n"
            f"  ──────────────────────────────\n"
            f"  💰 TỔNG THIỆT HẠI ƯỚC TÍNH: ${total:,.0f}"
        )
    else:
        damages = revenue * 0.15
        legal_fees = 100_000
        total = damages + legal_fees
        return (
            f"📊 ƯỚC TÍNH THIỆT HẠI TÀI CHÍNH (Contract Breach):\n"
            f"  • Bồi thường hợp đồng (15% doanh thu): ${damages:,.0f}\n"
            f"  • Phí pháp lý: ${legal_fees:,.0f}\n"
            f"  ──────────────────────────────\n"
            f"  💰 TỔNG THIỆT HẠI ƯỚC TÍNH: ${total:,.0f}"
        )


# ============================================================================
# CHALLENGE 4: Error Handling - Retry logic
# ============================================================================

async def call_llm_with_retry(prompt: str, max_retries: int = 3, delay: float = 2.0) -> str:
    """Gọi LLM với retry logic khi gặp lỗi.
    
    Args:
        prompt: Nội dung prompt gửi cho LLM
        max_retries: Số lần thử lại tối đa
        delay: Thời gian chờ giữa các lần thử (giây)
    
    Returns:
        Nội dung phản hồi từ LLM
    """
    llm = get_llm()
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            last_error = e
            print(f"  ⚠️ Lỗi lần thử {attempt}/{max_retries}: {type(e).__name__}: {e}")
            if attempt < max_retries:
                print(f"  ⏳ Đợi {delay}s rồi thử lại...")
                await asyncio.sleep(delay)
                delay *= 1.5  # exponential backoff

    return f"[LỖI] Không thể lấy phản hồi từ LLM sau {max_retries} lần thử. Lỗi cuối: {last_error}"


# ============================================================================
# CHALLENGE 2: Conversation Memory
# ============================================================================

class ConversationMemory:
    """Bộ nhớ hội thoại - lưu lại lịch sử các câu hỏi và kết quả."""

    def __init__(self):
        self.history: list[dict] = []

    def add(self, question: str, summary: str):
        """Thêm một lượt hội thoại vào bộ nhớ."""
        self.history.append({
            "turn": len(self.history) + 1,
            "question": question,
            "summary": summary[:500],  # giữ ngắn gọn
            "timestamp": time.strftime("%H:%M:%S"),
        })

    def get_context(self) -> str:
        """Trả về context từ lịch sử hội thoại."""
        if not self.history:
            return "Chưa có lịch sử hội thoại."
        lines = []
        for h in self.history:
            lines.append(f"[Lượt {h['turn']} - {h['timestamp']}] Q: {h['question']}\nA: {h['summary']}")
        return "\n---\n".join(lines)

    def __len__(self):
        return len(self.history)


# ============================================================================
# State và Agents (mở rộng từ Exercise 4)
# ============================================================================

def _last_wins(left: str | None, right: str | None) -> str:
    return right if right is not None else (left or "")


class State(TypedDict):
    question: str
    conversation_context: str  # Challenge 2: Memory
    law_analysis: Annotated[str, _last_wins]
    tax_analysis: Annotated[str, _last_wins]
    compliance_analysis: Annotated[str, _last_wins]
    privacy_analysis: Annotated[str, _last_wins]
    financial_analysis: Annotated[str, _last_wins]  # Challenge 1: Financial Agent
    final_response: str


# --- Law Agent (Challenge 4: có retry) ---
async def law_agent(state: State) -> dict:
    """Agent phân tích pháp lý tổng quát - có retry logic."""
    print("  🔄 [law_agent] Đang phân tích pháp lý...")

    # Challenge 3: Sử dụng Custom Tool
    tool_context = lookup_legal_database.invoke({"query": state["question"]})

    prompt = f"""Bạn là chuyên gia pháp lý. Phân tích câu hỏi sau:

{state['question']}

Thông tin tra cứu từ cơ sở dữ liệu pháp lý:
{tool_context}

Lịch sử hội thoại trước (nếu có):
{state.get('conversation_context', 'Không có')}

Tập trung vào: hợp đồng, trách nhiệm dân sự, quyền và nghĩa vụ pháp lý.
Hãy trả lời ngắn gọn, có cấu trúc."""

    result = await call_llm_with_retry(prompt)
    print(f"  ✅ [law_agent] Hoàn thành ({len(result)} ký tự)")
    return {"law_analysis": result}


def check_routing(state: State) -> list[Send]:
    """Quyết định gọi agents nào dựa trên nội dung câu hỏi."""
    question_lower = state["question"].lower()
    tasks = []

    if any(kw in question_lower for kw in ["tax", "irs", "thuế", "trốn thuế"]):
        tasks.append(Send("tax_agent", state))

    if any(kw in question_lower for kw in ["compliance", "sec", "regulation", "tuân thủ"]):
        tasks.append(Send("compliance_agent", state))

    if any(kw in question_lower for kw in [
        "data", "privacy", "gdpr", "dữ liệu", "rò rỉ",
        "data breach", "bảo mật", "thông tin cá nhân", "khách hàng"
    ]):
        tasks.append(Send("privacy_agent", state))

    # Challenge 1: Financial Agent - luôn chạy để ước tính thiệt hại
    if any(kw in question_lower for kw in [
        "thiệt hại", "tài chính", "financial", "damage", "phạt",
        "fine", "cost", "chi phí", "revenue", "doanh thu"
    ]):
        tasks.append(Send("financial_agent", state))

    return tasks if tasks else [Send("aggregate_results", state)]


# --- Tax Agent (Challenge 4: có retry) ---
async def tax_agent(state: State) -> dict:
    print("  🔄 [tax_agent] Đang phân tích thuế...")
    tool_context = lookup_legal_database.invoke({"query": "tax fraud IRS penalty"})

    prompt = f"""Bạn là chuyên gia thuế. Phân tích khía cạnh thuế:

Câu hỏi: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Tra cứu pháp lý:
{tool_context}

Tập trung: IRS, tax evasion, penalties, FBAR, FATCA.
Hãy trả lời ngắn gọn, có cấu trúc."""

    result = await call_llm_with_retry(prompt)
    print(f"  ✅ [tax_agent] Hoàn thành ({len(result)} ký tự)")
    return {"tax_analysis": result}


async def compliance_agent(state: State) -> dict:
    print("  🔄 [compliance_agent] Đang phân tích tuân thủ...")
    tool_context = lookup_legal_database.invoke({"query": "sox compliance sarbanes"})

    prompt = f"""Bạn là chuyên gia compliance. Phân tích khía cạnh tuân thủ:

Câu hỏi: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Tra cứu pháp lý:
{tool_context}

Tập trung: SEC, SOX, FCPA, AML, regulatory violations.
Hãy trả lời ngắn gọn, có cấu trúc."""

    result = await call_llm_with_retry(prompt)
    print(f"  ✅ [compliance_agent] Hoàn thành ({len(result)} ký tự)")
    return {"compliance_analysis": result}


async def privacy_agent(state: State) -> dict:
    print("  🔄 [privacy_agent] Đang phân tích quyền riêng tư...")
    tool_context = lookup_legal_database.invoke({"query": "bảo vệ dữ liệu gdpr rò rỉ data breach"})

    prompt = f"""Bạn là chuyên gia bảo vệ dữ liệu cá nhân và GDPR. Phân tích:

Câu hỏi: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Tra cứu pháp lý:
{tool_context}

Tập trung vào: GDPR, data protection, privacy rights, data breach,
nghĩa vụ thông báo 72 giờ, trách nhiệm bảo vệ dữ liệu.
Hãy trả lời ngắn gọn, có cấu trúc."""

    result = await call_llm_with_retry(prompt)
    print(f"  ✅ [privacy_agent] Hoàn thành ({len(result)} ký tự)")
    return {"privacy_analysis": result}


# ============================================================================
# CHALLENGE 1: Financial Agent
# ============================================================================

async def financial_agent(state: State) -> dict:
    """Agent chuyên phân tích thiệt hại tài chính - sử dụng Custom Tool."""
    print("  🔄 [financial_agent] Đang ước tính thiệt hại tài chính...")

    # Challenge 3: Sử dụng Custom Tool estimate_financial_damage
    try:
        damage_report = estimate_financial_damage.invoke({
            "revenue": 10_000_000,
            "breach_type": "data_breach",
            "num_affected_users": 50_000,
        })
    except Exception as e:
        # Challenge 4: Error handling cho tool
        print(f"  ⚠️ [financial_agent] Lỗi khi gọi tool: {e}")
        damage_report = "Không thể ước tính thiệt hại do lỗi hệ thống."

    prompt = f"""Bạn là chuyên gia tài chính doanh nghiệp. Dựa trên báo cáo thiệt hại bên dưới,
hãy phân tích và đưa ra khuyến nghị tài chính cho công ty:

Câu hỏi gốc: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')[:500]}

Báo cáo thiệt hại ước tính:
{damage_report}

Hãy đưa ra:
1. Đánh giá mức độ nghiêm trọng
2. Phân bổ ngân sách khắc phục
3. Khuyến nghị bảo hiểm
4. Kế hoạch tài chính phục hồi
Trả lời ngắn gọn, có cấu trúc."""

    result = await call_llm_with_retry(prompt)
    print(f"  ✅ [financial_agent] Hoàn thành ({len(result)} ký tự)")
    return {"financial_analysis": result}


# --- Aggregate (mở rộng với Financial) ---
async def aggregate_results(state: State) -> dict:
    """Tổng hợp kết quả từ tất cả agents."""
    sections = []
    if state.get("law_analysis"):
        sections.append(f"📋 PHÂN TÍCH PHÁP LÝ:\n{state['law_analysis']}")
    if state.get("tax_analysis"):
        sections.append(f"💰 PHÂN TÍCH THUẾ:\n{state['tax_analysis']}")
    if state.get("compliance_analysis"):
        sections.append(f"✅ PHÂN TÍCH TUÂN THỦ:\n{state['compliance_analysis']}")
    if state.get("privacy_analysis"):
        sections.append(f"🔐 PHÂN TÍCH QUYỀN RIÊNG TƯ:\n{state['privacy_analysis']}")
    if state.get("financial_analysis"):
        sections.append(f"💸 PHÂN TÍCH THIỆT HẠI TÀI CHÍNH:\n{state['financial_analysis']}")

    combined = "\n\n".join(sections)

    prompt = f"""Tổng hợp các phân tích sau thành một báo cáo pháp lý hoàn chỉnh:

{combined}

Câu hỏi gốc: {state['question']}

Hãy tạo một báo cáo có cấu trúc rõ ràng, bao gồm:
1. Tóm tắt vấn đề
2. Phân tích chi tiết theo từng khía cạnh
3. Ước tính thiệt hại (nếu có)
4. Khuyến nghị hành động
5. Kết luận"""

    result = await call_llm_with_retry(prompt)
    return {"final_response": result}


# --- Build Graph ---
def build_graph() -> StateGraph:
    graph = StateGraph(State)

    graph.add_node("law_agent", law_agent)
    graph.add_node("tax_agent", tax_agent)
    graph.add_node("compliance_agent", compliance_agent)
    graph.add_node("privacy_agent", privacy_agent)
    graph.add_node("financial_agent", financial_agent)  # Challenge 1
    graph.add_node("aggregate_results", aggregate_results)

    graph.add_edge(START, "law_agent")
    graph.add_conditional_edges("law_agent", check_routing)
    graph.add_edge("tax_agent", "aggregate_results")
    graph.add_edge("compliance_agent", "aggregate_results")
    graph.add_edge("privacy_agent", "aggregate_results")
    graph.add_edge("financial_agent", "aggregate_results")  # Challenge 1
    graph.add_edge("aggregate_results", END)

    return graph.compile()


# ============================================================================
# MAIN - Challenge 2: Conversation Memory demo
# ============================================================================

async def main():
    load_dotenv()

    memory = ConversationMemory()
    graph = build_graph()

    questions = [
        "Nếu công ty có doanh thu $10M bị rò rỉ dữ liệu 50,000 khách hàng, hậu quả pháp lý và thiệt hại tài chính là gì?",
        "Dựa trên phân tích trước, công ty cần làm gì để giảm thiểu rủi ro thuế?",
    ]

    for i, question in enumerate(questions, 1):
        print("\n" + "=" * 70)
        print(f"LƯỢT HỎI {i}/{len(questions)}")
        print("=" * 70)
        print(f"\n❓ Câu hỏi: {question}\n")

        # Challenge 2: Truyền context từ memory vào State
        context = memory.get_context()
        print(f"📝 Lịch sử hội thoại: {len(memory)} lượt trước\n")

        result = await graph.ainvoke({
            "question": question,
            "conversation_context": context,
            "law_analysis": "",
            "tax_analysis": "",
            "compliance_analysis": "",
            "privacy_analysis": "",
            "financial_analysis": "",
            "final_response": "",
        })

        final = result["final_response"]

        # Challenge 2: Lưu vào memory
        memory.add(question, final)

        print("\n" + "=" * 70)
        print(f"KẾT QUẢ LƯỢT {i}")
        print("=" * 70)
        print(final)
        print("=" * 70)

        # Chỉ chạy câu hỏi đầu tiên nếu muốn tiết kiệm thời gian
        # Bỏ comment dòng break bên dưới nếu muốn chạy cả 2 câu hỏi
        break

    print("\n\n📚 LỊCH SỬ HỘI THOẠI (Memory):")
    print("-" * 40)
    print(memory.get_context())


if __name__ == "__main__":
    asyncio.run(main())
