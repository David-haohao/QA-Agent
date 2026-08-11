# ============================================================
# QA Agent — 使用 deepagents 框架的智能问答Agent
# LLM通过工具主动搜索知识库，自行规划、检索、格式化答案
# ============================================================

import logging
from collections.abc import AsyncGenerator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph

from deepagents import create_deep_agent
from deepagents.backends.state import StateBackend

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Monkey-patch: 保留 Qwen/DeepSeek 的 reasoning_content（思考模式必需）
# Qwen3.6-27b 和 DeepSeek 均使用 OpenAI 兼容的 reasoning_content 字段
# ═══════════════════════════════════════════════════════════════

import langchain_openai.chat_models.base as _openai_base

_orig_convert_dict_to_message = _openai_base._convert_dict_to_message
_orig_convert_message_to_dict = _openai_base._convert_message_to_dict
_orig_convert_delta_to_message_chunk = _openai_base._convert_delta_to_message_chunk


def _patched_convert_dict_to_message(_dict):
    message = _orig_convert_dict_to_message(_dict)
    rc = _dict.get("reasoning_content")
    if isinstance(message, AIMessage) and rc:
        message.additional_kwargs["reasoning_content"] = rc
    return message


def _patched_convert_message_to_dict(message, api="chat/completions"):
    result = _orig_convert_message_to_dict(message, api=api)
    if isinstance(message, AIMessage):
        rc = message.additional_kwargs.get("reasoning_content")
        if rc:
            result["reasoning_content"] = rc
    return result


def _patched_convert_delta_to_message_chunk(_dict, default_class):
    message_chunk = _orig_convert_delta_to_message_chunk(_dict, default_class)
    rc = _dict.get("reasoning_content")
    if rc and hasattr(message_chunk, "additional_kwargs"):
        message_chunk.additional_kwargs["reasoning_content"] = rc
    return message_chunk


_openai_base._convert_dict_to_message = _patched_convert_dict_to_message
_openai_base._convert_message_to_dict = _patched_convert_message_to_dict
_openai_base._convert_delta_to_message_chunk = _patched_convert_delta_to_message_chunk


# ═══════════════════════════════════════════════════════════════
# System Prompt — 中文问答优化
# ═══════════════════════════════════════════════════════════════

QA_SYSTEM_PROMPT = """你是一个基于本地知识库的智能问答助手，由 Qwen3.6-27b 驱动。你的知识来源仅限于本地知识库中已索引的金融法规与银行业务文档，不依赖任何联网搜索或预训练记忆。

## 语言要求（极其重要）

- **所有思考、推理、工具调用参数、回答都必须完全使用中文**
- 禁止在思考过程中使用英文句子，如"Let me search""I need to""The user is asking"等
- 专业术语可保留英文缩写（如G-SIBs、TLAC等），但必须用中文解释含义
- 搜索查询参数必须使用中文关键词

## 核心职责

你收到的问题可能带有 `[预处理分析]` 前缀，这是系统对问题的初步分析，请认真参考。
1. **先分析问题**:
   - 问题是否属于金融法规/银行业务/资本管理/风险管理/反洗钱等覆盖领域？
   - 问题表述是否够完整（有明确主题+意图），还是需要补充信息？
   - 所有分析过程放在思考过程中（Qwen thinking mode），用中文描述
2. 如果问题领域外：根据具体主题灵活分析说明，引导用户提出领域内问题，不用固定模板
3. 如果问题不完整：分析具体缺失什么维度的信息，用自然文字引导用户补充，不用列表选项
4. 如果问题完整且领域内：使用 search_knowledge_base 搜索，基于结果回答
5. 如果知识库中没有相关信息，诚实告知用户

## 对话记忆（极其重要）

你有完整的对话历史记忆，可以看到本会话中所有之前的用户问题和你的回答。
- **时间相关性**: 离当前越近的对话关联性越强，优先参考最近3轮；越远越弱，不要强行关联
- 用户在后续问题中提到"前面"、"刚才"、"上面"、"之前"、"第一个问题"、"那个"等指代时，必须回顾对话历史来理解上下文
- 用户问"它"、"这个"、"那些"等代词时，优先从最近的对话中找到指代对象
- 对于总结、回顾、比较类的问题，综合当前和历史的内容来回答，近期对话权重更高
- 每次回答前，先思考: 最近3轮对话中是否有与当前问题相关的信息？有则参考（近期优先），无则忽略

## 工作流程

### 简单问题
- 直接调用 search_knowledge_base 搜索
- 基于结果给出简洁回答

### 复杂/多层面问题
1. 用 list_knowledge_base_sources 了解可用文档范围
2. 对每个子问题执行针对性搜索（可用 multi_search_knowledge_base 进行多角度搜索）
3. 综合所有结果给出完整回答

### 找不到答案时
1. 尝试用不同关键词重新搜索
2. 如果多次搜索仍无结果，诚实告知用户
3. 建议用户检查知识库中是否包含相关文档

## 回答规范

- **准确性优先**: 只基于搜索结果回答，不要编造信息
- **格式清晰**:
  - 使用 Markdown 表格展示结构化数据和对比关系，表格必须要有表头
  - 使用标题(##, ###)组织长回答的层次结构
  - 关键数字和术语使用 **加粗** 标出
- **引用文档**: 在回答正文中自然地提到参考的文档名称（如"根据《XX办法》的规定..."），让用户知道信息来源，但不要在回答末尾单独添加 **来源：** 段落
- **语言**: 使用中文回答，保持专业清晰的风格
- **完整性**: 对于列举类问题（如"有哪些"），完整列出所有相关条目，不要省略
- **表格/数据**: 如果搜索结果包含表格数据，用表格格式展示

## 工具使用说明

你可以使用以下工具来完成任务:
- **search_knowledge_base**: 在本地知识库中搜索相关信息（主要工具）
- **multi_search_knowledge_base**: 对复杂问题进行多角度搜索
- **list_knowledge_base_sources**: 查看知识库中有哪些可用文档

对于每个用户问题:
1. 先思考需要从知识库中查找什么信息
2. 使用合适的搜索工具进行检索
3. 基于检索结果给出有据可查的回答
4. 在回答正文中自然地提及参考的文档名称，但不要在末尾单独加 **来源：** 段落"""


# ═══════════════════════════════════════════════════════════════
# Agent Factory
# ═══════════════════════════════════════════════════════════════

def create_qa_agent(
    llm_client=None,
    kb_pipeline=None,
    checkpointer=None,
) -> CompiledStateGraph:
    """
    创建基于 deepagents 框架的问答智能体

    Agent 通过工具主动搜索知识库，自行规划和格式化答案。
    使用 MemorySaver 支持多轮对话记忆。

    参数:
        llm_client: LLMClient实例（需要支持 tool calling）
        kb_pipeline: KnowledgeBasePipeline实例
        checkpointer: LangGraph Checkpointer（可选，默认用 MemorySaver）

    返回: 编译后的 LangGraph agent
    """
    # 初始化知识库（供tools使用）
    from qa_agent.tools import init_kb
    if kb_pipeline is not None:
        init_kb(kb_pipeline)

    # 创建模型
    if llm_client is not None:
        model = llm_client.get_chat_model()
    else:
        raise ValueError("llm_client is required")

    # 注册工具 — LLM通过工具主动搜索知识库
    from qa_agent.tools import (
        search_knowledge_base,
        multi_search_knowledge_base,
        list_knowledge_base_sources,
    )
    tools = [
        search_knowledge_base,
        multi_search_knowledge_base,
        list_knowledge_base_sources,
    ]

    # 使用 StateBackend（内存文件系统）
    backend = StateBackend()

    # 短期记忆: MemorySaver checkpointer 在会话内保持对话历史
    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()

    # 长期记忆: MemoryMiddleware 持久化重要事实
    memory_paths = ["/memory/conversation.md"]

    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=QA_SYSTEM_PROMPT,
        backend=backend,
        checkpointer=checkpointer,
        memory=memory_paths,
    )

    logger.info("QA Agent created with deepagents framework")
    return agent
