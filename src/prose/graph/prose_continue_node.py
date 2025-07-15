# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
from langchain_core.messages import HumanMessage, SystemMessage
from src.config.config_loader import get_settings
from src.llms.llm import get_llm_by_type
from src.llms.error_handler import safe_llm_call
from src.prose.graph.state import ProseState
from src.utils.template import get_prompt_template

logger = logging.getLogger(__name__)


def prose_continue_node(state: ProseState):
    logger.info("Generating prose continue content...")
    settings = get_settings()
    llm_type = settings.agent_llm_map.get("prose_writer", "gpt-4o-mini")
    model = get_llm_by_type(llm_type)
    prompt_template = get_prompt_template("prose_continue")
    messages = [
        SystemMessage(content=prompt_template),
        HumanMessage(content=state["content"]),
    ]
    response = safe_llm_call(
        model.invoke,
        messages,
        operation_name="Prose Continue",
        context="Continuing prose content",
    )
    response_content = (
        response.content if hasattr(response, "content") else str(response)
    )
    logger.info(f"prose_continue_node response: {response_content}")
    return {"prose_content": response_content}
