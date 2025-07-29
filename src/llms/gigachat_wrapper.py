# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
GigaChat-specific wrapper to handle tool calling compatibility issues.
"""

import logging
from typing import List
from langchain_core.tools import BaseTool
from langchain_gigachat import GigaChat
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)


def create_gigachat_agent(model: GigaChat, tools: List[BaseTool], name: str = "gigachat_agent", prompt=None):
    """Create a GigaChat-compatible agent using standard create_react_agent.
    
    GigaChat supports tool calling through the standard LangGraph create_react_agent.
    No special wrapper is needed.
    """
    logger.info(f"Creating GigaChat agent '{name}' with {len(tools)} tools")
    
    return create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt,
    )