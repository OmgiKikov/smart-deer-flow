# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging

from langchain.schema import HumanMessage, SystemMessage

from src.config.config_loader import get_settings
from src.llms.llm import get_llm_by_type
from src.llms.error_handler import safe_llm_call
from src.utils.template import get_prompt_template

from ..types import Script
from .state import PodcastState

logger = logging.getLogger(__name__)


def script_writer_node(state: PodcastState):
    logger.info("Generating script for podcast...")
    settings = get_settings()
    llm_type = getattr(settings.agent_llm_map, "podcast_script_writer", "basic")
    model = get_llm_by_type(llm_type)
    
    # Check if this is GigaChat and use appropriate method
    if hasattr(model, '__class__') and 'GigaChat' in model.__class__.__name__:
        model = model.with_structured_output(Script, method="format_instructions")
    else:
        model = model.with_structured_output(Script, method="json_mode")
    script = safe_llm_call(
        model.invoke,
        [
            SystemMessage(content=get_prompt_template("podcast/podcast_script_writer")),
            HumanMessage(content=state["input"]),
        ],
        operation_name="Podcast Script Writer",
        context="Generating podcast script",
    )
    print(script)
    return {"script": script, "audio_chunks": []}
