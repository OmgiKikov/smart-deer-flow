# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from enum import Enum
from typing import List, Optional
import re

from pydantic import BaseModel, Field, field_validator, ConfigDict


class StepType(str, Enum):
    RESEARCH = "research"
    PROCESSING = "processing"


class Step(BaseModel):
    need_search: bool = Field(..., description="Must be explicitly set for each step")
    title: str = Field(..., max_length=200)
    description: str = Field(
        ..., description="Specify exactly what data to collect", max_length=1000
    )
    step_type: StepType = Field(..., description="Indicates the nature of the step")
    execution_res: Optional[str] = Field(
        default=None, description="The Step execution result", max_length=50000
    )

    @field_validator("title")
    @classmethod
    def validate_title(cls, v):
        if not v.strip():
            raise ValueError("Title cannot be empty")
        # Check for suspicious patterns
        suspicious_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Potentially unsafe content in title")
        return v

    @field_validator("description")
    @classmethod
    def validate_description(cls, v):
        if not v.strip():
            raise ValueError("Description cannot be empty")
        # Check for suspicious patterns
        suspicious_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Potentially unsafe content in description")
        return v

    @field_validator("execution_res")
    @classmethod
    def validate_execution_res(cls, v):
        if v is not None:
            # Check for suspicious patterns
            suspicious_patterns = [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
            ]
            for pattern in suspicious_patterns:
                if re.search(pattern, v, re.IGNORECASE):
                    raise ValueError("Potentially unsafe content in execution result")
        return v


class Plan(BaseModel):
    locale: str = Field(
        ...,
        description="'ru-RU'",
        pattern=r"^[a-z]{2}-[A-Z]{2}$",
        max_length=10,
    )
    has_enough_context: bool
    thought: str = Field(..., max_length=5000)
    title: str = Field(..., max_length=200)
    steps: List[Step] = Field(
        default_factory=list,
        description="Research & Processing steps to get more context",
    )

    @field_validator("thought")
    @classmethod
    def validate_thought(cls, v):
        if not v.strip():
            raise ValueError("Thought cannot be empty")
        # Check for suspicious patterns
        suspicious_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Potentially unsafe content in thought")
        return v

    @field_validator("title")
    @classmethod
    def validate_title(cls, v):
        if not v.strip():
            raise ValueError("Title cannot be empty")
        # Check for suspicious patterns
        suspicious_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Potentially unsafe content in title")
        return v

    @field_validator("steps")
    @classmethod
    def validate_steps(cls, v):
        if len(v) > 50:  # Limit number of steps
            raise ValueError("Too many steps in plan")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "locale": "ru-RU",
                    "has_enough_context": False,
                    "thought": (
                        "Чтобы понять текущие тенденции рынка ИИ, нам нужно собрать исчерпывающую информацию."
                    ),
                    "title": "План исследования рынка ИИ",
                    "steps": [
                        {
                            "need_search": True,
                            "title": "Анализ текущего рынка ИИ",
                            "description": (
                                "Собрать данные о размере рынка, темпах роста, основных игроках и инвестиционных трендах в секторе ИИ."
                            ),
                            "step_type": "research",
                        }
                    ],
                }
            ]
        }
    )
