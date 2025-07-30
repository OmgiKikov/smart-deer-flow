# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Промпты для рефлексивного анализа качества исследований.
Основан на механизме рефлексии GFLQ (Gemini Full-Stack LangGraph Quickstart).
"""

from typing import List, Dict, Any
from src.prompts.prompt_manager import get_prompt_with_variables


def get_reflection_instructions(
    research_topic: str = "",
    current_findings: List[str] = None,
    step_count: int = 0,
    previous_gaps: List[str] = None,
    locale: str = "ru-RU",
) -> str:
    """
    Получить основные инструкции для рефлексивного анализа.

    Args:
        research_topic: Основная тема исследования
        current_findings: Список текущих результатов исследования
        step_count: Количество выполненных шагов исследования
        previous_gaps: Ранее выявленные пробелы в знаниях
        locale: Языковая локаль (например, "ru-RU")

    Returns:
        str: Шаблон промпта для рефлексивного анализа
    """
    return get_prompt_with_variables(
        "reflection_instructions",
        {
            "research_topic": research_topic,
            "current_findings": current_findings or [],
            "step_count": step_count,
            "previous_gaps": previous_gaps or [],
            "locale": locale,
        },
    )


def get_context_analysis_prompt(
    research_topic: str,
    current_findings: List[str],
    step_count: int,
    previous_gaps: List[str] = None,
    locale: str = "ru-RU",
) -> str:
    """
    Сгенерировать контекстно-специфичный промпт для рефлексии.

    Args:
        research_topic: Основная тема исследования
        current_findings: Список текущих результатов исследования
        step_count: Количество выполненных шагов исследования
        previous_gaps: Ранее выявленные пробелы в знаниях
        locale: Языковая локаль (например, "ru-RU", "zh-CN")

    Returns:
        str: Отформатированный промпт для рефлексии
    """
    return get_reflection_instructions(
        research_topic=research_topic,
        current_findings=current_findings,
        step_count=step_count,
        previous_gaps=previous_gaps,
        locale=locale,
    )


def get_progressive_reflection_prompt(
    complexity_score: float, isolation_active: bool, context_size: int
) -> str:
    """
    Сгенерировать промпт для прогрессивного рефлексивного анализа.

    Args:
        complexity_score: Оценка сложности исследования (от 0.0 до 1.0)
        isolation_active: Активна ли изоляция контекста
        context_size: Текущий размер контекста

    Returns:
        str: Промпт для прогрессивной рефлексии
    """
    return f"""
Прогрессивный рефлексивный анализ

Контекст исследования:
- Оценка сложности: {complexity_score:.2f}
- Изоляция контекста активна: {isolation_active}
- Размер контекста: {context_size} токенов

Скорректируйте ваш рефлексивный анализ в зависимости от сложности исследования:
- Высокая сложность (>0.7): Сосредоточьтесь на всестороннем охвате и глубоком анализе
- Средняя сложность (0.3-0.7): Сбалансируйте широту и глубину соответствующим образом
- Низкая сложность (<0.3): Убедитесь, что основные требования выполнены эффективно

Если изоляция контекста активна, уделите особое внимание потенциальной фрагментации информации и обеспечьте непрерывность между изолированными контекстами.
"""


def get_integration_reflection_prompt(
    plan_update_needed: bool, researcher_context: Dict[str, Any]
) -> str:
    """
    Сгенерировать промпт для интеграции рефлексии с существующими компонентами.

    Args:
        plan_update_needed: Нужны ли обновления плана
        researcher_context: Информация о текущем контексте исследователя

    Returns:
        str: Промпт для интеграционной рефлексии
    """
    return f"""
Анализ интеграции рефлексии

Текущий контекст:
- Необходимо обновление плана: {plan_update_needed}
- Контекст исследователя: {researcher_context}

Обеспечьте рефлексивный анализ, который учитывает:
1. Как результаты интегрируются с текущим планом исследования
2. Необходимы ли изменения плана на основе новых данных
3. Как оптимизировать контекст исследователя для лучших результатов
4. Рекомендации по корректировке рабочего процесса

Убедитесь, что ваши рекомендации совместимы с существующей архитектурой SmartDeerFlow и могут быть реализованы без нарушения текущих исследовательских процессов.
"""


def get_metrics_reflection_prompt(session_metrics: Dict[str, Any]) -> str:
    """
    Сгенерировать промпт для рефлексивного анализа на основе метрик.

    Args:
        session_metrics: Метрики производительности текущей сессии

    Returns:
        str: Промпт для рефлексии на основе метрик
    """
    return f"""
Рефлексивный анализ на основе метрик

Метрики производительности сессии:
{session_metrics}

Проанализируйте эффективность исследования и предоставьте аналитику по:
1. Эффективности текущего подхода к исследованию
2. Тенденциям качества в результатах исследования
3. Оптимизации использования ресурсов
4. Потенциальным узким местам или областям для улучшения

Предоставьте конкретные, измеримые рекомендации для повышения эффективности исследования на основе данных метрик.
"""
