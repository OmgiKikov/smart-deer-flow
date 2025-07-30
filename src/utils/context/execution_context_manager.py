"""Единый менеджер контекста выполнения

Этот модуль предоставляет единую логику обработки контекста для последовательных и параллельных путей выполнения.
Решает проблемы накопления контекста и несогласованности.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain_core.messages import BaseMessage

from ..tokens.token_counter import count_tokens
from typing import NamedTuple
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ContextConfig:
    """Конфигурация обработки контекста"""

    max_context_steps: int = 3
    max_step_content_length: int = 2000
    max_observations_length: int = 10000
    token_budget_ratio: float = 0.7
    enable_content_deduplication: bool = True
    enable_smart_truncation: bool = True


class TokenAllocation(NamedTuple):
    """Результат распределения токенов"""

    allocated_tokens: int
    task_id: str
    model_name: str
    parallel_tasks: int


class ExecutionContextManager:
    """Единый менеджер контекста выполнения

    Предоставляет единую логику обработки контекста для последовательного и параллельного выполнения, включая:
    - Интеллектуальное управление историей шагов
    - Сжатие и дедупликация результатов наблюдений
    - Динамическое управление бюджетом токенов
    - Оптимизация истории сообщений
    - Оптимизация контекста планирования
    - Расширенное управление результатами наблюдений
    """

    def __init__(self, config: Optional[ContextConfig] = None):
        self.config = config or ContextConfig()
        self._task_allocations = {}  # Отслеживание распределения токенов задач
        self._observation_cache = {}  # Кэш результатов наблюдений

    def prepare_context_for_execution(
        self,
        completed_steps: List[Dict[str, Any]],
        current_step: Dict[str, Any],
        agent_type: str = "researcher",
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Подготовка оптимизированного контекста для выполнения агентом

        Args:
            completed_steps: Список завершенных шагов
            current_step: Текущий шаг
            agent_type: Тип агента

        Returns:
            Оптимизированный список шагов и отформатированная информация о контексте
        """
        # 1. Применение ограничения на количество шагов
        limited_steps = self._limit_context_steps(completed_steps)

        # 2. Дедупликация и усечение содержимого
        if self.config.enable_content_deduplication:
            limited_steps = self._deduplicate_step_content(limited_steps)

        if self.config.enable_smart_truncation:
            limited_steps = self._truncate_step_content(limited_steps)

        # 3. Форматирование информации о контексте
        context_info = self._format_context_info(limited_steps, current_step)

        logger.info(
            f"Context prepared for {agent_type}: {len(limited_steps)} steps, "
            f"{len(context_info)} chars"
        )

        return limited_steps, context_info

    def manage_observations(
        self, observations: List[str], new_observation: str
    ) -> List[str]:
        """Управление списком наблюдений, предотвращение бесконечного накопления

        Args:
            observations: Существующий список наблюдений
            new_observation: Новое наблюдение

        Returns:
            Оптимизированный список наблюдений
        """
        # Добавить новое наблюдение
        updated_observations = observations + [new_observation]

        # Рассчитать общую длину
        total_length = sum(len(obs) for obs in updated_observations)

        # Если лимит превышен, выполнить сжатие
        if total_length > self.config.max_observations_length:
            updated_observations = self._compress_observations(updated_observations)

        return updated_observations

    def optimize_messages(
        self, messages: List[BaseMessage], token_limit: Optional[int] = None
    ) -> List[BaseMessage]:
        """Оптимизация истории сообщений для предотвращения превышения лимита токенов

        Args:
            messages: Список сообщений
            token_limit: Лимит токенов

        Returns:
            Оптимизированный список сообщений
        """
        if not token_limit:
            return messages

        # Рассчитать текущее использование токенов
        current_tokens = sum(count_tokens(msg.content).total_tokens for msg in messages)

        if current_tokens <= token_limit:
            return messages

        # Использовать внутреннюю стратегию усечения
        return self._simple_message_truncation(messages, token_limit)

    def _limit_context_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ограничение количества шагов в контексте"""
        if len(steps) <= self.config.max_context_steps:
            return steps

        # Сохранить последние шаги
        return steps[-self.config.max_context_steps :]

    def _deduplicate_step_content(
        self, steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Удаление дублирующейся информации из содержимого шагов"""
        seen_content = set()
        deduplicated_steps = []

        for step in steps:
            execution_res = step.get("execution_res", "")

            # Простой отпечаток содержимого
            content_hash = hash(execution_res[:200])  # Использовать первые 200 символов в качестве отпечатка

            if content_hash not in seen_content:
                seen_content.add(content_hash)
                deduplicated_steps.append(step)
            else:
                # Сохранить структуру шага, но пометить как дубликат
                modified_step = step.copy()
                modified_step["execution_res"] = "[Дублирующееся содержимое опущено]"
                deduplicated_steps.append(modified_step)

        return deduplicated_steps

    def _truncate_step_content(
        self, steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Интеллектуальное усечение содержимого шагов"""
        truncated_steps = []

        for step in steps:
            execution_res = step.get("execution_res", "")

            if len(execution_res) > self.config.max_step_content_length:
                # Сохранить начало и конец, в середине многоточие
                keep_length = self.config.max_step_content_length // 2 - 50
                truncated_content = (
                    execution_res[:keep_length]
                    + "\n\n[... содержимое обрезано ...]\n\n"
                    + execution_res[-keep_length:]
                )

                modified_step = step.copy()
                modified_step["execution_res"] = truncated_content
                truncated_steps.append(modified_step)
            else:
                truncated_steps.append(step)

        return truncated_steps

    def _format_context_info(
        self, steps: List[Dict[str, Any]], current_step: Dict[str, Any]
    ) -> str:
        """Форматирование информации о контексте"""
        if not steps:
            return "Нет завершенных шагов."

        context_parts = []
        context_parts.append(f"Завершено {len(steps)} исследовательских шагов:\n")

        for i, step in enumerate(steps, 1):
            step_info = f"{i}. {step.get('step', 'Неизвестный шаг')}"
            execution_res = step.get("execution_res", "")

            if execution_res:
                # Ограничить длину каждого шага в контексте
                if len(execution_res) > 300:
                    execution_res = execution_res[:300] + "..."
                step_info += f"\n   Результат: {execution_res}"

            context_parts.append(step_info)

        return "\n".join(context_parts)

    def _compress_observations(self, observations: List[str]) -> List[str]:
        """Сжатие списка наблюдений"""
        if not observations:
            return observations

        # Сохранить последние наблюдения
        recent_count = max(1, len(observations) // 2)
        recent_observations = observations[-recent_count:]

        # Если все еще слишком длинно, дополнительно усечь каждое наблюдение
        compressed = []
        remaining_budget = self.config.max_observations_length

        for obs in reversed(recent_observations):
            if remaining_budget <= 0:
                break

            if len(obs) > remaining_budget:
                # Усечь это наблюдение
                truncated_obs = obs[: remaining_budget - 50] + "...[обрезано]"
                compressed.insert(0, truncated_obs)
                break
            else:
                compressed.insert(0, obs)
                remaining_budget -= len(obs)

        return compressed

    def _simple_message_truncation(
        self, messages: List[BaseMessage], token_limit: int
    ) -> List[BaseMessage]:
        """Простая стратегия усечения сообщений"""
        if not messages:
            return messages

        # Сохранить системные сообщения и последние сообщения пользователя/ассистента
        system_messages = [msg for msg in messages if msg.type == "system"]
        other_messages = [msg for msg in messages if msg.type != "system"]

        # Начать сохранение с самых новых сообщений
        truncated_messages = system_messages[:]
        current_tokens = sum(
            count_tokens(msg.content).total_tokens for msg in system_messages
        )

        for msg in reversed(other_messages):
            msg_tokens = count_tokens(msg.content).total_tokens
            if current_tokens + msg_tokens <= token_limit:
                truncated_messages.insert(
                    -len(system_messages) or len(truncated_messages), msg
                )
                current_tokens += msg_tokens
            else:
                break

        return truncated_messages

    def manage_observations_advanced(
        self, observations: List[str], optimization_level: str = "standard"
    ) -> List[str]:
        """Расширенное управление наблюдениями с поддержкой умного сжатия и дедупликации

        Args:
            observations: Список наблюдений
            optimization_level: Уровень оптимизации ("minimal", "standard", "aggressive")

        Returns:
            Оптимизированный список наблюдений
        """
        if not observations:
            return observations

        # Установить параметры в соответствии с уровнем оптимизации
        if optimization_level == "minimal":
            max_observations = len(observations)
            compression_ratio = 0.9
        elif optimization_level == "standard":
            max_observations = max(5, len(observations) // 2)
            compression_ratio = 0.7
        else:  # aggressive
            max_observations = max(3, len(observations) // 3)
            compression_ratio = 0.5

        # 1. Обработка дедупликации
        deduplicated = self._deduplicate_observations(observations)

        # 2. Оценка важности и сортировка
        scored_observations = self._score_observations(deduplicated)

        # 3. Выбор самых важных наблюдений
        selected = sorted(scored_observations, key=lambda x: x[1], reverse=True)[
            :max_observations
        ]
        selected_observations = [obs for obs, score in selected]

        # 4. Сжатие содержимого
        compressed = self._compress_observation_content(
            selected_observations, compression_ratio
        )

        logger.info(
            f"Advanced observation management: {len(observations)} -> {len(compressed)} "
            f"(level: {optimization_level})"
        )

        return compressed

    def optimize_planning_context(
        self, messages: List[BaseMessage], observations: List[str], plan_iterations: int
    ) -> Tuple[List[BaseMessage], List[str]]:
        """Оптимизация контекста на этапе планирования

        Args:
            messages: История сообщений
            observations: Наблюдения
            plan_iterations: Количество итераций планирования

        Returns:
            Оптимизированные списки сообщений и наблюдений
        """
        # Настроить интенсивность оптимизации в зависимости от количества итераций
        if plan_iterations <= 2:
            optimization_level = "minimal"
        elif plan_iterations <= 5:
            optimization_level = "standard"
        else:
            optimization_level = "aggressive"

        # Оптимизация истории сообщений
        optimized_messages = self._optimize_planning_messages(messages, plan_iterations)

        # Оптимизация наблюдений
        optimized_observations = self.manage_observations_advanced(
            observations, optimization_level
        )

        logger.info(
            f"Planning context optimization: messages {len(messages)} -> {len(optimized_messages)}, "
            f"observations {len(observations)} -> {len(optimized_observations)}"
        )

        return optimized_messages, optimized_observations

    def manage_token_budget(
        self, task_id: str, model_name: str, parallel_tasks: int = 1
    ) -> TokenAllocation:
        """Интегрированное управление бюджетом токенов

        Args:
            task_id: ID задачи
            model_name: Имя модели
            parallel_tasks: Количество параллельных задач

        Returns:
            Результат распределения токенов
        """
        # Базовое ограничение токенов (в зависимости от типа модели)
        base_limits = {
            "deepseek-chat": 32000,
            "deepseek-reasoner": 64000,
            "gpt-4": 8000,
            "gpt-3.5-turbo": 4000,
        }

        base_limit = base_limits.get(model_name, 8000)

        # Распределение в зависимости от количества параллельных задач и доли бюджета
        allocated_tokens = int(
            (base_limit * self.config.token_budget_ratio) / max(1, parallel_tasks)
        )

        allocation = TokenAllocation(
            allocated_tokens=allocated_tokens,
            task_id=task_id,
            model_name=model_name,
            parallel_tasks=parallel_tasks,
        )

        # Записать распределение
        self._task_allocations[task_id] = allocation

        logger.info(
            f"Token budget allocated: {allocated_tokens} tokens for task {task_id} "
            f"(model: {model_name}, parallel: {parallel_tasks})"
        )

        return allocation

    def release_task_resources(self, task_id: str) -> None:
        """Освобождение ресурсов задачи

        Args:
            task_id: ID задачи
        """
        if task_id in self._task_allocations:
            del self._task_allocations[task_id]
            logger.debug(f"Released resources for task {task_id}")

    def _deduplicate_observations(self, observations: List[str]) -> List[str]:
        """Дедупликация наблюдений"""
        seen_hashes = set()
        deduplicated = []

        for obs in observations:
            # Использовать хэш содержимого для дедупликации
            content_hash = hashlib.md5(str(obs).encode()).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                deduplicated.append(obs)

        return deduplicated

    def _score_observations(self, observations: List[str]) -> List[Tuple[str, float]]:
        """Оценка наблюдений"""
        scored = []

        for obs in observations:
            score = 0.0

            # Оценка по длине (умеренная длина получает более высокий балл)
            length = len(obs)
            if 100 <= length <= 2000:
                score += 1.0
            elif length > 2000:
                score += 0.5
            else:
                score += 0.3

            # Оценка по ключевым словам
            keywords = ["вывод", "открытие", "важно", "ключевой", "предложение", "итог", "анализ"]
            for keyword in keywords:
                if keyword in obs:
                    score += 0.2

            # Оценка за структурированное содержимое
            if any(marker in obs for marker in ["##", "**", "1.", "2.", "-"]):
                score += 0.3

            scored.append((obs, score))

        return scored

    def _compress_observation_content(
        self, observations: List[str], compression_ratio: float
    ) -> List[str]:
        """Сжатие содержимого наблюдений"""
        compressed = []

        for obs in observations:
            target_length = int(len(obs) * compression_ratio)

            if len(obs) <= target_length:
                compressed.append(obs)
            else:
                # Сохранить начало и конец, сжать середину
                keep_start = target_length // 2
                keep_end = target_length - keep_start - 20  # Оставить место для многоточия

                if keep_end > 0:
                    compressed_obs = (
                        obs[:keep_start] + "\n[...содержимое сжато...]\n" + obs[-keep_end:]
                    )
                else:
                    compressed_obs = obs[:target_length] + "..."

                compressed.append(compressed_obs)

        return compressed

    def _optimize_planning_messages(
        self, messages: List[BaseMessage], plan_iterations: int
    ) -> List[BaseMessage]:
        """Оптимизация истории сообщений для планирования"""
        if len(messages) <= 8:
            return messages

        # Сохранить важные сообщения
        important_messages = []
        regular_messages = []

        for msg in messages:
            content = msg.content.lower()

            # Распознать важные сообщения
            if any(
                keyword in content
                for keyword in ["запрос пользователя", "цель задачи", "ошибка", "неудача", "важно", "ключевой"]
            ):
                important_messages.append(msg)
            else:
                regular_messages.append(msg)

        # Определить количество сохраняемых обычных сообщений в зависимости от количества итераций
        if plan_iterations <= 3:
            keep_regular = len(regular_messages) // 2
        else:
            keep_regular = len(regular_messages) // 3

        # Сохранить последние обычные сообщения
        kept_regular = regular_messages[-keep_regular:] if keep_regular > 0 else []

        # Объединить и отсортировать по времени
        optimized = important_messages + kept_regular

        # Простая сортировка по времени (предполагается, что сообщения идут в хронологическом порядке)
        return optimized

    def evaluate_and_optimize_context_before_call_sync(
        self, llm_func, args: tuple, kwargs: dict, operation_name: str, context: str
    ) -> tuple:
        """Синхронная версия оценки и оптимизации контекста

        Args:
            llm_func: LLM-функция
            args: Позиционные аргументы
            kwargs: Именованные аргументы
            operation_name: Имя операции
            context: Информация о контексте

        Returns:
            Кортеж с оптимизированными (args, kwargs)
        """
        try:
            # Извлечь сообщения из параметров
            messages = self._extract_messages_from_args(args, kwargs)
            if not messages:
                return args, kwargs

            # Получить имя модели
            self._extract_model_name(llm_func, kwargs)

            # Применить оптимизацию сообщений
            optimized_messages = self.optimize_messages(messages)

            # Обновить сообщения в параметрах
            new_args, new_kwargs = self._update_args_with_messages(
                args, kwargs, optimized_messages
            )

            logger.debug(
                f"Context optimized for {operation_name}: "
                f"{len(messages)} -> {len(optimized_messages)} messages"
            )

            return new_args, new_kwargs

        except Exception as e:
            logger.warning(
                f"Context optimization failed in sync call: {e}, "
                "proceeding with original arguments"
            )
            return args, kwargs

    async def evaluate_and_optimize_context_before_call(
        self, llm_func, args: tuple, kwargs: dict, operation_name: str, context: str
    ) -> tuple:
        """Асинхронная версия оценки и оптимизации контекста

        Args:
            llm_func: LLM-функция
            args: Позиционные аргументы
            kwargs: Именованные аргументы
            operation_name: Имя операции
            context: Информация о контексте

        Returns:
            Кортеж с оптимизированными (args, kwargs)
        """
        try:
            # Извлечь сообщения из параметров
            messages = self._extract_messages_from_args(args, kwargs)
            if not messages:
                return args, kwargs

            # Получить имя модели
            self._extract_model_name(llm_func, kwargs)

            # Применить оптимизацию сообщений
            optimized_messages = self.optimize_messages(messages)

            # Обновить сообщения в параметрах
            new_args, new_kwargs = self._update_args_with_messages(
                args, kwargs, optimized_messages
            )

            logger.debug(
                f"Context optimized for {operation_name}: "
                f"{len(messages)} -> {len(optimized_messages)} messages"
            )

            return new_args, new_kwargs

        except Exception as e:
            logger.warning(
                f"Context optimization failed in async call: {e}, "
                "proceeding with original arguments"
            )
            return args, kwargs

    def _extract_messages_from_args(
        self, args: tuple, kwargs: dict
    ) -> List[BaseMessage]:
        """Извлечение списка сообщений из аргументов функции"""
        messages = []

        # Проверить сообщения в позиционных аргументах
        for arg in args:
            if isinstance(arg, list) and arg and isinstance(arg[0], BaseMessage):
                messages = arg
                break
            elif isinstance(arg, BaseMessage):
                messages = [arg]
                break

        # Проверить сообщения в именованных аргументах
        if not messages:
            for key in ["messages", "input", "prompt"]:
                if key in kwargs:
                    value = kwargs[key]
                    if (
                        isinstance(value, list)
                        and value
                        and isinstance(value[0], BaseMessage)
                    ):
                        messages = value
                        break
                    elif isinstance(value, BaseMessage):
                        messages = [value]
                        break

        return messages

    def _extract_model_name(self, llm_func, kwargs: dict) -> str:
        """Извлечение имени модели из LLM-функции или аргументов"""
        # Попытаться получить из kwargs
        if "model" in kwargs:
            return kwargs["model"]

        # Попытаться получить из атрибутов llm_func
        if hasattr(llm_func, "__self__"):
            llm_instance = llm_func.__self__
            if hasattr(llm_instance, "model_name"):
                return llm_instance.model_name
            elif hasattr(llm_instance, "model"):
                return llm_instance.model

        # Значение по умолчанию
        return "deepseek-chat"

    def _update_args_with_messages(
        self, args: tuple, kwargs: dict, optimized_messages: List[BaseMessage]
    ) -> tuple:
        """Обновление аргументов функции оптимизированными сообщениями"""
        new_args = list(args)
        new_kwargs = kwargs.copy()

        # Обновить сообщения в позиционных аргументах
        for i, arg in enumerate(args):
            if isinstance(arg, list) and arg and isinstance(arg[0], BaseMessage):
                new_args[i] = optimized_messages
                return tuple(new_args), new_kwargs
            elif isinstance(arg, BaseMessage):
                new_args[i] = optimized_messages[0] if optimized_messages else arg
                return tuple(new_args), new_kwargs

        # Обновить сообщения в именованных аргументах
        for key in ["messages", "input", "prompt"]:
            if key in kwargs:
                value = kwargs[key]
                if (
                    isinstance(value, list)
                    and value
                    and isinstance(value[0], BaseMessage)
                ):
                    new_kwargs[key] = optimized_messages
                    return tuple(new_args), new_kwargs
                elif isinstance(value, BaseMessage):
                    new_kwargs[key] = (
                        optimized_messages[0] if optimized_messages else value
                    )
                    return tuple(new_args), new_kwargs

        return tuple(new_args), new_kwargs
