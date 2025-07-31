"""Механизм слияния результатов для follow-up запросов

Этот модуль предоставляет интеллектуальные функции для слияния результатов follow-up запросов, включая:
- Дедупликацию контента и определение схожести
- Умную приоритизацию результатов
- Слияние структурированных данных
- Оценку и фильтрацию по качеству
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class MergedResult:
    """Структура данных для объединенного результата"""

    content: str
    sources: List[str]
    confidence_score: float
    relevance_score: float
    key_points: List[str]
    metadata: Dict[str, Any]
    original_count: int
    merged_count: int


@dataclass
class ResultMetrics:
    """Метрики качества результата"""

    content_length: int
    unique_information_ratio: float
    source_diversity: int
    temporal_relevance: float
    structural_completeness: float


class FollowUpResultMerger:
    """Интеллектуальный объединитель результатов для follow-up запросов"""

    def __init__(
        self,
        config: Optional[Any] = None,
        similarity_threshold: Optional[float] = None,
        min_content_length: Optional[int] = None,
        max_merged_results: Optional[int] = None,
        enable_semantic_grouping: Optional[bool] = None,
    ):
        """
        Инициализация объединителя

        Args:
            config: Объект конфигурации объединителя (наивысший приоритет)
            similarity_threshold: Порог схожести контента (для обратной совместимости)
            min_content_length: Минимальная длина контента (для обратной совместимости)
            max_merged_results: Максимальное количество объединенных результатов (для обратной совместимости)
            enable_semantic_grouping: Включить семантическую группировку (для обратной совместимости)
        """
        # Импорт конфигурации (отложенный импорт для избежания циклических зависимостей)
        if config is None:
            try:
                # Приоритетное использование новой единой системы конфигурации
                from src.config.config_loader import get_settings
                app_settings = get_settings()
                config = app_settings.get_followup_merger_config()
            except ImportError:
                try:
                    # Откат к старой системе конфигурации
                    from src.config.follow_up_merger_config import get_active_merger_config
                    config = get_active_merger_config()
                except ImportError:
                    # Если модуль конфигурации недоступен, используются значения по умолчанию
                    from src.config.follow_up_merger_config import FollowUpMergerConfig
                    config = FollowUpMergerConfig()

        # Применение переопределения параметров (для обратной совместимости)
        self.config = config
        if similarity_threshold is not None:
            self.config.similarity_threshold = similarity_threshold
        if min_content_length is not None:
            self.config.min_content_length = min_content_length
        if max_merged_results is not None:
            self.config.max_merged_results = max_merged_results
        if enable_semantic_grouping is not None:
            self.config.enable_semantic_grouping = enable_semantic_grouping

        # Конфигурация автоматически проверяется в модели Pydantic

        # Кэш отпечатков контента
        self._content_fingerprints: Set[str] = set()
        self._similarity_cache: Dict[Tuple[str, str], float] = {}

        # Статистика производительности
        self._stats = {
            "total_merges": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "deduplication_count": 0,
            "quality_filtered_count": 0,
        }

    def merge_follow_up_results(
        self,
        follow_up_results: List[Dict[str, Any]],
        original_findings: List[str],
        query_context: Optional[str] = None,
    ) -> List[MergedResult]:
        """
        Объединение результатов follow-up запросов

        Args:
            follow_up_results: Исходные результаты follow-up запросов
            original_findings: Исходные результаты исследования
            query_context: Контекст запроса

        Returns:
            Список объединенных результатов
        """
        logger.info(f"Начинаем слияние {len(follow_up_results)} новых и {len(original_findings)} существующих результатов")

        # Обновление статистики
        self._stats["total_merges"] += 1

        # 1. Стандартизация всех входных данных
        normalized_follow_ups = self._normalize_results(follow_up_results)
        normalized_originals = self._normalize_results(
            [{"content": f, "source": "original"} for f in original_findings]
        )
        
        all_normalized_results = normalized_originals + normalized_follow_ups

        # 2. Единовременная дедупликация всех результатов
        deduplicated_results = self._deduplicate_content(all_normalized_results)
        logger.info(
            f"После слияния и дедупликации, всего {len(deduplicated_results)}/{len(all_normalized_results)} уникальных результатов"
        )
        
        # 3. Семантическая группировка (если включена)
        if self.config.enable_semantic_grouping:
            grouped_results = self._group_by_semantic_similarity(deduplicated_results)
        else:
            grouped_results = [[result] for result in deduplicated_results]

        # 5. Слияние внутри группы
        merged_groups = []
        for group in grouped_results:
            merged_result = self._merge_group(group, query_context)
            if merged_result:
                merged_groups.append(merged_result)

        # 6. Оценка качества и сортировка
        scored_results = self._score_and_rank_results(merged_groups)

        # 7. Финальная фильтрация и ограничение количества
        final_results = self._apply_final_filters(scored_results)

        # Извлечение объединенного контента и наблюдений
        merged_findings = [res.content for res in final_results]
        merged_observations = [
            {"content": res.content, "source": res.sources} for res in final_results
        ]
        merge_stats = self.get_merge_statistics(final_results)

        logger.info(
            f"Слияние завершено, из {len(follow_up_results)} результатов объединено в {len(final_results)}"
        )
        return merged_findings, merged_observations, merge_stats

    def _normalize_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Стандартизация формата результатов"""
        normalized = []

        for i, result in enumerate(results):
            # Извлечение контента
            content = ""
            if isinstance(result, dict):
                content = result.get("content", "")
                if not content:
                    # Попытка использовать другие поля
                    content = result.get("observation", "")
                    if not content and "update" in result:
                        update = result["update"]
                        if isinstance(update, dict):
                            observations = update.get("observations", [])
                            if observations:
                                content = " ".join(
                                    (
                                        obs.get("content", str(obs))
                                        if isinstance(obs, dict)
                                        else str(obs)
                                    )
                                    for obs in observations
                                )
            elif isinstance(result, str):
                content = result
            else:
                content = str(result)

            # Очистка контента
            content = self._clean_content(content)

            if len(content) >= self.config.min_content_length:
                normalized.append(
                    {
                        "content": content,
                        "source": (
                            result.get("source", f"follow_up_{i+1}")
                            if isinstance(result, dict)
                            else f"follow_up_{i+1}"
                        ),
                        "metadata": result if isinstance(result, dict) else {},
                        "original_index": i,
                    }
                )

        return normalized

    def _clean_content(self, content: str) -> str:
        """Очистка текста контента"""
        if not content:
            return ""

        # Удаление маркеров Follow-up
        content = re.sub(r"\[Follow-up \d+\.\d+\]\s*", "", content)

        # Удаление лишних пробельных символов
        content = re.sub(r"\s+", " ", content).strip()

        # Удаление повторяющихся начал предложений
        lines = content.split("\n")
        cleaned_lines = []
        prev_line = ""

        for line in lines:
            line = line.strip()
            if line and line != prev_line:
                cleaned_lines.append(line)
                prev_line = line

        return "\n".join(cleaned_lines)

    def _deduplicate_content(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Дедупликация на основе отпечатков контента"""
        deduplicated = []
        seen_fingerprints = set()

        for result in results:
            content = result["content"]
            fingerprint = self._generate_content_fingerprint(content)

            if fingerprint not in seen_fingerprints:
                seen_fingerprints.add(fingerprint)
                deduplicated.append(result)
            else:
                logger.debug(f"Обнаружен дублирующийся контент, пропущено: {content[:100]}...")

        logger.info(f"После дедупликации сохранено {len(deduplicated)}/{len(results)} результатов")
        return deduplicated

    def _generate_content_fingerprint(self, content: str) -> str:
        """Генерация отпечатка контента"""
        # Стандартизация текста
        normalized = re.sub(r"\W+", " ", content.lower()).strip()

        # Извлечение ключевых слов (с удалением стоп-слов)
        words = normalized.split()
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is", "of", "the", "in", "and", "a", # Eng
            "是", "的", "了", "在", "有", "和", "与", # Chi
            "и", "в", "на", "с", "о", "не", "что", "это", # Rus
        }
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]

        # Взять первые 20 ключевых слов для генерации отпечатка
        key_content = " ".join(sorted(keywords[:20]))
        return hashlib.md5(key_content.encode()).hexdigest()

    def _filter_against_original(
        self, results: List[Dict[str, Any]], original_findings: List[str]
    ) -> List[Dict[str, Any]]:
        """Фильтрация контента, дублирующего исходные данные"""
        if not original_findings:
            return results

        # Проверка на дублирование с исходными данными
        original_fingerprints = set()
        for finding in original_findings:
            if (
                isinstance(finding, str)
                and len(finding) >= self.config.min_content_length
            ):
                fingerprint = self._generate_content_fingerprint(finding)
                original_fingerprints.add(fingerprint)

        # Фильтрация дублирующегося контента
        filtered = []
        for result in results:
            content_fingerprint = self._generate_content_fingerprint(result["content"])

            # Проверка на дублирование с исходными данными
            is_duplicate = content_fingerprint in original_fingerprints

            # Проверка схожести
            if not is_duplicate:
                max_similarity = 0.0
                for finding in original_findings:
                    if isinstance(finding, str):
                        similarity = self._calculate_similarity(
                            result["content"], finding
                        )
                        max_similarity = max(max_similarity, similarity)

                is_duplicate = max_similarity > self.config.similarity_threshold

            if not is_duplicate:
                filtered.append(result)
            else:
                self._stats["deduplication_count"] += 1
                logger.debug(f"Фильтрация дублирующегося контента: {result['content'][:100]}...")

        logger.info(f"После фильтрации по исходным данным сохранено {len(filtered)}/{len(results)} результатов")
        return filtered

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Вычисление схожести двух текстов (с кэшированием)
        """
        # Создание ключа кэша
        cache_key = (hash(text1), hash(text2))
        if cache_key[0] > cache_key[1]:
            cache_key = (cache_key[1], cache_key[0])

        # Проверка кэша
        if cache_key in self._similarity_cache:
            self._stats["cache_hits"] += 1
            return self._similarity_cache[cache_key]

        self._stats["cache_misses"] += 1

        try:
            # Использование простого расчета пересечения слов
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            if not words1 or not words2:
                similarity = 0.0
            else:
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                similarity = intersection / union if union > 0 else 0.0

            # Кэширование результата
            self._similarity_cache[cache_key] = similarity

            # Ограничение размера кэша
            if len(self._similarity_cache) > 1000:
                # Очистка старой половины кэша
                keys_to_remove = list(self._similarity_cache.keys())[:500]
                for key in keys_to_remove:
                    del self._similarity_cache[key]

            return similarity

        except Exception as e:
            logger.warning(f"Ошибка вычисления схожести: {e}")
            return 0.0

    def _group_by_semantic_similarity(
        self, results: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Группировка по семантической схожести"""
        if not results:
            return []

        groups = []
        ungrouped = results.copy()

        while ungrouped:
            # Взять первый элемент как основу для группы
            seed = ungrouped.pop(0)
            current_group = [seed]

            # Поиск схожих результатов
            remaining = []
            for result in ungrouped:
                similarity = self._calculate_similarity(
                    seed["content"], result["content"]
                )

                if similarity > self.config.similarity_threshold:
                    current_group.append(result)
                else:
                    remaining.append(result)

            ungrouped = remaining
            groups.append(current_group)

        logger.info(f"Семантическая группировка завершена: {len(results)} результатов разделено на {len(groups)} групп")
        return groups

    def _merge_group(
        self, group: List[Dict[str, Any]], query_context: Optional[str] = None
    ) -> Optional[MergedResult]:
        """Слияние результатов внутри одной группы"""
        if not group:
            return None

        if len(group) == 1:
            # Прямое преобразование одиночного результата
            result = group[0]
            return MergedResult(
                content=result["content"],
                sources=[result["source"]],
                confidence_score=0.8,
                relevance_score=0.7,
                key_points=self._extract_key_points(result["content"]),
                metadata=result["metadata"],
                original_count=1,
                merged_count=1,
            )

        # Необходимо слить несколько результатов
        merged_content = self._merge_content([r["content"] for r in group])
        sources = [r["source"] for r in group]

        # Слияние метаданных
        merged_metadata = {}
        for result in group:
            merged_metadata.update(result["metadata"])

        # Расчет коэффициентов уверенности и релевантности
        confidence_score = min(0.9, 0.6 + len(group) * 0.1)  # больше источников = выше уверенность
        relevance_score = self._calculate_relevance_score(merged_content, query_context)

        return MergedResult(
            content=merged_content,
            sources=sources,
            confidence_score=confidence_score,
            relevance_score=relevance_score,
            key_points=self._extract_key_points(merged_content),
            metadata=merged_metadata,
            original_count=len(group),
            merged_count=1,
        )

    def _merge_content(self, contents: List[str]) -> str:
        """Интеллектуальное слияние нескольких фрагментов контента"""
        if not contents:
            return ""

        if len(contents) == 1:
            return contents[0]

        # Сортировка по длине, приоритет у более детального контента
        sorted_contents = sorted(contents, key=len, reverse=True)

        # Извлечение всех предложений
        all_sentences = []
        for content in sorted_contents:
            sentences = [s.strip() for s in content.split(".") if s.strip()]
            all_sentences.extend(sentences)

        # Дедупликация предложений (с сохранением порядка)
        unique_sentences = []
        seen_sentences = set()

        for sentence in all_sentences:
            sentence_key = sentence.lower().strip()
            if sentence_key not in seen_sentences and len(sentence) > 10:
                seen_sentences.add(sentence_key)
                unique_sentences.append(sentence)

        # Реорганизация контента
        merged = ". ".join(unique_sentences)
        if merged and not merged.endswith("."):
            merged += "."

        return merged

    def _extract_key_points(self, content: str) -> List[str]:
        """Извлечение ключевых моментов"""
        if not content:
            return []

        sentences = [s.strip() for s in content.split(".") if s.strip()]

        # Простое извлечение ключевых моментов: выбор более длинных и информативных предложений
        key_points = []
        for sentence in sentences[:5]:  # не более 5 ключевых моментов
            if (
                len(sentence) > 20
                and len(sentence) < 200
                and any(char.isdigit() or char.isupper() for char in sentence)
            ):
                key_points.append(sentence)

        return key_points[:3]  # не более 3 ключевых моментов

    def _calculate_relevance_score(
        self, content: str, query_context: Optional[str] = None
    ) -> float:
        """Расчет коэффициента релевантности"""
        if not query_context:
            return 0.7  # оценка по умолчанию

        # Простое сопоставление по ключевым словам
        content_lower = content.lower()
        context_lower = query_context.lower()

        # Извлечение ключевых слов
        context_words = set(re.findall(r"\w+", context_lower))
        content_words = set(re.findall(r"\w+", content_lower))

        if not context_words:
            return 0.7

        # Расчет доли пересечения
        intersection = context_words.intersection(content_words)
        relevance = len(intersection) / len(context_words)

        return min(1.0, relevance + 0.3)  # базовая оценка + степень совпадения

    def _score_and_rank_results(
        self, results: List[MergedResult]
    ) -> List[MergedResult]:
        """Оценка и сортировка результатов"""
        if not results:
            return []

        # Расчет итоговой оценки
        for result in results:
            # итоговая оценка = уверенность * 0.4 + релевантность * 0.4 + качество контента * 0.2
            content_quality = self._calculate_content_quality(result.content)

            composite_score = (
                result.confidence_score * 0.4
                + result.relevance_score * 0.4
                + content_quality * 0.2
            )

            # Обновление метаданных
            result.metadata["composite_score"] = composite_score
            result.metadata["content_quality"] = content_quality

        # Сортировка по итоговой оценке
        sorted_results = sorted(
            results, key=lambda x: x.metadata.get("composite_score", 0), reverse=True
        )

        return sorted_results

    def _calculate_content_quality(self, content: str) -> float:
        """Расчет оценки качества контента"""
        if not content:
            return 0.0

        score = 0.0

        # Оценка по длине (предпочтительна средняя длина)
        length = len(content)
        if 100 <= length <= 500:
            score += 0.3
        elif 500 < length <= 1000:
            score += 0.2
        elif length > 50:
            score += 0.1

        # Оценка плотности информации
        if re.search(r"\d+", content):  # содержит цифры
            score += 0.2

        if re.search(r"[A-Z]{2,}", content):  # содержит аббревиатуры
            score += 0.1

        # Степень структурированности
        sentences = content.split(".")
        if len(sentences) >= 3:
            score += 0.2

        # Плотность профессиональной терминологии
        words = content.split()
        if not words:
            return min(1.0, score)
        long_words = [w for w in words if len(w) > 6]
        if len(long_words) / len(words) > 0.2:
            score += 0.2

        return min(1.0, score)

    def _apply_final_filters(self, results: List[MergedResult]) -> List[MergedResult]:
        """Применение финальных фильтров"""
        if not results:
            return []

        # Фильтрация низкокачественных результатов
        filtered = [
            result
            for result in results
            if result.metadata.get("content_quality", 0)
            >= self.config.quality_threshold
        ]

        # Статистика отфильтрованного количества
        self._stats["quality_filtered_count"] += len(results) - len(filtered)

        # Ограничение количества
        final_results = filtered[: self.config.max_merged_results]

        logger.info(
            f"Финальная фильтрация: {len(results)} -> {len(filtered)} -> {len(final_results)}"
        )
        return final_results

    def get_merge_statistics(self, results: List[MergedResult]) -> Dict[str, Any]:
        """Получение статистики слияния"""
        if not results:
            return {}

        total_original = sum(r.original_count for r in results)
        total_merged = len(results)

        avg_confidence = sum(r.confidence_score for r in results) / len(results)
        avg_relevance = sum(r.relevance_score for r in results) / len(results)

        source_diversity = len(
            set(source for result in results for source in result.sources)
        )

        return {
            "total_original_results": total_original,
            "total_merged_results": total_merged,
            "compression_ratio": (
                total_original / total_merged if total_merged > 0 else 0
            ),
            "average_confidence": avg_confidence,
            "average_relevance": avg_relevance,
            "source_diversity": source_diversity,
            "quality_distribution": {
                "high": len(
                    [r for r in results if r.metadata.get("content_quality", 0) > 0.7]
                ),
                "medium": len(
                    [
                        r
                        for r in results
                        if 0.4 <= r.metadata.get("content_quality", 0) <= 0.7
                    ]
                ),
                "low": len(
                    [r for r in results if r.metadata.get("content_quality", 0) < 0.4]
                ),
            },
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Получение статистики производительности
        """
        return {
            "total_merges": self._stats["total_merges"],
            "cache_performance": {
                "hits": self._stats["cache_hits"],
                "misses": self._stats["cache_misses"],
                "hit_rate": (
                    self._stats["cache_hits"]
                    / (self._stats["cache_hits"] + self._stats["cache_misses"])
                    if (self._stats["cache_hits"] + self._stats["cache_misses"]) > 0
                    else 0
                ),
                "cache_size": len(self._similarity_cache),
            },
            "filtering_stats": {
                "deduplication_count": self._stats["deduplication_count"],
                "quality_filtered_count": self._stats["quality_filtered_count"],
            },
            "config": {
                "similarity_threshold": self.config.similarity_threshold,
                "min_content_length": self.config.min_content_length,
                "max_merged_results": self.config.max_merged_results,
                "enable_semantic_grouping": self.config.enable_semantic_grouping,
                "enable_intelligent_merging": self.config.enable_intelligent_merging,
                "enable_deduplication": self.config.enable_deduplication,
                "enable_quality_filtering": self.config.enable_quality_filtering,
            },
        }

    def reset_stats(self):
        """
        Сброс статистики производительности
        """
        self._stats = {
            "total_merges": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "deduplication_count": 0,
            "quality_filtered_count": 0,
        }
        self._similarity_cache.clear()
        self._content_fingerprints.clear()
