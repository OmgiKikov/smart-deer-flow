"""Pydantic Settings-based configuration models for automatic environment variable loading."""

from typing import Any, Dict, List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum


class ReportStyle(str, Enum):
    """Report style enumeration."""
    ACADEMIC = "academic"
    BUSINESS = "business"
    TECHNICAL = "technical"
    CASUAL = "casual"


class SummaryType(str, Enum):
    """Summary type enumeration."""
    COMPREHENSIVE = "comprehensive"
    KEY_POINTS = "key_points"
    ABSTRACT = "abstract"


class IsolationLevel(str, Enum):
    """Researcher isolation level enumeration."""
    MINIMAL = "minimal"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class LLMSettings(BaseSettings):
    """LLM configuration settings with automatic environment variable loading."""
    model_config = SettingsConfigDict(
        env_prefix="DEER_LLM_",
        case_sensitive=False,
        extra="allow"
    )
    
    basic_model: Optional[Any] = None
    reasoning_model: Optional[Any] = None
    reflection_model: Optional[Any] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = None
    timeout: int = Field(default=30, ge=1)


class DatabaseSettings(BaseSettings):
    """Database configuration settings with automatic environment variable loading."""
    model_config = SettingsConfigDict(
        env_prefix="DEER_DATABASE_",
        case_sensitive=False,
        extra="allow"
    )
    
    connection_string: Optional[str] = None
    pool_size: int = Field(default=10, ge=1)
    max_overflow: int = Field(default=20, ge=0)
    pool_timeout: int = Field(default=30, ge=1)


class AgentSettings(BaseSettings):
    """Agent configuration settings with automatic environment variable loading."""
    model_config = SettingsConfigDict(
        env_prefix="DEER_",
        case_sensitive=False,
        extra="allow"
    )
    
    max_plan_iterations: int = Field(default=1, ge=1)
    max_step_num: int = Field(default=3, ge=1)
    max_search_results: int = Field(default=3, ge=1)
    enable_deep_thinking: bool = False
    enable_parallel_execution: bool = True
    max_parallel_tasks: int = Field(default=3, ge=1)
    max_context_steps_parallel: int = Field(default=1, ge=1)
    disable_context_parallel: bool = False


class ResearchSettings(BaseSettings):
    """Research-specific configuration settings with automatic environment variable loading."""
    model_config = SettingsConfigDict(
        env_prefix="DEER_RESEARCHER_",
        case_sensitive=False,
        extra="allow"
    )
    
    enable_researcher_isolation: bool = Field(default=True, alias="DEER_ENABLE_RESEARCHER_ISOLATION")
    researcher_isolation_level: IsolationLevel = IsolationLevel.MODERATE
    researcher_max_local_context: int = Field(default=5000, ge=100)
    researcher_isolation_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    researcher_auto_isolation: bool = False
    researcher_isolation_metrics: bool = False
    max_context_steps_researcher: int = Field(default=2, ge=1, alias="DEER_MAX_CONTEXT_STEPS_RESEARCHER")


class ReflectionSettings(BaseSettings):
    """Reflection configuration settings with automatic environment variable loading."""
    model_config = SettingsConfigDict(
        env_prefix="DEER_REFLECTION_",
        case_sensitive=False,
        extra="allow"
    )
    
    enable_enhanced_reflection: bool = Field(default=True, alias="DEER_ENABLE_ENHANCED_REFLECTION")
    max_reflection_loops: int = Field(default=3, ge=1, alias="DEER_MAX_REFLECTION_LOOPS")
    reflection_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    reflection_trigger_threshold: int = Field(default=2, ge=1)
    reflection_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    enable_reflection_integration: bool = True
    enable_progressive_reflection: bool = True
    enable_reflection_metrics: bool = True


class IterativeResearchSettings(BaseSettings):
    """Iterative research configuration settings with automatic environment variable loading."""
    model_config = SettingsConfigDict(
        env_prefix="DEER_",
        case_sensitive=False,
        extra="allow"
    )
    
    max_follow_up_iterations: int = Field(default=3, ge=1)
    sufficiency_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    enable_iterative_research: bool = True
    max_queries_per_iteration: int = Field(default=3, ge=1)
    follow_up_delay_seconds: float = Field(default=1.0, ge=0.0)


class ContentSettings(BaseSettings):
    """Content processing configuration settings with automatic environment variable loading."""
    model_config = SettingsConfigDict(
        env_prefix="DEER_",
        case_sensitive=False,
        extra="allow"
    )
    
    enable_content_summarization: bool = True
    enable_smart_filtering: bool = True
    summary_type: SummaryType = SummaryType.COMPREHENSIVE


class AdvancedContextConfig(BaseSettings):
    """Advanced context management configuration with automatic environment variable loading."""
    model_config = SettingsConfigDict(
        env_prefix="DEER_",
        case_sensitive=False,
        extra="allow"
    )
    
    max_context_ratio: float = Field(default=0.6, ge=0.0, le=1.0)
    sliding_window_size: int = Field(default=5, ge=1)
    overlap_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
    compression_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    default_strategy: str = "adaptive"
    priority_weights: Dict[str, float] = Field(default_factory=lambda: {
        "critical": 1.0,
        "high": 0.7,
        "medium": 0.4,
        "low": 0.1,
    })
    enable_caching: bool = True
    enable_analytics: bool = True
    debug_mode: bool = False


class MCPSettings(BaseSettings):
    """MCP (Model Context Protocol) settings with automatic environment variable loading."""
    model_config = SettingsConfigDict(
        env_prefix="DEER_MCP_",
        case_sensitive=False,
        extra="allow"
    )
    
    enabled: bool = False
    servers: List[Dict[str, Any]] = Field(default_factory=list)
    timeout: int = Field(default=30, ge=1)


class AppSettings(BaseSettings):
    """Main application configuration with automatic environment variable loading."""
    model_config = SettingsConfigDict(
        env_prefix="DEER_",
        case_sensitive=False,
        validate_assignment=True,
        extra="allow",
        env_nested_delimiter="__"
    )
    
    # Core settings
    report_style: ReportStyle = ReportStyle.ACADEMIC
    resources: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Sub-configurations - these will be automatically populated from environment variables
    llm: LLMSettings = Field(default_factory=LLMSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    agents: AgentSettings = Field(default_factory=AgentSettings)
    research: ResearchSettings = Field(default_factory=ResearchSettings)
    reflection: ReflectionSettings = Field(default_factory=ReflectionSettings)
    iterative_research: IterativeResearchSettings = Field(default_factory=IterativeResearchSettings)
    content: ContentSettings = Field(default_factory=ContentSettings)
    advanced_context: AdvancedContextConfig = Field(default_factory=AdvancedContextConfig)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    
    # Model token limits
    model_token_limits: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    
    def get_llm_config(self) -> LLMSettings:
        """Get LLM-specific configuration."""
        return self.llm

    def get_agent_config(self) -> AgentSettings:
        """Get agent-specific configuration."""
        return self.agents

    def get_research_config(self) -> ResearchSettings:
        """Get research-specific configuration."""
        return self.research

    def get_reflection_config(self) -> ReflectionSettings:
        """Get reflection-specific configuration."""
        return self.reflection

    def get_content_config(self) -> ContentSettings:
        """Get content-specific configuration."""
        return self.content

    def get_advanced_context_config(self) -> AdvancedContextConfig:
        """Get advanced context configuration."""
        return self.advanced_context

    def get_mcp_config(self) -> MCPSettings:
        """Get MCP configuration."""
        return self.mcp