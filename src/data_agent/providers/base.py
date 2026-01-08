"""
LLM Provider Base

Provider-agnostic LLM configuration and factory.
Defaults to OpenAI but supports easy switching to other providers.
Supports OpenAI-compatible APIs (Ollama, vLLM, LocalAI, Azure, etc.)
"""

from typing import Optional, Literal, Dict, Any
from dataclasses import dataclass, field
from langchain_core.language_models import BaseChatModel


@dataclass
class LLMConfig:
    """Configuration for LLM provider.
    
    Supports OpenAI-compatible APIs via base_url parameter.
    
    Examples:
        # Standard OpenAI
        config = LLMConfig(provider="openai", model="gpt-4o")
        
        # Ollama (OpenAI-compatible)
        config = LLMConfig(
            provider="openai_compatible",
            base_url="http://localhost:11434/v1",
            model="qwen2.5:7b",
            api_key="ollama"  # Ollama doesn't need real key
        )
        
        # Azure OpenAI
        config = LLMConfig(
            provider="azure",
            base_url="https://your-resource.openai.azure.com",
            model="gpt-4o",
            api_key="your-azure-key",
            extra_params={"api_version": "2024-02-01", "deployment_name": "my-deployment"}
        )
        
        # vLLM / LocalAI / Other OpenAI-compatible
        config = LLMConfig(
            provider="openai_compatible",
            base_url="http://localhost:8000/v1",
            model="your-model",
            api_key="not-needed"
        )
    """
    
    provider: Literal["openai", "anthropic", "google", "openai_compatible", "azure", "ollama"] = "openai"
    model: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # For OpenAI-compatible APIs
    extra_params: Optional[Dict[str, Any]] = None  # Azure deployment_name, api_version, etc.
    
    # Provider-specific defaults
    _default_models: dict = field(default_factory=lambda: {
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-20250514",
        "google": "gemini-1.5-pro",
        "openai_compatible": "gpt-3.5-turbo",
        "azure": "gpt-4o",
        "ollama": "qwen2.5:7b",
    }, repr=False)
    
    def get_model(self) -> str:
        """Get the model name, using default if not specified."""
        if self.model:
            return self.model
        return self._default_models.get(self.provider, "gpt-4o")


# Global default configuration
_default_config: Optional[LLMConfig] = None


def configure_llm(config: LLMConfig) -> None:
    """
    Set the default LLM configuration.
    
    Args:
        config: LLMConfig instance to use as default
    """
    global _default_config
    _default_config = config


def get_llm(config: Optional[LLMConfig] = None) -> BaseChatModel:
    """
    Get a LangChain chat model instance.
    
    Args:
        config: Optional configuration, uses global default if not provided
        
    Returns:
        BaseChatModel instance for the configured provider
    """
    cfg = config or _default_config or LLMConfig()
    
    if cfg.provider == "openai":
        return _get_openai_llm(cfg)
    elif cfg.provider in ("openai_compatible", "ollama"):
        return _get_openai_compatible_llm(cfg)
    elif cfg.provider == "azure":
        return _get_azure_llm(cfg)
    elif cfg.provider == "anthropic":
        return _get_anthropic_llm(cfg)
    elif cfg.provider == "google":
        return _get_google_llm(cfg)
    else:
        raise ValueError(f"Unknown provider: {cfg.provider}")


def _get_openai_llm(config: LLMConfig) -> BaseChatModel:
    """Get OpenAI chat model."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai not installed. "
            "Install with: pip install 'vanna-langgraph[openai]'"
        )
    
    kwargs = {
        "model": config.get_model(),
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    
    return ChatOpenAI(**kwargs)


def _get_openai_compatible_llm(config: LLMConfig) -> BaseChatModel:
    """Get OpenAI-compatible chat model (Ollama, vLLM, LocalAI, etc.)."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai not installed. "
            "Install with: pip install 'vanna-langgraph[openai]'"
        )
    
    # For Ollama, set default base_url if not provided
    base_url = config.base_url
    if config.provider == "ollama" and not base_url:
        base_url = "http://localhost:11434/v1"
    
    if not base_url:
        raise ValueError(
            "base_url is required for openai_compatible provider. "
            "Example: base_url='http://localhost:8000/v1'"
        )
    
    kwargs = {
        "model": config.get_model(),
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "base_url": base_url,
    }
    
    # For OpenAI-compatible APIs, a dummy key may be needed
    api_key = config.api_key or "not-needed"
    kwargs["api_key"] = api_key
    
    return ChatOpenAI(**kwargs)


def _get_azure_llm(config: LLMConfig) -> BaseChatModel:
    """Get Azure OpenAI chat model."""
    try:
        from langchain_openai import AzureChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai not installed. "
            "Install with: pip install 'vanna-langgraph[openai]'"
        )
    
    extra = config.extra_params or {}
    
    kwargs = {
        "model": config.get_model(),
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["azure_endpoint"] = config.base_url
    if "api_version" in extra:
        kwargs["api_version"] = extra["api_version"]
    if "deployment_name" in extra:
        kwargs["deployment_name"] = extra["deployment_name"]
    
    return AzureChatOpenAI(**kwargs)


def _get_anthropic_llm(config: LLMConfig) -> BaseChatModel:
    """Get Anthropic chat model."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError(
            "langchain-anthropic not installed. "
            "Install with: pip install 'vanna-langgraph[anthropic]'"
        )
    
    kwargs = {
        "model": config.get_model(),
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.api_key:
        kwargs["api_key"] = config.api_key
    
    return ChatAnthropic(**kwargs)


def _get_google_llm(config: LLMConfig) -> BaseChatModel:
    """Get Google Generative AI chat model."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError(
            "langchain-google-genai not installed. "
            "Install with: pip install 'vanna-langgraph[google]'"
        )
    
    kwargs = {
        "model": config.get_model(),
        "temperature": config.temperature,
        "max_output_tokens": config.max_tokens,
    }
    if config.api_key:
        kwargs["google_api_key"] = config.api_key
    
    return ChatGoogleGenerativeAI(**kwargs)
