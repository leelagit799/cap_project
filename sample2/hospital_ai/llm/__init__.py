"""LLM access through LiteLLM — AWS Bedrock Nova Lite and Cohere Command R+."""

from hospital_ai.llm.gateway import Completion, LLMGateway, get_gateway, reset_gateway

__all__ = ["Completion", "LLMGateway", "get_gateway", "reset_gateway"]
