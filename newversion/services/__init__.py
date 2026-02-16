"""
Services package for newversion module.
"""

from newversion.services.gemini_rule_generator import GeminiRuleGenerator
from newversion.services import rule_base_mongodb

__all__ = ["GeminiRuleGenerator", "rule_base_mongodb"]
