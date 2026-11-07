"""Provisional offline handoff checks. Does not import or execute a model."""
from .contract import ContractError, adapt_candidates, validate_bundle

__all__ = ['ContractError', 'adapt_candidates', 'validate_bundle']
