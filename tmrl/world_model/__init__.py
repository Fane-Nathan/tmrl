"""Experimental latent world-model training path for Trackmania.

This package intentionally lives beside the original SAC/REDQ pipeline so the
existing TMRL baseline remains unchanged and directly comparable.
"""

from tmrl.world_model.config import WorldModelConfig

__all__ = ["WorldModelConfig"]
