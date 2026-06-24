"""
embeddings/factory.py
======================
This module acts as a structural proxy for the Embeddings Factory.
The actual implementation resides in config.py to prevent circular imports
between the configuration layer and the components.
"""

from config import get_embeddings

__all__ = ["get_embeddings"]
