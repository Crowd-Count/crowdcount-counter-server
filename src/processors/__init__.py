"""
Stream Processors Package
Contains base and example stream processors for handling video streams
"""

from .base_processor import BaseStreamProcessor
from .example_processor import ExampleProcessor, PassthroughProcessor

__all__ = ['BaseStreamProcessor', 'ExampleProcessor', 'PassthroughProcessor']
