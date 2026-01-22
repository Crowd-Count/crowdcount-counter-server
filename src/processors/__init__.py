"""
Stream Processors Package
Contains base and example stream processors for handling video streams
"""

from .base_processor import BaseStreamProcessor
from .example_processor import ExampleProcessor, PassthroughProcessor
from .display_processor import DisplayProcessor, MultiStreamDisplayProcessor
from .people_counter_processor import PeopleCounterProcessor
from .frame_buffer import FrameJitterBuffer

__all__ = [
    'BaseStreamProcessor',
    'ExampleProcessor',
    'PassthroughProcessor',
    'DisplayProcessor',
    'MultiStreamDisplayProcessor',
    'PeopleCounterProcessor',
    'FrameJitterBuffer'
]
