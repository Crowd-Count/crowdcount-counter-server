#!/usr/bin/env python3
"""
Main entry point for the CrowdCount Counter Server
Starts the stream orchestrator which manages MediaMTX and stream processors
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from streaming.stream_orchestrator import StreamOrchestrator
from processors import ExampleProcessor
import multiprocessing


def main():
    """Main entry point"""
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║                                                            ║
    ║         CrowdCount Counter Server                          ║
    ║         Stream Reception & Processing System               ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)

    # Set up multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    # Create orchestrator with your desired processor
    # Change ExampleProcessor to your custom processor class when ready
    orchestrator = StreamOrchestrator(processor_class=ExampleProcessor)

    print("Starting orchestrator...")
    print("Press Ctrl+C to stop\n")

    # Run (this blocks until stopped)
    orchestrator.run()


if __name__ == "__main__":
    main()
