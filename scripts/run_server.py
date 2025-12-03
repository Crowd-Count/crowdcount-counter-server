#!/usr/bin/env python3
"""
Main entry point for the CrowdCount Counter Server
Starts the stream orchestrator which manages MediaMTX and stream processors
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from streaming.stream_orchestrator import StreamOrchestrator
from processors import (
    ExampleProcessor,
    PassthroughProcessor,
    DisplayProcessor,
    MultiStreamDisplayProcessor,
    PeopleCounterProcessor
)
import multiprocessing


# Available processor classes
PROCESSORS = {
    'example': ExampleProcessor,
    'passthrough': PassthroughProcessor,
    'display': DisplayProcessor,
    'multi': MultiStreamDisplayProcessor,
    'counter': PeopleCounterProcessor
}


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='CrowdCount Counter Server - Stream Reception & Processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Processors:
  example      - Example processor with edge detection demo
  passthrough  - Minimal processor that just logs frames
  display      - Shows stream in a window with info overlay
  multi        - Lightweight display for monitoring multiple streams
  counter      - People counter with YOLO detection and tracking

Examples:
  # Run with people counter (YOLO-based person counting)
  python3 scripts/run_server.py --processor counter

  # Run with display processor (shows video in window)
  python3 scripts/run_server.py --processor display

  # Run with example processor
  python3 scripts/run_server.py --processor example

  # Run with multi-stream display
  python3 scripts/run_server.py --processor multi
        """
    )

    parser.add_argument(
        '--processor', '-p',
        type=str,
        default='display',
        choices=PROCESSORS.keys(),
        help='Processor type to use for streams (default: display)'
    )

    args = parser.parse_args()

    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║                                                            ║
    ║         CrowdCount Counter Server                          ║
    ║         Stream Reception & Processing System               ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)

    # Get selected processor class
    processor_class = PROCESSORS[args.processor]

    print(f"Processor: {processor_class.__name__}")
    print()

    # Set up multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    # Create orchestrator with selected processor
    orchestrator = StreamOrchestrator(processor_class=PeopleCounterProcessor)

    print("Starting orchestrator...")
    print("Press Ctrl+C to stop")
    print()

    # Run (this blocks until stopped)
    orchestrator.run()


if __name__ == "__main__":
    main()
