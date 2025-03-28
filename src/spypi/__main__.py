#!/usr/bin/env python
"""
Entry point for running the pi_sentinel application.
"""

import socket
from .camera import Camera
from .detection import DetectionProcessor
from .streaming import StreamingServer, StreamingHandler, StreamingOutput

# Get host IP address.
hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)
print("Your Computer Name is:", hostname)
print("Your Computer IP Address is:", IPAddr)


def main():
    # Configure the camera (you can adjust resolution to improve performance on Raspberry Pi).
    camera = Camera(width=640, height=480)

    # Create the streaming output object.
    output = StreamingOutput()

    # Initialize the detection processor.
    processor = DetectionProcessor(
        camera=camera,
        max_results=2,
        score_threshold=0.25,
        detection_interval=5,  # Increase interval to reduce load if needed.
        video_sleep=0.05,
        greeting_duration=3
    )
    # Start the detection thread.
    processor.start(output_callback=output.update_frame)

    # Start the streaming server.
    address = ('', 7123)
    server_instance = StreamingServer(address, StreamingHandler, output)
    print(f"Server running at http://{IPAddr}:7123")
    try:
        server_instance.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down server.")


if __name__ == '__main__':
    main()
