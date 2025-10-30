#!/usr/bin/env python3
import zmq
import time
import numpy as np

class RobotController:
    def __init__(self):
        """Initialize robot control interface here"""
        # This is where you would add code to connect to your specific robot
        self.connected = False
        print("Robot controller initialized")
    
    def connect(self):
        """Connect to robot hardware"""
        # Replace with actual connection code
        self.connected = True
        print("Robot connected")
    
    def move_to_pose(self, position, orientation):
        """Move robot to specified pose"""
        if not self.connected:
            print("Robot not connected")
            return False
        
        # Convert orientation (rotation matrix) to a format suitable for your robot
        # This depends on what your robot control API expects
        
        # Example of a simple robot movement
        print(f"Moving robot to position: {position}")
        print(f"With orientation matrix: {orientation}")
        
        # Here you would add the actual commands to move your robot
        return True
    
    def close(self):
        """Close connection to robot"""
        if self.connected:
            # Add cleanup code here
            self.connected = False
            print("Robot disconnected")

def main():
    # Initialize ZMQ context and socket
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5555")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all topics
    
    # Initialize robot controller
    robot = RobotController()
    robot.connect()
    
    # Set target marker ID to track
    target_marker_id = 0  # Change this to your target ArUco marker ID
    
    print(f"Robot control client started. Subscribed to tcp://localhost:5555")
    print(f"Tracking ArUco marker ID: {target_marker_id}")
    
    try:
        while True:
            # Receive pose data
            try:
                data = socket.recv_json(flags=zmq.NOBLOCK)
                timestamp = data["timestamp"]
                poses = data["poses"]
                
                # Check if our target marker is detected
                if str(target_marker_id) in poses:
                    target_pose = poses[str(target_marker_id)]
                    position = target_pose["position"]
                    orientation = target_pose["rotation_matrix"]
                    
                    print(f"Received pose for marker {target_marker_id} at time {timestamp:.2f}")
                    
                    # Move robot to the detected pose
                    robot.move_to_pose(position, orientation)
                else:
                    print(f"Target marker {target_marker_id} not detected")
                
            except zmq.Again:
                # No message received yet, continue
                pass
            
            # Rate limiting
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("Shutting down")
    
    finally:
        # Clean up
        robot.close()
        socket.close()
        context.term()

if __name__ == "__main__":
    main()
