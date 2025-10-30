import pyrealsense2 as rs

realsense_ctx = rs.context()
connected_devices = []
for i in range(len(realsense_ctx.devices)):
    device = realsense_ctx.devices[i]
    serial_number = device.get_info(rs.camera_info.serial_number)
    camera_name = device.get_info(rs.camera_info.name)
    
    print(f"[{camera_name}] --> {serial_number}")
    # connected_devices.append((camera_name, serial_number))
