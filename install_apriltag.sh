git clone https://github.com/AprilRobotics/apriltag
mv apriltag local_apriltag
cd local_apriltag
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build