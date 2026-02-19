cmake_minimum_required(VERSION 3.10)
project(ur_description)

find_package(ament_cmake REQUIRED)

# If you have Xacro files you want to check during build (optional but good practice)
# find_package(ament_cmake_ros REQUIRED) 

# --- Install Rules ---
# This replicates your old "install(DIRECTORY...)" command.
# In ROS 2, we install to share/${PROJECT_NAME}
install(
  DIRECTORY 
    cfg 
    config 
    launch 
    meshes 
    urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()