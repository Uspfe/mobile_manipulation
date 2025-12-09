cmake_minimum_required(VERSION 3.0.2)
project(mm_utils)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
   scripts/plot_logs.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )
