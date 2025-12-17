cmake_minimum_required(VERSION 3.0.2)
project(mm_control)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
            scripts/generate_acados_code.py
            tests/test_robot_model.py
            DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
          )
