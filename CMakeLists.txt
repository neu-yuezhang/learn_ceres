cmake_minimum_required(VERSION 3.0.0)
project(ceres VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(test_ceres0 test_ceres0.cpp)
target_link_libraries(test_ceres0 ${CERES_LIBRARIES} ${OpenCV_LIBS})

add_executable(test_ceres1 test_ceres1.cpp)
target_link_libraries(test_ceres1 ${CERES_LIBRARIES} ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
