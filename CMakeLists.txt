cmake_minimum_required(VERSION 3.20)
project(MiniVectorDB LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_executable(minivectordb
  src/main.cpp
  src/sql.cpp
  src/storage.cpp
  src/vector_index.cpp
  src/engine.cpp
)

target_compile_options(minivectordb PRIVATE -Wall -Wextra -Wpedantic)
