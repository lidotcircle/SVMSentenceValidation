cmake_minimum_required(VERSION 3.4)
project(SentenceValidation VERSION 0.1.0)

include("cmake/CPM.cmake")
include("cmake/dlib.cmake")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)

file(GLOB_RECURSE SentenceValidationLib_SOURCES   CONFIGURE_DEPENDS ./lib/**.cpp)
file(GLOB_RECURSE SentenceValidationLib_SOURCES_C CONFIGURE_DEPENDS ./lib/**.c)
list(APPEND SentenceValidationLib_SOURCES ${SentenceValidationLib_SOURCES_C})
add_library(SentenceValidationLib STATIC ${SentenceValidationLib_SOURCES})
target_link_libraries(SentenceValidationLib PUBLIC dlib)
set_property(TARGET SentenceValidationLib PROPERTY CXX_STANDARD 14)
target_include_directories(SentenceValidationLib PUBLIC include)

set(SentenceValidation_SOURCES "")
list(APPEND SentenceValidation_SOURCES
	"src/main.cpp"
)
add_executable(SentenceValidation)
target_sources(SentenceValidation PRIVATE ${SentenceValidation_SOURCES})
target_link_libraries(SentenceValidation PRIVATE SentenceValidationLib)
set_property(TARGET SentenceValidation PROPERTY CXX_STANDARD 14)
target_include_directories(SentenceValidation PRIVATE include)

