cmake_minimum_required(VERSION 3.4)
project(SentenceValidation VERSION 0.1.0)

include("cmake/CPM.cmake")
include("cmake/dlib.cmake")
include("cmake/googletest.cmake")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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


# testing
enable_testing()
include(GoogleTest)
file(GLOB TEST_FILES_CPP "${CMAKE_CURRENT_LIST_DIR}/test/*.cpp")
file(GLOB TEST_FILES_CX  "${CMAKE_CURRENT_LIST_DIR}/test/*.cx")
file(GLOB TEST_FILES     "${CMAKE_CURRENT_LIST_DIR}/test/*.c")
list(APPEND TEST_FILES ${TEST_FILES_CPP} ${TEST_FILES_CX})
foreach (test_file IN LISTS TEST_FILES)
    get_filename_component(filenamewe ${test_file} NAME_WE)
    string(CONCAT execname "test_" ${filenamewe})
    add_executable(${execname} ${test_file})
    set_property(TARGET ${execname} PROPERTY CXX_STANDARD 14)
    target_link_libraries(${execname} gtest_main)
    target_link_libraries(${execname} SentenceValidationLib)
    gtest_discover_tests(${execname})
endforeach()

