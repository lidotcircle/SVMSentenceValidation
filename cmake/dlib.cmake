
CPMAddPackage(
    NAME dlib
    GITHUB_REPOSITORY davisking/dlib
    DOWNLOAD_ONLY ON
    VERSION 19.22
)

if(dlib_ADDED)
    add_subdirectory(${dlib_SOURCE_DIR})
endif()

