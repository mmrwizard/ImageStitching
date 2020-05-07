file(GLOB SOURCES "*.cpp" "*.h")
h2o_add_executable(ImageStitching ${SOURCES})
target_link_libraries(ImageStitching PRIVATE h2o_base h2o_match)


