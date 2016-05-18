file(GLOB VRIPLIB_SRC
    "vrip/*.h"
    "vrip/*.cpp"
)
add_library(vriplib ${VRIPLIB_SRC})

file(GLOB PLY2VRI_SRC
    "*.h"
    "*.cpp"
    "*.c"
)
add_executable(ply2vri ${PLY2VRI_SRC})
target_link_libraries(ply2vri vriplib)
install(TARGETS ply2vri DESTINATION bin)
