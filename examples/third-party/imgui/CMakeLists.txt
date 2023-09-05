if (APPLE)
    set(ADDITIONAL_LIBRARIES "-framework Cocoa")
else (APPLE)
    unset(ADDITIONAL_LIBRARIES)
endif (APPLE)

add_library(imgui
    imgui/imgui.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_demo.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
    )

target_include_directories(imgui INTERFACE
    .
    )

target_include_directories(imgui PRIVATE
    imgui
    )

target_link_libraries(imgui PRIVATE
    ${ADDITIONAL_LIBRARIES}
    )

target_compile_features(imgui PUBLIC cxx_std_11)

# target_compile_definitions(imgui PUBLIC
#     IMGUI_USER_CONFIG="../imgui-extra/imconfig-vtx32.h"
#     )
# if (EMSCRIPTEN)
#     add_library(imgui-sdl2
#         imgui-extra/imgui_impl.cpp
#         imgui/backends/imgui_impl_sdl.cpp
#         imgui/backends/imgui_impl_opengl3.cpp
#         )

#     target_include_directories(imgui-sdl2 PRIVATE
#         imgui
#         imgui-extra
#         )

#     target_link_libraries(imgui-sdl2 PUBLIC
#         imgui
#         )
# else()
find_package(OpenGL REQUIRED)

add_library(imgui-sdl2
    imgui-extra/imgui_impl.cpp
    imgui/backends/imgui_impl_sdl2.cpp
    imgui/backends/imgui_impl_opengl3.cpp
    )

target_include_directories(imgui-sdl2 PUBLIC
    ${SDL2_INCLUDE_DIRS}
    )

target_include_directories(imgui-sdl2 PRIVATE
    imgui
    imgui-extra
    )

target_link_libraries(imgui-sdl2 PUBLIC
    imgui
    ${OPENGL_LIBRARIES}
    ${SDL2_LIBRARIES}
    )

target_link_libraries(imgui-sdl2 PRIVATE
    ${CMAKE_DL_LIBS}
    ${ADDITIONAL_LIBRARIES}
        )
# endif()
