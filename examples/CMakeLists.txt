set(SAM_TARGET sam)

add_executable(${SAM_TARGET}
    main.cpp
    stb_image.h
    stb_image_write.h
)

find_package(SDL2 REQUIRED)
string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)

add_subdirectory(third-party)

install(TARGETS ${SAM_TARGET} RUNTIME)
target_link_libraries(${SAM_TARGET} PRIVATE sam.cpp imgui-sdl2 ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${SAM_TARGET} PUBLIC cxx_std_11)

if(MSVC)
    target_compile_definitions(${SAM_TARGET} PRIVATE -D_CRT_SECURE_NO_WARNINGS=1)
endif()
