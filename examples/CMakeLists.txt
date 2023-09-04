set(SAM_TARGET sam)

add_executable(${SAM_TARGET}
    main.cpp
    stb_image.h
    stb_image_write.h
)

find_package(glfw3 REQUIRED)

install(TARGETS ${SAM_TARGET} RUNTIME)
target_link_libraries(${SAM_TARGET} PRIVATE sam.cpp ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${SAM_TARGET} PUBLIC cxx_std_11)
