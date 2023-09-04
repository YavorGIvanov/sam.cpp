#include "imgui-extra/imgui_impl.h"

#include "imgui/backends/imgui_impl_sdl2.h"
#include "imgui/backends/imgui_impl_opengl3.h"

#include <SDL.h>

bool ImGui_PreInit() {
    // Decide GL+GLSL versions
#if __APPLE__
    // GL 3.2 Core + GLSL 150
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG); // Always required on Mac
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
#elif __EMSCRIPTEN__
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
#ifdef USE_LINE_SHADER
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
#else
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
#endif
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#else
    // GL 3.0 + GLSL 130
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#endif

    // Create window with graphics context
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

    return true;
}

ImGuiContext* ImGui_Init(SDL_Window* window, SDL_GLContext gl_context) {
    // Decide GL+GLSL versions
#if __APPLE__
    // GL 3.2 Core + GLSL 150
    const char* glsl_version = "#version 150";
#elif __EMSCRIPTEN__
#ifdef USE_LINE_SHADER
    const char* glsl_version = "#version 300 es";
#else
    const char* glsl_version = "#version 100";
#endif
#else
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
#endif

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    auto ctx = ImGui::CreateContext();
    ImGui::SetCurrentContext(ctx);

    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls

    // Setup Platform/Renderer bindings
    bool res = true;
    res &= ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
    res &= ImGui_ImplOpenGL3_Init(glsl_version);

    return res ? ctx : nullptr;
}

void ImGui_Shutdown() { ImGui_ImplOpenGL3_Shutdown(); ImGui_ImplSDL2_Shutdown(); }
void ImGui_NewFrame(SDL_Window* window) { ImGui_ImplOpenGL3_NewFrame(); ImGui_ImplSDL2_NewFrame(window); }
bool ImGui_ProcessEvent(const SDL_Event* event) { return ImGui_ImplSDL2_ProcessEvent(event); }

void ImGui_RenderDrawData(ImDrawData* draw_data)    { ImGui_ImplOpenGL3_RenderDrawData(draw_data); }

bool ImGui_CreateFontsTexture()     { return ImGui_ImplOpenGL3_CreateFontsTexture(); }
void ImGui_DestroyFontsTexture()    { ImGui_ImplOpenGL3_DestroyFontsTexture(); }
bool ImGui_CreateDeviceObjects()    { return ImGui_ImplOpenGL3_CreateDeviceObjects(); }
void ImGui_DestroyDeviceObjects()   { ImGui_ImplOpenGL3_DestroyDeviceObjects(); }
