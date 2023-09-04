#include "sam.h"

#include "ggml.h"
#include "ggml-alloc.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "imgui-extra/imgui_impl.h"

#define SDL_DISABLE_ARM_NEON_H 1
#include <SDL.h>
#include <SDL_opengl.h>

#include <string>
#include <thread>

static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}

static bool sam_image_load_from_file(const std::string & fname, sam_image_u8 & img) {
    int nx, ny, nc;
    auto data = stbi_load(fname.c_str(), &nx, &ny, &nc, 3);
    if (!data) {
        fprintf(stderr, "%s: failed to load '%s'\n", __func__, fname.c_str());
        return false;
    }

    img.nx = nx;
    img.ny = ny;
    img.data.resize(nx * ny * 3);
    memcpy(img.data.data(), data, nx * ny * 3);

    stbi_image_free(data);

    return true;
}

static const size_t tensor_alignment = 32;

struct sam_params {
    int32_t seed      = -1; // RNG seed
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());

    std::string model     = "models/sam-vit-b/ggml-model-f16.bin"; // model path
    std::string fname_inp = "img.jpg";
    std::string fname_out = "img.out";
};

static void sam_print_usage(int argc, char ** argv, const sam_params & params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1)\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "  -i FNAME, --inp FNAME\n");
    fprintf(stderr, "                        input file (default: %s)\n", params.fname_inp.c_str());
    fprintf(stderr, "  -o FNAME, --out FNAME\n");
    fprintf(stderr, "                        output file (default: %s)\n", params.fname_out.c_str());
    fprintf(stderr, "\n");
}

static bool sam_params_parse(int argc, char ** argv, sam_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-s" || arg == "--seed") {
            params.seed = std::stoi(argv[++i]);
        } else if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-m" || arg == "--model") {
            params.model = argv[++i];
        } else if (arg == "-i" || arg == "--inp") {
            params.fname_inp = argv[++i];
        } else if (arg == "-o" || arg == "--out") {
            params.fname_out = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            sam_print_usage(argc, argv, params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            sam_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

bool ImGui_BeginFrame(SDL_Window * window) {
    ImGui_NewFrame(window);

    return true;
}

bool ImGui_EndFrame(SDL_Window * window) {
    // Rendering
    int display_w, display_h;
    SDL_GetWindowSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    ImGui::Render();
    ImGui_RenderDrawData(ImGui::GetDrawData());

    SDL_GL_SwapWindow(window);

    return true;
}

GLuint createGLTexture(const sam_image_u8 & img, GLint format) {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same

    // Upload pixels into texture
#if defined(GL_UNPACK_ROW_LENGTH) && !defined(__EMSCRIPTEN__)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
#endif
    glTexImage2D(GL_TEXTURE_2D, 0, format, img.nx, img.ny, 0, format, GL_UNSIGNED_BYTE, img.data.data());

    return tex;
}

bool load_model_and_compute_embd_img(const sam_image_u8 & img, const sam_params & params, sam_model & model, sam_state & state, int64_t & t_load_us) {
    // preprocess to f32
    sam_image_f32 img1;
    if (!sam_image_preprocess(img, img1)) {
        fprintf(stderr, "%s: failed to preprocess image\n", __func__);
        return false;
    }
    fprintf(stderr, "%s: preprocessed image (%d x %d)\n", __func__, img1.nx, img1.ny);

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!sam_model_load(params.model, model)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return false;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    sam_state_init_img(model, state);

    // Encode the image
    {
        state.buf_compute_img_enc.resize(ggml_tensor_overhead()*GGML_MAX_NODES + ggml_graph_overhead());
        state.allocr = ggml_allocr_new_measure(tensor_alignment);
        struct ggml_cgraph * gf_measure = sam_encode_image(model, state, img1);
        if (!gf_measure) {
            fprintf(stderr, "%s: failed to encode image\n", __func__);
            return false;
        }

        size_t alloc_size = ggml_allocr_alloc_graph(state.allocr, gf_measure) + tensor_alignment;
        ggml_allocr_free(state.allocr);

        // recreate allocator with exact memory requirements
        state.buf_alloc_img_enc.resize(alloc_size);
        state.allocr = ggml_allocr_new(state.buf_alloc_img_enc.data(), state.buf_alloc_img_enc.size(), tensor_alignment);

        // compute the graph with the measured exact memory requirements from above
        ggml_allocr_reset(state.allocr);

        struct ggml_cgraph  * gf = sam_encode_image(model, state, img1);
        if (!gf) {
            fprintf(stderr, "%s: failed to encode image\n", __func__);
            return false;
        }

        ggml_allocr_alloc_graph(state.allocr, gf);

        ggml_graph_compute_helper(state.work_buffer, gf, params.n_threads);

        ggml_allocr_free(state.allocr);
        state.allocr = NULL;
        state.work_buffer.clear();
    }

    return true;
}

std::map<std::string, sam_image_u8> compute_masks(const sam_image_u8 & img, const sam_params & params, sam_point pt, sam_model & model, sam_state & state) {
    {
        state.buf_compute_fast.resize(ggml_tensor_overhead()*GGML_MAX_NODES + ggml_graph_overhead());
        state.allocr = ggml_allocr_new_measure(tensor_alignment);

        // measure memory requirements for the graph
        struct ggml_cgraph  * gf_measure = sam_build_fast_graph(model, state, img.nx, img.ny, pt);
        if (!gf_measure) {
            fprintf(stderr, "%s: failed to build fast graph to measure\n", __func__);
            return {};
        }

        size_t alloc_size = ggml_allocr_alloc_graph(state.allocr, gf_measure) + tensor_alignment;
        ggml_allocr_free(state.allocr);

        // recreate allocator with exact memory requirements
        state.buf_alloc_fast.resize(alloc_size);
        state.allocr = ggml_allocr_new(state.buf_alloc_fast.data(), state.buf_alloc_fast.size(), tensor_alignment);

        // compute the graph with the measured exact memory requirements from above
        ggml_allocr_reset(state.allocr);

        struct ggml_cgraph  * gf = sam_build_fast_graph(model, state, img.nx, img.ny, pt);
        if (!gf) {
            fprintf(stderr, "%s: failed to build fast graph\n", __func__);
            return {};
        }

        ggml_allocr_alloc_graph(state.allocr, gf);

        ggml_graph_compute_helper(state.work_buffer, gf, params.n_threads);

        //print_t_f32("iou_predictions", state.iou_predictions);
        //print_t_f32("low_res_masks", state.low_res_masks);
        ggml_allocr_free(state.allocr);
        state.allocr = NULL;
        state.buf_compute_fast.clear();
        state.buf_alloc_fast.clear();
    }

    std::map<std::string, sam_image_u8> masks = sam_postprocess_masks(model.hparams, img.nx, img.ny, state);
    if (masks.empty()) {
        fprintf(stderr, "%s: failed to postprocess masks\n", __func__);
    }

    return masks;
}

int main_loop(sam_image_u8 & img, const sam_params & params, sam_model & model, sam_state & state) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "Error: %s\n", SDL_GetError());
        return -1;
    }

    ImGui_PreInit();

    const char * windowTitle = "SAM.cpp";
    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_OPENGL | SDL_WINDOW_ALLOW_HIGHDPI);
    SDL_Window * window = SDL_CreateWindow(windowTitle, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, img.nx, img.ny, window_flags);
    if (!window) {
        fprintf(stderr, "Error: %s\n", SDL_GetError());
        return -1;
    }

    void * gl_context = SDL_GL_CreateContext(window);

    SDL_GL_MakeCurrent(window, gl_context);
    SDL_GL_SetSwapInterval(1); // Enable vsync

    GLuint tex = createGLTexture(img, GL_RGB);

    ImGui_Init(window, gl_context);
    ImGui::GetIO().IniFilename = nullptr;

    ImGui_BeginFrame(window);
    ImGui::NewFrame();
    ImGui::EndFrame();
    ImGui_EndFrame(window);

    bool done = false;
    float x = 0.f, y = 0.f;
    std::map<std::string, sam_image_u8> masks;
    std::vector<GLuint> maskTextures;
    while (!done) {
        bool hasNewInputPoint = false;
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ProcessEvent(&event);
            if (event.type == SDL_QUIT) {
                done = true;
            }
            if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE && event.window.windowID == SDL_GetWindowID(window)) {
                done = true;
            }
            if (event.type == SDL_MOUSEBUTTONDOWN) {
                if (event.button.button == SDL_BUTTON_LEFT) {
                    hasNewInputPoint = true;
                    x = event.button.x;
                    y = event.button.y;
                }
            }
        }

        if (hasNewInputPoint) {
            sam_point pt;
            pt.x = x;
            pt.y = y;
            printf("pt = (%f, %f)\n", pt.x, pt.y);
            sam_state_init_masks(model, state);
            masks = compute_masks(img, params, pt, model, state);
            if (!maskTextures.empty()) {
                glDeleteTextures(maskTextures.size(), maskTextures.data());
                maskTextures.clear();
            }
            for (auto& mask : masks) {
                maskTextures.push_back(createGLTexture(mask.second, GL_LUMINANCE));
            }
            ggml_free(state.ctx_masks);
        }

        ImGui_BeginFrame(window);
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(0,0));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::Begin("SAM.cpp", NULL, ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoScrollbar|ImGuiWindowFlags_NoScrollWithMouse);
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        draw_list->AddImage((void*)(intptr_t)tex, ImVec2(0,0), ImVec2(img.nx, img.ny));
        int ii = 0;
        for (auto& maskTex : maskTextures) {
            bool isRed = ii == 0;
            bool isGreen = ii == 1;
            bool isBlue = ii == 2;
            draw_list->AddImage((void*)(intptr_t)maskTex, ImVec2(0,0), ImVec2(img.nx, img.ny), ImVec2(0,0), ImVec2(1,1), IM_COL32(isRed ? 255 : 0, isGreen ? 255 : 0, isBlue ? 255 : 0, 128));
            ++ii;
        }
        draw_list->AddCircleFilled(ImVec2(x, y), 5, IM_COL32(255, 0, 0, 255));
        ImGui::End();
        ImGui::EndFrame();
        ImGui_EndFrame(window);
    }

    SDL_DestroyWindow(window);

    return 0;
}

int main(int argc, char ** argv) {
    const int64_t t_main_start_us = ggml_time_us();

    sam_params params;
    params.model = "models/sam-vit-b/ggml-model-f16.bin";

    int64_t t_load_us = 0;

    if (sam_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }
    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    // load the image
    sam_image_u8 img0;
    if (!sam_image_load_from_file(params.fname_inp, img0)) {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, params.fname_inp.c_str());
        return 1;
    }
    fprintf(stderr, "%s: loaded image '%s' (%d x %d)\n", __func__, params.fname_inp.c_str(), img0.nx, img0.ny);

    sam_model model;
    sam_state state;
    load_model_and_compute_embd_img(img0, params, model, state, t_load_us);
    main_loop(img0, params, model, state);

    return 0;
}
