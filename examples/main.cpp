#include "sam.h"

#include "ggml.h"
#include "ggml-alloc.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

int main(int argc, char ** argv) {
    const int64_t t_main_start_us = ggml_time_us();

    sam_params params;
    params.model = "models/sam-vit-b/ggml-model-f16.bin";

    sam_model model;
    sam_state state;
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

    // preprocess to f32
    sam_image_f32 img1;
    if (!sam_image_preprocess(img0, img1)) {
        fprintf(stderr, "%s: failed to preprocess image\n", __func__);
        return 1;
    }
    fprintf(stderr, "%s: preprocessed image (%d x %d)\n", __func__, img1.nx, img1.ny);


    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!sam_model_load(params.model, model)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    {
        static size_t buf_size = 256u*1024*1024;

        struct ggml_init_params ggml_params = {
            /*.mem_size   =*/ buf_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };

        state.ctx = ggml_init(ggml_params);

        state.embd_img = ggml_new_tensor_3d(state.ctx, GGML_TYPE_F32,
                model.hparams.n_img_embd(), model.hparams.n_img_embd(), model.hparams.n_enc_out_chans);

        state.low_res_masks = ggml_new_tensor_3d(state.ctx, GGML_TYPE_F32,
                model.hparams.n_enc_out_chans, model.hparams.n_enc_out_chans, 3);

        state.iou_predictions = ggml_new_tensor_1d(state.ctx, GGML_TYPE_F32, 3);
    }


    static const size_t tensor_alignment = 32;
    {
        state.buf_compute_img_enc.resize(ggml_tensor_overhead()*GGML_MAX_NODES + ggml_graph_overhead());
        state.allocr = ggml_allocr_new_measure(tensor_alignment);
        struct ggml_cgraph * gf_measure = sam_encode_image(model, state, img1);
        if (!gf_measure) {
            fprintf(stderr, "%s: failed to encode image\n", __func__);
            return 1;
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
            return 1;
        }

        ggml_allocr_alloc_graph(state.allocr, gf);

        ggml_graph_compute_helper(state.work_buffer, gf, params.n_threads);

        ggml_allocr_free(state.allocr);
        state.allocr = NULL;
        state.work_buffer.clear();
    }

    {
        state.buf_compute_fast.resize(ggml_tensor_overhead()*GGML_MAX_NODES + ggml_graph_overhead());
        state.allocr = ggml_allocr_new_measure(tensor_alignment);

        // TODO: user input
        const sam_point pt = { 414.375f, 162.796875f, };
        // measure memory requirements for the graph
        struct ggml_cgraph  * gf_measure = sam_build_fast_graph(model, state, img0.nx, img0.ny, pt);
        if (!gf_measure) {
            fprintf(stderr, "%s: failed to build fast graph to measure\n", __func__);
            return 1;
        }

        size_t alloc_size = ggml_allocr_alloc_graph(state.allocr, gf_measure) + tensor_alignment;
        ggml_allocr_free(state.allocr);

        // recreate allocator with exact memory requirements
        state.buf_alloc_fast.resize(alloc_size);
        state.allocr = ggml_allocr_new(state.buf_alloc_fast.data(), state.buf_alloc_fast.size(), tensor_alignment);

        // compute the graph with the measured exact memory requirements from above
        ggml_allocr_reset(state.allocr);

        struct ggml_cgraph  * gf = sam_build_fast_graph(model, state, img0.nx, img0.ny, pt);
        if (!gf) {
            fprintf(stderr, "%s: failed to build fast graph\n", __func__);
            return 1;
        }

        ggml_allocr_alloc_graph(state.allocr, gf);

        ggml_graph_compute_helper(state.work_buffer, gf, params.n_threads);

        //print_t_f32("iou_predictions", state.iou_predictions);
        //print_t_f32("low_res_masks", state.low_res_masks);
        ggml_allocr_free(state.allocr);
        state.allocr = NULL;
    }

    std::map<std::string, sam_image_u8> masks = sam_postprocess_masks(model.hparams, img0.nx, img0.ny, state);
    if (masks.empty()) {
        fprintf(stderr, "%s: failed to postprocess masks\n", __func__);
        return 1;
    }

    for (auto& [filename, mask] : masks) {
        if (!stbi_write_png(filename.c_str(), mask.nx, mask.ny, 1, mask.data.data(), mask.nx)) {
            printf("%s: failed to write mask %s\n", __func__, filename.c_str());
            return 1;
        }
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        fprintf(stderr, "\n\n");
        fprintf(stderr, "%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        fprintf(stderr, "%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    ggml_free(model.ctx);

    return 0;
}
