#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <thread>
#include <memory>

struct sam_point {
    float x = 0;
    float y = 0;
};

// RGB uint8 image
struct sam_image_u8 {
    int nx = 0;
    int ny = 0;

    std::vector<uint8_t> data;
};

struct sam_params {
    int32_t seed      = -1; // RNG seed
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());

    std::string model     = "../checkpoints/ggml-model-f16-b.bin"; // model path
    std::string fname_inp = "../img.jpg";
    std::string fname_out = "img.out";
};

struct sam_ggml_state;
struct sam_ggml_model;
struct sam_state {
    std::unique_ptr<sam_ggml_state> state;
    std::unique_ptr<sam_ggml_model> model;
    int t_load_ms = 0;
    int t_compute_img_ms = 0;
    int t_compute_masks_ms = 0;
};

std::shared_ptr<sam_state> sam_load_model(
        const sam_params & params);

bool sam_compute_embd_img(
        const sam_image_u8 & img,
        int                  n_threads ,
        sam_state          & state);

// returns masks sorted by the sum of the iou_score and stability_score in descending order
std::vector<sam_image_u8> sam_compute_masks(
        const sam_image_u8 & img,
        int                  n_threads,
        sam_point            pt,
        sam_state          & state,
        int                  mask_on_val  = 255,
        int                  mask_off_val = 0);

void sam_deinit(
        sam_state & state);
