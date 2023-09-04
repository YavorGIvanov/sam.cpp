#pragma once

#include <vector>
#include <cstdint>
#include <map>
#include <string>

struct ggml_cgraph;
struct ggml_tensor;
struct ggml_context;
struct ggml_allocr;

struct sam_model;

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

// RGB float32 image
// Memory layout: RGBRGBRGB...
struct sam_image_f32 {
    int nx;
    int ny;

    std::vector<float> data;
};

// default hparams (ViT-B SAM)
struct sam_hparams {
    int32_t n_enc_state               = 768;
    int32_t n_enc_layer               = 12;
    int32_t n_enc_head                = 12;
    int32_t n_enc_out_chans           = 256;
    int32_t n_pt_embd                 = 4;
    int32_t n_dec_heads               = 8;
    int32_t ftype                     = 1;
    float   mask_threshold            = 0.f;
    float   iou_threshold             = 0.85f; // Default in PyTorch is 0.88f
    float   stability_score_threshold = 0.90f; // Default in PyTorch is 0.95f
    float   stability_score_offset    = 1.0f;
    float   eps                       = 1e-6f;
    float   eps_decoder_transformer   = 1e-5f;

    int32_t n_enc_head_dim() const { return n_enc_state / n_enc_head; }
    int32_t n_img_size()     const { return 1024; }
    int32_t n_window_size()  const { return 14; }
    int32_t n_patch_size()   const { return 16; }
    int32_t n_img_embd()     const { return n_img_size() / n_patch_size(); }

    std::vector<int32_t> global_attn_indices() const {
        switch (n_enc_state) {
            case  768: return {  2,  5,  8, 11 };
            case 1024: return {  5, 11, 17, 23 };
            case 1280: return {  7, 15, 23, 31 };
            default:
                {
                    printf("%s: unsupported n_enc_state = %d\n", __func__, n_enc_state);
                } break;
        };

        return {};
    }

    bool is_global_attn(int32_t layer) const {
        const auto indices = global_attn_indices();

        for (const auto & idx : indices) {
            if (layer == idx) {
                return true;
            }
        }

        return false;
    }
};

struct sam_layer_enc {
    struct ggml_tensor * norm1_w;
    struct ggml_tensor * norm1_b;

    struct ggml_tensor * rel_pos_w;
    struct ggml_tensor * rel_pos_h;

    struct ggml_tensor * qkv_w;
    struct ggml_tensor * qkv_b;

    struct ggml_tensor * proj_w;
    struct ggml_tensor * proj_b;

    struct ggml_tensor * norm2_w;
    struct ggml_tensor * norm2_b;

    struct ggml_tensor * mlp_lin1_w;
    struct ggml_tensor * mlp_lin1_b;

    struct ggml_tensor * mlp_lin2_w;
    struct ggml_tensor * mlp_lin2_b;
};

struct sam_encoder_image {
    struct ggml_tensor * pe;

    struct ggml_tensor * proj_w;
    struct ggml_tensor * proj_b;

    struct ggml_tensor * neck_conv_0;
    struct ggml_tensor * neck_norm_0_w;
    struct ggml_tensor * neck_norm_0_b;
    struct ggml_tensor * neck_conv_1;
    struct ggml_tensor * neck_norm_1_w;
    struct ggml_tensor * neck_norm_1_b;

    std::vector<sam_layer_enc> layers;
};

struct sam_encoder_prompt {
    struct ggml_tensor * pe;

    struct ggml_tensor * not_a_pt_embd_w;
    std::vector<struct ggml_tensor *> pt_embd;

    struct ggml_tensor * no_mask_embd_w;
    //std::vector<struct ggml_tensor *> mask_down_w;
    //std::vector<struct ggml_tensor *> mask_down_b;
};

struct  sam_layer_dec_transformer_attn {
    // q_proj
    struct ggml_tensor * q_w;
    struct ggml_tensor * q_b;

    // k_proj
    struct ggml_tensor * k_w;
    struct ggml_tensor * k_b;

    // v_proj
    struct ggml_tensor * v_w;
    struct ggml_tensor * v_b;

    // out_proj
    struct ggml_tensor * out_w;
    struct ggml_tensor * out_b;
};

struct sam_layer_dec_transformer {
    sam_layer_dec_transformer_attn self_attn;

    // norm1
    struct ggml_tensor * norm1_w;
    struct ggml_tensor * norm1_b;

    sam_layer_dec_transformer_attn cross_attn_token_to_img;

    // norm2
    struct ggml_tensor * norm2_w;
    struct ggml_tensor * norm2_b;

    // mlp.lin1
    struct ggml_tensor * mlp_lin1_w;
    struct ggml_tensor * mlp_lin1_b;

    // mlp.lin2
    struct ggml_tensor * mlp_lin2_w;
    struct ggml_tensor * mlp_lin2_b;

    // norm3
    struct ggml_tensor * norm3_w;
    struct ggml_tensor * norm3_b;

    // norm4
    struct ggml_tensor * norm4_w;
    struct ggml_tensor * norm4_b;

    sam_layer_dec_transformer_attn cross_attn_img_to_token;
};

struct sam_layer_dec_output_hypernet_mlps {
    // mlps_*.layers.0
    struct ggml_tensor * w_0;
    struct ggml_tensor * b_0;

    // mlps_*.layers.1
    struct ggml_tensor * w_1;
    struct ggml_tensor * b_1;

    // mlps_*.layers.2
    struct ggml_tensor * w_2;
    struct ggml_tensor * b_2;
};

struct sam_decoder_mask {
    std::vector<sam_layer_dec_transformer> transformer_layers;

    // trasnformer.final_attn_token_to_image
    sam_layer_dec_transformer_attn transformer_final_attn_token_to_img;

    // transformer.norm_final
    struct ggml_tensor * transformer_norm_final_w;
    struct ggml_tensor * transformer_norm_final_b;

    // output_upscaling.0
    struct ggml_tensor * output_upscaling_0_w;
    struct ggml_tensor * output_upscaling_0_b;

    // output_upscaling.1
    struct ggml_tensor * output_upscaling_1_w;
    struct ggml_tensor * output_upscaling_1_b;

    // output_upscaling.3
    struct ggml_tensor * output_upscaling_3_w;
    struct ggml_tensor * output_upscaling_3_b;

    // output_hypernetworks_mlps
    std::vector<sam_layer_dec_output_hypernet_mlps> output_hypernet_mlps;

    // iou_prediction_head.0
    struct ggml_tensor * iou_prediction_head_0_w;
    struct ggml_tensor * iou_prediction_head_0_b;

    // iou_prediction_head.1
    struct ggml_tensor * iou_prediction_head_1_w;
    struct ggml_tensor * iou_prediction_head_1_b;

    // iou_prediction_head.2
    struct ggml_tensor * iou_prediction_head_2_w;
    struct ggml_tensor * iou_prediction_head_2_b;

    // iou_token.weight
    struct ggml_tensor * iou_token_w;

    // mask_tokens.weight
    struct ggml_tensor * mask_tokens_w;
};

struct sam_model {
    sam_hparams hparams;

    sam_encoder_image  enc_img;
    sam_encoder_prompt enc_prompt;
    sam_decoder_mask   dec;

    //
    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

struct sam_state {
    struct ggml_tensor * embd_img = {};
    struct ggml_context * ctx_img = {};

    struct ggml_tensor * low_res_masks = {};
    struct ggml_tensor * iou_predictions = {};
    struct ggml_context * ctx_masks = {};

    //struct ggml_tensor * tmp_save = {};


    // buffer for `ggml_graph_plan.work_data`
    std::vector<uint8_t> work_buffer;
    // buffers to evaluate the model
    std::vector<uint8_t> buf_alloc_img_enc;
    std::vector<uint8_t> buf_compute_img_enc;

    std::vector<uint8_t> buf_alloc_fast;
    std::vector<uint8_t> buf_compute_fast;

    struct ggml_allocr  * allocr = {};
};

bool sam_model_load(
        const std::string & fname,
        sam_model & model);

void sam_state_init_img(
        const sam_model & model,
        sam_state       & state);


void sam_state_init_masks(
        const sam_model & model,
        sam_state       & state);

bool sam_image_preprocess(
        const sam_image_u8 & img,
        sam_image_f32      & res);

struct ggml_cgraph  * sam_encode_image(
        const sam_model     & model,
        sam_state           & state,
        const sam_image_f32 & img);

struct ggml_cgraph  * sam_build_fast_graph(
        const sam_model & model,
        sam_state       & state,
        int               nx,
        int               ny,
        sam_point         point);

std::map<std::string, sam_image_u8> sam_postprocess_masks(
        const sam_hparams & hparams,
        int                 nx,
        int                 ny,
        const sam_state   & state);
