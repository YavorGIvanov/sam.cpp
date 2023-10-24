# SAM.cpp

Inference of Meta's [Segment Anything Model](https://github.com/facebookresearch/segment-anything/) in pure C/C++

https://github.com/YavorGIvanov/sam.cpp/assets/1991296/a69be66f-8e27-43a0-8a4d-6cfe3b1d9335

## Quick start
```bash
git clone --recursive https://github.com/YavorGIvanov/sam.cpp
cd sam.cpp
```

Note: you need to download the model checkpoint below (`sam_vit_b_01ec64.pth`) first from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and place it in the `checkpoints` folder

```bash
# Download PTH model and place it under checkpoints
mkdir checkpoints
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Convert PTH model to ggml. Requires python3, torch and numpy
python convert-pth-to-ggml.py checkpoints/sam_vit_b_01ec64.pth checkpoints 1

# You need CMake and SDL2 - Used for GUI windows & input [libsdl](https://www.libsdl.org)

[Ubuntu]
$ sudo apt install libsdl2-dev

[Mac OS with brew]
$ brew install sdl2

[MSYS2]
$ pacman -S git cmake make mingw-w64-x86_64-dlfcn mingw-w64-x86_64-gcc mingw-w64-x86_64-SDL2

# Build sam.cpp.
mkdir build && cd build
cmake .. && make -j4

# run inference
./bin/sam -t 16 -i ../img.jpg -m ../checkpoints/ggml-model-f16.bin
```
Note: The optimal threads parameter ("-t") value should be manually selected based on the specific machine running the inference.

Note: If you have problems with the Windows build, you can check [this issue](https://github.com/YavorGIvanov/sam.cpp/issues/8) for more details

## Downloading and converting the model checkpoints

You can download a [model checkpoint](https://github.com/facebookresearch/segment-anything/tree/main#model-checkpoints) and convert it to `ggml` format using the script `convert-pth-to-ggml.py`:

```
# Convert PTH model to ggml
python convert-pth-to-ggml.py sam_vit_b_01ec64.pth . 1
```

## Example output on M2 Ultra
```
 $ ▶ make -j sam && time ./bin/sam -t 8 -i img.jpg
[ 28%] Built target common
[ 71%] Built target ggml
[100%] Built target sam
main: seed = 1693224265
main: loaded image 'img.jpg' (680 x 453)
sam_image_preprocess: scale = 0.664062
main: preprocessed image (1024 x 1024)
sam_model_load: loading model from 'models/sam-vit-b/ggml-model-f16.bin' - please wait ...
sam_model_load: n_enc_state      = 768
sam_model_load: n_enc_layer      = 12
sam_model_load: n_enc_head       = 12
sam_model_load: n_enc_out_chans  = 256
sam_model_load: n_pt_embd        = 4
sam_model_load: ftype            = 1
sam_model_load: qntvr            = 0
operator(): ggml ctx size = 202.32 MB
sam_model_load: ...................................... done
sam_model_load: model size =   185.05 MB / num tensors = 304
embd_img
dims: 64 64 256 1 f32
First & Last 10 elements:
-0.05117 -0.06408 -0.07154 -0.06991 -0.07212 -0.07690 -0.07508 -0.07281 -0.07383 -0.06779
0.01589 0.01775 0.02250 0.01675 0.01766 0.01661 0.01811 0.02051 0.02103 0.03382
sum:  12736.272313

Skipping mask 0 with iou 0.705935 below threshold 0.880000
Skipping mask 1 with iou 0.762136 below threshold 0.880000
Mask 2: iou = 0.947081, stability_score = 0.955437, bbox (371, 436), (144, 168)


main:     load time =    51.28 ms
main:    total time =  2047.49 ms

real	0m2.068s
user	0m16.343s
sys	0m0.214s
```

Input point is (414.375, 162.796875) (currently hardcoded)

Input image:

![llamas](https://user-images.githubusercontent.com/8558655/261301565-37b7bf4b-bf91-40cf-8ec1-1532316e1612.jpg)

Output mask (mask_out_2.png in build folder):

![mask_glasses](https://user-images.githubusercontent.com/8558655/265732931-e7e31285-7efc-4009-98c8-57fd819bdfc1.png)

## References

- [ggml](https://github.com/ggerganov/ggml)
- [ggml SAM example](https://github.com/ggerganov/ggml/tree/master/examples/sam)
- [SAM](https://segment-anything.com/)
- [SAM demo](https://segment-anything.com/demo)

## Next steps

- [X] Reduce memory usage by utilizing the new ggml-alloc
- [X] Remove redundant graph nodes
- [X] Fix the difference in output masks compared to the PyTorch implementation
- [X] Filter masks based on stability score
- [X] Add support for point user input
- [X] Support bigger model checkpoints
- [ ] Make inference faster
- [ ] Support F16 for heavy F32 ops
- [ ] Test quantization
- [ ] Add support for mask and box input + #14
- [ ] GPU support
