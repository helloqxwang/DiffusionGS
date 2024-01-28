# [DiffusioGS: Regularizing 3D Gaussians with Denoising Diffusion Models]


## Quickstart

This section will walk you through setting up DiffusioNeRF and using it to fit a NeRF to a scene from LLFF.

### Hardware requirements

You will need a relatively powerful graphics card to run DiffusioGS. All of our experiments were performed on an A100.

### Conda environment
### & Downloading the pretrained diffusion model 
### & Prepare the LLFF dataset
Just run 
```
bash install.sh
```


### Run on the LLFF dataset

You can now fit a NeRF to an LLFF scene using our regularizers by running from the root of the repo:

```
bash scripts/run_diffusionerf_example.sh
```

The arguments passed in this script correspond to the configuration reported as ours in the paper.

Image-by-image metrics will be written to the output folder (which with the above script will be `./runs/example/3_poses/room/`) under `metrics.json`. You should obtain an average test PSNR of about 21.6 with this script.

To change the script to run a full LLFF evaluation, just delete the `--only_run_on room` argument to run on all scenes, and change `--num_train 3` to `--num_train 3 6 9` to run each scene with 3, 6 and 9 training views.

To run without our learned diffusion model regularizer, just drop the `--patch_regulariser_path` argument; 

### Run on other scenes

`nerf/evaluate.py`, which is used in the above steps, is just a wrapper around `main.py`; if you want to runata, you should use `main.py`. The data should be in the NeRF 'blender' format, i.e. it should contain a `transforms.json` file. on other d


