conda env create -f environment.yml
conda activate diffusionGS
mkdir models && wget https://storage.googleapis.com/niantic-lon-static/research/diffusionerf/rgbd-patch-diffusion.pt -O models/rgbd-patch-diffusion.pt
cd data
unzip nerf_llff_data.zip
cd nerf_llff_data
bash ../../scripts/preprocess_llff.sh
cd ../..