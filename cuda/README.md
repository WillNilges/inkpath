# THIS IS SO, SO EXPERIMENTAL JESUS LORD PLEASE DON'T USE THIS

An implementation of Otsu thresholding and OpenCV's Adaptive Thresholding, parallelized in CUDA.

## Usage
The most reliable way to install this is probably through Podman. Ensure you have Podman installed, as well as the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installation-guide).

```
cd HACKING/
podman build . --tag inkpath-cuda
cd ..
./HACKING/launch_env.sh
```
