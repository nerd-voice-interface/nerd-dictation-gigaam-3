# Arch bug report draft: onnxruntime-cuda provider library lacks NEEDED libcudnn.so.9

To be filed at https://gitlab.archlinux.org/archlinux/packaging/packages/onnxruntime/-/issues

**Title:** onnxruntime-cuda 1.29.0-3: libonnxruntime_providers_cuda.so does not link libcudnn.so.9 (undefined symbol cudnnGetConvolutionBackwardDataAlgorithm_v7)

**Package:** onnxruntime-cuda 1.29.0-3 (x86_64)

**Description:**
`/usr/lib/libonnxruntime_providers_cuda.so` references cuDNN symbols but has
no `NEEDED` entry for `libcudnn.so.9`. onnxruntime dlopen()s the provider
library when the CUDA execution provider is appended, and the load fails:

    undefined symbol: cudnnGetConvolutionBackwardDataAlgorithm_v7

so every application asking for the CUDA EP (e.g.
`SessionOptionsAppendExecutionProvider_CUDA_V2`) gets an error, and those
that fall back to the CPU do so silently.

**Evidence** (from onnxruntime-cuda-1.29.0-3-x86_64.pkg.tar.zst, geo.mirror.pkgbuild.com):

    $ readelf -d usr/lib/libonnxruntime_providers_cuda.so | grep NEEDED
     (NEEDED) Shared library: [libcublasLt.so.13]
     (NEEDED) Shared library: [libcublas.so.13]
     (NEEDED) Shared library: [libcudart.so.13]
     (NEEDED) Shared library: [libnccl.so.2]
     (NEEDED) Shared library: [libstdc++.so.6]
     (NEEDED) Shared library: [libm.so.6]
     (NEEDED) Shared library: [libgcc_s.so.1]
     (NEEDED) Shared library: [libc.so.6]
     (NEEDED) Shared library: [ld-linux-x86-64.so.2]

    $ nm -D --undefined-only usr/lib/libonnxruntime_providers_cuda.so | grep cudnn
     U cudnnBatchNormalizationBackward
     U cudnnConvolutionBackwardBias
     U cudnnConvolutionBackwardFilter
     U cudnnFindConvolutionBackwardFilterAlgorithmEx
     U cudnnGetConvolutionBackwardDataAlgorithm_v7
     U cudnnGetConvolutionBackwardDataWorkspaceSize
     U cudnnGetConvolutionBackwardFilterAlgorithm_v7
     U cudnnGetConvolutionBackwardFilterWorkspaceSize

(For comparison, an onnxruntime 1.23.2 CUDA build has
`(NEEDED) Shared library: [libcudnn.so.9]` in the same library.)

**Reproducer (one line):**

    python -c 'import ctypes; ctypes.CDLL("/usr/lib/libonnxruntime_providers_cuda.so")'

fails with `undefined symbol: cudnnGetConvolutionBackwardDataAlgorithm_v7`
(add `ctypes.CDLL("libcudnn.so.9", ctypes.RTLD_GLOBAL)` first and it loads).

**Expected:** the provider library links libcudnn.so.9 (`NEEDED`), e.g. by
making sure the cuDNN libraries are on the link line of
onnxruntime_providers_cuda (the symbols are only partly undefined, which
suggests the link resolves part of cuDNN statically/through another path and
drops the dependency, e.g. via `-Wl,--as-needed` ordering).

**Observed on:** NVIDIA T500, onnxruntime-cuda 1.29.0-3, cudnn 9.x.
**Workaround in applications:** `dlopen("libcudnn.so.9", RTLD_NOW | RTLD_GLOBAL)`
before appending the CUDA EP.
