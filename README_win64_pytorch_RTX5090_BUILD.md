# Building PyTorch for RTX 5090 (CUDA 13) on Windows

The NVIDIA RTX 5090 (Blackwell architecture) requires CUDA 13. As of this writing, official PyTorch wheels do not support CUDA 13.

This guide details how to build a **standalone, portable wheel** from source. It uses static linking and `delvewheel` to bundle necessary dependencies (cuDNN, NVRTC, cuBLAS, ZLIB) so the wheel can be installed on other machines without manual DLL patching.

## 1. Prerequisites

*   **OS:** Windows 10/11 (64-bit).
*   **Visual Studio 2022:** Install the "Desktop development with C++" workload.
*   **CUDA Toolkit 13.0 (Preview):** Install the full toolkit.
*   **Anaconda or Miniconda:** For managing build dependencies.
*   **Git:** To clone the repository.
*   **Visual C++ Redistributable (x64):** [Download here](https://aka.ms/vs/17/release/vc_redist.x64.exe). **Required on any machine running the finished wheel.**

## 2. Initialize Build Environment

**Crucial:** Do not use the standard PowerShell or the generic "Developer Command Prompt". You must use the **x64 Native Tools** to ensure a pure 64-bit build environment.

1.  Open a standard Command Prompt (`cmd.exe`).
2.  Run the initialization script.
    *   *Note: Adjust the path if you use VS Professional or Enterprise.*

```cmd
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

If you see `[vcvarsall.bat] Environment initialized for: 'x64'`, you are ready to proceed.

## 3. Install Build Dependencies

We use Conda to create a clean environment and install MKL (Math Kernel Library), which allows better static linking than standard MSVC OpenMP.

```cmd
:: 1. Create and activate environment
conda create -n build_torch python=3.12 -y
conda activate build_torch

:: 2. Install build tools and MKL
:: mkl-include is required for USE_MKL=1
conda install cmake ninja numpy pyyaml typing_extensions mkl mkl-include -c conda-forge -y

:: 3. Install wheel packaging tool
pip install delvewheel packaging
```

## 4. Clone PyTorch

```cmd
cd C:\git
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
:: Optional: Checkout a specific tag if you don't want the bleeding edge main branch
:: git checkout v2.6.0-rc1
```

## 5. Configure & Build

We use `USE_CUDA_STATIC_LINK=1` to merge the main CUDA runtime into the PyTorch binaries. This drastically reduces the number of loose DLLs required at runtime.

```cmd
:: --- Configuration Flags ---
set CMAKE_GENERATOR=Ninja
set CMAKE_BUILD_TYPE=Release

:: Force CUDA 13 Paths (Adjust if you installed elsewhere)
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set CUDA_HOME=%CUDA_PATH%

:: --- Linking Strategy ---
:: Static link CUDA to avoid "DLL Hell"
set USE_CUDA_STATIC_LINK=1
:: Use Intel MKL for Math/OpenMP
set USE_MKL=1
:: Ensure NumPy integration
set USE_NUMPY=1

:: --- Build Versioning (Optional) ---
set PYTORCH_BUILD_VERSION=2.10.0
set PYTORCH_BUILD_NUMBER=1

:: --- Start Build ---
python setup.py bdist_wheel
```

**Note:** This step may take 30-90 minutes depending on your CPU.

## 6. Package & Repair (The Critical Step)

A raw Windows wheel produced by `setup.py` is not portable. It is missing dynamic libraries like cuDNN, NVRTC (required for `torch.compile`), CUPTI, and ZLIB.

We use `delvewheel` to bundle these.

**Note on CUDA 13:** The preview version splits binaries between `bin` and `bin\x64`. We must explicitly include both paths to ensure NVRTC and Sparse libraries are found.

```cmd
cd dist

:: We use %CONDA_PREFIX% to automatically find Python and ZLIB DLLs
:: We use %CUDA_PATH% to find NVRTC, cuDNN, and CUPTI

delvewheel repair torch-*.whl ^
    -w fixed ^
    --ignore-existing ^
    --add-path "%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;%CUDA_PATH%\extras\CUPTI\lib64;%CONDA_PREFIX%;%CONDA_PREFIX%\Library\bin;%CONDA_PREFIX%\Scripts"
```

## 7. Installation & Verification

The final, production-ready wheel is located in `dist\fixed\`.

### Installation

```cmd
pip install fixed\torch-*.whl --force-reinstall
```

### Verification Script

Run this Python command to verify that PyTorch loads, recognizes the CUDA 13 driver, and can allocate tensors on the RTX 5090.

```python
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version:    {torch.version.cuda}")
print(f"Device Name:     {torch.cuda.get_device_name(0)}")
print(f"CUDA Available:  {torch.cuda.is_available()}")

# Test Tensor Allocation
x = torch.randn(2, 2).cuda()
print("Tensor on GPU:\n", x)
```

## Troubleshooting

### [WinError 126] Error loading `aoti_custom_ops.dll`

This library depends on the Standard C++ Runtime.

**Fix:** Install the Visual C++ Redistributable (x64) on the machine running the code.

### RuntimeError: NVRTC Error or `caffe2_nvrtc.dll` not found

This indicates the NVRTC compiler libraries were missing from the wheel.

**Fix:** Ensure you included `%CUDA_PATH%\bin\x64` in the `delvewheel` command (Step 6), as CUDA 13 often hides `nvrtc*.dll` there.