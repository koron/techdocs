# merlin-modelsのインストール

Windows 11ネイティブで[merlin models](https://github.com/NVIDIA-Merlin/models)を試すためのインストール手順を紹介します。

## インストールするソフト

それぞれ少しずつ古いソフトで

1. [Python 3.10.12](https://www.python.org/ftp/python/3.10.12/)
2. [CUDA 11.2](https://developer.nvidia.com/cuda-11.2.2-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork)
3. [cuDNN v8.9.4 for CUDA 11.x](https://developer.nvidia.com/rdp/cudnn-download)

    展開して出てくる bin/, include/, lib/ を CUDA のインストールディレクトリにコピーするのが良い。
    ただ本当に必要なのかは自身が無い。
    起動だけなら無くてもできた。

4. merlin models

    ```
    > pip install merlin-models
    ```

5. Tensorflow 2.10.1

    ```
    > pip install tensorflow==2.10.1
    ```

6. Other dependencies

    ```
    > pip install torch scipy
    ```

MerlinにCUDAが必須のためNVIDIAのGPUが必須。

ここまでくれば `import merlin.models.tf as mm` が成功する。
実行例:

```
> python
Python 3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import merlin.models.tf as mm
2023-09-05 22:15:16.395250: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-09-05 22:15:16.746535: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 6140 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 4070, pci bus id: 0000:01:00.0, compute capability: 8.9
>>>
```

## インストールに至るまでの経緯

MerlinにはGPUが有効化されたTensorflowが必要。
しかしTensorflowは2.10.1までしかWindowsでのGPUをサポートしていない。
(以降はWSL2のみなっている)
cf. <https://www.tensorflow.org/install/source_windows?hl=ja#gpu>

Tensorflow 2.10.1のビルド済みバイナリはPython 3.10用までしかない。
(Pythonは3.11が最新)
cf. <https://pypi.org/project/tensorflow/2.10.1/#files>

MerlinはTensorflow 2.12.xまでしかサポートしてない。
(Tensorflowは2.13.0が最新)

幾つかの依存モジュール(tensorflowやtorchやscipy)が、
`pip install merlin-models` ではインストールできない。
