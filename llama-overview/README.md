# LLaMa(.cpp) 周辺のサーベイ

発端: [CPUだけでも使える1ファイル版フリーの量子化省メモリローカルLLMのllamafileを入れてJava,PythonプログラムをAIに書かせてみた (toggter)](https://togetter.com/li/2273536)

* 関連: https://huggingface.co/jartine/mistral-7b.llamafile/tree/main

Facebookが作った [LLaMaモデル](https://ai.meta.com/llama/) をC++から使えるようにするのが [llama.cpp](https://github.com/ggerganov/llama.cpp)

* https://ja.wikipedia.org/wiki/LLaMA
* [Llama.cpp で Llama 2 を試す (note) ](https://note.com/npaka/n/n0ad63134fbe2)

> 「Llama.cpp」を利用するには、「Llama 2」モデルをGGML形式に変換する必要があります。HuggingFaceには、変換済みのモデルが公開されています。

GGMLとは、モデルを量子化して圧縮したフォーマット。
同様の目的を持つフォーマットにはGGUFやGPTQやAWQがある。
GGUFはGGMLの発展形で、いまではGGMLは廃れてる。
またAWQはGPTQの発展形らしい。
llama.cppでゃGGUFを使うことになる。

Hugging FaceというのはAIモデルのGitHubみたいなもの。
特に [TheBloke (Tom Jobbins)](https://huggingface.co/TheBloke) 氏がめちゃくちゃ多くのモデルをアップロードしている。
何してる人なんだ…

発端のポストでは [llamafile](https://github.com/Mozilla-Ocho/llamafile) なるものを使ってる。llamafileとはなんぞや?

* https://github.com/Mozilla-Ocho/llamafile

llamafileはllama.cppベースで、CLIやHTTP serverをラッピングし、モデルを埋め込んだ実行ファイルらしい。
またその実行ファイルを作るための仕組みの名前を兼ねている。
さらにその実行ファイルは [Cosmopolitan (cosmocc)](https://github.com/jart/cosmopolitan) を使ってマルチプラットフォームで実行できるバイナリになってるっぽい。

なのでモデルの埋め込まれたllmafileをダウンロードして実行するか、
llamafile (コマンド)をGGUFモデルを指定 (`-m`) して使うかの2通りの使い方がある。
前者のほうがお手軽に試せるが、最新のモデルを使うには後者が必要だろう。

## モデル埋め込み llamafile の検証

<https://huggingface.co/jartine/mistral-7b.llamafile/tree/main> から main.llamafile をダウンロード。

拡張子をexeに変えて実行してみたが起動せず。
`unzip -l` で中身を見てみると、コンパイラが要りそうな雰囲気はある。

```console
$ unzip -l mistral-7b-instruct-v0.1-Q4_K_M-main.llamafile
Archive:  mistral-7b-instruct-v0.1-Q4_K_M-main.llamafile
  Length      Date    Time    Name
---------  ---------- -----   ----
   393216  2023-07-29 00:00   .symtab.amd64
   458752  2023-07-29 00:00   .symtab.arm64
      958  2022-03-17 07:00   llamafile/compcap.cu
      969  2022-03-17 07:00   llamafile/llamafile.h
   308805  2022-03-17 07:00   llama.cpp/ggml-cuda.cu
     2057  2022-03-17 07:00   llama.cpp/ggml-cuda.h
     6869  2022-03-17 07:00   llama.cpp/ggml-impl.h
     3558  2022-03-17 07:00   llama.cpp/ggml-metal.h
    84425  2022-03-17 07:00   llama.cpp/ggml-metal.m
   103217  2022-03-17 07:00   llama.cpp/ggml-metal.metal
    10231  2022-03-17 07:00   llama.cpp/ggml-quants.h
    78804  2022-03-17 07:00   llama.cpp/ggml.h
      977  2022-03-17 07:00   usr/share/zoneinfo/Anchorage
      582  2022-03-17 07:00   usr/share/zoneinfo/Beijing
     2335  2022-03-17 07:00   usr/share/zoneinfo/Berlin
     2453  2022-03-17 07:00   usr/share/zoneinfo/Boulder
     3585  2022-03-17 07:00   usr/share/zoneinfo/Chicago
      114  2022-03-17 07:00   usr/share/zoneinfo/GMT
     2845  2022-03-17 07:00   usr/share/zoneinfo/GST
      338  2022-03-17 07:00   usr/share/zoneinfo/Honolulu
     2397  2022-03-17 07:00   usr/share/zoneinfo/Israel
      318  2022-03-17 07:00   usr/share/zoneinfo/Japan
     3687  2022-03-17 07:00   usr/share/zoneinfo/London
     2223  2022-03-17 07:00   usr/share/zoneinfo/Melbourne
     3545  2022-03-17 07:00   usr/share/zoneinfo/New_York
      127  2022-03-17 07:00   usr/share/zoneinfo/UTC
        0  2022-03-17 07:00   usr/share/zoneinfo/
        0  2022-03-17 07:00   .cosmo
       44  2023-11-15 22:13   .args
4368438944  2023-11-15 22:13   mistral-7b-instruct-v0.1.Q4_K_M.gguf
---------                     -------
4369916375                     30 files
```

エラーメッセージを頼りに検証すると、WSL2からなら動きそうな気もする。

```console
$ ./mistral-7b-instruct-v0.1-Q4_K_M-main.llamafile.exe -h
./mistral-7b-instruct-v0.1-Q4_K_M-main.llamafile.exe: 行 64: /home/koron/.ape-1.9: バイナリファイルを実行できません: Exec format error
./mistral-7b-instruct-v0.1-Q4_K_M-main.llamafile.exe: 64 行: exec: /home/koron/.ape-1.9: 実行できません: Permission denied

$ file ~/.ape-1.9
/home/koron/.ape-1.9: ELF 64-bit LSB executable, x86-64, version 1 (FreeBSD), for OpenBSD, statically linked, no section header
```

WSL2、ダメだった。
一旦この方向はストップ。

```console
$ ./mistral-7b-instruct-v0.1-Q4_K_M-main.llamafile
./mistral-7b-instruct-v0.1-Q4_K_M-main.llamafile: Invalid argument

$ ./mistral-7b-instruct-v0.1-Q4_K_M-main.llamafile -p "### Instruction: Good Morning! ### RESPONSE:"
./mistral-7b-instruct-v0.1-Q4_K_M-main.llamafile: Invalid argument

$ ll
total 4270244
drwxr-xr-x 3 koron koron       4096 Dec 12 12:00 ./
drwxr-x--- 9 koron koron       4096 Dec  5 00:17 ../
-rwxr-xr-x 1 koron koron 4372701937 Dec 12 12:00 mistral-7b-instruct-v0.1-Q4_K_M-main.llamafile*
```

## llamafile コマンド+モデル

llamafile-0.3 ファイルをダウンロード後、ウィルススキャン、制限解除して、拡張子.exeを付与。
後に実行した結果動いた。

<details>
<summary>実行した際のログと出力</summary>

```console
$ ./llamafile-0.3.exe -m mistral-7b-instruct-v0.2.Q3_K_M.gguf -p "### Instruction: Good Morning! ### RESPONSE:"
protip: pass the --n-gpu-layers N flag to link NVIDIA cuBLAS support
Log start
warning: this OS doesn't support pledge() security
main: llamafile version 0.3.0
main: seed  = 1702350513
llama_model_loader: loaded meta data with 24 key-value pairs and 291 tensors from mistral-7b-instruct-v0.2.Q3_K_M.gguf (version GGUF V3 (latest))
llama_model_loader: - tensor    0:                token_embd.weight q3_K     [  4096, 32000,     1,     1 ]
llama_model_loader: - tensor    1:              blk.0.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor    2:              blk.0.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor    3:              blk.0.attn_v.weight q5_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor    4:         blk.0.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor    5:            blk.0.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor    6:              blk.0.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor    7:            blk.0.ffn_down.weight q5_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor    8:           blk.0.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor    9:            blk.0.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor   10:              blk.1.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor   11:              blk.1.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor   12:              blk.1.attn_v.weight q5_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor   13:         blk.1.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor   14:            blk.1.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor   15:              blk.1.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor   16:            blk.1.ffn_down.weight q5_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor   17:           blk.1.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor   18:            blk.1.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor   19:              blk.2.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor   20:              blk.2.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor   21:              blk.2.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor   22:         blk.2.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor   23:            blk.2.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor   24:              blk.2.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor   25:            blk.2.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor   26:           blk.2.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor   27:            blk.2.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor   28:              blk.3.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor   29:              blk.3.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor   30:              blk.3.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor   31:         blk.3.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor   32:            blk.3.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor   33:              blk.3.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor   34:            blk.3.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor   35:           blk.3.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor   36:            blk.3.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor   37:              blk.4.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor   38:              blk.4.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor   39:              blk.4.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor   40:         blk.4.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor   41:            blk.4.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor   42:              blk.4.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor   43:            blk.4.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor   44:           blk.4.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor   45:            blk.4.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor   46:              blk.5.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor   47:              blk.5.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor   48:              blk.5.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor   49:         blk.5.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor   50:            blk.5.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor   51:              blk.5.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor   52:            blk.5.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor   53:           blk.5.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor   54:            blk.5.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor   55:              blk.6.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor   56:              blk.6.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor   57:              blk.6.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor   58:         blk.6.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor   59:            blk.6.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor   60:              blk.6.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor   61:            blk.6.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor   62:           blk.6.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor   63:            blk.6.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor   64:              blk.7.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor   65:              blk.7.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor   66:              blk.7.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor   67:         blk.7.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor   68:            blk.7.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor   69:              blk.7.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor   70:            blk.7.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor   71:           blk.7.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor   72:            blk.7.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor   73:              blk.8.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor   74:              blk.8.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor   75:              blk.8.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor   76:         blk.8.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor   77:            blk.8.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor   78:              blk.8.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor   79:            blk.8.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor   80:           blk.8.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor   81:            blk.8.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor   82:              blk.9.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor   83:              blk.9.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor   84:              blk.9.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor   85:         blk.9.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor   86:            blk.9.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor   87:              blk.9.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor   88:            blk.9.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor   89:           blk.9.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor   90:            blk.9.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor   91:             blk.10.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor   92:             blk.10.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor   93:             blk.10.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor   94:        blk.10.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor   95:           blk.10.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor   96:             blk.10.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor   97:           blk.10.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor   98:          blk.10.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor   99:           blk.10.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  100:             blk.11.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  101:             blk.11.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  102:             blk.11.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  103:        blk.11.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  104:           blk.11.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  105:             blk.11.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  106:           blk.11.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor  107:          blk.11.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  108:           blk.11.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  109:             blk.12.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  110:             blk.12.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  111:             blk.12.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  112:        blk.12.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  113:           blk.12.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  114:             blk.12.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  115:           blk.12.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor  116:          blk.12.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  117:           blk.12.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  118:             blk.13.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  119:             blk.13.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  120:             blk.13.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  121:        blk.13.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  122:           blk.13.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  123:             blk.13.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  124:           blk.13.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor  125:          blk.13.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  126:           blk.13.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  127:             blk.14.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  128:             blk.14.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  129:             blk.14.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  130:        blk.14.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  131:           blk.14.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  132:             blk.14.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  133:           blk.14.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor  134:          blk.14.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  135:           blk.14.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  136:             blk.15.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  137:             blk.15.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  138:             blk.15.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  139:        blk.15.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  140:           blk.15.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  141:             blk.15.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  142:           blk.15.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor  143:          blk.15.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  144:           blk.15.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  145:             blk.16.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  146:             blk.16.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  147:             blk.16.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  148:        blk.16.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  149:           blk.16.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  150:             blk.16.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  151:           blk.16.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor  152:          blk.16.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  153:           blk.16.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  154:             blk.17.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  155:             blk.17.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  156:             blk.17.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  157:        blk.17.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  158:           blk.17.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  159:             blk.17.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  160:           blk.17.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor  161:          blk.17.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  162:           blk.17.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  163:             blk.18.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  164:             blk.18.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  165:             blk.18.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  166:        blk.18.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  167:           blk.18.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  168:             blk.18.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  169:           blk.18.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor  170:          blk.18.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  171:           blk.18.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  172:             blk.19.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  173:             blk.19.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  174:             blk.19.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  175:        blk.19.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  176:           blk.19.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  177:             blk.19.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  178:           blk.19.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor  179:          blk.19.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  180:           blk.19.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  181:             blk.20.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  182:             blk.20.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  183:             blk.20.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  184:        blk.20.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  185:           blk.20.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  186:             blk.20.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  187:           blk.20.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor  188:          blk.20.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  189:           blk.20.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  190:             blk.21.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  191:             blk.21.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  192:             blk.21.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  193:        blk.21.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  194:           blk.21.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  195:             blk.21.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  196:           blk.21.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor  197:          blk.21.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  198:           blk.21.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  199:             blk.22.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  200:             blk.22.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  201:             blk.22.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  202:        blk.22.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  203:           blk.22.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  204:             blk.22.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  205:           blk.22.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor  206:          blk.22.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  207:           blk.22.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  208:             blk.23.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  209:             blk.23.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  210:             blk.23.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  211:        blk.23.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  212:           blk.23.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  213:             blk.23.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  214:           blk.23.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor  215:          blk.23.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  216:           blk.23.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  217:             blk.24.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  218:             blk.24.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  219:             blk.24.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  220:        blk.24.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  221:           blk.24.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  222:             blk.24.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  223:           blk.24.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor  224:          blk.24.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  225:           blk.24.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  226:             blk.25.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  227:             blk.25.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  228:             blk.25.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  229:        blk.25.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  230:           blk.25.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  231:             blk.25.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  232:           blk.25.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor  233:          blk.25.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  234:           blk.25.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  235:             blk.26.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  236:             blk.26.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  237:             blk.26.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  238:        blk.26.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  239:           blk.26.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  240:             blk.26.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  241:           blk.26.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor  242:          blk.26.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  243:           blk.26.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  244:             blk.27.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  245:             blk.27.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  246:             blk.27.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  247:        blk.27.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  248:           blk.27.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  249:             blk.27.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  250:           blk.27.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor  251:          blk.27.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  252:           blk.27.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  253:             blk.28.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  254:             blk.28.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  255:             blk.28.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  256:        blk.28.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  257:           blk.28.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  258:             blk.28.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  259:           blk.28.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor  260:          blk.28.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  261:           blk.28.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  262:             blk.29.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  263:             blk.29.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  264:             blk.29.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  265:        blk.29.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  266:           blk.29.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  267:             blk.29.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  268:           blk.29.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor  269:          blk.29.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  270:           blk.29.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  271:             blk.30.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  272:             blk.30.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  273:             blk.30.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  274:        blk.30.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  275:           blk.30.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  276:             blk.30.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  277:           blk.30.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor  278:          blk.30.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  279:           blk.30.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  280:             blk.31.attn_q.weight q3_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  281:             blk.31.attn_k.weight q3_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  282:             blk.31.attn_v.weight q4_K     [  4096,  1024,     1,     1 ]
llama_model_loader: - tensor  283:        blk.31.attn_output.weight q4_K     [  4096,  4096,     1,     1 ]
llama_model_loader: - tensor  284:           blk.31.ffn_gate.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  285:             blk.31.ffn_up.weight q3_K     [  4096, 14336,     1,     1 ]
llama_model_loader: - tensor  286:           blk.31.ffn_down.weight q4_K     [ 14336,  4096,     1,     1 ]
llama_model_loader: - tensor  287:          blk.31.attn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  288:           blk.31.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  289:               output_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - tensor  290:                    output.weight q6_K     [  4096, 32000,     1,     1 ]
llama_model_loader: - kv   0:                       general.architecture str
llama_model_loader: - kv   1:                               general.name str
llama_model_loader: - kv   2:                       llama.context_length u32
llama_model_loader: - kv   3:                     llama.embedding_length u32
llama_model_loader: - kv   4:                          llama.block_count u32
llama_model_loader: - kv   5:                  llama.feed_forward_length u32
llama_model_loader: - kv   6:                 llama.rope.dimension_count u32
llama_model_loader: - kv   7:                 llama.attention.head_count u32
llama_model_loader: - kv   8:              llama.attention.head_count_kv u32
llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32
llama_model_loader: - kv  10:                       llama.rope.freq_base f32
llama_model_loader: - kv  11:                          general.file_type u32
llama_model_loader: - kv  12:                       tokenizer.ggml.model str
llama_model_loader: - kv  13:                      tokenizer.ggml.tokens arr
llama_model_loader: - kv  14:                      tokenizer.ggml.scores arr
llama_model_loader: - kv  15:                  tokenizer.ggml.token_type arr
llama_model_loader: - kv  16:                tokenizer.ggml.bos_token_id u32
llama_model_loader: - kv  17:                tokenizer.ggml.eos_token_id u32
llama_model_loader: - kv  18:            tokenizer.ggml.unknown_token_id u32
llama_model_loader: - kv  19:            tokenizer.ggml.padding_token_id u32
llama_model_loader: - kv  20:               tokenizer.ggml.add_bos_token bool
llama_model_loader: - kv  21:               tokenizer.ggml.add_eos_token bool
llama_model_loader: - kv  22:                    tokenizer.chat_template str
llama_model_loader: - kv  23:               general.quantization_version u32
llama_model_loader: - type  f32:   65 tensors
llama_model_loader: - type q3_K:  129 tensors
llama_model_loader: - type q4_K:   92 tensors
llama_model_loader: - type q5_K:    4 tensors
llama_model_loader: - type q6_K:    1 tensors
llm_load_vocab: special tokens definition check successful ( 259/32000 ).
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 32000
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: n_ctx_train      = 32768
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_layer          = 32
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_gqa            = 4
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: n_ff             = 14336
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_yarn_orig_ctx  = 32768
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: model type       = 7B
llm_load_print_meta: model ftype      = mostly Q3_K - Medium
llm_load_print_meta: model params     = 7.24 B
llm_load_print_meta: model size       = 3.28 GiB (3.89 BPW)
llm_load_print_meta: general.name   = mistralai_mistral-7b-instruct-v0.2
llm_load_print_meta: BOS token = 1 '<s>'
llm_load_print_meta: EOS token = 2 '</s>'
llm_load_print_meta: UNK token = 0 '<unk>'
llm_load_print_meta: PAD token = 0 '<unk>'
llm_load_print_meta: LF token  = 13 '<0x0A>'
llm_load_tensors: ggml ctx size =    0.11 MB
llm_load_tensors: mem required  = 3355.37 MB
llm_load_tensors: offloading 0 repeating layers to GPU
llm_load_tensors: offloaded 0/35 layers to GPU
llm_load_tensors: VRAM used: 0.00 MB
warning: posix_madvise(.., POSIX_MADV_WILLNEED) failed: No error information (win32 error 998)
.................................................................................................
llama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_new_context_with_model: kv self size  =   64.00 MB
llama_build_graph: non-view tensors processed: 740/740
llama_new_context_with_model: compute buffer total size = 79.63 MB

system_info: n_threads = 8 / 16 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 |
sampling:
        repeat_last_n = 64, repeat_penalty = 1.100, frequency_penalty = 0.000, presence_penalty = 0.000
        top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.800
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
generate: n_ctx = 512, n_batch = 512, n_predict = -1, n_keep = 0


### Instruction: Good Morning! ### RESPONSE: Good morning! How can I assist you today?

I'm an assistant designed to help make your day easier. Whether it's setting reminders, providing information, or just answering questions, I'm here for you. Let me know what I can do for you today.
```
</details>

サーバー (llama-server-0.3) でも同じことをやってみる。
起動したらWeb UIが表示され、会話ができた。
見る限りCPUをぶん回してる。

## まとめ

* Facebookが開発したLLMである LLaMa
* LLaMa をC++から依存なしで使えるようにする llama.cpp
* llama.cpp をポータブルで実行可能にする llamafile
* llamafile が受け付けるモデルフォーマットは GGUF
* 各種モデルはHugging FaceでThe Blokeがいろいろ配ってる

## 保留した疑問

Q. [配布されてるモデル](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/tree/main)の命名規則がよくわからない。
`Q{n}` は量子化サイズだと推定される。
一方で `K`, `K_L`, `K_M` という suffix が不明。
量子化方法(Quant method)という推測はあるが、
それぞれのメソッドがどんなものかがわかんない。
GGUF側を見れば、手がかりがあるかも。

A. モデルの量子化後サイズだった。
`M` にはQ3だけでなくQ4~6も含まれるが、`S`にはQ3だけしかなかった。
結果としてBit Per Weight (BPW)は `M` が 3.89 なのに対し、
`S` は 3.50 になっていた。

~`K` は K-quant の意味らしいが、 K-quant が何者なのかはまだわかっていない。~
[quauntization.md](./quauntization.md) で調査中。

Q. WindowsでGPU (NVIDIA CUDA)を使う方法がわかってない。
llamafileのドキュメントにはCUDA12.xとVC2022が必要と書いてあった。
またcmakeを利用するとも。
ハマったら時間がかかりそうだから、試すのはしばらく保留。


