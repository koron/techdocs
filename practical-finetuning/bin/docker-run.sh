#!/bin/sh
#
# Paligemmaでfinetune実験を行うためのコンテナを起動する
#
# モデルをダウンロードするにはローカル側の ~/.kaggle/kaggle.json に
# kaggleの鍵情報(usernameとkey)が必要
#

set -eu

img="paligemma-playground:latest"
cachedir="$(pwd)"/cache
datadir="$(pwd)"/playground
kaggledir="$HOME"/.kaggle
shmsize=12g

while getopts c:d:i:k:s: OPT ; do
  case $OPT in
    c) cachedir="$OPTARG" ;;
    d) datadir="$OPTARG" ;;
    i) img="$OPTARG" ;;
    k) kaggle="$OPTARG" ;;
    r) shmsize="$OPTARG" ;;
    *) exit 1 ;;
  esac
done


case $(uname -s) in
  MSYS*)
    cachedir=$(cygpath -w $cachedir)
    datadir=$(cygpath -w $datadir)
    kaggledir=$(cygpath -w $kaggledir)
    ;;
esac

docker run --rm -it \
  --name paligemma_playground \
  -v "${cachedir}:/root/.cache" \
  -v "${datadir}:/root/playground" \
  -v "${kaggledir}:/root/.kaggle" \
  -p 8888:8888 \
  --gpus all \
  --shm-size="$shmsize" \
  "$img" \
  "$@"
