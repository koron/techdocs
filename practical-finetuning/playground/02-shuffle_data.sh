#!/bin/sh
#
# Script to separate data for training and validation
#
# データを学習用と検証用に分けるスクリプト
# dataset/data.jsonl を シャッフルして
# dataset/{出力名}/valid.jsonl と dataset/{出力名}/train.jsonl に切り分ける
#
# オプション:
#
#   -o  出力名。デフォルトは set1 で dataset/set1/ に出力する
#   -v  検証用データの数。デフォルトは 10 で残りは学習用に使う


set -eu

outset="set1"
vnum=10

while getopts o:v: OPT ; do
  case $OPT in
    o) outset="$OPTARG" ;;
    v) vnum="$OPTARG" ;;
    *) exit 1 ;;
  esac
done

rm -f dataset/.shuffle.jsonl
shuf dataset/data.jsonl > dataset/.shuffle.jsonl

mkdir -p "dataset/$outset"

head -n $vnum dataset/.shuffle.jsonl > "dataset/$outset/valid.jsonl"
tail -n +$(expr $vnum + 1) dataset/.shuffle.jsonl > "dataset/$outset/train.jsonl"

rm -f dataset/.shuffle.jsonl
