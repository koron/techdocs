#!/bin/sh
#
# Paligemmaでfinetune実験を行うためのdockerイメージをビルドする

set -eu

docker build -t paligemma-playground:latest .
