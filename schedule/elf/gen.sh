#!/bin/bash
set -ux

workdir="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)"
filenames="$(ls ${workdir}/*.cu)"
elf=".elf"
ptx=".ptx"

default_all_archs="--offload-arch=mp_21 --offload-arch=mp_22 --offload-arch=mp_31"
if [ $# -ge 1 ]; then
    arch_option="--offload-arch=$1"
else
    arch_option=$default_all_archs
fi

if [ -z "${TEST_ON_NVIDIA:-}" ]; then
  for eachfile in $filenames
   do
      echo ${eachfile%.*}$elf
      mcc $eachfile -o ${eachfile%.*}$elf --cuda-device-only -mtgpu $arch_option
   done
else
  for eachfile in $filenames
   do
      echo ${eachfile%.*}$ptx
      nvcc -ptx $eachfile -o ${eachfile%.*}$ptx
   done
fi
