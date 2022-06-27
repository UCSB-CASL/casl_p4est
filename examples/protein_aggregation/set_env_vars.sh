#!/bin/sh




export OUT_DIR_ERR="$(echo $PWD)/outputs/ERR"
echo $OUT_DIR_ERR

export OUT_DIR_VTK="$(echo $PWD)/outputs/VTK"
echo $OUT_DIR_VTK

export OUT_DIR_MEM="$(echo $PWD)/outputs/MEM"
echo $OUT_DIR_MEM



if [[ ! -e $OUT_DIR_ERR ]]; then
    mkdir $OUT_DIR_ERR
elif [[ ! -d $OUT_DIR_ERR ]]; then
    echo "$OUT_DIR_ERR already exists but is not a directory" 1>&2
fi

if [[ ! -e $OUT_DIR_VTK ]]; then
    mkdir $OUT_DIR_VTK
elif [[ ! -d $OUT_DIR_VTK ]]; then
    echo "$OUT_DIR_VTK already exists but is not a directory" 1>&2
fi


if [[ ! -e $OUT_DIR_MEM ]]; then
    mkdir $OUT_DIR_MEM
elif [[ ! -d $OUT_DIR_MEM ]]; then
    echo "$OUT_DIR_MEM already exists but is not a directory" 1>&2
fi

