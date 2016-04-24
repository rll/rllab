#!/bin/bash

if [ "$(uname)" == "Darwin" ]; then
    mujoco_file="libmujoco131.dylib"
    glfw_file="libglfw.3.dylib"
    zip_file="mjpro131_osx.zip"
    mktemp_cmd="mktemp -d /tmp/mujoco"
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    mujoco_file="libmujoco131.so"
    glfw_file="libglfw.so.3"
    zip_file="mjpro131_linux.zip"
    mktemp_cmd="mktemp -d"
fi

if [ ! -f vendor/mujoco/$mujoco_file ]; then
    read -e -p "Please enter the path to the mujoco zip file [$zip_file]:" path
    path=${path:-$zip_file} 
    eval path=\"$path\"
    if [ ! -f $path ]; then
        echo "No file found at $path"
        exit 0
    fi
    rm -r /tmp/mujoco
    dir=`$mktemp_cmd`
    unzip $path -d $dir
    if [ ! -f $dir/mjpro131/bin/$mujoco_file ]; then
        echo "mjpro/$mujoco_file not found. Make sure you have the correct file (most likely named $zip_file)"
        exit 0
    fi
    if [ ! -f $dir/mjpro131/bin/$glfw_file ]; then
        echo "mjpro/$glfw_file not found. Make sure you have the correct file (most likely named $zip_file)"
        exit 0
    fi

    mkdir -p vendor/mujoco
    cp $dir/mjpro131/bin/$mujoco_file vendor/mujoco/
    cp $dir/mjpro131/bin/$glfw_file vendor/mujoco/
fi

if [ ! -f vendor/mujoco/mjkey.txt ]; then
    read -e -p "Please enter the path to the mujoco license file [mjkey.txt]:" path
    path=${path:-mjkey.txt}
    eval path=$path
    if [ ! -f $path ]; then
        echo "No file found at $path"
        exit 0
    fi
    cp $path vendor/mujoco/mjkey.txt
fi

echo "Mujoco has been set up!"
