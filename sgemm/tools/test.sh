#!/bin/bash
# 1. build if bin file is not exist
if [ ! -f "build/main" ]; then
    mkdir -p build && cd build
    cmake ..
    make
    if [ $? -ne 0 ]; then
        echo "compile error"
        exit 1
    fi
    cd --
fi

# 2. get the nummber of custom kernel
include_folder="include"
kernel_count=$(find "$include_folder" -type f -name "kernel*.cuh" | grep -v "kernel.cuh" | wc -l)
echo "found ${kernel_count} custom kernel"

# 3. test
rm -rf test
mkdir -p test
for((i=0;i<=${kernel_count};i++)); do
    if [ "$i" -eq 0 ]; then
        echo -n "test cuBLAS..."
    else
        echo -n "test custom kernel: ${i}..."
    fi
    file_name="./test/test_kernel_${i}.log"
    ./build/main ${i} > ${file_name}
    if [ $? -ne 0 ]; then
        echo "kernel${i} error"
        exit 1
    fi

    # 3. if not cuBLAs, plot and save to images/
    if [ ${i} -gt 0 ]; then
        echo -n "Done. Ploting..."
        python3 tools/plot.py 0 ${i}
        if [ ${i} -gt 0 ]; then
            python3 tools/plot.py $(expr $i - 1) ${i}
        fi
    fi
    echo "Done."
done
