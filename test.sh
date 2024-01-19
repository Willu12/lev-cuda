#!/bin/bash

cd data
tr -dc '[:print:]' </dev/urandom | head -c $1 > jeden.txt
tr -dc '[:print:]' </dev/urandom | head -c $2 > dwa.txt
cd ..
./cuda_lev -gc && diff cpu_results gpu_results