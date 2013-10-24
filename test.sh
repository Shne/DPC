#!/bin/bash

cd miro/build;
filename="output.data"
touch "$filename"
#blocks=(64 96 128 160 192 224 256 288 320 352 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 1024) # multiplums of 32
blocks=(64 128 256 512 1024) # powers of two, requiered by partial sum implementation.

#Test to find best block size
for block in ${blocks[@]}
do
	./miro $block >> "$filename"
done
echo "" >> "$filename"