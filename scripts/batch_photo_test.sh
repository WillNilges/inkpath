#!/bin/bash

make ipcv-debug

rm -r /xopp-dev/batch_photo_test_output/*

for filename in /xopp-dev/batch_photo_test_dir/*; do
		b=$(basename ${filename})
		echo "test ${filename}"
		mkdir /xopp-dev/batch_photo_test_output/${b}
		./build/ipcv-debug -f ${filename} -o /xopp-dev/batch_photo_test_output/${b}/${b}_output.jpg &
done

