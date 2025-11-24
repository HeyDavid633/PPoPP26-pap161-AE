#!/bin/bash
cd ../src

rm -rf build 

rm *.so


cd ByteTransformer
rm -rf build 
rm -f *.so
cd ../


cd SPLAT-reproduce
make clean
cd ../