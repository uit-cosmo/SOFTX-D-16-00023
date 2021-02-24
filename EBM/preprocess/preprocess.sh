#!/bin/sh

ifort extract.f90 -o extract
ifort prepare_geography.f90 -o prepare_geography -I/usr/include -L/usr/lib -lnetcdff -lnetcdf
ifort prepare_albedo.f90 -o prepare_albedo -I/usr/include
 -L/usr/lib -lnetcdff -lnetcdf

./extract
./prepare_geography
./prepare_albedo
mv -f *.nc ../input
rm -rf *~



