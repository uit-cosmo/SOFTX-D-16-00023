
! I keep fortran code in directory /Users/martinrypdal/Dropbox/FRIPRO/FortranCode
! To get to this directory type cd
! /Users/martinrypdal/Dropbox/FRIPRO/FortranCode  in the Terminal window   
! To compile the code write the following in the Terminal: gfortran -o helloworld helloworld.f
! To run the program write the following in the Terminal: /Users/martinrypdal/Dropbox/FRIPRO/FortranCode/helloworld 
! Installing gfortran compiler using Homebrew: https://www.macinchem.org/reviews/cheminfo/cheminfoMacUpdate.php     


! I installed netcdf using this command in the Terminal: brew install netcdf
! After that I modified the makefile in the EBM program:
! NA=/usr/local/Cellar/netcdf/4.7.4_2//lib and NI=/usr/local/Cellar/netcdf/4.7.4_2//include 
! I do the same thing in the file preprocess.sh
! I change to FC=gfortran in the makefile 
! In the Terminal: make -f /Users/martinrypdal/Dropbox/FRIPRO/FortranCode/EBM/src/Makefile 
! Change directory cd /Users/martinrypdal/Dropbox/FRIPRO/FortranCode/EBM/src
! Unlock ebm.sh: chmod +x ebm.sh
! In Terminal type:  /Users/martinrypdal/Dropbox/FRIPRO/FortranCode/EBM/src/ebm.sh 
      
        program hello
                implicit none
                 write(*,*) 'Hello world!'
        end program hello
