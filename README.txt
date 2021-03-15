Functionality implementation:

* 4-digit-year handling in ./EBM fortran code: 
    single commit: 4604369309001e7e0a49b0a889006ad5026e8bdd
    affected files: 
            ./EBM/src/EBM.f90, 
            ./EBM/src/timesteps_output.f90

* plotting sea-ice area as function of global temeprature (extracted & estimated from timesteps-output.nc output file from the ./EBM/src/EBM.f90 run)
    last commit: 321f80edcfc570f5d426d269c57ff9e48394aced 
                + all earlier commits for ./postprocess_python/main.py file.
    affected file: ./postprocess_python/main.py
