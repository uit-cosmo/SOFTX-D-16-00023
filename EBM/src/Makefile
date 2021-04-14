FC=gfortran 
NAME=EBM

NA=/usr/lib
NI=/usr/include

all: EBM.f90 grid.f90 app.f90 geography_input.f90 albedo_input.f90 albedo_update.f90 A_value.f90 orbital_params.f90 monthly_output.f90 timesteps_output.f90
	${FC} EBM.f90 grid.f90 app.f90 geography_input.f90 albedo_input.f90 albedo_update.f90 A_value.f90 orbital_params.f90  monthly_output.f90 albedo_output.f90 timesteps_output.f90 -o ${NAME} -I${NI} -L${NA} -lnetcdff -lnetcdf

clean:
	rm -rf *~


