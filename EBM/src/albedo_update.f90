! This program calculates albedo values based on land masks and Legende polynomials
! See reference:
! Gerald R. North and James A. Coakley, Jr., 1979,
! Differences between Seasonal and Mean Annual Energy Balance Model Calculations
! of Climate and Climate Sensistivity.
! Journal of the Atmospheric Sciences
! Vol 36, pp 1189-1204.

    SUBROUTINE update_albedo_timestep (Pcoalbedo, Temp, geography_0, geography_updated)
    implicit none
    
    integer,parameter:: nx=128, ny=65
    real:: albedo(nx,ny), legende, Temp(nx,ny), Pcoalbedo(nx,ny)
    integer:: i, j, geo, temp_mask_int
    integer:: geography_0(nx,ny), temp_mask(nx,ny), geography_updated(nx,ny)
    
    CALL temperature_mask (Temp, temp_mask)
 
    ! ================  newly added ==================================
    do j=1,ny
       legende=0.5*(3*sin((90.0-(j-1)*2.8125)*3.1415926/180.0)*sin((90.0-(j-1)*2.8125)*3.1415926/180.0)-1)
        do i=1,nx
            geo=geography_0(i,j)
            temp_mask_int=temp_mask(i,j)
            !landmask: 1. land;  2. sea ice; 3. land ice; 5 ocean.
            if((geo.eq.1).and.(temp_mask_int.eq.0)) then 
                 albedo(i,j)=0.30 +0.09*legende !land without ice 
                 geography_updated(i,j) = 1
                 !print *, "land without ice"
            end if
            if((geo.eq.2).and.(temp_mask_int.eq.1)) then
                albedo(i,j)=0.60 !sea ice
               
                geography_updated(i,j) = 2
                !print *, "sea ice"
            end if
            if((geo.eq.1).and.(temp_mask_int.eq.1)) then 
                albedo(i,j)=0.70 !land with ice
                geography_updated(i,j) = 3
                !print *, "land ice"
            end if
            if((geo.eq.2).and.(temp_mask_int.eq.0)) then 
                albedo(i,j)=0.29 +0.09*legende !ocean without ice
                geography_updated(i,j) = 5
                !print *, "ocean"
            end if
            Pcoalbedo(i, j) = 1.0 - albedo(i, j)
        end do
    end do
    END SUBROUTINE


    SUBROUTINE temperature_mask (Temp, temp_mask)
        ! creates temperature mask for albedo calculation:
        ! 1 corresponds to ice
        ! 0 corresponds to no ice
        implicit none
        
        integer,parameter:: nx=128, ny=65
        real:: Temp(nx,ny)
        integer:: temp_mask(nx,ny)
        integer:: i, j
        
        ! ================  newly added ==================================
        do j=1,ny
            do i=1,nx
                if(Temp(i, j).le.-1)  then ! freezing temp of water 
                    temp_mask(i,j) = 1 ! 1 corresponds to ice
                else
                    temp_mask(i,j) = 0  ! 0 corresponds to no ice
                end if 
            end do
        end do
    END SUBROUTINE
        
    SUBROUTINE read_geography_0 (geography_0)
        ! reads the Earth geography (continents & oceans) with 1 = land, 2 = ocean
        ! see geography_0.py for more info
        implicit none
        
        integer,parameter:: nx=128, ny=65
        integer:: geography_0(nx,ny)
        integer:: i, j

        open(33 , file= '../preprocess/geography_0.dat', status='old')
    
        do j = ny, 1, -1
            read (33,100) (geography_0(i,j),i=1,nx)
        100  format (128I1)
        end do
        close(33)

    END SUBROUTINE

