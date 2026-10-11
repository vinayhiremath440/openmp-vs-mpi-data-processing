@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo  Compiling Sequential, OpenMP, and MPI Executables
echo ========================================================

rem Ensure bin directory exists
if not exist "bin" mkdir "bin"

rem Configure MinGW compiler path
set "PATH=C:\Program Files\CodeBlocks\MinGW\bin;%CD%\tools\mpi\bin;%PATH%"

rem 1. Sequential Compilation
echo [1/3] Compiling Sequential Baseline (src/sequential.c)...
gcc -O3 src\sequential.c -o bin\sequential.exe -lm
if %errorlevel% neq 0 (
    echo [ERROR] Failed to compile sequential.c
    exit /b 1
)
echo    [OK] Successfully created bin\sequential.exe

rem 2. OpenMP Compilation
echo [2/3] Compiling OpenMP Parallel (src/openmp_processing.c)...
gcc -O3 -fopenmp src\openmp_processing.c -o bin\openmp_processing.exe -lm
if %errorlevel% neq 0 (
    echo [ERROR] Failed to compile openmp_processing.c
    exit /b 1
)
echo    [OK] Successfully created bin\openmp_processing.exe

rem 3. MPI Compilation
echo [3/3] Compiling MPI Distributed (src/mpi_processing.c)...
gcc -O3 -I"tools\mpi\include" src\mpi_processing.c -L"tools\mpi\lib" -lmsmpi -o bin\mpi_processing.exe -lm
if %errorlevel% neq 0 (
    echo [ERROR] Failed to compile mpi_processing.c
    exit /b 1
)
echo    [OK] Successfully created bin\mpi_processing.exe

rem 4. Copy required runtime DLLs to bin for portable execution
if exist "tools\mpi\bin\msmpi.dll" copy /y "tools\mpi\bin\msmpi.dll" "bin\" >nul
if exist "C:\Program Files\CodeBlocks\MinGW\bin\libgomp-1.dll" copy /y "C:\Program Files\CodeBlocks\MinGW\bin\libgomp-1.dll" "bin\" >nul
if exist "C:\Program Files\CodeBlocks\MinGW\bin\libwinpthread-1.dll" copy /y "C:\Program Files\CodeBlocks\MinGW\bin\libwinpthread-1.dll" "bin\" >nul
if exist "C:\Program Files\CodeBlocks\MinGW\bin\libgcc_s_seh-1.dll" copy /y "C:\Program Files\CodeBlocks\MinGW\bin\libgcc_s_seh-1.dll" "bin\" >nul

echo ========================================================
echo  All 3 targets built successfully in 'bin\' directory!
echo ========================================================
