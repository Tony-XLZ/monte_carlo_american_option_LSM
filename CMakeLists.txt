cmake_minimum_required(VERSION 3.15)
project(monte_carlo_american_option_LSM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(monte_carlo_american_option_LSM main.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(monte_carlo_american_option_LSM PUBLIC OpenMP::OpenMP_CXX)
endif()

find_package(Eigen3 REQUIRED)
include_directories(S:\\directiories\\eigen-3.3.9)

if(MSVC)
    # MSVC-specific compiler options
    set(RELEASE_OPTIONS /O2 /fp:fast /arch:AVX2 /favor:INTEL64 /openmp /GL)
    set(DEBUG_OPTIONS /Od /RTC1 /openmp)

    # Apply compiler options
    target_compile_options(monte_carlo_american_option_LSM PRIVATE
            $<$<CONFIG:Release>:${RELEASE_OPTIONS}>
            $<$<CONFIG:Debug>:${DEBUG_OPTIONS}>
    )

    # Link-time code generation for MSVC
    target_link_options(monte_carlo_american_option_LSM PRIVATE
            $<$<CONFIG:Release>:/LTCG>
    )
else()
    # GCC/Clang-specific compiler options
    target_compile_options(monte_carlo_american_option_LSM PRIVATE
            $<$<CONFIG:Release>:-O3 -ffast-math -march=native -fopenmp>
            $<$<CONFIG:Debug>:-O0 -g -fopenmp>
    )
endif()

# Add definitions for better debugging and release performance
target_compile_definitions(monte_carlo_american_option_LSM PRIVATE
        $<$<CONFIG:Release>:NDEBUG>
)
