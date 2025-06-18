set(HPP_HEADERS
    Fourier.hpp
)

set(CPP_SOURCES
)

target_sources(${CPP_TARGET} PRIVATE ${CPP_SOURCES} ${HPP_HEADERS})

message(STATUS "Math included")

if(BUILD_TESTING)
add_subdirectory(tests)
endif()