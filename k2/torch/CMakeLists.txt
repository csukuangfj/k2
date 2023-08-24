add_subdirectory(csrc)

if(K2_BUILD_TORCH_EXAMPLES)
  add_subdirectory(bin)
endif()
