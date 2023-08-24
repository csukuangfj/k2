add_subdirectory(csrc)

if(K2_BUILD_TORCH_EXAMPLES)
  include(kaldifeat)
  add_subdirectory(bin)
endif()
