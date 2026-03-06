# Copyright      2020  Mobvoi AI Lab, Beijing, China (authors: Fangjun Kuang)
# See ../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

function(download_moderngpu)
  if(CMAKE_VERSION VERSION_LESS 3.11)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
  endif()

  include(FetchContent)

  # this is the latest commit of modern gpu as of 2026-03-06
  set(moderngpu_URL  "https://github.com/moderngpu/moderngpu/archive/267cc8d02c03f00f1656f1205c61c946a1e530b0.zip")
  set(moderngpu_HASH "SHA256=433f8b9ba12fce80a932110a75d03d638b90d08a1cd41255c4a992121fdfe696")

  # If you don't have access to the Internet,
  # please pre-download moderngpu
  set(possible_file_locations
    $ENV{HOME}/Downloads/moderngpu-267cc8d02c03f00f1656f1205c61c946a1e530b0.zip
    ${CMAKE_SOURCE_DIR}/moderngpu-267cc8d02c03f00f1656f1205c61c946a1e530b0.zip
    ${CMAKE_BINARY_DIR}/moderngpu-267cc8d02c03f00f1656f1205c61c946a1e530b0.zip
    /tmp/moderngpu-267cc8d02c03f00f1656f1205c61c946a1e530b0.zip
    /star-fj/fangjun/download/github/moderngpu-267cc8d02c03f00f1656f1205c61c946a1e530b0.zip
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(moderngpu_URL  "file://${f}")
      break()
    endif()
  endforeach()

  FetchContent_Declare(moderngpu
    URL
      ${moderngpu_URL}
    URL_HASH          ${moderngpu_HASH}
  )

  FetchContent_GetProperties(moderngpu)
  if(NOT moderngpu)
    message(STATUS "Downloading moderngpu from ${moderngpu_URL}")
    FetchContent_Populate(moderngpu)
  endif()
  message(STATUS "moderngpu is downloaded to ${moderngpu_SOURCE_DIR}")
  add_library(moderngpu INTERFACE)
  target_include_directories(moderngpu INTERFACE ${moderngpu_SOURCE_DIR}/src)
endfunction()

download_moderngpu()
