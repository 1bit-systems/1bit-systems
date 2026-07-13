#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ftxui::screen" for configuration "Release"
set_property(TARGET ftxui::screen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ftxui::screen PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libftxui-screen.a"
  )

list(APPEND _cmake_import_check_targets ftxui::screen )
list(APPEND _cmake_import_check_files_for_ftxui::screen "${_IMPORT_PREFIX}/lib/libftxui-screen.a" )

# Import target "ftxui::dom" for configuration "Release"
set_property(TARGET ftxui::dom APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ftxui::dom PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libftxui-dom.a"
  )

list(APPEND _cmake_import_check_targets ftxui::dom )
list(APPEND _cmake_import_check_files_for_ftxui::dom "${_IMPORT_PREFIX}/lib/libftxui-dom.a" )

# Import target "ftxui::component" for configuration "Release"
set_property(TARGET ftxui::component APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ftxui::component PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libftxui-component.a"
  )

list(APPEND _cmake_import_check_targets ftxui::component )
list(APPEND _cmake_import_check_files_for_ftxui::component "${_IMPORT_PREFIX}/lib/libftxui-component.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
