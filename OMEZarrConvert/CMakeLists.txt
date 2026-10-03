#-----------------------------------------------------------------------------
set(MODULE_NAME OMEZarrConvert)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${MODULE_NAME}Lib/__init__.py
  ${MODULE_NAME}Lib/to_omezarr.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)
endif()
