#-----------------------------------------------------------------------------
set(MODULE_NAME Agent_CLI)
set(FOLDER_LIBRARY Agent_CLI_utils)

set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${FOLDER_LIBRARY}/__init__.py
  ${FOLDER_LIBRARY}/utils.py
  ${FOLDER_LIBRARY}/parameter_extraction_improved.py
  ${FOLDER_LIBRARY}/parameter_validator.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  manifest.yaml
  )

SlicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
)
