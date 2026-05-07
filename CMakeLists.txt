cmake_minimum_required(VERSION 3.20)

# This project uses CMake as an *orchestration/task runner* (no compilation).
project(ec_cad_orchestration LANGUAGES NONE)

set(ROOT_DIR "${CMAKE_SOURCE_DIR}")

# Prefer the project's venv python if present; otherwise fall back to `python`.
set(DEFAULT_VENV_PY "${ROOT_DIR}/.venv/Scripts/python.exe")
if(EXISTS "${DEFAULT_VENV_PY}")
  set(PYTHON_EXECUTABLE "${DEFAULT_VENV_PY}" CACHE FILEPATH "Python executable to run scripts")
else()
  set(PYTHON_EXECUTABLE "python" CACHE STRING "Python executable to run scripts")
endif()

function(add_py_target TARGET_NAME SCRIPT_REL)
  set(options)
  set(oneValueArgs WORKING_DIRECTORY)
  set(multiValueArgs ARGS DEPENDS)
  cmake_parse_arguments(PYT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT PYT_WORKING_DIRECTORY)
    set(PYT_WORKING_DIRECTORY "${ROOT_DIR}")
  endif()

  add_custom_target(${TARGET_NAME}
    COMMAND "${PYTHON_EXECUTABLE}" "${ROOT_DIR}/${SCRIPT_REL}" ${PYT_ARGS}
    WORKING_DIRECTORY "${PYT_WORKING_DIRECTORY}"
    USES_TERMINAL
    COMMENT "Running ${SCRIPT_REL}"
  )

  if(PYT_DEPENDS)
    add_dependencies(${TARGET_NAME} ${PYT_DEPENDS})
  endif()
endfunction()

# -------- Stage-3 (WSI) utility: generate slide-level labels from folders --------
add_py_target(gen_slide_labels
  scripts/tcga_02_generate_slide_labels.py
)

# -------- Stage-1 (Risk engine) --------
set(RISK_CFG "${ROOT_DIR}/config/risk_engine/base.yaml")
set(XGB_CFG  "${ROOT_DIR}/config/risk_engine/xgboost_default.yaml")

add_py_target(risk_preprocess
  scripts/risk_01_preprocess_tabular_data.py
  ARGS --config "${RISK_CFG}"
)

add_py_target(risk_train
  scripts/risk_02_train_xgboost_model.py
  ARGS --risk-config "${RISK_CFG}" --xgb-config "${XGB_CFG}"
  DEPENDS risk_preprocess
)

add_custom_target(risk_all
  DEPENDS risk_preprocess risk_train
)

# -------- Stage-2 (Segmentation: PanNuke U-Net) --------
set(SEG_CFG "${ROOT_DIR}/config/segmentation/pannuke_unet_resnet34_base.yaml")
add_py_target(seg_train
  scripts/segmentation_legacy_train_unet_pannuke.py
  ARGS --config "${SEG_CFG}"
)

# Convenience target: show available targets.
add_custom_target(show_targets
  COMMAND "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}" --target help
  USES_TERMINAL
)

