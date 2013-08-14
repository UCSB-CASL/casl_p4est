# Define the HEAD_DIR to point to the top level directory of the project
# HEAD_DIR=???

# Project info
vpath %.cpp    $(HEAD_DIR)/src 
vpath %.c      $(HEAD_DIR)/src 
INCLUDE_FLAGS += -I$(HEAD_DIR)
SRCS           = utils.cpp my_p4est_vtk.cpp semi_lagrangian.cpp interpolating_function.cpp refine_coarsen.cpp
#MAIN_SRC      = main_test.cpp
#MAIN_SRC      = main_semi_lagrangian.cpp
#MAIN_SRC      = main_refine_coarsen.cpp
MAIN_SRC       = main_create_loop_output.cpp
SRCS += $(MAIN_SRC)
CSRCS          = my_p4est_tools.c my_p4est_nodes.c
BIN_NAME       = $(patsubst %.cpp,%,$(MAIN_SRC))
BIN_FOLDER     = .

# What build type to use? __MUST__ either be 'debug' or 'release'
BUILD_TYPE = debug

# CXX compiler options;
# CXX_COMPILER_TYPE: This variable is used to set correct flags for the compiler; it can be GNU (for g++) or INTEL (for icpc)
# CXX_USE_MPI_COMPILER: This variable is used to determine whether MPI should be used
CXX_COMPILER_TYPE    = GNU
CXX_DEFINES          = -DCASL_THROWS
CXX_EXTRA_FLAGS      = 

# Verbose control
PRINT_WARNINGS    = NO
PRINT_MAKE_OUTPUT = NO

# Where to put all temporary objets?
TEMP_OBJS_DIR      = objs

# Use any external libraries?
CASL_HAVE_OPENMP   = NO
CASL_HAVE_MPI      = YES
CASL_HAVE_PETSC    = YES
CASL_HAVE_P4EST    = YES

include $(HEAD_DIR)/common.mk

OBJS := \
	$(patsubst %.cpp, $(TEMP_OBJS_DIR)/%.o,$(SRCS)) \
	$(patsubst %.c, $(TEMP_OBJS_DIR)/%.o,$(CSRCS))
DEPS := \
	$(patsubst %.cpp, $(TEMP_OBJS_DIR)/.%.d,$(SRCS)) \
	$(patsubst %.c, $(TEMP_OBJS_DIR)/.%.d,$(CSRCS))

.PHONY: all
.DEFAULT_GOAL:=all
all:
	@echo	" ============== Build information =============== "
	@echo   " Build type          = $(BUILD_TYPE) "
	@echo   " CXX compiler type   = $(CXX_COMPILER_TYPE) " 
	@echo   " CXX compiler called = $(CXX_FULL_PATH)"
	@echo   " Compiler flags      = $(CXX_FLAGS)" 
	@echo   " Include path        = $(INCLUDE_FLAGS)" 
ifeq ($(CASL_HAVE_PETSC), YES)
	@echo   ""
	@echo   " Using external library PETSc:"
	@echo   "    PETSC_DIR  = $(PETSC_HOME_DIR)"
	@echo   "    PETSC_ARCH = $(PETSC_ARCH)"
	@echo	"    PETSC_LIBS = $(PETSC_LIB)"
endif
ifeq ($(CASL_HAVE_P4EST), YES)
	@echo   ""
	@echo   " Using external library p4est:"
	@echo   "    P4EST_DIR  = $(P4EST_HOME_DIR)"
	@echo   "    P4EST_ARCH = $(P4EST_ARCH)"
	@echo	"    P4EST_LIBS = $(P4EST_LIBS)"
endif
	@echo   ""
	@echo	" =========== Building process started =========== "
	@$(MKDIR) $(TEMP_OBJS_DIR)
	@$(MAKE)  $(OBJS)
	@$(MKDIR) $(BIN_FOLDER)
	@$(MAKE)  $(BIN_NAME)
	@echo	" ===================== done ===================== "

-include $(DEPS)

$(TEMP_OBJS_DIR)/%.o: %.cpp
	@echo " building $(@F) ..."
	$(CXX) -c $(CXX_FLAGS) $(INCLUDE_FLAGS) -MMD $< -o $@
	@mv $(TEMP_OBJS_DIR)/$*.d $(TEMP_OBJS_DIR)/.$*.d

$(TEMP_OBJS_DIR)/%.o: %.c
	@echo " building $(@F) ..."
	$(CXX) -c $(CXX_FLAGS) $(INCLUDE_FLAGS) -MMD $< -o $@
	@mv $(TEMP_OBJS_DIR)/$*.d $(TEMP_OBJS_DIR)/.$*.d

$(BIN_NAME): $(OBJS)
	@echo " linking $(@F) ..."
	$(CXX) $(LINK_FLAGS) $^ $(LINK_LIBS) -o $(BIN_FOLDER)/$(BIN_NAME)


# .PHONY: clean
# clean::
# 	@$(RM) $(TEMP_OBJS_DIR) $(BIN_FOLDER)/$(BIN_NAME)

.PHONY: cleanall
cleanall::
	@$(RM) $(TEMP_OBJS_DIR) $(BIN_FOLDER)/$(BIN_NAME) $(BIN_FOLDER)/*.vtu $(BIN_FOLDER)/*.xmf $(BIN_FOLDER)/*.hvy $(BIN_FOLDER)/*.pvtu $(BIN_FOLDER)/*.vtk $(BIN_FOLDER)/*.bin $(BIN_FOLDER)/*.visit
