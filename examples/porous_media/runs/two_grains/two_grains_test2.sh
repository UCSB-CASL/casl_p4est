#!bash/sh

#
# This is example 1 in Fisrt_Author et al, "Title", Journal Volume (Year).
# Simulation of precipitation in a porous media with 2 grains.
# The precipitate at a given concentrations flows from left to right and contributes to the growth of each grain.
# The simulation output is saved in the folder CASL/simulation_output/porous_media/two_grains/
# In addition to the standard graphics software, the python code "plot_porous_media.py" can be used to plot 
# the concentration and the norm of the velocity field.
#





# -------------------------------------------------
# Toggle whether to run in debug or release mode:
# -------------------------------------------------
export DEBUG_OR_RELEASE=cmake-build-release	# cmake-build-debug

# -------------------------------------------------
# Output directories:
# -------------------------------------------------
export OUTPUT_BASE_DIR=$HOME/CASL/workspace/simulations_output/porous_media
export OUTPUT_DIR_NAME=two_grains
export OUT_DIR_VTK=$OUTPUT_BASE_DIR/$OUTPUT_DIR_NAME
export OUT_DIR_FILES=$OUTPUT_BASE_DIR/$OUTPUT_DIR_NAME
mkdir $OUT_DIR_VTK									# make the directory in case it doesn't exist

# -------------------------------------------------
# Save and load information:
# -------------------------------------------------
# loading from previous state?
export LOADING_FROM_PREV_STATE=0
#Load dir info:
export INIT_TRANSIENCE=$OUTPUT_BASE_DIR/initial_transience
export LOAD_STATE_BASE=$OUTPUT_BASE_DIR/$OUTPUT_DIR_NAME #$INIT_TRANSIENCE
export LOAD_STATE_BACKUP_NO=18
# it will auto-name the load state paths with the refinement level we are considering
# Save state info:
export OUT_DIR_SAVE_STATE=$OUTPUT_BASE_DIR/$OUTPUT_DIR_NAME

# -------------------------------------------------
# Geometry directory:
# -------------------------------------------------
export GEOMETRY_DIR=$HOME/CASL/casl_code_base/casl_p4est/examples/porous_media/runs/two_grains/geometry/two_grains_test2

# ------------------------------------------------------------------------------------------------------
# Input variables:
# ------------------------------------------------------------------------------------------------------
# type of nondimensionalization:
# ------------------------
export NONDIM_TYPE=1 # 0 = by fluid vel, 1 = by diffusivity

# ------------------------
# physics to solve
# ------------------------
export SOLVE_STEFAN=1
export SOLVE_NS=1
export IS_DISSOLUTION_CASE=1

# ------------------------
# settings for regularize front
# ------------------------
export USE_REG_FRONT=1
export PROX_SMOOTHING=2.0
export TRACK_EVOLVING_GEOM=1

# ------------------------
# Domain settings 
# ------------------------
export XMIN=0.0
export XMAX=6.0
export YMIN=0.0
export YMAX=3.0 #2.0
export NX=2
export NY=1
export LMIN=4				# Minimum level of the tree.
export LINT=0				
export LMAX=10				# Maximum level of the tree.
export DA=0.1

# ------------------------
# Refinement settings
# ------------------------
export UNIFORM_BAND=8.0
export REFINE_BY_D2T=1
export D2T_REFINE_THRESH=50.0 #10.0 #80.0 # was 100 for theta_init=2, so for theta_init=100 I multiplied that by 100
export D2T_COARSEN_THRESH=1.0 #10.0

# ------------------------
# CFL settings
# ------------------------
export CFL=0.5
export CFL_NS=1. #2.0
export NS_MAX=1000.0 # Max NS norm allowed before it "blows up"
export VINT_MAX=1000.0 # max interfacial velocity allowed before it "blows up"
export DO_PHI_ADV_SUBSTEPS=1
export CFL_PHI_ADV_SUBSTEP=1e-3

# ------------------------
# Initial fluid nondim conc and wall nondim conc
# ------------------------
export THETA_INFTY=40.0   # Using theta = C/Csat, this sets wall and initial domain at 2x the saturation , and we should see precipitation
export THETA_INIT=2.0 # the initial C/Csat value in the fluid domain

# ------------------------
# Settings for governing the dissolution/precip problem
# ------------------------
export GAMMA_DISS=-1.e-3 #-1.0e-10  
# Note: we set gamma_diss = -1 because when we nondim as C/Csat, it needs to have a negative. When we nondim by C/Cinf, as used in dissolution benchmark, we need gamma_diss to be positive. I may change the convention in the code later but for now this works
#export DA=200.0 #.0
export SC=1000.0 # 1000.0 #1000.0

# ------------------------
# Pressure drop
# ------------------------
export PRESS_DROP=0.10			# dimensional pressure drop in Pascals

# ------------------------
# Frequency of reinitialization
# ------------------------
export REINIT_EVERY_N=50

# ------------------------
# Settings for the LSF advection substep procedure:
# ------------------------
#export DO_PHI_ADV_SUBSTEPS=0
#export CFL_PHI_ADV_SUBSTEP=1.e-3
export PHI_ADV_SUBSTEP_STARTUP_TIME=3.0  # seconds

# ------------------------
# Characteristic length scale
# ------------------------
export LCHAR=1.0e-4 # length scale that corresponds to unit 1 in simulation. In this case, corresponds average grain size 

# ------------------------
# Duration and startup time
# ------------------------
export DURATION=780. #75.0 #2.0 # [in minutes]  --0.05 = 3 seconds,  0.0333 = equivalent to 2 seconds 
export STARTUP_DIM_TIME=-0.01 #0.03 #0.75 # try -- 0.75 seconds  1/19/21
export FORCE_VGAMMA_ZERO=0

# ------------------------
# Flushing information:
# ------------------------
export THETA_FLUSH=0.
export FLUSH_EVERY_DT=-0.15 #0.01 # next try 0.03 # really want to try 0.4
export FLUSH_DUR=-0.075 #0.005 # next try 0.01 # really want to try 0.05

# ------------------------
# Grain geometry properties
# ------------------------
export NUM_GRAINS=2
export USE_INNER_SURFACE=0
export OUTER_SURFACE_THICKNESS=4.0
export START_W_MERGED_GRAINS=1

# ------------------------
# Save settings
# ------------------------
export SAVE_TO_VTK=1
export SAVE_USING_ITER=0
export SAVE_EVERY_ITER=1
export SAVE_USING_DT=1
export SAVE_EVERY_DT=0.1 # in seconds

# ------------------------
# Save STATE settings
# ------------------------
export SAVE_STATE=1
export SAVE_STATE_USING_ITER=0
export SAVE_STATE_USING_DT=1
export SAVE_STATE_EVERY_DT=3600.0 #seconds
export SAVE_STATE_EVERY_ITER=100000

# ------------------------
# Getting timing info
# ------------------------
export TIMING_EVERY_N=1000

# ---------------------------------------------------
# Naming the load state path appropriately: 
# ---------------------------------------------------
export LOAD_STATE_NAME=$LOAD_STATE_BASE/"save_states_output_lmin_"$LMIN"_lmax_"$LMAX"_advection_order_2_example_8"
export LOAD_STATE_PATH=$LOAD_STATE_NAME"-1"/"backup_18" #$LOAD_STATE_BACKUP_NO
export LOAD_STATE_PATH_NS=$LOAD_STATE_NAME"_navier_stokes-1"/"backup_19" #$LOAD_STATE_BACKUP_NO

# ---------------------------------------------------
# Logfile name: 
# ---------------------------------------------------
export LOGNAME=$OUT_DIR_VTK/logfile"lmin"$LMIN"lmax"$LMAX"_reload"$LOADING_FROM_PREV_STATE
#valgrind --leak-check=full --track-origins=yes --log-file=valgrind_clogging_porous_reload

# ---------------------------------------------------
# Executable: 
# ---------------------------------------------------
export EXECUTABLE=$HOME/CASL/workspace/built_examples/porous_media/$DEBUG_OR_RELEASE/porous_media



# ---------------------------------------------------
# Command to run the case:
# ---------------------------------------------------
mpirun -np 10 $EXECUTABLE \
\
-example_ 8 \
\
-solve_stefan $SOLVE_STEFAN -solve_navier_stokes $SOLVE_NS -is_dissolution_case $IS_DISSOLUTION_CASE  \
\
-reinit_every_iter $REINIT_EVERY_N \
\
-do_phi_advection_substeps $DO_PHI_ADV_SUBSTEPS -cfl_phi_advection_substep $CFL_PHI_ADV_SUBSTEP -phi_advection_substep_startup_time $PHI_ADV_SUBSTEP_STARTUP_TIME \
\
-use_regularize_front $USE_REG_FRONT -proximity_smoothing $PROX_SMOOTHING -start_w_merged_grains $START_W_MERGED_GRAINS -track_evolving_geometries $TRACK_EVOLVING_GEOM \
\
-lmin $LMIN -lint $LINT -lmax $LMAX \
\
-xmin $XMIN -xmax $XMAX -ymin $YMIN -ymax $YMAX -nx $NX -ny $NY \
\
-nondim_type_used $NONDIM_TYPE \
\
 -uniform_band $UNIFORM_BAND  -refine_by_d2T $REFINE_BY_D2T -d2T_refine_threshold $D2T_REFINE_THRESH -d2T_coarsen_threshold $D2T_COARSEN_THRESH \
\
-cfl $CFL -cfl_NS $CFL_NS -NS_max_allowed $NS_MAX -v_int_max_allowed $VINT_MAX \
\
-l_char $LCHAR -theta_infty $THETA_INFTY -theta_initial $THETA_INIT -gamma_diss $GAMMA_DISS -Da $DA -Sc $SC \
\
-theta_flush $THETA_FLUSH -flush_every_dt $FLUSH_EVERY_DT -flush_duration $FLUSH_DUR \
\
-pressure_drop $PRESS_DROP \
\
-num_grains $NUM_GRAINS \
\
-use_inner_surface_porous_media $USE_INNER_SURFACE -porous_media_initial_thickness $OUTER_SURFACE_THICKNESS \
\
-duration_overwrite $DURATION -startup_dim_time $STARTUP_DIM_TIME \
\
-force_interfacial_velocity_to_zero $FORCE_VGAMMA_ZERO \
\
-save_to_vtk $SAVE_TO_VTK  -save_using_iter $SAVE_USING_ITER -save_every_iter $SAVE_EVERY_ITER -save_using_dt $SAVE_USING_DT -save_every_dt $SAVE_EVERY_DT \
\
-save_state $SAVE_STATE \
\
-save_state_using_iter $SAVE_STATE_USING_ITER -save_state_using_dt $SAVE_STATE_USING_DT \
\
-save_state_every_iter $SAVE_STATE_EVERY_ITER -save_state_every_dt $SAVE_STATE_EVERY_DT \
\
-timing_every_n $TIMING_EVERY_N \
\
-save_fluid_forces 0 -print_checkpoints 0 -loading_from_previous_state $LOADING_FROM_PREV_STATE >& $LOGNAME  | tee -a $OUTPUT_DIR/$LOGNAME
