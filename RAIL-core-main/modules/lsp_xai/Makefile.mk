
MAZE_XAI_BASENAME ?= maze
FLOORPLAN_XAI_BASENAME ?= floorplan
SP_LIMIT_NUM ?= -1
XAI_LEARNING_SEED ?= 8616
XAI_LEARNING_XENT_FACTOR ?= 1
XAI_UNITY_BASENAME ?= dungeon_env_2020

## ==== Core arguments ====

TEST_ADDITIONAL_ARGS += --lsp-xai-maze-net-0SG-path=/resources/testing/lsp_xai_maze_0SG.ExpNavVisLSP.pt

SIM_ROBOT_ARGS ?= --step_size 1.8 \
		--num_primitives 32 \
		--field_of_view_deg 360
INTERP_ARGS ?= --summary_frequency 100 \
		--num_epochs 1 \
		--learning_rate 2.0e-2 \
		--batch_size 4

## ==== Maze Arguments and Experiments ====
MAZE_CORE_ARGS ?= --unity_path /unity/$(XAI_UNITY_BASENAME).x86_64 \
		--map_type maze \
		--base_resolution 1.0 \
		--inflation_rad 2.5 \
		--laser_max_range_m 60 \
		--save_dir /data/lsp_xai/$(MAZE_XAI_BASENAME)/
MAZE_DATA_GEN_ARGS = $(MAZE_CORE_ARGS) --logdir /data/lsp_xai/$(MAZE_XAI_BASENAME)/training/data_gen
MAZE_EVAL_ARGS = $(MAZE_CORE_ARGS) --logdir /data/lsp_xai/$(MAZE_XAI_BASENAME)/training/$(EXPERIMENT_NAME) \
		--logfile_name logfile_final.txt

lsp-xai-maze-dir = $(DATA_BASE_DIR)/lsp_xai/$(MAZE_XAI_BASENAME)

# Initialize the Learning
xai-maze-init-learning = $(lsp-xai-maze-dir)/training/data_gen/ExpNavVisLSP.init.pt
$(xai-maze-init-learning):
	@echo "Writing the 'initial' neural network: $@"
	@mkdir -p $(lsp-xai-maze-dir)/training/$(EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_train_eval \
		$(MAZE_DATA_GEN_ARGS) \
		$(SIM_ROBOT_ARGS) \
		$(INTERP_ARGS) \
		--do_init_learning

# Generate Data
xai-maze-data-gen-seeds = $(shell for ii in $$(seq 1000 1009); do echo "$(lsp-xai-maze-dir)/data_collect_plots/learned_planner_$$ii.png"; done)
$(xai-maze-data-gen-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(xai-maze-data-gen-seeds): $(xai-maze-init-learning)
	@echo "Generating Data: $@"
	@$(call xhost_activate)
	@$(call arg_check_unity)
	@rm -f $(lsp-xai-maze-dir)/lsp_data_$(seed).*.csv
	@mkdir -p $(lsp-xai-maze-dir)/data
	@mkdir -p $(lsp-xai-maze-dir)/data_collect_plots
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_train_eval \
		$(MAZE_DATA_GEN_ARGS) \
	 	$(SIM_ROBOT_ARGS) \
	 	$(INTERP_ARGS) \
	 	--do_data_gen \
	 	--current_seed $(seed)

# Train the Network
xai-maze-train-learning = $(lsp-xai-maze-dir)/training/$(EXPERIMENT_NAME)/ExpNavVisLSP.final.pt
$(xai-maze-train-learning): $(xai-maze-data-gen-seeds)
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_train_eval \
		$(MAZE_EVAL_ARGS) \
		$(SIM_ROBOT_ARGS) \
		$(INTERP_ARGS) \
		--sp_limit_num $(SP_LIMIT_NUM) \
		--learning_seed $(XAI_LEARNING_SEED) \
		--xent_factor $(XAI_LEARNING_XENT_FACTOR) \
		--do_train

# Evaluate Performance
xai-maze-eval-seeds = $(shell for ii in $$(seq 11000 11009); do echo "$(lsp-xai-maze-dir)/results/$(EXPERIMENT_NAME)/learned_planner_$$ii.png"; done)
$(xai-maze-eval-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(xai-maze-eval-seeds): $(xai-maze-train-learning)
	@echo "Evaluating Performance: $@"
	@$(call xhost_activate)
	@$(call arg_check_unity)
	@mkdir -p $(lsp-xai-maze-dir)/results/$(EXPERIMENT_NAME)
	$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_train_eval \
		$(MAZE_EVAL_ARGS) \
		$(SIM_ROBOT_ARGS) \
		$(INTERP_ARGS) \
		--do_eval \
		--save_dir /data/lsp_xai/$(MAZE_XAI_BASENAME)/results/$(EXPERIMENT_NAME) \
		--current_seed $(seed)


## ==== University Building (Floorplan) Environment Experiments ====
FLOORPLAN_CORE_ARGS ?= --unity_path /unity/$(XAI_UNITY_BASENAME).x86_64 \
		--map_type ploader \
		--base_resolution 0.6 \
		--inflation_radius_m 1.5 \
		--laser_max_range_m 72 \
		--save_dir /data/lsp_xai/$(FLOORPLAN_XAI_BASENAME)/
FLOORPLAN_DATA_GEN_ARGS ?= $(FLOORPLAN_CORE_ARGS) \
		--map_file /resources/university_building_floorplans/train/*.pickle \
		--logdir /data/lsp_xai/$(FLOORPLAN_XAI_BASENAME)/training/data_gen
FLOORPLAN_EVAL_ARGS ?= $(FLOORPLAN_CORE_ARGS) \
		--map_file /resources/university_building_floorplans/test/*.pickle \
		--logdir /data/lsp_xai/$(FLOORPLAN_XAI_BASENAME)/training/$(EXPERIMENT_NAME) \
		--logfile_name logfile_final.txt
lsp-xai-floorplan-dir = $(DATA_BASE_DIR)/lsp_xai/$(FLOORPLAN_XAI_BASENAME)


# Initialize the Learning
xai-floorplan-init-learning = $(lsp-xai-floorplan-dir)/training/data_gen/ExpNavVisLSP.init.pt
$(xai-floorplan-init-learning):
	@echo "Writing the 'initial' neural network [Floorplan: $(FLOORPLAN_XAI_BASENAME)]"
	@mkdir -p $(lsp-xai-floorplan-dir)/training/$(EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_train_eval \
		$(FLOORPLAN_DATA_GEN_ARGS) \
		$(SIM_ROBOT_ARGS) \
		$(INTERP_ARGS) \
		--do_init_learning

# Generate Data
xai-floorplan-data-gen-seeds = $(shell for ii in $$(seq 1000 1009); do echo "$(lsp-xai-floorplan-dir)/data_collect_plots/learned_planner_$$ii.png"; done)
$(xai-floorplan-data-gen-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(xai-floorplan-data-gen-seeds): $(xai-floorplan-init-learning)
	@echo "Generating Data: $@"
	@$(call xhost_activate)
	@rm -f $(lsp-xai-floorplan-dir)/lsp_data_$(seed).*.csv
	@mkdir -p $(lsp-xai-floorplan-dir)/data
	@mkdir -p $(lsp-xai-floorplan-dir)/data_collect_plots
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_train_eval \
		$(FLOORPLAN_DATA_GEN_ARGS) \
	 	$(SIM_ROBOT_ARGS) \
	 	$(INTERP_ARGS) \
	 	--do_data_gen \
	 	--current_seed $(seed)

# Train the Network
xai-floorplan-train-learning = $(lsp-xai-floorplan-dir)/training/$(EXPERIMENT_NAME)/ExpNavVisLSP.final.pt
$(xai-floorplan-train-learning): $(xai-floorplan-data-gen-seeds)
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_train_eval \
		$(FLOORPLAN_EVAL_ARGS) \
		$(SIM_ROBOT_ARGS) \
		$(INTERP_ARGS) \
		--sp_limit_num $(SP_LIMIT_NUM) \
		--learning_seed $(XAI_LEARNING_SEED) \
		--xent_factor $(XAI_LEARNING_XENT_FACTOR) \
		--do_train

# Evaluate Performance
xai-floorplan-eval-seeds = $(shell for ii in $$(seq 11000 11009); do echo "$(lsp-xai-floorplan-dir)/results/$(EXPERIMENT_NAME)/learned_planner_$$ii.png"; done)
$(xai-floorplan-eval-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(xai-floorplan-eval-seeds): $(xai-floorplan-train-learning)
	@echo "Evaluating Performance: $@"
	@$(call xhost_activate)
	@$(call arg_check_unity)
	@mkdir -p $(lsp-xai-floorplan-dir)/results/$(EXPERIMENT_NAME)
	$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_train_eval \
		$(FLOORPLAN_EVAL_ARGS) \
		$(SIM_ROBOT_ARGS) \
		$(INTERP_ARGS) \
		--do_eval \
		--current_seed $(seed) \
		--save_dir /data/lsp_xai/$(FLOORPLAN_XAI_BASENAME)/results/$(EXPERIMENT_NAME) \


# Some helper targets to run code individually
xai-floorplan-intervene-seeds-4SG = $(shell for ii in 11304 11591 11870 11336 11245 11649 11891 11315 11069 11202 11614 11576 11100 11979 11714 11430 11267 11064 11278 11367 11193 11670 11385 11180 11923 11195 11642 11462 11010 11386 11913 11103 11474 11855 11823 11641 11408 11899 11449 11393 11041 11435 11101 11610 11422 11546 11048 11070 11699 11618; do echo "$(lsp-xai-floorplan-dir)/results/$(EXPERIMENT_NAME)/learned_planner_$${ii}_intervened_4SG.png"; done)
xai-floorplan-intervene-seeds-allSG = $(shell for ii in 11304 11591 11870 11336 11245 11649 11891 11315 11069 11202 11614 11576 11100 11979 11714 11430 11267 11064 11278 11367 11193 11670 11385 11180 11923 11195 11642 11462 11010 11386 11913 11103 11474 11855 11823 11641 11408 11899 11449 11393 11041 11435 11101 11610 11422 11546 11048 11070 11699 11618; do echo "$(lsp-xai-floorplan-dir)/results/$(EXPERIMENT_NAME)/learned_planner_$${ii}_intervened_allSG.png"; done)
$(xai-floorplan-intervene-seeds-4SG): $(xai-floorplan-train-learning)
	@mkdir -p $(DATA_BASE_DIR)/$(FLOORPLAN_XAI_BASENAME)/results/$(EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_train_eval \
		$(FLOORPLAN_EVAL_ARGS) \
		$(SIM_ROBOT_ARGS) \
		$(INTERP_ARGS) \
		--do_intervene \
		--sp_limit_num 4 \
	 	--current_seed $(shell echo $@ | grep -Eo '[0-9]+' | tail -2 | head -1) \
		--save_dir /data/lsp_xai/$(FLOORPLAN_XAI_BASENAME)/results/$(EXPERIMENT_NAME) \
		--logfile_name logfile_intervene_4SG.txt

$(xai-floorplan-intervene-seeds-allSG): $(xai-floorplan-train-learning)
	@mkdir -p $(DATA_BASE_DIR)/$(FLOORPLAN_XAI_BASENAME)/results/$(EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_train_eval \
		$(FLOORPLAN_EVAL_ARGS) \
		$(SIM_ROBOT_ARGS) \
		$(INTERP_ARGS) \
		--do_intervene \
	 	--current_seed $(shell echo $@ | grep -Eo '[0-9]+' | tail -1) \
		--save_dir /data/lsp_xai/$(FLOORPLAN_XAI_BASENAME)/results/$(EXPERIMENT_NAME) \
		--logfile_name logfile_intervene_allSG.txt \

## ==== Results & Plotting ====
.PHONY: xai-process-results
xai-process-results:
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_results \
		--data_file /data/$(MAZE_XAI_BASENAME)/results/$(EXPERIMENT_NAME)/logfile_final.txt \
		--output_image_file /data/tmp.png
	@echo "==== Maze Results ===="
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_results \
		--data_file /data/$(MAZE_XAI_BASENAME)/results/base_allSG/logfile_final.txt \
			/data/$(MAZE_XAI_BASENAME)/results/base_4SG/logfile_final.txt \
			/data/$(MAZE_XAI_BASENAME)/results/base_0SG/logfile_final.txt \
		--output_image_file /data/maze_results.png
	@echo "==== Floorplan Results ===="
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_results \
		--data_file /data/$(FLOORPLAN_XAI_BASENAME)/results/base_allSG/logfile_final.txt \
			/data/$(FLOORPLAN_XAI_BASENAME)/results/base_4SG/logfile_final.txt \
			/data/$(FLOORPLAN_XAI_BASENAME)/results/base_0SG/logfile_final.txt \
		--output_image_file /data/floorplan_results.png
	@echo "==== Floorplan Intervention Results ===="
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_results \
		--data_file /data/$(FLOORPLAN_XAI_BASENAME)/results/base_4SG/logfile_intervene_4SG.txt \
			/data/$(FLOORPLAN_XAI_BASENAME)/results/base_4SG/logfile_intervene_allSG.txt \
		--do_intervene \
		--xpassthrough $(XPASSTHROUGH) \
		--output_image_file /data/floorplan_intervene_results.png \

.PHONY: xai-explanations
xai-explanations:
	@mkdir -p $(DATA_BASE_DIR)/explanations/
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_train_eval \
	 	$(MAZE_EVAL_ARGS) \
	 	$(SIM_ROBOT_ARGS) \
	 	$(INTERP_ARGS) \
		--do_explain \
		--explain_at 20 \
	 	--sp_limit_num 4 \
	  	--current_seed 1037 \
	 	--save_dir /data/explanations/ \
	 	--logdir /data/$(MAZE_XAI_BASENAME)/training/base_0SG
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_train_eval \
		$(FLOORPLAN_EVAL_ARGS) \
		$(SIM_ROBOT_ARGS) \
		$(INTERP_ARGS) \
		--do_explain \
		--explain_at 289 \
		--sp_limit_num 4 \
	 	--current_seed 11591 \
		--save_dir /data/explanations/ \
		--logdir /data/$(FLOORPLAN_XAI_BASENAME)/training/base_4SG

## ==== Some helper targets to run code individually ====
# Maze
xai-maze-data-gen: $(xai-maze-data-gen-seeds)
xai-maze-train: $(xai-maze-train-learning)
xai-maze-eval: $(xai-maze-eval-seeds)
xai-maze: xai-maze-eval

# Floorplan
xai-floorplan-data-gen: $(xai-floorplan-data-gen-seeds)
xai-floorplan-train: $(xai-floorplan-train-learning)
xai-floorplan-eval: $(xai-floorplan-eval-seeds)
xai-floorplan: xai-floorplan-eval

xai-floorplan-data-gen: $(xai-floorplan-data-gen-seeds)
xai-floorplan-intervene: $(xai-floorplan-intervene-seeds-allSG) $(xai-floorplan-intervene-seeds-4SG)
