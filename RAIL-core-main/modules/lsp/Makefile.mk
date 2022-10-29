
help::
	@echo "Learned Subgoal Planning (lsp):"
	@echo "  lsp-maze	Runs the 'guided maze' experiments."
	@echo ""

LSP_MAZE_BASENAME ?= lsp/maze
LSP_UNITY_BASENAME ?= dungeon_env_2020
LSP_MAZE_NUM_TRAINING_SEEDS ?= 500
LSP_MAZE_NUM_TESTING_SEEDS ?= 100
LSP_MAZE_NUM_EVAL_SEEDS ?= 1000

LSP_SIM_ROBOT_ARGS ?= --step_size 1.8 \
		--num_primitives 32 \
		--field_of_view_deg 360

LSP_MAZE_CORE_ARGS ?= --unity_path /unity/$(LSP_UNITY_BASENAME).x86_64 \
		--map_type maze \
		--base_resolution 1.0 \
		--inflation_rad 2.5 \
		--laser_max_range_m 60 \
		--save_dir /data/$(LSP_MAZE_BASENAME)/
LSP_MAZE_DATA_GEN_ARGS = $(LSP_MAZE_CORE_ARGS) \
		--save_dir /data/$(LSP_MAZE_BASENAME)/training_data/
LSP_MAZE_TRAINING_ARGS = \
		--save_dir /data/$(LSP_MAZE_BASENAME)/training_logs/$(EXPERIMENT_NAME) \
		--data_csv_dir /data/$(LSP_MAZE_BASENAME)/training_data/
LSP_MAZE_EVAL_ARGS = $(LSP_MAZE_CORE_ARGS) \
		--save_dir /data/$(LSP_MAZE_BASENAME)/results/$(EXPERIMENT_NAME) \
		--network_file /data/$(LSP_MAZE_BASENAME)/training_logs/$(EXPERIMENT_NAME)/VisLSPOriented.pt

lsp-maze-data-gen-seeds = \
	$(shell for ii in $$(seq 1000 $$((1000 + $(LSP_MAZE_NUM_TRAINING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_MAZE_BASENAME)/training_data/data_collect_plots/data_training_$${ii}.png"; done) \
	$(shell for ii in $$(seq 2000 $$((2000 + $(LSP_MAZE_NUM_TESTING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_MAZE_BASENAME)/training_data/data_collect_plots/data_testing_$${ii}.png"; done)

$(lsp-maze-data-gen-seeds): traintest = $(shell echo $@ | grep -Eo '(training|testing)' | tail -1)
$(lsp-maze-data-gen-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-maze-data-gen-seeds):
	@echo "Generating Data [$(MAZE_XAI_BASENAME) | seed: $(seed) | $(traintest)"]
	@$(call xhost_activate)
	@-rm -f $(DATA_BASE_DIR)/$(LSP_MAZE_BASENAME)/training_data/data_$(traintest)_$(seed).csv
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_MAZE_BASENAME)/training_data/data
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_MAZE_BASENAME)/training_data/data_collect_plots
	@$(DOCKER_PYTHON) -m lsp.scripts.vis_lsp_generate_data \
		$(LSP_MAZE_DATA_GEN_ARGS) $(LSP_SIM_ROBOT_ARGS) \
	 	--current_seed $(seed) --data_file_base_name data_$(traintest)

lsp-maze-train-file = $(DATA_BASE_DIR)/$(LSP_MAZE_BASENAME)/training_logs/$(EXPERIMENT_NAME)/VisLSPOriented.pt
$(lsp-maze-train-file): $(lsp-maze-data-gen-seeds)
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_MAZE_BASENAME)/training_logs/$(EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m lsp.scripts.vis_lsp_train_net \
		$(LSP_MAZE_TRAINING_ARGS)

lsp-maze-eval-seeds = \
	$(shell for ii in $$(seq 10000 $$((10000 + $(LSP_MAZE_NUM_EVAL_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_MAZE_BASENAME)/results/$(EXPERIMENT_NAME)/maze_learned_$${ii}.png"; done)
$(lsp-maze-eval-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-maze-eval-seeds): $(lsp-maze-train-file)
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_MAZE_BASENAME)/results/$(EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m lsp.scripts.vis_lsp_eval \
		$(LSP_MAZE_EVAL_ARGS) \
	 	--current_seed $(seed) --image_filename maze_learned_$(seed).png

LSP_MAZE_PLOTTING_ARGS = \
		--data_file /data/$(LSP_MAZE_BASENAME)/results/$(EXPERIMENT_NAME)/logfile.txt \
		--output_image_file /data/$(LSP_MAZE_BASENAME)/results/results_$(EXPERIMENT_NAME).png

lsp-maze-results:
	@$(DOCKER_PYTHON) -m lsp.scripts.vis_lsp_plotting \
		$(LSP_MAZE_PLOTTING_ARGS)

.PHONY: lsp-maze-data-gen lsp-maze-train lsp-maze-eval lsp-maze-results lsp-maze
lsp-maze-data-gen: $(lsp-maze-data-gen-seeds)
lsp-maze-train: $(lsp-maze-train-file)
lsp-maze-eval: $(lsp-maze-eval-seeds)
lsp-maze: lsp-maze-eval
	$(MAKE) lsp-maze-results

lsp-maze-check: DATA_BASE_DIR = $(shell pwd)/data/check
lsp-maze-check: LSP_MAZE_NUM_TRAINING_SEEDS = 12
lsp-maze-check: LSP_MAZE_NUM_TESTING_SEEDS = 4
lsp-maze-check: LSP_MAZE_NUM_EVAL_SEEDS = 12
lsp-maze-check: build
	$(MAKE) lsp-maze DATA_BASE_DIR=$(DATA_BASE_DIR) \
		LSP_MAZE_NUM_TRAINING_SEEDS=$(LSP_MAZE_NUM_TRAINING_SEEDS) \
		LSP_MAZE_NUM_TESTING_SEEDS=$(LSP_MAZE_NUM_TESTING_SEEDS) \
		LSP_MAZE_NUM_EVAL_SEEDS=$(LSP_MAZE_NUM_EVAL_SEEDS)
