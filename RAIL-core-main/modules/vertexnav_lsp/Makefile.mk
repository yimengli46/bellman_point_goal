
VERTEXNAV_LSP_MAZE_BASENAME ?= vertexnav_lsp/maze
VERTEXNAV_LSP_UNITY_BASENAME ?= $(VERTEXNAV_UNITY_BASENAME)
VERTEXNAV_LSP_MAZE_NUM_TRAINING_SEEDS ?= 500
VERTEXNAV_LSP_MAZE_NUM_TESTING_SEEDS ?= 100
VERTEXNAV_LSP_MAZE_NUM_EVAL_SEEDS ?= 500
VERTEXNAV_CORE_ARGS = \
	--unity_path /unity/$(VERTEXNAV_UNITY_BASENAME).x86_64 \
	--xpassthrough $(XPASSTHROUGH) \
	--max_range 100 \
	--num_range 32 \
	--num_bearing 128
vertexnav_dungeon_base_dir = $(DATA_BASE_DIR)/vertexnav/dungeon

vertexnav-lsp-maze-data-gen-seeds = \
	$(shell for ii in $$(seq 1000 $$((1000 + $(VERTEXNAV_LSP_MAZE_NUM_TRAINING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(VERTEXNAV_LSP_MAZE_BASENAME)/training_data/data_collect_plots/data_training_$${ii}.png"; done) \
	$(shell for ii in $$(seq 2000 $$((2000 + $(VERTEXNAV_LSP_MAZE_NUM_TESTING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(VERTEXNAV_LSP_MAZE_BASENAME)/training_data/data_collect_plots/data_testing_$${ii}.png"; done)

$(vertexnav-lsp-maze-data-gen-seeds): traintest = $(shell echo $@ | grep -Eo '(training|testing)' | tail -1)
$(vertexnav-lsp-maze-data-gen-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(vertexnav-lsp-maze-data-gen-seeds):
	@echo "Generating Data [$(MAZE_XAI_BASENAME) | seed: $(seed) | $(traintest)"]
	@$(call xhost_activate)
	@-rm -f $(DATA_BASE_DIR)/$(VERTEXNAV_LSP_MAZE_BASENAME)/training_data/data_$(traintest)_$(seed).csv
	@mkdir -p $(DATA_BASE_DIR)/$(VERTEXNAV_LSP_MAZE_BASENAME)/training_data/data
	@mkdir -p $(DATA_BASE_DIR)/$(VERTEXNAV_LSP_MAZE_BASENAME)/training_data/data_collect_plots
	@$(DOCKER_PYTHON) -m vertexnav_lsp.scripts.generate_data \
		--unity_path /unity/$(VERTEXNAV_UNITY_BASENAME).x86_64 \
		--num_range 32 \
		--num_bearing 128 \
		--save_dir /data/$(VERTEXNAV_LSP_MAZE_BASENAME)/training_data \
		--base_resolution 1.0 \
		--inflation_rad 2.5 \
		--laser_max_range_m 120 \
		--map_type maze \
	 	--current_seed $(seed) --data_file_base_name data_$(traintest)

vertexnav-lsp-maze-train-file = $(DATA_BASE_DIR)/$(VERTEXNAV_LSP_MAZE_BASENAME)/training_logs/$(EXPERIMENT_NAME)/VertexLSPOmni.pt
$(vertexnav-lsp-maze-train-file): $(vertexnav-lsp-maze-data-gen-seeds)
	@mkdir -p $(DATA_BASE_DIR)/$(VERTEXNAV_LSP_MAZE_BASENAME)/training_logs/$(EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m vertexnav_lsp.scripts.train_net \
	 	--training_data_file /data/$(VERTEXNAV_LSP_MAZE_BASENAME)/training_data/*train*.csv \
	 	--test_data_file /data/$(VERTEXNAV_LSP_MAZE_BASENAME)/training_data/*test*.csv \
	 	--logdir /data/$(VERTEXNAV_LSP_MAZE_BASENAME)/training_logs/$(EXPERIMENT_NAME)/ \
	 	--mini_summary_frequency 100 --summary_frequency 1000 \
	 	--num_epochs 6 \
	 	--learning_rate 0.001 \

vertexnav-lsp-maze-eval-seeds = \
	$(shell for ii in $$(seq 10000 $$((10000 + $(VERTEXNAV_LSP_MAZE_NUM_EVAL_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(VERTEXNAV_LSP_MAZE_BASENAME)/results/$(EXPERIMENT_NAME)/maze_learned_$${ii}.png"; done) \

$(vertexnav-lsp-maze-eval-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(vertexnav-lsp-maze-eval-seeds): $(vertexnav-lsp-maze-train-file)
	@echo "Generating Data [$(MAZE_XAI_BASENAME) | seed: $(seed) | $(traintest)"]
	@$(call xhost_activate)
	@mkdir -p $(DATA_BASE_DIR)/$(VERTEXNAV_LSP_MAZE_BASENAME)/results/$(EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m vertexnav_lsp.scripts.eval \
		--unity_path /unity/$(VERTEXNAV_UNITY_BASENAME).x86_64 \
		--num_range 32 \
		--num_bearing 128 \
		--network_file /data/$(VERTEXNAV_LSP_MAZE_BASENAME)/training_logs/$(EXPERIMENT_NAME)/VertexLSPOmni.pt \
		--save_dir /data/$(VERTEXNAV_LSP_MAZE_BASENAME)/results/$(EXPERIMENT_NAME) \
		--base_resolution 1.0 \
		--inflation_rad 2.5 \
		--laser_max_range_m 120 \
		--map_type maze \
		--sig_r 5.0 --sig_th 0.1 --nn_peak_thresh 0.5 \
	 	--current_seed $(seed) --data_file_base_name data_$(traintest)

vertexnav-lsp-maze-results:
	@$(DOCKER_PYTHON) -m vertexnav_lsp.scripts.plot \
		--data_file /data/$(VERTEXNAV_LSP_MAZE_BASENAME)/results/$(EXPERIMENT_NAME)/logfile.txt \
		--output_image_file /data/$(VERTEXNAV_LSP_MAZE_BASENAME)/results/results_$(EXPERIMENT_NAME).png

vertexnav-lsp-maze-data-gen: $(vertexnav-lsp-maze-data-gen-seeds)
vertexnav-lsp-maze-train: $(vertexnav-lsp-maze-train-file)
vertexnav-lsp-maze-eval: $(vertexnav-lsp-maze-eval-seeds)
vertexnav-lsp-maze: vertexnav-lsp-maze-eval
	$(MAKE) vertexnav-lsp-maze-results
