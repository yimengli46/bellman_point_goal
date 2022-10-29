help::
	@echo "Conditional Learned Subgoal Planning (lsp-cond):"
	@echo "  lsp-cond-gen-graph-data creates the data for the experiments."
	@echo "  lsp-cond-train-autoencoder trains the auto encoder model."
	@echo "  lsp-cond-train trains the GCN model."
	@echo "  lsp-cond-eval plans using the trained model vs baseline."
	@echo ""

LSP_COND_BASENAME ?= lsp_conditional
LSP_COND_NUM_TRAINING_SEEDS ?= 500
LSP_COND_NUM_TESTING_SEEDS ?= 100
LSP_COND_NUM_EVAL_SEEDS ?= 500

LSP_COND_CORE_ARGS ?= --save_dir /data/ \
		--unity_path /unity/$(DUNGEON_UNITY_BASENAME).x86_64 \
		--map_type maze \
		--base_resolution 1.0 \
		--inflation_radius_m 2.5 \
		--field_of_view_deg 360 \
		--laser_max_range_m 120 \
		--pickle_directory data_pickles

LSP_COND_DATA_GEN_ARGS ?= $(LSP_COND_CORE_ARGS) \
		--save_dir /data/$(LSP_COND_BASENAME)/ \
		--do_randomize_start_pose

LSP_COND_TRAINING_ARGS ?= --test_log_frequency 10 \
		--save_dir /data/$(LSP_COND_BASENAME)/logs/$(EXPERIMENT_NAME) \
		--data_csv_dir /data/$(LSP_COND_BASENAME)/

LSP_COND_EVAL_ARGS = $(LSP_COND_CORE_ARGS) \
		--save_dir /data/$(LSP_COND_BASENAME)/results/$(EXPERIMENT_NAME) \
		--network_file /data/$(LSP_COND_BASENAME)/logs/$(EXPERIMENT_NAME)/model.pt \
		--autoencoder_network_file /data/$(LSP_COND_BASENAME)/logs/$(EXPERIMENT_NAME)/AutoEncoder.pt \
		--do_randomize_start_pose

lsp-cond-data-gen-seeds = \
	$(shell for ii in $$(seq 1000 $$((1000 + $(LSP_COND_NUM_TRAINING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/data_completion_logs/data_training_$${ii}.txt"; done) \
	$(shell for ii in $$(seq 2000 $$((2000 + $(LSP_COND_NUM_TESTING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/data_completion_logs/data_testing_$${ii}.txt"; done)

$(lsp-cond-data-gen-seeds): traintest = $(shell echo $@ | grep -Eo '(training|testing)' | tail -1)
$(lsp-cond-data-gen-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-cond-data-gen-seeds):
	@echo "Generating Data [seed: $(seed) | $(traintest)"]
	@$(call xhost_activate)
	@-rm -f $(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/data_$(traintest)_$(seed).csv
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/pickles
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/data_completion_logs
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/error_logs
	@$(DOCKER_PYTHON) -m lsp_cond.scripts.gen_graph_data \
		$(LSP_COND_DATA_GEN_ARGS) \
	 	--current_seed $(seed) \
	 	--data_file_base_name data_$(traintest)

.PHONY: lsp-cond-gen-graph-data
lsp-cond-gen-graph-data: $(lsp-cond-data-gen-seeds)

lsp-cond-autoencoder-train-file = $(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/logs/$(EXPERIMENT_NAME)/AutoEncoder.pt
$(lsp-cond-autoencoder-train-file): $(lsp-cond-data-gen-seeds)
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/logs/$(EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m lsp_cond.scripts.lsp_cond_train \
		$(LSP_COND_TRAINING_ARGS) \
		--num_steps 100000 \
		--learning_rate 1e-3

lsp-cond-train-file = $(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/logs/$(EXPERIMENT_NAME)/model.pt 
$(lsp-cond-train-file): $(lsp-cond-autoencoder-train-file)
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/logs/$(EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m lsp_cond.scripts.lsp_cond_train \
		$(LSP_COND_TRAINING_ARGS) \
		--num_steps 100000 \
		--learning_rate 1e-4 \
		--autoencoder_network_file /data/$(LSP_COND_BASENAME)/logs/$(EXPERIMENT_NAME)/AutoEncoder.pt 

.PHONY: lsp-cond-train lsp-cond-train-autoencoder
lsp-cond-train-autoencoder: DOCKER_ARGS ?= -it
lsp-cond-train-autoencoder: $(lsp-cond-autoencoder-train-file)
lsp-cond-train: DOCKER_ARGS ?= -it
lsp-cond-train: $(lsp-cond-train-file)


LSP_COND_PLOTTING_ARGS = \
		--data_file /data/$(LSP_COND_BASENAME)/results/$(EXPERIMENT_NAME)/logfile.txt \
		--output_image_file /data/$(LSP_COND_BASENAME)/results/results_$(EXPERIMENT_NAME).png

.PHONY: lsp-cond-results
lsp-cond-results:
	@$(DOCKER_PYTHON) -m lsp.scripts.vis_lsp_plotting \
		$(LSP_COND_PLOTTING_ARGS)

lsp-cond-eval-seeds = \
	$(shell for ii in $$(seq 10000 $$((10000 + $(LSP_COND_NUM_EVAL_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/results/$(EXPERIMENT_NAME)/maze_learned_$${ii}.png"; done)
$(lsp-cond-eval-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-cond-eval-seeds): $(lsp-cond-train-file)
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/results/$(EXPERIMENT_NAME)
	@$(call xhost_activate)
	@$(DOCKER_PYTHON) -m lsp_cond.scripts.lsp_cond_evaluate \
		$(LSP_COND_EVAL_ARGS) \
	 	--current_seed $(seed) \
	 	--image_filename maze_learned_$(seed).png

.PHONY: lsp-cond-eval
lsp-cond-eval: $(lsp-cond-eval-seeds)
	$(MAKE) lsp-cond-results
