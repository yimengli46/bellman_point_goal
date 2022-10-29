# This target is to make an image by calling a script within 'example'

envprep_base_dir = $(DATA_BASE_DIR)/envprep
env_type = blockworld
envprep-data-gen-seeds = \
	$(shell for ii in $$(seq 1000 5999); do echo "$(envprep_base_dir)/$(env_type)/data/training_env_plots/training_env_$$ii.png"; done) \
	$(shell for ii in $$(seq 6000 8499); do echo "$(envprep_base_dir)/$(env_type)/data/training_env_plots/testing_env_$$ii.png"; done)

$(envprep-data-gen-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(envprep-data-gen-seeds): traintest = $(shell echo $@ | grep -Eo '(training|testing)' | tail -1)
$(envprep-data-gen-seeds):
	@$(call arg_check_data)
	@mkdir -p $(envprep_base_dir)/$(env_type)/data/pickles/
	@mkdir -p $(envprep_base_dir)/$(env_type)/data/training_env_plots/
	@rm -f $(envprep_base_dir)/$(env_type)/data/env_*_$(seed).csv
	@echo "env_$(seed)_$(traintest)_data"
	@$(DOCKER_PYTHON) -m envprep.scripts.generate_gcn_data \
		--base_data_path /data/envprep/$(env_type) \
		--data_file_base_name env_$(traintest) \
		--data_plot_name $(traintest)_env \
		--seed $(seed)

envprep-train-net = $(envprep_base_dir)/$(env_type)/training_logs/$(EXPERIMENT_NAME)/PrepareEnvGCN.pt
$(envprep-train-net): $(envprep-data-gen-seeds)
		@mkdir -p $(envprep_base_dir)/$(env_type)/training_logs/$(EXPERIMENT_NAME)/
		@$(call arg_check_data)
		@$(DOCKER_PYTHON) -m envprep.scripts.gcn_train \
			 --training_data_file /data/envprep/$(env_type)/data/*train*.csv \
			 --test_data_file /data/envprep/$(env_type)/data/*test*.csv \
			 --logdir /data/envprep/$(env_type)/training_logs/$(EXPERIMENT_NAME)/ \
			 --mini_summary_frequency 100 --summary_frequency 1000 \
			 --num_epochs 10 \
			 --learning_rate 0.01

ENVPREP_PLOTTING_ARGS = \
			--data_file /data/envprep/$(env_type)/results/$(EXPERIMENT_NAME)/logfile.txt \
			--states_file /data/envprep/$(env_type)/results/$(EXPERIMENT_NAME)/statesfile.txt \
			--output_image_file /data/envprep/$(env_type)/results/results_$(EXPERIMENT_NAME).png \
			--pie_chart /data/envprep/$(env_type)/results/results_$(EXPERIMENT_NAME)_piechart.png


envprep-eval-seeds = \
	$(shell for ii in $$(seq 9000 9999); do echo "$(envprep_base_dir)/$(env_type)/results/$(EXPERIMENT_NAME)/optimal_state_env_$${ii}.png"; done)

$(envprep-eval-seeds): seed = $(shell echo '$@' | grep -Eo '[0-9]+' | tail -1)
$(envprep-eval-seeds): $(envprep-train-net)
	@echo "Random Seed: $(seed)"
	@mkdir -p $(envprep_base_dir)/$(env_type)/results/$(EXPERIMENT_NAME)
	@mkdir -p $(envprep_base_dir)/$(env_type)/results/$(EXPERIMENT_NAME)/non_learned/
	@mkdir -p $(envprep_base_dir)/$(env_type)/results/$(EXPERIMENT_NAME)/learned/
	@$(DOCKER_PYTHON) -m envprep.scripts.eval_gcn \
		 --seed $(seed) \
		 --network_file /data/envprep/$(env_type)/training_logs/$(EXPERIMENT_NAME)/PrepareEnvGCN.pt \
		 --base_results_path /data/envprep/$(env_type)/results/$(EXPERIMENT_NAME)/ \
		 --eval_folder /data/envprep/$(env_type)/results/$(EXPERIMENT_NAME)/ \
		 --log_file /data/envprep/$(env_type)/results/$(EXPERIMENT_NAME)/logfile.txt \
		 --states_file /data/envprep/$(env_type)/results/$(EXPERIMENT_NAME)/statesfile.txt



envprep-results:
	@$(DOCKER_PYTHON) -m envprep.scripts.eval_result \
		$(ENVPREP_PLOTTING_ARGS)

# Convenience Targets
.PHONY: envprep-generate-data envprep-train envprep-eval envprep-results envprep
envprep-generate-data: $(envprep-data-gen-seeds)
envprep-train: $(envprep-train-net)
envprep-eval: $(envprep-eval-seeds)
envprep: envprep-eval
	$(MAKE) envprep-results
