# This target is to make an image by calling a script within 'example'

organization_base_dir = $(DATA_BASE_DIR)/organization
env_type = blockworld
organization-data-gen-seeds = \
	$(shell for ii in $$(seq 1000 1249); do echo "$(organization_base_dir)/$(env_type)/data/training_env_plots/training_env_$$ii.png"; done) \
	$(shell for ii in $$(seq 2000 2149); do echo "$(organization_base_dir)/$(env_type)/data/training_env_plots/testing_env_$$ii.png"; done)

$(organization-data-gen-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(organization-data-gen-seeds): traintest = $(shell echo $@ | grep -Eo '(training|testing)' | tail -1)
$(organization-data-gen-seeds):
	@$(call arg_check_data)
	@mkdir -p $(organization_base_dir)/$(env_type)/data/pickles/
	@mkdir -p $(organization_base_dir)/$(env_type)/data/training_env_plots/
	@rm -f $(organization_base_dir)/$(env_type)/data/env_*_$(seed).csv
	@echo "env_$(seed)_$(traintest)_data"
	@$(DOCKER_PYTHON) -m organization.scripts.generate_data \
		--base_data_path /data/organization/$(env_type) \
		--data_file_base_name env_$(traintest) \
		--data_plot_name $(traintest)_env \
		--seed $(seed)

organization-train-net = $(organization_base_dir)/$(env_type)/training_logs/$(EXPERIMENT_NAME)/OrganizationNet.pt
$(organization-train-net): $(organization-data-gen-seeds)
		@mkdir -p $(organization_base_dir)/$(env_type)/training_logs/$(EXPERIMENT_NAME)/
		@$(call arg_check_data)
		@$(DOCKER_PYTHON) -m organization.scripts.train \
			 --training_data_file /data/organization/$(env_type)/data/*train*.csv \
			 --test_data_file /data/organization/$(env_type)/data/*test*.csv \
			 --logdir /data/organization/$(env_type)/training_logs/$(EXPERIMENT_NAME)/ \
			 --mini_summary_frequency 100 --summary_frequency 1000 \
			 --num_epochs 10 \
			 --learning_rate 0.001


ORGANIZATION_PLOTTING_ARGS = \
			--data_file /data/organization/$(env_type)/results/$(EXPERIMENT_NAME)/logfile.txt \
			--output_image_file /data/organization/$(env_type)/results/results_$(EXPERIMENT_NAME).png

organization-eval-seeds = \
	$(shell for ii in $$(seq 5000 5099); do echo "$(organization_base_dir)/$(env_type)/results/$(EXPERIMENT_NAME)/optimal_state_env_$${ii}.png"; done)

$(organization-eval-seeds): seed = $(shell echo '$@' | grep -Eo '[0-9]+' | tail -1)
$(organization-eval-seeds): $(organization-train-net)
	@echo "Random Seed: $(seed)"
	@mkdir -p $(organization_base_dir)/$(env_type)/results/$(EXPERIMENT_NAME)
	@mkdir -p $(organization_base_dir)/$(env_type)/results/$(EXPERIMENT_NAME)/non_learned/
	@mkdir -p $(organization_base_dir)/$(env_type)/results/$(EXPERIMENT_NAME)/learned/
	@$(DOCKER_PYTHON) -m organization.scripts.eval_organization \
		 --seed $(seed) \
		 --num_blocks 2 \
		 --num_tasks 5 \
		 --network_file /data/organization/$(env_type)/training_logs/$(EXPERIMENT_NAME)/OrganizationNet.pt \
		 --base_results_path /data/organization/$(env_type)/results/$(EXPERIMENT_NAME)/ \
		 --eval_folder /data/organization/$(env_type)/results/$(EXPERIMENT_NAME)/ \
		 --log_file /data/organization/$(env_type)/results/$(EXPERIMENT_NAME)/logfile.txt


organization-results:
	@$(DOCKER_PYTHON) -m organization.scripts.eval_result_hist \
		$(ORGANIZATION_PLOTTING_ARGS)

# Convenience Targets
.PHONY: organization-generate-data organization-train organization-eval organization-results organization
organization-generate-data: $(organization-data-gen-seeds)
organization-train: $(organization-train-net)
organization-eval: $(organization-eval-seeds)
organization: organization-eval
	$(MAKE) organization-results
