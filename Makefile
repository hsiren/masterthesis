SHELL=/bin/bash
include cfg.mk 


cfg_dir=configs
res_dir=res


all_configs=$(shell ls $(cfg_dir)/$(exp_dir))
all_targets=$(addprefix $(res_dir)/$(exp_dir)/,$(foreach fname,$(all_configs),$(shell echo $(fname) | sed 's/.json/.pkl/g')))
all_converted=$(addprefix $(res_dir)/$(exp_dir)/,$(foreach fname,$(all_configs),$(shell echo $(fname) | sed 's/.json/.csv/g')))


all: $(all_targets)
	@echo $^

init: $(cfg_dir)/$(exp_dir).json
	rm -f $(cfg_dir)/$(exp_dir)/*.json
	$(PYTHON) gen_json.py -i $^

pkg:
	tar cfz $(exp_dir).tar.gz $(res_dir)/$(exp_dir)
cv:$(all_converted)

$(res_dir)/$(exp_dir)/%.csv: $(res_dir)/$(exp_dir)/%.pkl
	$(PYTHON) cv_res_file.py -i $^ > $@

$(res_dir)/$(exp_dir)/%.pkl: $(cfg_dir)/$(exp_dir)/%.json
	$(PYTHON) main.py -i $^
