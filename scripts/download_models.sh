# Actor SFT
mkdir -p /workspace/CodeRL_Reproduce/models/coderl-actor-sft
cd /workspace/CodeRL_Reproduce/models/coderl-actor-sft
wget https://huggingface.co/lck0328/coderl-actor-sft/resolve/main/config.json
wget https://huggingface.co/lck0328/coderl-actor-sft/resolve/main/pytorch_model.bin

# Actor RL epoch-6
mkdir -p /workspace/CodeRL_Reproduce/models/coderl-actor-rl-epoch-6
cd /workspace/CodeRL_Reproduce/models/coderl-actor-rl-epoch-6
wget https://huggingface.co/lck0328/coderl-actor-rl-epoch-6/resolve/main/config.json
wget https://huggingface.co/lck0328/coderl-actor-rl-epoch-6/resolve/main/pytorch_model.bin

# Critic
mkdir -p /workspace/CodeRL_Reproduce/models/coderl-critic
cd /workspace/CodeRL_Reproduce/models/coderl-critic
wget https://huggingface.co/lck0328/coderl-critic/resolve/main/config.json
wget https://huggingface.co/lck0328/coderl-critic/resolve/main/pytorch_model.bin

# Tokenizer (from Salesforce directly)
mkdir -p /workspace/CodeRL_Reproduce/models/codet5-base-tokenizer
cd /workspace/CodeRL_Reproduce/models/codet5-base-tokenizer
wget https://huggingface.co/Salesforce/codet5-base/resolve/main/config.json
wget https://huggingface.co/Salesforce/codet5-base/resolve/main/special_tokens_map.json
wget https://huggingface.co/Salesforce/codet5-base/resolve/main/tokenizer_config.json
wget https://huggingface.co/Salesforce/codet5-base/resolve/main/vocab.json
wget https://huggingface.co/Salesforce/codet5-base/resolve/main/merges.txt


