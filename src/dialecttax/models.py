####################
# LANGUAGE MODELS  #
####################

LANGUAGE_MODELS = {
    # Llama — base
    "llama_1b_base": "meta-llama/Llama-3.2-1B",
    "llama_3b_base": "meta-llama/Llama-3.2-3B",
    "llama_8b_base": "meta-llama/Llama-3.1-8B",
    "llama_70b_base": "meta-llama/Llama-3.1-70B",
    # Llama — instruct
    "llama_1b_instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "llama_3b_instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "llama_8b_instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama_70b_instruct": "meta-llama/Llama-3.1-70B-Instruct",
    # Gemma — base
    "gemma_1b_base": "google/gemma-3-1b-pt",
    "gemma_4b_base": "google/gemma-3-4b-pt",
    "gemma_12b_base": "google/gemma-3-12b-pt",
    "gemma_27b_base": "google/gemma-3-27b-pt",
    # Gemma — instruct
    "gemma_1b_instruct": "google/gemma-3-1b-it",
    "gemma_4b_instruct": "google/gemma-3-4b-it",
    "gemma_12b_instruct": "google/gemma-3-12b-it",
    "gemma_27b_instruct": "google/gemma-3-27b-it",
    # Qwen — base
    "qwen_1.7b_base": "Qwen/Qwen3-1.7B-Base",
    "qwen_4b_base": "Qwen/Qwen3-4B-Base",
    "qwen_8b_base": "Qwen/Qwen3-8B-Base",
    # Qwen — instruct
    "qwen_1.7b_instruct": "Qwen/Qwen3-1.7B",
    "qwen_4b_instruct": "Qwen/Qwen3-4B",
    "qwen_8b_instruct": "Qwen/Qwen3-8B",
    "qwen_32b_instruct": "Qwen/Qwen3-32B",
}

LANGUAGE_MODELS_BASE = {k: v for k, v in LANGUAGE_MODELS.items() if k.endswith("_base")}
LANGUAGE_MODELS_INST = {k: v for k, v in LANGUAGE_MODELS.items() if k.endswith("_instruct")}


############
# MESSAGES #
############

def get_message(prompt: str, system: str | None = None, instruct: bool = True) -> list[dict]:
    # Remove trailing spaces for instruction-tuned models
    if instruct:
        system = system.strip()
        prompt = prompt.strip()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages
