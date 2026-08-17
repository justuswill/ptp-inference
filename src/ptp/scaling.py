CONFIGS = {
    "qm9": dict(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
    ),
    "4k": dict(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
    ),
    "66k": dict(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
    ),
    "525k": dict(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
    ),
    "4.2M": dict(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
    ),
    "34M": dict(
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
    ),
    "268M": dict(
        hidden_size=1024,
        num_hidden_layers=16,
        num_attention_heads=16,
    ),
    "1.1B": dict(
        hidden_size=2048,
        num_hidden_layers=22,
        num_attention_heads=32,
    ),
}

for key, value in CONFIGS.items():
    value["intermediate_size"] = 4 * value["hidden_size"]


def make_scaling_llama(config_name: str, attn_implementation: str = "flex_attention", **kwargs):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = CONFIGS[config_name]
    model_config = LlamaConfig(**kwargs, **config)
    model_config._attn_implementation = attn_implementation
    return LlamaForCausalLM(model_config)
