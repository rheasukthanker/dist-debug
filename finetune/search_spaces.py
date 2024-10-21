
from syne_tune.config_space import randint, lograndint


class SMALL:

    def __init__(self, gpt_model_specification):
        self.config_space = {
            'n_embd': lograndint(64, gpt_model_specification.n_embd),
            'n_layers': randint(1, gpt_model_specification.n_layer),
            'heads': randint(1, gpt_model_specification.n_head),
            'intermediate_size': randint(1, gpt_model_specification.intermediate_size)
        }

    @staticmethod
    def cast(config):
        return {
            'sub_network_n_embd': config['n_embd'],
            'sub_network_intermediate_size': [config[f'intermediate_size']] * config['n_layers'],
            'sub_network_num_heads': [config[f'heads']] * config['n_layers'],
            'sub_network_n_layers': config['n_layers']
        }


class MEDIUM:

    def __init__(self, gpt_model_specification):
        self.config_space = {
            'n_embd': lograndint(64, gpt_model_specification.n_embd),
            'n_layers': randint(1, gpt_model_specification.n_layer)
        }

        for li in range(gpt_model_specification.n_layer):
            self.config_space[f'heads_{li}'] = randint(1, gpt_model_specification.n_head)
            self.config_space[f'intermediate_size_{li}'] = randint(1, gpt_model_specification.intermediate_size)

    @staticmethod
    def cast(config):
        return {
            'sub_network_n_embd': config['n_embd'],
            'sub_network_intermediate_size': [config[f'intermediate_size_{li}'] for li in range(config['n_layers'])],
            'sub_network_num_heads': [config[f'heads_{li}'] for li in range(config['n_layers'])],
            'sub_network_n_layers': config['n_layers']
        }


class HWGPTBench:

    def __init__(self, gpt_model_specification):
        self.config_space = {
            "embed_dim": lograndint(1, gpt_model_specification.n_embd),
            "num_heads": randint(1, gpt_model_specification.n_head),
            "mlp_ratio": randint(1, 4),
            "depth": randint(1, gpt_model_specification.n_layer),
        }

    @staticmethod
    def cast(config):
        return {
            'sub_network_n_embd': config['embed_dim'],
            'sub_network_intermediate_size': [config[f'mlp_ratio'] * config['embed_dim']
                                              for _ in range(config['depth'])],
            'sub_network_num_heads': [config[f'num_heads'] for _ in range(config['depth'])],
            'sub_network_n_layers': config['depth']
        }


search_spaces = {
    'small': SMALL,
    'medium': MEDIUM,
    "hw_gpt_bench": HWGPTBench
}