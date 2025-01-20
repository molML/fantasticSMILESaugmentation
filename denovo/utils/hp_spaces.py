#%%
hp_space = {
    "lstm": {
        "size_layers": [[256, 256], [512, 512], [256, 256, 256], [512, 512, 512]],
        "lstm_activation": ["tanh"],
        "lstm_recurrentactivation": ["sigmoid"],
    },
    "common": {
        "n_dense": [1],
        "dense_layer_size": [256],
        "dense_activation": ["softmax"],
        "dropout_rate": [0.0],
        "optimizer_name": ["adam"],    
        "learning_rate": [1e-4, 5e-4, 1e-3],
        "batch_size": [32, 64, 128],
        "n_epochs": [500],
        "maxlen": [150],
        "loss": ["categorical_crossentropy"],
        "metric": ["accuracy"]     
    },
}


def sample_hp_space(sampled_space_size: int = 50):
    sampled_hps = list()
    hp_values = set()
    while len(sampled_hps) < sampled_space_size:
        hp_config = dict()
        model_type = random.choice(["lstm"])
        hp_config["model_type"] = model_type
        model_params = hp_space[model_type]
        for param_name, param_values in model_params.items():
            param_value = random.choice(param_values)
            hp_config[param_name] = param_value

        for param_name, param_values in hp_space["common"].items():
            param_value = random.choice(param_values)
            hp_config[param_name] = param_value

        hp_config_tuple = tuple(
            (param_name, tuple(param_value) if isinstance(param_value, list) else param_value)
            for param_name, param_value in hp_config.items()
        )

        if hp_config_tuple not in hp_values:
            hp_values.add(hp_config_tuple)
            sampled_hps.append(hp_config)

    return sampled_hps


if __name__ == "__main__":
    import json
    import random
    import pandas as pd

    random.seed(1)
    sampled_space = sample_hp_space(36)

    df_space = pd.DataFrame(sampled_space)
    df_space.to_csv("./scripts/base_scripts/sampled_hp_space.csv", index=False)      # HELENA CHANGED IT
    
    with open("./scripts/base_scripts/sampled_hp_space.json", "w") as f:     # HELENA CHANGED IT
        json.dump(sampled_space, f, indent=4)
    