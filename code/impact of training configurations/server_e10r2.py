# server_e10r2.py
import flwr as fl
from model import MaskCNN
import torch
from flwr.common import parameters_to_ndarrays


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_parameters = None

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, _ = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters is not None:
            model = MaskCNN()
            weights = parameters_to_ndarrays(aggregated_parameters)
            state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), weights)}
            model.load_state_dict(state_dict, strict=True)
            torch.save(model.state_dict(), f"global_model_round_{rnd}.pt")
            print(f"✅ Saved global model to global_model_round_{rnd}.pt")
        return aggregated_parameters, {}

def fit_config(rnd: int):
    return {"local_epochs": 10}

strategy = SaveModelStrategy(
    fraction_fit=1.0,
    min_fit_clients=5,
    min_available_clients=5,
    on_fit_config_fn=fit_config,
)

if __name__ == "__main__":
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=strategy
    )
