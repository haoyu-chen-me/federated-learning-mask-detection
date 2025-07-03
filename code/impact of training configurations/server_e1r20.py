# server_e1r20.py
import flwr as fl
from model import MaskCNN
import torch
from flwr.common import parameters_to_ndarrays

# 定义保存模型的策略类
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_parameters = None

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, _ = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters is not None:
            self.final_parameters = aggregated_parameters
            model = MaskCNN()
            weights = parameters_to_ndarrays(aggregated_parameters)
            params_dict = zip(model.state_dict().keys(), weights)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)

            model_path = f"global_model_round_{rnd}.pt"
            torch.save(model.state_dict(), model_path)
            print(f"Saved global model to {model_path}")
        return aggregated_parameters, {}

# 本地训练轮数设置函数
def fit_config(rnd: int):
    return {"local_epochs": 1}  # E=1 R=20

# 构造策略
strategy = SaveModelStrategy(
    fraction_fit=1.0,
    min_fit_clients=5,
    min_available_clients=5,
    on_fit_config_fn=fit_config,
)

# 启动服务器
if __name__ == "__main__":
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy
    )
