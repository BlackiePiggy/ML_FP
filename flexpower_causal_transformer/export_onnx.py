import torch, yaml
from models.causal_transformer import CausalTransformerFlex

def main(cfg_path="config.yaml", out="model.onnx"):
    cfg = yaml.safe_load(open(cfg_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalTransformerFlex(
        num_cont=cfg["num_cont"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
        use_film=cfg["use_film"],
        film_dim=cfg["film_dim"],
        num_stations=cfg["num_stations"],
        num_receivers=cfg["num_receivers"],
        num_antennas=cfg["num_antennas"],
        num_constellations=cfg["num_constellations"],
        num_prns=cfg["num_prns"],
    ).to(device).eval()

    B, L, C = 4, cfg["seq_len"], cfg["num_cont"]
    x = torch.randn(B, L, C, device=device)
    meta = {
        "station_id": torch.zeros(B, 1, dtype=torch.long, device=device),
        "receiver_id": torch.zeros(B, 1, dtype=torch.long, device=device),
        "antenna_id": torch.zeros(B, 1, dtype=torch.long, device=device),
        "constellation_id": torch.zeros(B, 1, dtype=torch.long, device=device),
        "prn_id": torch.zeros(B, 1, dtype=torch.long, device=device),
    }

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x, station_id, receiver_id, antenna_id, constellation_id, prn_id):
            meta = {
                "station_id": station_id.long(),
                "receiver_id": receiver_id.long(),
                "antenna_id": antenna_id.long(),
                "constellation_id": constellation_id.long(),
                "prn_id": prn_id.long(),
            }
            return self.model(x, meta)

    wrapper = Wrapper(model)
    torch.onnx.export(
        wrapper, 
        (x, meta["station_id"], meta["receiver_id"], meta["antenna_id"], meta["constellation_id"], meta["prn_id"]),
        out,
        input_names=["x", "station_id", "receiver_id", "antenna_id", "constellation_id", "prn_id"],
        output_names=["logits"],
        dynamic_axes={"x": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
        opset_version=17
    )
    print(f"Exported to {out}")

if __name__ == "__main__":
    main()
