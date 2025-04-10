import torch
from fl4health.strategies.fed_avg import FedAvg
from fl4health.server.base_server import BaseServer
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from coco_dataset import CocoClientDataset
from torch.utils.data import DataLoader

# Configuration
NUM_CLIENTS = 10
BATCH_SIZE = 4

# Initialize model
def get_model(num_classes: int) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
    return model

# Federated Server and Clients
class CocoServer(BaseServer):
    def __init__(self, num_classes: int):
        super().__init__(FedAvg())
        self.num_classes = num_classes
    
    def get_global_model(self) -> torch.nn.Module:
        return get_model(self.num_classes)

class CocoClient:
    def __init__(self, client_id: int, data_path: str, split_dir: str):
        self.client_id = client_id
        self.dataset = CocoClientDataset(
            root=data_path,
            annFile="annotations/instances_val2017.json",
            client_split_file=f"{split_dir}/client_{client_id}.json",
            transform=torchvision.transforms.ToTensor(),
        )
        self.loader = DataLoader(self.dataset, batch_size=BATCH_SIZE, collate_fn=lambda x: tuple(zip(*x)))
        self.model = get_model(num_classes=80)  # COCO has 80 classes
    
    def local_train(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.005, momentum=0.9)
        self.model.train()
        for images, targets in self.loader:
            images = list(image.to(device) for image in images)
            targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        return self.model.state_dict()

# Run Federated Learning
if __name__ == "__main__":
    server = CocoServer(num_classes=80)
    for client_id in range(NUM_CLIENTS):
        client = CocoClient(client_id, "coco/test2017", "client_splits")
        server.register_client(client)
    server.run_rounds(num_rounds=10)