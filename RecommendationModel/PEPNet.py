import torch.nn as nn


class GateNU(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, scale=2.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        self.scale = scale  # 缩放因子

    def forward(self, x):
        output = self.scale * self.gate(x)
        return output


class EPNet(nn.Module):
    def __init__(self, domain_feature_dim, embed_dim):
        super().__init__()
        self.gate = GateNU(domain_feature_dim, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        return self.gate(x) * self.embed_dim


class PPNet(nn.Module):
    def __init__(self, user_item_author_feature_dim, dnn_hidden_dim):
        super().__init__()
        self.gate = GateNU(user_item_author_feature_dim, dnn_hidden_dim)
        self.dnn_hidden_dim = dnn_hidden_dim

    def forward(self, x):
        return self.gate(x) * self.dnn_hidden_dim


class PEPNet:
    def __init__(self, user_feature_dim, item_feature_dim, embed_dim, domain_feature_dim, dnn_hidden_dim):
        super().__init__()

        self.user_embedding = nn.Embedding(user_feature_dim, embed_dim)
        self.item_embedding = nn.Embedding(item_feature_dim, embed_dim)

        self.ep_net = EPNet(embed_dim, domain_feature_dim)

        self.shared_dnn = nn.Sequential(
            nn.Linear(embed_dim, dnn_hidden_dim),
            nn.ReLU()
        )
