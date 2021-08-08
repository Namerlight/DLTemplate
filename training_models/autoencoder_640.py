import torch
import torch.nn as nn


class autoencoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.encoder_hidden_layer_1 = nn.Linear(in_features=640, out_features=320)
        self.encoder_hidden_layer_2 = nn.Linear(in_features=320, out_features=160)
        self.encoder_hidden_layer_3 = nn.Linear(in_features=160, out_features=80)
        self.encoder_output_layer = nn.Linear(in_features=80, out_features=2)
        self.decoder_hidden_layer_0 = nn.Linear(in_features=2, out_features=80)
        self.decoder_hidden_layer_1 = nn.Linear(in_features=80, out_features=160)
        self.decoder_hidden_layer_2 = nn.Linear(in_features=160, out_features=320)
        self.decoder_output_layer = nn.Linear(in_features=320, out_features=640)
        self.fc = nn.Linear(640, 2)

    def forward(self, x):
        # print(x.shape)

        x = x.view(-1, 640)
        x = self.encoder_hidden_layer_1(x)
        x = torch.relu(x)
        x = self.encoder_hidden_layer_2(x)
        x = torch.relu(x)
        x = self.encoder_hidden_layer_3(x)
        x = torch.relu(x)
        x = self.encoder_output_layer(x)
        x = torch.relu(x)

        x = self.decoder_hidden_layer_0(x)
        x = torch.relu(x)
        x = self.decoder_hidden_layer_1(x)
        x = torch.relu(x)
        x = self.decoder_hidden_layer_2(x)
        x = torch.relu(x)
        x = self.decoder_output_layer(x)
        x = torch.relu(x)
        x = self.fc(x)

        return x
