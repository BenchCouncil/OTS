import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Autoregressive GRU forecaster for long-term forecasting.

    The encoder reads the full history window once. The decoder then rolls out
    pred_len steps by feeding each predicted step back as the next input.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers

        dropout = configs.dropout if self.num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.decoder = nn.GRU(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.projection = nn.Linear(self.hidden_size, self.c_out)
        self.feedback_projection = (
            nn.Identity() if self.c_out == self.enc_in else nn.Linear(self.c_out, self.enc_in)
        )

    def forecast(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        _, hidden = self.encoder(x)
        decoder_input = x[:, -1:, :]
        outputs = []
        for _ in range(self.pred_len):
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            step = self.projection(decoder_output[:, -1, :])
            outputs.append(step.unsqueeze(1))
            decoder_input = self.feedback_projection(step).unsqueeze(1)

        output = torch.cat(outputs, dim=1)
        output_means = means[:, :, -self.c_out:]
        output_stdev = stdev[:, :, -self.c_out:]
        return output * output_stdev + output_means

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            return self.forecast(x_enc)[:, -self.pred_len:, :]
        return None
