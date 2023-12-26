import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base model with: a encoder
    """

    def __init__(self, base_encoder, args):
        super(BaseModel, self).__init__()

        # create the encoders
        self.encoder = base_encoder(
            low_dim=args.feature_dim, head_type=args.head_type)
