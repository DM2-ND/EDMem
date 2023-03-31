import torch
import numpy as np
from typing import Union
from transformers.trainer_pt_utils import nested_numpify


class DetailLoss:
    """
    Loss values for EMAG training
    """

    def __init__(self, loss_dict=None):

        # each element should be a torch.Tensor
        self.lm_loss = torch.tensor(0.0)
        self.el_loss = torch.tensor(0.0)
        self.encoder_elloss = torch.tensor(0.0)
        self.decoder_elloss = torch.tensor(0.0)
        self.final_elloss = torch.tensor(0.0)

        if isinstance(loss_dict, dict):
            self.lm_loss = loss_dict["lm_loss"]
            self.el_loss = loss_dict["el_loss"]
            self.encoder_elloss = loss_dict["encoder_elloss"]
            self.decoder_elloss = loss_dict["decoder_elloss"]
            self.final_elloss = loss_dict["final_elloss"]
        elif isinstance(loss_dict, tuple) or isinstance(loss_dict, list):
            self.lm_loss = loss_dict[0]
            self.el_loss = loss_dict[1]
            self.encoder_elloss = loss_dict[2]
            self.decoder_elloss = loss_dict[3]
            self.final_elloss = loss_dict[4]
        elif isinstance(loss_dict, map):
            loss_list = list(loss_dict)
            self.lm_loss = loss_list[0]
            self.el_loss = loss_list[1]
            self.encoder_elloss = loss_list[2]
            self.decoder_elloss = loss_list[3]
            self.final_elloss = loss_list[4]
        elif loss_dict is not None:
            raise ValueError("Invalid parameter type of loss_dict!")

        self.__post_init__()

    def __post_init__(self):
        if self.lm_loss is None:
            self.lm_loss = torch.tensor(0.0)
        if self.el_loss is None:
            self.el_loss = torch.tensor(0.0)
        if self.encoder_elloss is None:
            self.encoder_elloss = torch.tensor(0.0)
        if self.decoder_elloss is None:
            self.decoder_elloss = torch.tensor(0.0)
        if self.final_elloss is None:
            self.final_elloss = torch.tensor(0.0)

    def to(self, device):
        """
        Place loss tensors to a certain device.
        """
        self.lm_loss = self.lm_loss.to(device)
        self.el_loss = self.el_loss.to(device)
        self.encoder_elloss = self.encoder_elloss.to(device)
        self.decoder_elloss = self.decoder_elloss.to(device)
        self.final_elloss = self.final_elloss.to(device)
        return self

    def __str__(self):
        return f"lm_loss: {self.lm_loss}, " \
               f"el_loss: {self.el_loss}, " \
               f"encoder_elloss: {self.encoder_elloss}, " \
               f"decoder_elloss: {self.decoder_elloss}, " \
               f"final_elloss: {self.final_elloss}"

    def __add__(self, other):
        if type(other) == DetailLoss:
            lm_loss = self.lm_loss + other.lm_loss
            el_loss = self.el_loss + other.el_loss
            encoder_elloss = self.encoder_elloss + other.encoder_elloss
            decoder_elloss = self.decoder_elloss + other.decoder_elloss
            final_elloss = self.final_elloss + other.final_elloss
        elif type(other) == int or type(other) == float:
            lm_loss = self.lm_loss + other
            el_loss = self.el_loss + other
            encoder_elloss = self.encoder_elloss + other
            decoder_elloss = self.decoder_elloss + other
            final_elloss = self.final_elloss + other
        else:
            raise TypeError(f"Invalid add operation: DetailLoss and {type(other)}")
        return DetailLoss(loss_dict=(lm_loss, el_loss, encoder_elloss, decoder_elloss, final_elloss))

    def __sub__(self, other):
        if type(other) == DetailLoss:
            lm_loss = self.lm_loss - other.lm_loss
            el_loss = self.el_loss - other.el_loss
            encoder_elloss = self.encoder_elloss - other.encoder_elloss
            decoder_elloss = self.decoder_elloss - other.decoder_elloss
            final_elloss = self.final_elloss - other.final_elloss
        elif type(other) == int or type(other) == float:
            lm_loss = self.lm_loss - other
            el_loss = self.el_loss - other
            encoder_elloss = self.encoder_elloss - other
            decoder_elloss = self.decoder_elloss - other
            final_elloss = self.final_elloss - other
        else:
            raise TypeError(f"Invalid add operation: DetailLoss and {type(other)}")
        return DetailLoss(loss_dict=(lm_loss, el_loss, encoder_elloss, decoder_elloss, final_elloss))

    def __truediv__(self, other):
        if type(other) == int or type(other) == float:
            lm_loss = self.lm_loss / other
            el_loss = self.el_loss / other
            encoder_elloss = self.encoder_elloss / other
            decoder_elloss = self.decoder_elloss / other
            final_elloss = self.final_elloss / other
        else:
            raise TypeError(f"Invalid divide operation: DetailLoss and {type(other)}")
        return DetailLoss(loss_dict=(lm_loss, el_loss, encoder_elloss, decoder_elloss, final_elloss))

    def clear(self):
        self.lm_loss.zero_()
        self.el_loss.zero_()
        self.encoder_elloss.zero_()
        self.decoder_elloss.zero_()
        self.final_elloss.zero_()

    def mean(self):
        lm_loss = self.lm_loss.mean()
        el_loss = self.el_loss.mean()
        encoder_elloss = self.encoder_elloss.mean()
        decoder_elloss = self.decoder_elloss.mean()
        final_elloss = self.final_elloss.mean()
        return DetailLoss(loss_dict=(lm_loss, el_loss, encoder_elloss, decoder_elloss, final_elloss))

    def detach(self):
        lm_loss = self.lm_loss.detach()
        el_loss = self.el_loss.detach()
        encoder_elloss = self.encoder_elloss.detach()
        decoder_elloss = self.decoder_elloss.detach()
        final_elloss = self.final_elloss.detach()
        return DetailLoss(loss_dict=(lm_loss, el_loss, encoder_elloss, decoder_elloss, final_elloss))

    def __iter__(self):
        for t in (self.lm_loss,
                  self.el_loss,
                  self.encoder_elloss,
                  self.decoder_elloss,
                  self.final_elloss):
            yield t

    def tolist(self):
        return [self.lm_loss.item(),
                self.el_loss.item(),
                self.encoder_elloss.item(),
                self.decoder_elloss.item(),
                self.final_elloss.item()]

    def totuple(self):
        return (self.lm_loss.item(),
                self.el_loss.item(),
                self.encoder_elloss.item(),
                self.decoder_elloss.item(),
                self.final_elloss.item())

    def item(self):
        return self.totuple()

    def to_tupled_tensor(self):
        return (
            self.lm_loss,
            self.el_loss,
            self.encoder_elloss,
            self.decoder_elloss,
            self.final_elloss
        )

    def repeat(self, size: Union[torch.Size, int]):
        lm_loss = self.lm_loss.repeat(size)
        el_loss = self.el_loss.repeat(size)
        encoder_elloss = self.encoder_elloss.repeat(size)
        decoder_elloss = self.decoder_elloss.repeat(size)
        final_elloss = self.final_elloss.repeat(size)
        return DetailLoss(loss_dict=(lm_loss, el_loss, encoder_elloss, decoder_elloss, final_elloss))

    def concat(self, other, dim):
        lm_loss = torch.concat((self.lm_loss, other.lm_loss), dim=dim)
        el_loss = torch.concat((self.el_loss, other.el_loss), dim=dim)
        encoder_elloss = torch.concat((self.encoder_elloss, other.encoder_elloss), dim=dim)
        decoder_elloss = torch.concat((self.decoder_elloss, other.decoder_elloss), dim=dim)
        final_elloss = torch.concat((self.final_elloss, other.final_elloss), dim=dim)
        return DetailLoss(loss_dict=(lm_loss, el_loss, encoder_elloss, decoder_elloss, final_elloss))

    def np_concat(self, other, axis):
        assert type(self.lm_loss) == np.ndarray
        assert type(other.lm_loss) == np.ndarray
        lm_loss = np.concatenate((self.lm_loss, other.lm_loss), axis=0)
        el_loss = np.concatenate((self.el_loss, other.el_loss), axis=0)
        encoder_elloss = np.concatenate((self.encoder_elloss, other.encoder_elloss), axis=0)
        decoder_elloss = np.concatenate((self.decoder_elloss, other.decoder_elloss), axis=0)
        final_elloss = np.concatenate((self.final_elloss, other.final_elloss), axis=0)
        return DetailLoss(loss_dict=(lm_loss, el_loss, encoder_elloss, decoder_elloss, final_elloss))

    def numpify(self):
        lm_loss = nested_numpify(self.lm_loss)
        el_loss = nested_numpify(self.el_loss)
        encoder_elloss = nested_numpify(self.encoder_elloss)
        decoder_elloss = nested_numpify(self.decoder_elloss)
        final_elloss = nested_numpify(self.final_elloss)
        return DetailLoss(loss_dict=(lm_loss, el_loss, encoder_elloss, decoder_elloss, final_elloss))

    def truncate(self, limit):
        lm_loss = self.lm_loss[:limit]
        el_loss = self.el_loss[:limit]
        encoder_elloss = self.encoder_elloss[:limit]
        decoder_elloss = self.decoder_elloss[:limit]
        final_elloss = self.final_elloss[:limit]
        return DetailLoss(loss_dict=(lm_loss, el_loss, encoder_elloss, decoder_elloss, final_elloss))


# a = DetailLoss({
#     "lm_loss": torch.tensor(1.0),
#     "el_loss": torch.tensor(2.0),
#     "encoder_elloss": torch.tensor(3.0),
#     "decoder_elloss": torch.tensor(4.0),
#     "final_elloss": torch.tensor(5.0)
# })
#
# b = DetailLoss({
#     "lm_loss": torch.tensor(1.0),
#     "el_loss": torch.tensor(2.0),
#     "encoder_elloss": torch.tensor(3.0),
#     "decoder_elloss": torch.tensor(4.0),
#     "final_elloss": torch.tensor(5.0)
# })
#
# c = a - 1
# print(type(c))
# print(c)
