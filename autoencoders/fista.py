from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from torchtyping import TensorType
from ..autoencoders.learned_dict import (LearnedDict, ReverseSAE, TiedSAE,
                                       UntiedSAE)
from ..autoencoders.ensemble import DictSignature
from ..autoencoders.learned_dict import LearnedDict
# function that writes a thing (its cool):

_n_samples, _activation_size = None, None


class Fista(DictSignature):
    @staticmethod
    def init(
            activation_size,
            n_dict_components,
            l1_alpha,
            device=None,
            dtype=None,

            bias_decay=0.0,
            translation=None,
            rotation=None,
            scaling=None,
    ):
        params = {}
        buffers = {}

        if rotation is None:
            rotation = torch.eye(activation_size, device=device, dtype=dtype)

        if translation is None:
            translation = torch.zeros(activation_size, device=device, dtype=dtype)

        if scaling is None:
            scaling = torch.ones(activation_size, device=device, dtype=dtype)

        buffers["center_rot"] = rotation
        buffers["center_trans"] = translation
        buffers["center_scale"] = scaling

        params["encoder"] = torch.empty((n_dict_components, activation_size), device=device, dtype=dtype)
        nn.init.xavier_uniform_(params["encoder"])

        params["encoder_bias"] = torch.empty((n_dict_components,), device=device, dtype=dtype)
        nn.init.zeros_(params["encoder_bias"])

        buffers["l1_alpha"] = torch.tensor(l1_alpha, device=device, dtype=dtype)

        buffers["device"] = device

        return params, buffers

    @staticmethod
    def to_learned_dict(params, buffers):
        return TiedSAE(params["encoder"], params["encoder_bias"],
                       centering=(buffers["center_trans"], buffers["center_rot"], buffers["center_scale"]),
                       norm_encoder=True)

    @staticmethod
    def center(buffers, batch):
        return torch.einsum("cu,bu->bc", buffers["center_rot"], batch - buffers["center_trans"][None, :]) * buffers[
                                                                                                                "center_scale"][
                                                                                                            None, :]

    @staticmethod
    def uncenter(buffers, batch):
        return torch.einsum("cu,bc->bu", buffers["center_rot"], batch / buffers["center_scale"][None, :]) + buffers[
                                                                                                                "center_trans"][
                                                                                                            None, :]

    @staticmethod
    def loss(params, buffers, batch):
        decoder_norms = torch.norm(params["encoder"], 2, dim=-1)
        learned_dict = params["encoder"] / torch.clamp(decoder_norms, 1e-8)[:, None]
        batch_centered = Fista.center(buffers, batch)

        c = torch.einsum("nd,bd->bn", learned_dict, batch_centered)
        c = c + params["encoder_bias"]
        c = torch.clamp(c, min=0.0)

        x_hat_centered = torch.einsum("nd,bn->bd", learned_dict, c)
        x_hat = Fista.uncenter(buffers, x_hat_centered)

        l_reconstruction = (x_hat_centered - batch_centered).pow(2).mean()
        l_l1 = buffers["l1_alpha"] * torch.norm(c, 1, dim=-1).mean()
        l_bias_decay = buffers["bias_decay"] * torch.norm(params["encoder_bias"], 2)

        #Fista component
        c_fista, res = Fista.fista(batch_centered, learned_dict, buffers["l1_alpha"], c, 50, buffers["device"], eta=None)
        l_reconstruction_fista = res.pow(2).mean()

        overall_reconstruction = l_reconstruction + l_reconstruction_fista

        loss_data = {
            "loss": overall_reconstruction + l_l1 + l_bias_decay,
            "autoencoder_reconstruction": l_reconstruction,
            "fista_reconstruction": l_reconstruction_fista,
            "l_l1": l_l1,
        }

        aux_data = {
            "c": c,
            "c_fista": c_fista,
        }

        return overall_reconstruction + l_l1 + l_bias_decay, (loss_data, aux_data)


    @staticmethod
    def fista_loss(params, buffers, batch, c):
        decoder_norms = torch.norm(params["encoder"], 2, dim=-1)
        learned_dict = params["encoder"] / torch.clamp(decoder_norms, 1e-8)[:, None]

        batch_centered = Fista.center(buffers, batch)
        #batch_centered.shape = (batch_size, activation_size)

        c_fista, res = Fista.fista(batch_centered, learned_dict, buffers["l1_alpha"], c, 50, buffers["device"], eta=None)
        l_reconstruction = res.pow(2).mean()

        return l_reconstruction, ({"loss": l_reconstruction}, {"c_fista": c_fista})

    @staticmethod
    def fista(batch, learned_dict, l1_coef, coefficients, num_iter, device, eta=None):
        # shapes: batch = (b_size, h_dim), learned_dict = (dict_size, h_dim), coefficients = (b_size, dict_size)

        # batch_size = batch.size(0)
        # M = learned_dict.size(0)
        if eta is None:
            eigenvalues = torch.linalg.eigvalsh(learned_dict @ learned_dict.T)
            eta = 1.0 / eigenvalues.max().item()
            eta = torch.tensor(eta, dtype=torch.float32).to(device)

        tk_n = 1.
        tk = 1.
        # Res = torch.FloatTensor(I.size()).fill_(0).to(cfg.device)

        # ahat = torch.FloatTensor(M,batch_size).fill_(0).to(cfg.device)
        # ahat_y = torch.FloatTensor(M,batch_size).fill_(0).to(cfg.device)
        ahat = coefficients
        ahat_y = coefficients

        for t in range(num_iter):
            tk = tk_n
            tk_n = (1 + np.sqrt(1 + 4 * tk ** 2)) / 2
            ahat_pre = ahat
            Res = batch - torch.mm(ahat_y, learned_dict)
            ahat_y = ahat_y.add(eta * torch.mm(Res, learned_dict.t()))
            ahat = ahat_y.sub(eta * l1_coef).clamp(min=0.)
            ahat_y = ahat.add(ahat.sub(ahat_pre).mul((tk - 1) / (tk_n)))
        Res = batch - torch.mm(ahat, learned_dict)
        return ahat, Res


