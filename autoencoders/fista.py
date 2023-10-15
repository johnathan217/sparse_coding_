from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from torchtyping import TensorType
from autoencoders.learned_dict import (LearnedDict, ReverseSAE, TiedSAE,
                                       UntiedSAE)
from autoencoders.ensemble import DictSignature
from autoencoders.learned_dict import LearnedDict


_n_samples, _activation_size = None, None


class FunctionalFista(DictSignature):
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
        buffers["bias_decay"] = torch.tensor(bias_decay, device=device, dtype=dtype)

        buffers["hessian_diag"] = torch.zeros(n_dict_components).to(device)

        return params, buffers

    @staticmethod
    def to_learned_dict(params, buffers):
        return Fista(params["encoder"], params["encoder_bias"], centering=(buffers["center_trans"], buffers["center_rot"], buffers["center_scale"]), norm_encoder=True)


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

        batch_centered = FunctionalFista.center(buffers, batch)

        c = torch.einsum("nd,bd->bn", learned_dict, batch_centered)
        c = c + params["encoder_bias"]
        c = torch.clamp(c, min=0.0)

        x_hat_centered = torch.einsum("nd,bn->bd", learned_dict, c)
        x_hat = FunctionalFista.uncenter(buffers, x_hat_centered)
        l_reconstruction = (x_hat_centered - batch_centered).pow(2).mean()
        l_l1 = buffers["l1_alpha"] * torch.norm(c, 1, dim=-1).mean()
        l_bias_decay = buffers["bias_decay"] * torch.norm(params["encoder_bias"], 2)

        loss_data = {
            "loss": l_reconstruction + l_l1 + l_bias_decay,
            "l_reconstruction": l_reconstruction,
            "l_l1": l_l1,
        }

        aux_data = {
            "c": c,
        }

        return l_reconstruction + l_l1 + l_bias_decay, (loss_data, aux_data)


    @staticmethod
    def loss2(params, buffers, batch):
        decoder_norms = torch.norm(params["encoder"], 2, dim=-1)
        learned_dict = params["encoder"] / torch.clamp(decoder_norms, 1e-8)[:, None]
        batch_centered = FunctionalFista.center(buffers, batch)

        c = torch.einsum("nd,bd->bn", learned_dict, batch_centered)
        c = c + params["encoder_bias"]
        c = torch.clamp(c, min=0.0)

        x_hat_centered = torch.einsum("nd,bn->bd", learned_dict, c)
        x_hat = FunctionalFista.uncenter(buffers, x_hat_centered)

        l_reconstruction = (x_hat_centered - batch_centered).pow(2).mean()
        l_l1 = buffers["l1_alpha"] * torch.norm(c, 1, dim=-1).mean()
        l_bias_decay = buffers["bias_decay"] * torch.norm(params["encoder_bias"], 2)

        # with torch.no_grad():
        FunctionalFista.dictionary_update(params, buffers, batch_centered, c, learned_dict)
        # coeffs_fista, res = FunctionalFista.fista(batch_centered, learned_dict, buffers["l1_alpha"], c, 50, params["encoder"].device,
        #                                           eta=None)

        loss_data = {
            "loss": l_reconstruction + l_l1 + l_bias_decay,
            "l_reconstruction": l_reconstruction,
            "l_l1": l_l1,
        }
        aux_data = {
            "c": c,
        }
        return l_reconstruction + l_l1 + l_bias_decay, (loss_data, aux_data)

    @staticmethod
    def dictionary_update(params, buffers, batch_centered, coeffs, learned_dict):
        device = params["encoder"].device
        coeffs_fista, res = FunctionalFista.fista(batch_centered, learned_dict, buffers["l1_alpha"], coeffs, 1, device, eta=None)
        ACT_HISTORY_LEN = 300
        buffers["hessian_diag"] = buffers["hessian_diag"].mul((ACT_HISTORY_LEN - 1.0) / ACT_HISTORY_LEN ) + torch.pow(coeffs_fista, 2).mean(0) / ACT_HISTORY_LEN

        return FunctionalFista.quadraticBasisUpdate(learned_dict, res, coeffs_fista, 0.001, buffers["hessian_diag"])




        # #Fista component of loss function
        # c_fista, res = FunctionalFista.fista(batch_centered, learned_dict, buffers["l1_alpha"], c, 50, device, eta=None)
        # l_reconstruction_fista = res.pow(2).mean()
        #
        # overall_reconstruction = l_reconstruction + l_reconstruction_fista
        #
        # loss_data = {
        #     "loss": overall_reconstruction + l_l1 + l_bias_decay,
        #     "autoencoder_reconstruction": l_reconstruction,
        #     "fista_reconstruction": l_reconstruction_fista,
        #     "l_l1": l_l1,
        # }
        #
        # aux_data = {
        #     "c": c,
        #     "c_fista": c_fista,
        # }
        #
        # return overall_reconstruction + l_l1 + l_bias_decay, (loss_data, aux_data)

    @staticmethod
    def fista_loss(params, buffers, batch, c):
        decoder_norms = torch.norm(params["encoder"], 2, dim=-1)
        learned_dict = params["encoder"] / torch.clamp(decoder_norms, 1e-8)[:, None]
        batch_centered = Fista.center(buffers, batch)
        device = params["encoder"].device
        #batch_centered.shape = (batch_size, activation_size)

        c_fista, res = FunctionalFista.fista(batch_centered, learned_dict, buffers["l1_alpha"], c, 50, device, eta=None)
        l_reconstruction = res.pow(2).mean()

        return l_reconstruction, ({"loss": l_reconstruction}, {"c_fista": c_fista})

    @staticmethod
    def fista(batch, learned_dict, l1_coef, coefficients, num_iter, device, eta=None):
        # shapes: batch = (b_size, h_dim), learned_dict = (dict_size, h_dim), coefficients = (b_size, dict_size)
        dtype = batch.dtype
        # batch_size = batch.size(0)
        # M = learned_dict.size(0)
        if eta is None:
            eigenvalues = torch.linalg.eigvalsh(learned_dict @ learned_dict.T)
            eta = 1.0 / eigenvalues.max().to(dtype)
        # eta = torch.tensor(eta, dtype=torch.float32).to(device)

        tk_n = 1.
        tk = 1.

        ahat = coefficients
        # print("ahat dtype 1:", ahat.dtype)
        ahat_y = coefficients

        for t in range(num_iter):
            tk = tk_n
            tk_n = (1 + np.sqrt(1 + 4 * tk ** 2)) / 2
            ahat_pre = ahat
            Res = batch - torch.mm(ahat_y, learned_dict)
            ahat_y = ahat_y.add(eta * torch.mm(Res, learned_dict.t()))
            ahat = ahat_y.sub(eta * l1_coef).clamp(min=0.)
            ahat_y = ahat.add(ahat.sub(ahat_pre).mul((tk - 1) / (tk_n)))
            # should probably figure out what changes ahat's dtype instead of just changing it here but whatever
            ahat = ahat.to(dtype)
        Res = batch - torch.mm(ahat, learned_dict)
        # Res is the residual, the difference between reconstructed and original data. ahat is the coefficients.
        return ahat, Res


    @staticmethod
    def quadraticBasisUpdate(learned_dict, Res, ahat, lowestActivation, HessianDiag, stepSize = 0.001, Noneg = False):
        dBasis = stepSize * torch.mm(Res.t(), ahat) / ahat.size(0)
        dBasis = dBasis / (HessianDiag + lowestActivation)
        learned_dict = learned_dict + (dBasis.t())
        if Noneg:
            learned_dict = learned_dict.clamp(min=0.)
        learned_dict = learned_dict / (learned_dict.norm(2, 0))
        return learned_dict

class Fista(LearnedDict):
    def __init__(self, encoder, encoder_bias, centering=(None, None, None), norm_encoder=False):
        self.encoder = encoder
        self.encoder_bias = encoder_bias
        self.norm_encoder = norm_encoder
        self.n_feats, self.activation_size = self.encoder.shape

        center_trans, center_rot, center_scale = centering

        if center_trans is None:
            center_trans = torch.zeros(self.activation_size)

        if center_rot is None:
            center_rot = torch.eye(self.activation_size)

        if center_scale is None:
            center_scale = torch.ones(self.activation_size)

        self.center_trans = center_trans
        self.center_rot = center_rot
        self.center_scale = center_scale

    def initialize_missing(self):
        if not hasattr(self, "center_trans"):
            self.center_trans = torch.zeros(self.activation_size, device=self.encoder.device)

        if not hasattr(self, "center_rot"):
            self.center_rot = torch.eye(self.activation_size, device=self.encoder.device)

        if not hasattr(self, "center_scale"):
            self.center_scale = torch.ones(self.activation_size, device=self.encoder.device)

    def center(self, batch):
        return torch.einsum("cu,bu->bc", self.center_rot, batch - self.center_trans[None, :]) * self.center_scale[None,
                                                                                                :]

    def uncenter(self, batch):
        return torch.einsum("cu,bc->bu", self.center_rot, batch / self.center_scale[None, :]) + self.center_trans[None,
                                                                                                :]

    def get_learned_dict(self):
        norms = torch.norm(self.encoder, 2, dim=-1)
        return self.encoder / torch.clamp(norms, 1e-8)[:, None]

    def to_device(self, device):
        self.initialize_missing()

        self.encoder = self.encoder.to(device)
        self.encoder_bias = self.encoder_bias.to(device)

        self.center_trans = self.center_trans.to(device)
        self.center_rot = self.center_rot.to(device)
        self.center_scale = self.center_scale.to(device)

    def encode(self, batch):
        if self.norm_encoder:
            norms = torch.norm(self.encoder, 2, dim=-1)
            encoder = self.encoder / torch.clamp(norms, 1e-8)[:, None]
        else:
            encoder = self.encoder

        c = torch.einsum("nd,bd->bn", encoder, batch)
        c = c + self.encoder_bias
        c = torch.clamp(c, min=0.0)
        return c

    def fista(self, batch, coefficients, device, l1_coef, num_iter, eta=None):
        # shapes: batch = (b_size, h_dim), learned_dict = (dict_size, h_dim), coefficients = (b_size, dict_size)
        learned_dict = self.get_learned_dict()
        # batch_size = batch.size(0)
        # M = learned_dict.size(0)
        if eta is None:
            eigenvalues = torch.linalg.eigvalsh(learned_dict @ learned_dict.T)
            eta = 1.0 / eigenvalues.max()

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