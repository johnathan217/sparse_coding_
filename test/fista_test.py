import unittest
import torch
from autoencoders.fista import FunctionalFista


class TestFistaLoss(unittest.TestCase):

    def setUp(self):
        activation_size = 500
        n_dict_components = 2000
        b_size = 50
        l1_alpha = 0.1
        device = torch.device('cpu')

        self.params, self.buffers = FunctionalFista.init(
            activation_size,
            n_dict_components,
            l1_alpha,
            device=device
        )

        self.buffers["bias_decay"] = 1
        self.batch = torch.randn((b_size, activation_size), device=device)
        self.coefficients = torch.randn((b_size, n_dict_components), device=device)


    def test_loss(self):
        actual_loss, (loss_data, aux_data) = FunctionalFista.loss(self.params, self.buffers, self.batch)
        print(actual_loss, loss_data, aux_data["c"].shape)
        self.assertIsInstance(actual_loss, torch.Tensor)


    def test_fista_loss(self):
        actual_loss, (loss_data, aux_data) = FunctionalFista.fista_loss(self.params, self.buffers, self.batch, self.coefficients)
        print(actual_loss, loss_data, aux_data["c_fista"].shape)
        self.assertIsInstance(actual_loss, torch.Tensor)



if __name__ == '__main__':
    unittest.main()
