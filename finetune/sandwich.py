from __future__ import annotations

from whittle.training_strategies.base_strategy import BaseTrainingStrategy


class SandwichStrategy(BaseTrainingStrategy):
    """
    Sandwich strategy.

    In each step, the sandwich strategy updates the super-network, the smallest, and a set of randomly sampled
    sub-networks.

    refs:
        Universally Slimmable Networks and Improved Training Techniques
        Jiahui Yu, Thomas Huang
        International Conference on Computer Vision 2019
        https://arxiv.org/abs/1903.05134
    """

    def __init__(self, random_samples=2, **kwargs):
        """
        Initialises a `SandwichStrategy`

        Args:
            random_samples: the number of randomly sampled sub-networks to sample and update in each step
            **kwargs: kwargs of `BaseTrainingStrategy`
        """
        super().__init__(**kwargs)
        self.random_samples = random_samples
        self.lora = False

    def __call__(self, model, inputs, outputs, step, **kwargs):
        total_loss = 0
        #lora=False
        # update super-network
        #for n,p in model.named_parameters():
        #    if "lora" in n:
        #        lora = True
        #        break
        if self.lora:
          y_supernet = model(inputs, lm_head_chunk_size=128)
          y_supernet[-1] = y_supernet[-1][..., :-1, :]
        else:
          y_supernet = model(inputs)
          y_supernet = y_supernet[..., :-1, :]
          #y_supernet[-1] = y_supernet[-1][..., :-1, :]
        loss = self.loss_function(y_supernet, outputs[..., 1:])
        # loss = self.loss_function(y_supernet, outputs)
        self.fabric.backward(loss / (step * (2+self.random_samples)))
        total_loss += loss.item()

        # update random sub-networks
        for i in range(self.random_samples):
            sub_network_config = self.sampler.sample()
            model.set_sub_network(**sub_network_config)
            if self.lora:
               y_hat = model(inputs, lm_head_chunk_size=128)
               y_hat[-1] = y_hat[-1][..., :-1, :]
            else:
               y_hat = model(inputs)
               y_hat = y_hat[..., :-1, :]
            #y_hat[-1] = y_hat[-1][..., :-1, :]
            if self.kd_loss is not None:
                loss = self.kd_loss(y_hat, outputs[..., 1:], y_supernet.detach())
            else:
                loss = self.loss_function(y_hat, outputs[..., 1:])
                # loss = self.loss_function(y_hat, outputs)
            self.fabric.backward(loss / (step * (2+self.random_samples)))
            model.reset_super_network()
            total_loss += loss.item()

        # smallest network
        sub_network_config = self.sampler.get_smallest_sub_network()
        model.set_sub_network(**sub_network_config)
        if self.lora:
           y_hat = model(inputs, lm_head_chunk_size=128)
           y_hat[-1] = y_hat[-1][..., :-1, :]
        else:
           y_hat = model(inputs)
           y_hat = y_hat[..., :-1, :]
        #y_hat[-1] = y_hat[-1][..., :-1, :]
        if self.kd_loss is not None:
            loss = self.kd_loss(y_hat, outputs[..., 1:], y_supernet.detach())
        else:
            loss = self.loss_function(y_hat, outputs[..., 1:])
        self.fabric.backward(loss / (step * (2+self.random_samples)))
        model.reset_super_network()
        total_loss += loss.item()

        return total_loss/(2+self.random_samples)