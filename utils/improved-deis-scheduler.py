# Copyright 2022 Stanford University Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, randn_tensor
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin


@dataclass
class DEISSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample (x_{0}) based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.

    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)


class DEISInverseScheduler(SchedulerMixin, ConfigMixin):
    """
    Denoising Exponential Integrator Sampler (DEIS) is a scheduler that extends the denoising procedure
    with high-order exponential integrators for diffusion ODEs.

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
        clip_sample (`bool`, default `True`):
            option to clip predicted sample between -1 and 1 for numerical stability.
        set_alpha_to_one (`bool`, default `True`):
            each diffusion step uses the value of alphas product at that step and at the previous one. For the final
            step there is no previous alpha. When this option is `True` the previous alpha product is fixed to `1`,
            otherwise it uses the value of alpha at step 0.
        steps_offset (`int`, default `0`):
            an offset added to the inference steps. You can use a combination of `offset=1` and
            `set_alpha_to_one=False`, to make the last step use step 0 for the previous alpha product, as done in
            stable diffusion.
        prediction_type (`str`, default `epsilon`, optional):
            prediction type of the scheduler function, one of `epsilon` (predicting the noise of the diffusion
            process), `sample` (directly predicting the noisy sample`) or `v_prediction` (see section 2.4
            https://imagen.research.google/video/paper.pdf)
        order (`int`, default `2`):
            the integration order of the DEIS sampler. Higher orders provide better approximations but require
            more computation. Typical values are 1, 2, or 3.
        eps (`float`, default `1e-6`):
            epsilon value to avoid division by zero.
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 2  # Default higher order for DEIS

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        order: int = 2,
        eps: float = 1e-6,
    ):
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # At every step in deis, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # whether we use the final alpha of the "non-previous" one.
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))
        
        # Store previous model outputs for higher-order DEIS, with timesteps
        self.model_outputs_dict = {}
        self.integration_order = order
        self.eps = eps
        
    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`): input sample
            timestep (`int`, optional): current timestep

        Returns:
            `torch.FloatTensor`: scaled input sample
        """
        return sample

    def _get_variance(self, timestep, prev_timestep):
        """
        Compute the variance for the previous step given the current timestep.
        
        Args:
            timestep: Current timestep index
            prev_timestep: Previous timestep index
            
        Returns:
            Variance value
        """
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, optional):
                the device to which the timesteps should be moved to.
        """
        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.config.num_train_timesteps} timesteps."
            )

        self.num_inference_steps = num_inference_steps
        step_ratio = self.config.num_train_timesteps // self.num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)
        self.timesteps += self.config.steps_offset
        
        # Reset model outputs dictionary for the new timestep sequence
        self.model_outputs_dict = {}

    def _predict_x0_from_noise(self, sample, timestep, noise):
        """
        Predict x0 (original sample) from the noise.
        Handles different prediction types: epsilon, sample, v_prediction
        
        Args:
            sample: Current noisy sample x_t
            timestep: Current timestep
            noise: Predicted noise epsilon_t
            
        Returns:
            Predicted x0
        """
        alpha_prod_t = self.alphas_cumprod[timestep-1]
        eps = self.eps
        
        if self.config.prediction_type == "epsilon":
            # Original formula: (x - sqrt(1-alpha_t) * noise) / sqrt(alpha_t)
            pred_x0 = (sample - (1 - alpha_prod_t + eps).sqrt() * noise) / (alpha_prod_t + eps).sqrt()
        elif self.config.prediction_type == "sample":
            # If prediction_type is 'sample', the model directly outputs x0
            pred_x0 = noise
        elif self.config.prediction_type == "v_prediction":
            # v-prediction: https://imagen.research.google/video/paper.pdf equation 33
            pred_x0 = (alpha_prod_t + eps).sqrt() * sample - (1 - alpha_prod_t + eps).sqrt() * noise
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )
            
        if self.config.clip_sample:
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
        return pred_x0

    def _compute_exponential_coefficients(self, timestep, prev_timestep):
        """
        Compute the coefficients for the exponential integrator based on the order.
        Higher order coefficients provide more accurate approximations.
        
        Args:
            timestep: Current timestep index
            prev_timestep: Previous timestep index
            
        Returns:
            List of coefficients for the exponential integrator
        """
        a_t = self.alphas_cumprod[timestep-1]
        a_prev = self.alphas_cumprod[prev_timestep-1] if prev_timestep >= 0 else self.final_alpha_cumprod
        eps = self.eps
        
        # Improved coefficient calculation for better numerical stability
        log_alpha_ratio = torch.log((a_prev + eps) / (a_t + eps)) / 2.0
        
        # First order coefficient (Same as DDIM but more numerically stable)
        coef_1 = torch.exp(log_alpha_ratio)
        
        if self.integration_order == 1:
            return [coef_1]
        
        # Calculate normalized time step for better integration
        t = timestep / self.config.num_train_timesteps
        prev_t = prev_timestep / self.config.num_train_timesteps
        dt = prev_t - t  # Note: in reverse process, prev_t > t
        
        # Additional coefficients for higher-order integration with improved numerical stability
        if self.integration_order >= 2:
            # Second-order coefficient with exp stabilization
            coef_2 = torch.exp(-dt) * torch.sqrt((1 - a_prev + eps) / (1 - a_t + eps))
            
            if self.integration_order == 2:
                return [coef_1, coef_2]
            
            # Third-order coefficient with better numerical stability
            coef_3 = torch.exp(-2 * dt) * ((1 - a_prev + eps) / (1 - a_t + eps)) ** (3/4)
            
            return [coef_1, coef_2, coef_3]
        
        return [coef_1]  # Default fallback

    def get_model_output_index_pairs(self, timestep):
        """
        Get indices of previous model outputs needed for the current timestep
        based on integration order.
        
        Args:
            timestep: Current timestep
            
        Returns:
            List of timesteps to use for higher-order integration
        """
        indices = []
        current_idx = np.where(self.timesteps.cpu().numpy() == timestep)[0][0]
        
        # We need up to 'order' previous outputs
        for i in range(self.integration_order):
            if current_idx + i + 1 < len(self.timesteps):
                indices.append(self.timesteps[current_idx + i + 1].item())
            else:
                break
                
        return indices
    
    def store_model_output(self, model_output, timestep):
        """
        Store model output for the given timestep.
        
        Args:
            model_output: Output tensor from the model
            timestep: Current timestep
        """
        self.model_outputs_dict[timestep] = model_output.detach().clone()
        
        # Keep dictionary size manageable by removing old entries
        if len(self.model_outputs_dict) > self.integration_order * 2:
            # Remove oldest entries
            oldest_keys = sorted(list(self.model_outputs_dict.keys()))[:-(self.integration_order * 2)]
            for k in oldest_keys:
                del self.model_outputs_dict[k]

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        reverse: bool = False
    ) -> Union[DEISSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE.
        This function propagates the diffusion process from the learned model outputs.
        
        Args:
            model_output: The model output at current timestep
            timestep: Current discrete timestep in the diffusion chain
            sample: Current instance of sample being created
            eta: Corresponds to η in DDIM paper, controls stochasticity
            use_clipped_model_output: If True, clip the model output
            generator: Random number generator for stochastic sampling
            variance_noise: Optional noise to add for stochastic sampling
            return_dict: If True, return a DEISSchedulerOutput instead of tuple
            reverse: If True, reverse the diffusion process
            
        Returns:
            prev_sample: Sample from the previous timestep
            pred_original_sample: Predicted original sample (x_0)
        """
        # Store current model output for future higher-order steps
        self.store_model_output(model_output, timestep)
        
        e_t = model_output  # Current noise prediction
        if use_clipped_model_output and self.config.clip_sample:
            e_t = torch.clamp(model_output, -1, 1)
            
        x = sample  # Current noisy sample
        
        # Find previous timestep
        step_idx = (self.timesteps == timestep).nonzero().item()
        prev_timestep = self.timesteps[step_idx + 1].item() if step_idx < len(self.timesteps) - 1 else timestep
        
        # Get alphas for current and previous timesteps (ensure valid indices)
        timestep_idx = min(timestep, len(self.alphas_cumprod) - 1)
        prev_timestep_idx = min(prev_timestep, len(self.alphas_cumprod) - 1)
        
        a_t = self.alphas_cumprod[timestep_idx-1]
        a_prev = self.alphas_cumprod[prev_timestep_idx-1] if prev_timestep_idx > 0 else self.final_alpha_cumprod
        
        # Predict x0 (clean image) based on prediction type
        pred_x0 = self._predict_x0_from_noise(x, timestep_idx, e_t)
        
        # Basic direction to x_t (first order - same as DDIM)
        dir_xt = (1. - a_prev + self.eps).sqrt() * e_t
        
        # Get previous model outputs needed for higher-order integration
        prev_indices = self.get_model_output_index_pairs(timestep)
        
        # Enhance with higher-order integration if previous outputs are available
        if self.integration_order > 1 and prev_indices:
            # Compute coefficients for exponential integrator
            coefs = self._compute_exponential_coefficients(timestep_idx, prev_timestep_idx)
            
            # Apply available higher-order corrections
            for i, prev_t in enumerate(prev_indices):
                if i >= len(coefs) - 1 or prev_t not in self.model_outputs_dict:
                    continue
                    
                # Get previous model output and apply correction
                if i == 0 and len(prev_indices) > 0:  # Second-order correction
                    prev_e_t = self.model_outputs_dict[prev_t]
                    second_order_term = coefs[1] * (e_t - prev_e_t)
                    dir_xt = dir_xt + second_order_term
                elif i == 1 and len(prev_indices) > 1:  # Third-order correction
                    prev_e_t_1 = self.model_outputs_dict[prev_indices[0]]
                    prev_e_t_2 = self.model_outputs_dict[prev_t]
                    third_order_term = coefs[2] * (
                        e_t - 2 * prev_e_t_1 + prev_e_t_2
                    )
                    dir_xt = dir_xt + third_order_term
        
        # Add variance (if eta > 0)
        variance = 0
        if eta > 0:
            variance = self._get_variance(timestep_idx, prev_timestep_idx)
            variance = variance * eta
            
            if variance_noise is None:
                variance_noise = torch.randn(
                    model_output.shape, generator=generator, device=model_output.device
                )
                
            # Add stochasticity from Langevin diffusion
            variance = variance_noise * variance.sqrt()
        
        # Compute the next sample
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + variance
        
        if not return_dict:
            return (x_prev,)
            
        return DEISSchedulerOutput(prev_sample=x_prev, pred_original_sample=pred_x0)

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        Add noise to the original samples at the specified timesteps.
        
        Args:
            original_samples: Clean images (x_0)
            noise: Noise to add (epsilon)
            timesteps: Timesteps at which to add noise
            
        Returns:
            Noisy samples (x_t)
        """
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        # Clip timesteps to valid range
        timesteps = torch.clamp(timesteps, 0, len(self.alphas_cumprod) - 1)

        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def get_velocity(
        self, sample: torch.FloatTensor, noise: torch.FloatTensor, timesteps: torch.IntTensor
    ) -> torch.FloatTensor:
        """
        Compute the velocity (drift) of the ODE at the current sample and timestep.
        
        Args:
            sample: Current sample (x_t)
            noise: Predicted noise (epsilon)
            timesteps: Current timesteps
            
        Returns:
            Velocity of the ODE
        """
        # Make sure alphas_cumprod and timestep have compatible device and dtype
        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        timesteps = timesteps.to(sample.device)
        
        # Clip timesteps to valid range
        timesteps = torch.clamp(timesteps, 0, len(self.alphas_cumprod) - 1)

        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # The velocity formula for the probability flow ODE
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    def __len__(self):
        return self.config.num_train_timesteps
