import torch
from typing import Optional
from dataclasses import dataclass

from diffusers.utils.torch_utils import randn_tensor

@dataclass
class BaseOutput:
    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None

def randn_tensor_like(tensor, generator=None):
    return randn_tensor(tensor.shape, generator=generator, device=tensor.device, dtype=tensor.dtype)

class BaseScheduler:
    def __init__(
        self,
        v_prediction: bool = False,
        schedule: Optional[str] = None,
        variance_exploring: bool = False,
    ):
        self.order = None
        self.ancestral = False
        self.v_prediction = v_prediction  # velocity予測かどうか
        self.schedule = schedule  # Karrasのスケジュールを使うかどうか
        self.variance_exploring = variance_exploring  # 分散発散型のsamplerかどうか

        self.make_alpha_beta()

    def make_alpha_beta(self, beta_start=0.00085, beta_end=0.012, num_timesteps=1000):

        self.num_timesteps = num_timesteps

        # beta_1, ... , beta_T
        self.betas = (
            torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps, dtype=torch.float32) ** 2
        )
        # beta_0, ... , beta_T
        # beta_0は通常定義されないが、0とすればx0 =  1 * x0 + 0 * noiseとなって便利
        # ただしdiffusersでは基本的にt in [0, 999]と扱うので注意（１ずれる）
        self.betas = torch.cat([torch.zeros(1, dtype=torch.float32), self.betas])

        # alpha_0, ... , alpha_T
        self.alphas = 1 - self.betas

        # with bar
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.betas_bar = 1 - self.alphas_bar

        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_betas_bar = torch.sqrt(self.betas_bar)

        # sigmas for variance exploring
        # sigma_0, ... , sigma_T
        self.sigmas = self.sqrt_betas_bar / self.sqrt_alphas_bar

    def set_timesteps(self, num_inference_steps, device="cuda"):
        self.num_inference_steps = num_inference_steps
        self.device = device

        if self.schedule == "karras":
            # sigmaの7乗根が線形になるようにスケジューリング
            karras_sigmas = torch.linspace(
                self.sigmas[1] ** (1 / 7),
                self.sigmas[-1] ** (1 / 7),
                num_inference_steps
            ) ** 7

            #　めんどくさいので最近傍探索でやる
            diff =  karras_sigmas.view(-1, 1) - self.sigmas.view(1, -1)
            timesteps = torch.argmin(diff.abs(), dim=1) - 1 # [1, 1000] -> [0, 999]
        elif self.schedule == "linspace":
            # 0から999まで線形にスケジューリング
            timesteps = torch.linspace(0, self.num_timesteps - 1, num_inference_steps, dtype=float).round().clone()
        else:
            step_ratio = self.num_timesteps // num_inference_steps
            timesteps = (torch.arange(0, num_inference_steps) * step_ratio)
            timesteps = timesteps + 1

        self.timesteps = timesteps.flip(0).clone().long().to(device)

    # predict x0 from xt and model_output
    def get_original_sample_and_noise(self, noisy_latents, model_output, t):
        if self.v_prediction:
            pred_original_sample = self.sqrt_alphas_bar[t] * noisy_latents - self.sqrt_betas_bar[t] * model_output
            noise_pred = (pred_original_sample - self.sqrt_alphas_bar[t] * noisy_latents) / self.sqrt_betas_bar[t]
        else: # noise_predicialtion
            pred_original_sample = (noisy_latents - self.sqrt_betas_bar[t] * model_output) / self.sqrt_alphas_bar[t]
            noise_pred = model_output
            
        return pred_original_sample, noise_pred
    
    def get_alpha_respacing(self, t, prev_t):
        return self.alphas_bar[t] / self.alphas_bar[prev_t]
    
    def get_beta_respacing(self, t, prev_t):
        return 1 - self.get_alpha_respacing(t, prev_t)
    
    @property
    def init_noise_sigma(self):
        if self.variance_exploring:
            return self.sigmas[-1] # 初期ノイズを分散発散型に置き換える
        else:
            return 1.0
    
    # ve -> vp
    def scale_model_input(self, sample, t):
        t = t + 1 # [0, 999] -> [1, 1000]
        if self.variance_exploring:
            return sample * self.sqrt_alphas_bar[t] # 潜在変数を分散発散型に置き換える
        else:
            return sample

    # x0 -> xt    
    def add_noise(self, sample, noise, t):
        t = t + 1 # [0, 999] -> [1, 1000]
        if self.variance_exploring:
            return sample + noise * self.sigmas[t]
        else:
            return self.sqrt_alphas_bar[t] * sample +  self.sqrt_betas_bar[t] * noise
    
    # return index of t
    def get_timesteps_id(self, t):
        return (self.timesteps == t).nonzero(as_tuple=True)[0][0]
    
    def get_t_and_prev_t(self, t):
        if t == self.timesteps[-1]:
            return t+1, 0
        else:
            t_id = self.get_timesteps_id(t)
            return t+1, self.timesteps[t_id + 1].item() + 1
        
    def get_ancestral_sigma(self, t, prev_t):
        sigma_from = self.sigmas[t]
        sigma_to = self.sigmas[prev_t]
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

        return sigma_up, sigma_down
