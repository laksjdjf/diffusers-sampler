import torch
from base import BaseOutput, BaseScheduler, randn_tensor_like

class SimpleDPMScheduler(BaseScheduler):
    def __init__(
            self,
            v_prediction=False,
            schedule="linspace", 
            mode="dpm-solver",
            order=1,
            multi_step=False,
        ):
        super().__init__(v_prediction, schedule, variance_exploring=False)
        self.order = order
        self.mode = mode

    def step(self, model_output, timestep, sample, return_dict=False, generator=None, **kwargs):
        t, prev_t = self.get_t_and_prev_t(timestep)
        pred_original_sample, noise_pred = self.get_original_sample_and_noise(sample, model_output, t)

        prev_sample = self.one_step_update(t, prev_t, pred_original_sample, noise_pred, sample)

        if return_dict:
            return BaseOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
        else:
            return (prev_sample,)
        
    def one_step_update(self, t, prev_t, pred_original_sample, noise_pred, sample):
        sigma_t = self.sigmas[t]
        sigma_prev_t = self.sigmas[prev_t] if prev_t != 0 else 1e-5 # 0除算を防ぐため^^?

        sqrt_alpha_t = self.get_alpha_respacing(t, prev_t).sqrt()
        sqrt_beta_t = self.sqrt_betas_bar[t] / self.sqrt_betas_bar[prev_t]

        if self.mode == "dpm-solver":
            prev_sample = sample / sqrt_alpha_t - self.sqrt_betas_bar[prev_t] * (sigma_t/sigma_prev_t - 1) * noise_pred
        elif self.mode == "dpm-solver++":
            prev_sample = sample / sqrt_beta_t - self.sqrt_alphas_bar[prev_t] * (sigma_prev_t/sigma_t - 1) * pred_original_sample
        else:
            raise NotImplementedError

        return prev_sample