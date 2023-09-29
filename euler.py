import torch
from base import BaseOutput, BaseScheduler, randn_tensor_like


class SimpleEulerDiscreteScheduler(BaseScheduler):
    def __init__(self, v_prediction=False, schedule="linspace", ancestral=True):
        super().__init__(v_prediction, schedule, variance_exploring=True)
        self.order = 1
        self.ancestral = ancestral

    def step(self, model_output, timestep, sample, return_dict=False, generator=None, **kwargs):
        t, prev_t = self.get_t_and_prev_t(timestep)

        if self.ancestral:
            sigma_up, sigma_down = self.get_ancestral_sigma(t, prev_t)
            dsigma = sigma_down - self.sigmas[t]
        else:
            dsigma = self.sigmas[prev_t] - self.sigmas[t]

        pred_original_sample, noise_pred = self.get_original_sample_and_noise(sample, model_output, t)
        prev_sample = sample + noise_pred * dsigma

        if self.ancestral:
            ancestral_noise = randn_tensor_like(model_output, generator=generator)
            prev_sample = prev_sample + ancestral_noise * sigma_up

        if return_dict:
            return BaseOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
        else:
            return (prev_sample,)


class SimpleHeunDiscreteScheduler(BaseScheduler):
    def __init__(self, v_prediction=False, schedule="linspace"):
        super().__init__(v_prediction, schedule, variance_exploring=True)
        self.order = 2
        self.first_sample = None

    # heun法は (t_0, t_1), (t_1, t_2), (t_2, t_3), ... という順番で計算する
    def set_timesteps(self, num_inference_steps, device="cuda"):
        super().set_timesteps(num_inference_steps, device)
        self.timesteps = torch.cat([self.timesteps[:1], self.timesteps[1:].repeat_interleave(2)])

    def step(self, model_output, timestep, sample, return_dict=False, generator=None, **kwargs):
        t, prev_t = self.get_t_and_prev_t(timestep)
        pred_original_sample, noise_pred = self.get_original_sample_and_noise(sample, model_output, t)
        # first_stage, predict prev_sample by euler
        if self.first_sample is None:
            dsigma = self.sigmas[prev_t] - self.sigmas[t]

            # 次ステップのために保存
            self.first_sample = sample
            self.first_noise_pred = noise_pred
            self.dsigma = dsigma

            prev_sample = sample + noise_pred * dsigma  # euler法による予測
            
        # second_stage, predict prev_sample by heun
        else:
            dsigma = self.dsigma
            noise_pred_heun = (self.first_noise_pred + noise_pred) / 2

            prev_sample = self.first_sample + noise_pred_heun * dsigma  # heun法による予測
            
            # reset
            self.first_sample = None  
            self.first_noise_pred = None
            self.dsigma = None

        if return_dict:
            return BaseOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
        else:
            return (prev_sample,)
