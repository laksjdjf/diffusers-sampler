from base import BaseOutput, BaseScheduler, randn_tensor_like

class SimpleDDPMScheduler(BaseScheduler):
    def __init__(self, v_prediction=False, schedule=None):
        super().__init__(v_prediction, schedule, variance_exploring=False)
        self.order = 1

    def step(self, model_output, timestep, sample, return_dict=False, generator=None, **kwargs):
        t, prev_t = self.get_t_and_prev_t(timestep)

        current_alpha_t = self.get_alpha_respacing(t, prev_t)
        current_beta_t = self.get_beta_respacing(t, prev_t)

        # 第一項
        pred_original_sample, _ = self.get_original_sample_and_noise(sample, model_output, t)
        mean_x0 = self.sqrt_alphas_bar[prev_t] * current_beta_t / self.betas_bar[t] * pred_original_sample
        
        # 第二項
        mean_xt = current_alpha_t.sqrt() * self.betas_bar[prev_t] / self.betas_bar[t] * sample

        # 第三項
        noise = randn_tensor_like(model_output, generator=generator)
        variance = self.betas_bar[prev_t] / self.betas_bar[t] * current_beta_t

        # x_s
        prev_sample = mean_x0 + mean_xt + variance.sqrt() * noise
        
        if return_dict:
            return BaseOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
        else:
            return (prev_sample,)
        
class SimpleDDIMScheduler(BaseScheduler):
    def __init__(self, v_prediction=False, schedule=None):
        super().__init__(v_prediction, schedule, variance_exploring=False)
        self.order = 1

    # fix eta = 0
    def step(self, model_output, timestep, sample, return_dict=False, generator=None, **kwargs):
        t, prev_t = self.get_t_and_prev_t(timestep)

        pred_original_sample, noise_pred = self.get_original_sample_and_noise(sample, model_output, t)

        mean = self.sqrt_alphas_bar[prev_t] * pred_original_sample

        prev_sample = mean + self.betas_bar[prev_t].sqrt() * noise_pred
        
        if return_dict:
            return BaseOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
        else:
            return (prev_sample,)