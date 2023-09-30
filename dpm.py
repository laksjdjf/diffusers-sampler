import torch
from base import BaseOutput, BaseScheduler

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
        self.order = order if not multi_step else 1 # multi stepはself.order=1とする(pipelineでも使われるため)
        self.solver_order = order # multi stepを含めたorder
        self.mode = mode
        self.multi_step = multi_step
        self.first_sample = None
        
        self.pre_pred_original_sample = None # for 2M

        self.lambdas = - self.sigmas.log()
        assert not (self.mode == "dpm-solver" and self.multi_step), "dpm-solver does not support multi_step"

    def set_timesteps(self, num_inference_steps, device="cuda"):
        super().set_timesteps(num_inference_steps, device)
        if self.order == 2: # 2S
            # timestepsにlambdaを基準にした中点を追加する。
            
            # 一旦適当に拡張
            self.timesteps = torch.cat([self.timesteps[:1], self.timesteps[1:].repeat_interleave(self.order)])
            for i in range(0, self.timesteps.shape[0]-1, 2):
                t = self.timesteps[i]
                prev_t = self.timesteps[i+1]
        
                lambda_t = self.lambdas[t]
                lambda_prev_t = self.lambdas[prev_t]

                lambda_midpoint = (lambda_t + lambda_prev_t) / 2

                if prev_t == 0:
                    t_midpoint = t // 2 # lambda_0は無限に発散するので回避策
                else:
                    t_midpoint = (self.lambdas - lambda_midpoint).abs().argmin()
                self.timesteps[i+1] = t_midpoint

    def step(self, model_output, timestep, sample, return_dict=False, generator=None, **kwargs):
        t, prev_t = self.get_t_and_prev_t(timestep)
        pred_original_sample, noise_pred = self.get_original_sample_and_noise(sample, model_output, t)

        if self.solver_order == 1:
            prev_sample = self.one_step_update(t, prev_t, pred_original_sample, noise_pred, sample)
        # 2S(中点法)
        elif self.solver_order == 2 and not self.multi_step:
            # first_stage, predict prev_sample by dpm
            if self.first_sample is None:
                prev_sample = self.one_step_update(t, prev_t, pred_original_sample, noise_pred, sample)
                
                if prev_t > 0:
                    self.first_sample = sample
                    self.first_t = t
            # second_stage, predict prev_sample by dpm-midpoint
            else:
                prev_sample = self.one_step_update(self.first_t, prev_t, pred_original_sample, noise_pred, self.first_sample)
                self.first_sample = None
        # 2M(Adams-Bashforth法)
        elif self.solver_order == 2 and self.multi_step:
            h = self.lambdas[prev_t] - self.lambdas[t]
            if self.pre_pred_original_sample is not None and prev_t > 0: # sの発散を防ぐためprev_t = 0のときはなし
                r = self.h / h
                s = 1 / (2 * r)
                pred_original_sample_input = (1 + s) * pred_original_sample - s * self.pre_pred_original_sample
            else:
                pred_original_sample_input = pred_original_sample

            prev_sample = self.one_step_update(t, prev_t, pred_original_sample_input, noise_pred, sample)
            self.pre_pred_original_sample = pred_original_sample if prev_t > 0 else None
            self.h = h
        else:
            raise NotImplementedError
        
        if return_dict:
            return BaseOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
        else:
            return (prev_sample,)
        
    def one_step_update(self, t, prev_t, pred_original_sample, noise_pred, sample):
        sigma_t = self.sigmas[t]
        sigma_prev_t = self.sigmas[prev_t] if prev_t != 0 else 1e-5 # 0除算を防ぐため^^


        if self.mode == "dpm-solver":
            sqrt_alpha_t = self.get_alpha_respacing(t, prev_t).sqrt()
            prev_sample = sample / sqrt_alpha_t - self.sqrt_betas_bar[prev_t] * (sigma_t/sigma_prev_t - 1) * noise_pred
        elif self.mode == "dpm-solver++":
            sqrt_beta_t = self.sqrt_betas_bar[t] / self.sqrt_betas_bar[prev_t]
            prev_sample = sample / sqrt_beta_t - self.sqrt_alphas_bar[prev_t] * (sigma_prev_t/sigma_t - 1) * pred_original_sample
        else:
            raise NotImplementedError

        return prev_sample
    
