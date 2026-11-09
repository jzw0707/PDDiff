import torch
from einops import rearrange, pack, unpack
from torch import nn
from model.smsp import SMSP, TPALogSNRSchedule

from denoising_diffusion_pytorch.simple_diffusion import *


class CondGaussianDiffusion(GaussianDiffusion):
    def __init__(
            self,
            model,
            *,
            image_size,
            channels=1,
            extra_channels=0,
            cond_channels=3,
            pred_objective='v',
            loss_type='l2',
            noise_schedule=logsnr_schedule_cosine,
            noise_d=None,
            noise_d_low=None,
            noise_d_high=None,
            num_sample_steps=500,
            clip_sample_denoised=True,
            auxiliary_loss_weights=(0.4, 0.2, 0.1),
            auxiliary_dilation_kernels=(),
            targetness_loss_type=None,
            targetness_loss_weight=0.0,
            targetness_sigma=2.0,
            stratified_time_sampling=False,
            smsp=None,
            tpa=None,
    ):
        if tpa is not None:
            if any(value is not None for value in (noise_d, noise_d_low, noise_d_high)):
                raise ValueError(
                    "TPA replaces noise_d/noise_d_low/noise_d_high; set them to null "
                    "to avoid applying the log-SNR shift twice"
                )
            noise_schedule = TPALogSNRSchedule(noise_schedule, **tpa)
        super(CondGaussianDiffusion, self).__init__(model, image_size=image_size, channels=channels,
                                                    pred_objective=pred_objective, noise_schedule=noise_schedule,
                                                    noise_d=noise_d, noise_d_low=noise_d_low, noise_d_high=noise_d_high,
                                                    num_sample_steps=num_sample_steps,
                                                    clip_sample_denoised=clip_sample_denoised)
        if loss_type not in ['l2', 'l1', 'l1+l2', 'mean(l1, l2)']:
            try:
                from utils.import_utils import get_obj_from_str
                loss_type = get_obj_from_str(loss_type)
            except:
                raise NotImplementedError

        self.loss_type = loss_type
        self.extra_channels = extra_channels
        self.cond_channels = cond_channels
        self.auxiliary_loss_weights = tuple(auxiliary_loss_weights)
        self.auxiliary_dilation_kernels = tuple(auxiliary_dilation_kernels)
        if isinstance(targetness_loss_type, str):
            from utils.import_utils import get_obj_from_str
            targetness_loss_type = get_obj_from_str(targetness_loss_type)
        self.targetness_loss_type = targetness_loss_type
        self.targetness_loss_weight = float(targetness_loss_weight)
        self.targetness_sigma = float(targetness_sigma)
        self.stratified_time_sampling = bool(stratified_time_sampling)
        self.train_time_max = 1.0
        self.smsp = None
        if smsp is not None:
            if pred_objective != 'x0' or channels != 1:
                raise ValueError("SMSP currently supports binary masks with pred_objective='x0' only")
            self.smsp = SMSP(**smsp)


        self.history = []

    def forward(self, img, cond_img, seg=None, extra_cond=None, *args, **kwargs):
        b, channels, h, w = img.shape
        cond_channels = cond_img.shape[1]
        assert channels == self.channels
        assert h == w == self.image_size
        assert cond_channels == self.cond_channels

        img = normalize_to_neg_one_to_one(img)
        seg = normalize_to_neg_one_to_one(seg) if seg is not None else None
        extra_cond = default(extra_cond, torch.zeros((b, self.extra_channels, h, w), device=self.device))
        if self.stratified_time_sampling and img.shape[0] > 1:
            batch_size = img.shape[0]
            times = (
                torch.arange(batch_size, device=self.device, dtype=torch.float32)
                + torch.rand(batch_size, device=self.device)
            ) / batch_size
            times = times[torch.randperm(batch_size, device=self.device)] * self.train_time_max
        else:
            times = torch.zeros((img.shape[0],), device=self.device).float().uniform_(0, self.train_time_max)
        return self.p_losses(img, times, cond_img, seg, extra_cond, *args, **kwargs)

    def _targetness_heatmap(self, target):
        sigma = self.targetness_sigma
        radius = max(1, int(round(3 * sigma)))
        coordinates = torch.arange(-radius, radius + 1, device=target.device, dtype=torch.float32)
        gaussian = torch.exp(-(coordinates ** 2) / (2 * sigma ** 2))
        kernel = torch.outer(gaussian, gaussian)
        kernel = kernel.view(1, 1, *kernel.shape).to(target.dtype)
        heatmap = F.conv2d(target, kernel, padding=radius)
        maximum = heatmap.amax(dim=(-2, -1), keepdim=True)
        return torch.where(maximum > 0, heatmap / maximum.clamp_min(1e-6), heatmap).clamp(0, 1)

    def p_losses(self, x_start, times, cond_img, seg=None, extra_cond=None, noise=None, *args, **kwargs):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # SMSP perturbs only the diffusion input; supervision stays on the original mask.
        diffusion_source = seg if seg is not None else x_start
        if self.training and self.smsp is not None:
            diffusion_source = self.smsp(diffusion_source, times)
        x, log_snr = self.q_sample(x_start=diffusion_source, times=times, noise=noise)

        model_args = (torch.cat([x, extra_cond], dim=1), log_snr, cond_img)
        model_out = self.model(*model_args)
        auxiliary_outputs = ()
        targetness_output = None
        if isinstance(model_out, dict):
            targetness_output = model_out.get("targetness")
            auxiliary_outputs = tuple(model_out.get("auxiliary_outputs", ()))
            model_out = model_out["prediction"]
        elif isinstance(model_out, (tuple, list)):
            model_out, *auxiliary_outputs = model_out

        if self.pred_objective == 'v':
            padded_log_snr = right_pad_dims_to(x, log_snr)
            alpha, sigma = padded_log_snr.sigmoid().sqrt(), (-padded_log_snr).sigmoid().sqrt()
            target = alpha * noise - sigma * x_start


        elif self.pred_objective == 'eps':
            target = noise

        elif self.pred_objective == 'x0':

            target = (x_start + 1)/2

        def calculate_loss(output, loss_target=target):
            if self.loss_type == 'l2':
                return F.mse_loss(output, loss_target)
            if self.loss_type == 'l1':
                return F.l1_loss(output, loss_target)
            if self.loss_type == 'l1+l2':
                return F.mse_loss(output, loss_target) + F.l1_loss(output, loss_target)
            if self.loss_type == 'mean(l1, l2)':
                return (F.mse_loss(output, loss_target) + F.l1_loss(output, loss_target)) / 2
            return self.loss_type(output, loss_target)

        loss = calculate_loss(model_out)
        for index, (weight, auxiliary_output) in enumerate(
            zip(self.auxiliary_loss_weights, auxiliary_outputs)
        ):
            auxiliary_target = target
            if index < len(self.auxiliary_dilation_kernels):
                kernel_size = int(self.auxiliary_dilation_kernels[index])
                if kernel_size > 1:
                    auxiliary_target = F.max_pool2d(
                        target,
                        kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                    )
            loss = loss + weight * calculate_loss(auxiliary_output, auxiliary_target)
        if targetness_output is not None and self.targetness_loss_weight > 0:
            targetness_target = self._targetness_heatmap(target)
            if self.targetness_loss_type is None:
                targetness_loss = F.binary_cross_entropy_with_logits(
                    targetness_output,
                    targetness_target,
                )
            else:
                targetness_loss = self.targetness_loss_type(
                    targetness_output,
                    targetness_target,
                )
            loss = loss + self.targetness_loss_weight * targetness_loss
        return loss

    @torch.no_grad()
    def sample(self, cond_img, extra_cond=None, verbose=True):
        b, c, h, w = cond_img.shape
        extra_cond = default(extra_cond, torch.zeros((b, self.extra_channels, h, w), device=self.device))
        return self.p_sample_loop((b, self.channels, self.image_size, self.image_size),
                                  cond_img,
                                  extra_cond=extra_cond, verbose=verbose)

    @torch.no_grad()
    def p_sample_loop(self, shape, cond_img, extra_cond, verbose=True):
        self.history = []
        img = torch.randn(shape, device=self.device)
        conditioning_features = self.model.extract_features(cond_img)
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device=self.device)

        for i in tqdm(range(self.num_sample_steps), desc='sampling loop time step', total=self.num_sample_steps,
                      disable=not verbose):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, conditioning_features, extra_cond, times, times_next)

        img.clamp_(-1., 1.)
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def p_sample(self, x, cond, extra_cond, time, time_next):
        batch, *_, device = *x.shape, x.device

        model_mean, model_variance = self.p_mean_variance(x=x, cond=cond, extra_cond=extra_cond, time=time,
                                                          time_next=time_next)

        if time_next == 0:
            return model_mean

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise


    def p_mean_variance(self, x, cond, extra_cond, time, time_next):

        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b=x.shape[0])
        model_args = (torch.cat([x, extra_cond], dim=1), batch_log_snr, cond)
        pred = self.model.sample_unet(*model_args)

        if self.pred_objective == 'v':
            x_start = alpha * x - sigma * pred

        elif self.pred_objective == 'eps':
            x_start = (x - sigma * pred) / alpha

        elif self.pred_objective == 'x0':


            x_start = pred.tanh()


        x_start.clamp_(-1., 1.)
        self.history.append(x_start)

        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)

        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance


