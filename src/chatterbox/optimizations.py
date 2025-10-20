"""
Comprehensive optimizations for ChatterBox TTS

This module provides:
1. Mixed precision (BF16) support
2. torch.compile optimizations
3. CUDA backend optimizations
4. Performance logging
"""

import logging
import time
from functools import wraps
from typing import Optional, Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """
    Centralized performance optimization manager for ChatterBox
    """

    def __init__(self, enable_bf16: bool = True, enable_compile: bool = True):
        """
        Args:
            enable_bf16: Enable BF16 mixed precision
            enable_compile: Enable torch.compile optimization
        """
        self.enable_bf16 = enable_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.enable_compile = enable_compile

        # Initialize CUDA optimizations
        if torch.cuda.is_available():
            self._init_cuda_optimizations()

        logger.info(f"PerformanceOptimizer initialized:")
        logger.info(f"  - BF16: {self.enable_bf16}")
        logger.info(f"  - torch.compile: {self.enable_compile}")

    def _init_cuda_optimizations(self):
        """Initialize CUDA backend optimizations"""
        # Enable TF32 for matmul and convolutions (Ada GPUs)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cuDNN benchmark mode for consistent input sizes
        torch.backends.cudnn.benchmark = True

        # Enable cuDNN for better performance
        torch.backends.cudnn.enabled = True

        logger.info("CUDA optimizations enabled: TF32, cuDNN benchmark")

    def optimize_model(
        self,
        model: nn.Module,
        compile_mode: str = "reduce-overhead",
        compile_dynamic: bool = False
    ) -> nn.Module:
        """
        Apply optimizations to a model

        Args:
            model: PyTorch module to optimize
            compile_mode: torch.compile mode ('reduce-overhead', 'max-autotune', 'default')
            compile_dynamic: Enable dynamic shapes

        Returns:
            Optimized model
        """
        # Convert to BF16 if enabled
        if self.enable_bf16 and model.training is False:
            model = model.to(dtype=torch.bfloat16)
            logger.info(f"Converted {model.__class__.__name__} to BF16")

        # Apply torch.compile if enabled
        if self.enable_compile:
            try:
                model = torch.compile(
                    model,
                    mode=compile_mode,
                    dynamic=compile_dynamic,
                    fullgraph=False  # Allow graph breaks for complex models
                )
                logger.info(f"Compiled {model.__class__.__name__} with mode={compile_mode}")
            except Exception as e:
                logger.warning(f"Failed to compile {model.__class__.__name__}: {e}")

        return model

    @staticmethod
    def timed_inference(func: Callable) -> Callable:
        """
        Decorator to log inference timing
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = (time.time() - start) * 1000
            logger.info(f"[TIMING] {func.__name__}: {elapsed:.1f}ms")
            return result
        return wrapper


def optimize_hifigan_inference(hifigan, enable_bf16: bool = True, enable_compile: bool = True):
    """
    Optimize HiFiGAN vocoder for faster inference

    Args:
        hifigan: HiFTGenerator instance
        enable_bf16: Enable BF16 mixed precision via autocast
        enable_compile: Enable torch.compile
    """
    optimizer = PerformanceOptimizer(enable_bf16=enable_bf16, enable_compile=enable_compile)

    # Keep model in FP32 - autocast will handle conversion
    logger.info("[VOCODER] Model kept in FP32, will use autocast for BF16")

    # Store original inference
    if not hasattr(hifigan, '_original_inference'):
        hifigan._original_inference = hifigan.inference

    # Create optimized inference with logging and autocast
    @torch.inference_mode()
    def optimized_inference(speech_feat: torch.Tensor, cache_source: torch.Tensor = None):
        """Optimized vocoder inference with timing logs and autocast"""
        if cache_source is None:
            cache_source = torch.zeros(1, 1, 0).to(speech_feat.device)

        t_start = time.time()

        # Use autocast context for automatic BF16 conversion
        autocast_enabled = enable_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=autocast_enabled):
            # mel->f0
            t0 = time.time()
            f0 = hifigan.f0_predictor(speech_feat)
            t1 = time.time()
            logger.debug(f"[VOCODER] F0 prediction: {(t1-t0)*1000:.1f}ms")

            # f0->source
            s = hifigan.f0_upsamp(f0[:, None]).transpose(1, 2)
            s, _, _ = hifigan.m_source(s)
            s = s.transpose(1, 2)
            t2 = time.time()
            logger.debug(f"[VOCODER] Source generation: {(t2-t1)*1000:.1f}ms")

            # use cache_source to avoid glitch
            if cache_source.shape[2] != 0:
                s[:, :, :cache_source.shape[2]] = cache_source

            # decode (mel + source -> waveform)
            generated_speech = hifigan.decode(x=speech_feat, s=s)
            t3 = time.time()
            logger.debug(f"[VOCODER] Decode: {(t3-t2)*1000:.1f}ms")

        # Convert back to FP32 for compatibility after exiting autocast
        generated_speech = generated_speech.float()
        s = s.float()

        t_end = time.time()
        logger.info(f"[VOCODER] Total inference: {(t_end-t_start)*1000:.1f}ms")

        return generated_speech, s

    # Replace inference method
    hifigan.inference = optimized_inference

    # Optionally compile sub-modules (disabled for now due to complexity)
    if enable_compile:
        # Note: Disabling torch.compile for vocoder sub-modules as they're complex
        # and may cause graph breaks. The performance gain is minimal vs risk.
        logger.info("[VOCODER] torch.compile disabled for vocoder (too complex, minimal gain)")

    logger.info("[VOCODER] Optimization complete")
    return hifigan


def optimize_flow_decoder(flow_model, n_timesteps: int = 4, enable_bf16: bool = True, enable_compile: bool = True):
    """
    Optimize flow matching decoder for faster inference using torch.autocast

    Args:
        flow_model: CausalMaskedDiffWithXvec instance
        n_timesteps: Number of timesteps (reduced from 10)
        enable_bf16: Enable BF16 mixed precision via autocast
        enable_compile: Enable torch.compile
    """
    import torch.nn.functional as F
    from chatterbox.models.s3gen.utils.mask import make_pad_mask

    # Keep model in FP32 - autocast will handle conversion
    logger.info("[FLOW] Model kept in FP32, will use autocast for BF16")

    if not hasattr(flow_model, '_original_inference'):
        flow_model._original_inference = flow_model.inference

    @torch.inference_mode()
    def optimized_inference(
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        prompt_feat_len,
        embedding,
        finalize,
    ):
        """Optimized flow inference with reduced timesteps and autocast"""
        t_start = time.time()

        assert token.shape[0] == 1

        # Use autocast context for automatic BF16 conversion
        autocast_enabled = enable_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=autocast_enabled):
            t0 = time.time()

            # xvec projection - autocast handles dtype automatically
            embedding = F.normalize(embedding, dim=1)
            embedding = flow_model.spk_embed_affine_layer(embedding)

            # concat text and prompt_text
            token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
            mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
            token = flow_model.input_embedding(
                torch.clamp(token, min=0, max=flow_model.input_embedding.num_embeddings-1)
            ) * mask

            t1 = time.time()
            logger.debug(f"[FLOW] Embedding: {(t1-t0)*1000:.1f}ms")

            # text encode
            h, h_lengths = flow_model.encoder(token, token_len)
            if finalize is False:
                h = h[:, :-flow_model.pre_lookahead_len * flow_model.token_mel_ratio]
            mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
            h = flow_model.encoder_proj(h)

            t2 = time.time()
            logger.debug(f"[FLOW] Encoding: {(t2-t1)*1000:.1f}ms")

            # get conditions
            conds = torch.zeros([1, mel_len1 + mel_len2, flow_model.output_size], device=token.device).to(h.dtype)
            conds[:, :mel_len1] = prompt_feat
            conds = conds.transpose(1, 2)

            mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)

            t3 = time.time()

            # CRITICAL: Use reduced n_timesteps
            feat, _ = flow_model.decoder(
                mu=h.transpose(1, 2).contiguous(),
                mask=mask.unsqueeze(1),
                spks=embedding,
                cond=conds,
                n_timesteps=n_timesteps  # REDUCED from 10
            )

            t4 = time.time()
            logger.info(f"[FLOW] Decoder ({n_timesteps} steps): {(t4-t3)*1000:.1f}ms")

            feat = feat[:, :, mel_len1:]
            assert feat.shape[2] == mel_len2

        # Convert back to FP32 after exiting autocast
        feat = feat.float()

        t_end = time.time()
        logger.info(f"[FLOW] Total inference: {(t_end-t_start)*1000:.1f}ms")

        return feat, None

    # Replace inference method
    flow_model.inference = optimized_inference

    # Optionally compile decoder (disabled for now due to complexity)
    if enable_compile:
        # Note: Disabling torch.compile for flow decoder as it's complex
        # and may cause graph breaks. BF16 + reduced timesteps give sufficient speedup.
        logger.info(f"[FLOW] torch.compile disabled for flow decoder (too complex)")

    return flow_model


def optimize_t3_model(t3_model, enable_bf16: bool = True, enable_compile: bool = True):
    """
    Optimize T3 (LLaMA-based) model for faster inference

    Args:
        t3_model: T3 model instance
        enable_bf16: Enable BF16 mixed precision
        enable_compile: Enable torch.compile
    """
    if not hasattr(t3_model, '_original_inference'):
        t3_model._original_inference = t3_model.inference

    # Wrap inference with timing
    original_inference = t3_model.inference

    def timed_inference(*args, **kwargs):
        t_start = time.time()
        result = original_inference(*args, **kwargs)
        t_end = time.time()
        logger.info(f"[T3] Inference: {(t_end-t_start)*1000:.1f}ms")
        return result

    t3_model.inference = timed_inference

    # Optionally compile
    if enable_compile:
        try:
            # Note: LLaMA models are complex, compile carefully
            logger.info("[T3] Model compilation skipped (LLaMA is complex)")
        except Exception as e:
            logger.warning(f"[T3] Failed to compile: {e}")

    return t3_model


def optimize_s3gen_complete(s3gen, n_timesteps: int = 4, enable_bf16: bool = True, enable_compile: bool = True):
    """
    Complete S3Gen optimization (flow + vocoder)

    Args:
        s3gen: S3Gen (S3Token2Wav) instance
        n_timesteps: Number of timesteps for flow decoder
        enable_bf16: Enable BF16 mixed precision
        enable_compile: Enable torch.compile
    """
    # Optimize flow decoder
    if hasattr(s3gen, 'flow'):
        optimize_flow_decoder(s3gen.flow, n_timesteps=n_timesteps, enable_bf16=enable_bf16, enable_compile=enable_compile)

    # Optimize vocoder
    if hasattr(s3gen, 'mel2wav'):
        optimize_hifigan_inference(s3gen.mel2wav, enable_bf16=enable_bf16, enable_compile=enable_compile)

    logger.info(f"[S3GEN] Complete optimization applied (n_timesteps={n_timesteps})")
    return s3gen


def optimize_chatterbox_tts(
    tts_model,
    n_timesteps: int = 4,
    enable_bf16: bool = True,
    enable_compile: bool = True,
    disable_watermark: bool = False
):
    """
    Comprehensive ChatterBox TTS optimization

    Args:
        tts_model: ChatterboxTTS instance
        n_timesteps: Number of timesteps for flow matching
        enable_bf16: Enable BF16 mixed precision
        enable_compile: Enable torch.compile
        disable_watermark: Disable watermarking (saves ~300ms)
    """
    logger.info("[CHATTERBOX] Starting comprehensive optimization...")

    # Optimize T3 model
    if hasattr(tts_model, 't3'):
        optimize_t3_model(tts_model.t3, enable_bf16=enable_bf16, enable_compile=enable_compile)

    # Optimize S3Gen (flow + vocoder)
    if hasattr(tts_model, 's3gen'):
        optimize_s3gen_complete(tts_model.s3gen, n_timesteps=n_timesteps, enable_bf16=enable_bf16, enable_compile=enable_compile)

    # Optionally disable watermark
    if disable_watermark:
        if hasattr(tts_model, 'watermarker'):
            class NoOpWatermarker:
                def apply_watermark(self, audio, sample_rate):
                    return audio
            tts_model.watermarker = NoOpWatermarker()
            logger.info("[CHATTERBOX] Watermarking disabled")

    logger.info("[CHATTERBOX] Optimization complete!")
    return tts_model
