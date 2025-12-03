"""
AudioTextHTDemucs v2 - Text-conditioned source separation.

Changes from v1:
- Custom trainable decoder that outputs 1 source (not 4)
- HTDemucs encoder kept (frozen)
- CLAP text encoder (frozen)
- Cross-attention conditioning at bottleneck
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Any
from fractions import Fraction
from einops import rearrange

from demucs.htdemucs import HTDemucs
from transformers import ClapModel, ClapTextModelWithProjection, RobertaTokenizerFast

class TextCrossAttention(nn.Module):
    """Cross-attention: audio features attend to text embeddings."""

    def __init__(self, feat_dim, text_dim, n_heads=8, dropout=0.0):
        super().__init__()
        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(text_dim, feat_dim)
        self.v_proj = nn.Linear(text_dim, feat_dim)
        self.attn = nn.MultiheadAttention(feat_dim, n_heads, batch_first=True, dropout=dropout)
        self.out_mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )
        self.norm_q = nn.LayerNorm(feat_dim)
        self.norm_out = nn.LayerNorm(feat_dim)

    def forward_attend(self, queries, text_emb):
        q = self.norm_q(queries)
        if text_emb.dim() == 2:
            text_emb = text_emb.unsqueeze(1)
        k = self.k_proj(text_emb)
        v = self.v_proj(text_emb)
        q_proj = self.q_proj(q)
        attn_out, _ = self.attn(query=q_proj, key=k, value=v)
        out = queries + attn_out
        out = out + self.out_mlp(out)
        return self.norm_out(out)

    def forward(self, x, xt, text_emb):
        B, C, F, T = x.shape
        x_seq = rearrange(x, "b c f t -> b (f t) c")
        xt_seq = rearrange(xt, "b c t -> b t c")
        x_seq = self.forward_attend(x_seq, text_emb)
        xt_seq = self.forward_attend(xt_seq, text_emb)
        x = rearrange(x_seq, "b (f t) c -> b c f t", f=F, t=T)
        xt = rearrange(xt_seq, "b t c -> b c t")
        return x, xt


class FreqDecoder(nn.Module):
    """Frequency-domain decoder: mirrors HTDemucs encoder structure but outputs 1 source."""

    def __init__(self, channels: List[int], kernel_size: int = 8, stride: int = 4):
        """
        channels: List of channel dims from bottleneck to output, e.g. [384, 192, 96, 48, 2]
        """
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            is_last = (i == len(channels) - 2)

            self.layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(kernel_size//4, 0)),
                nn.GroupNorm(1, out_ch) if not is_last else nn.Identity(),
                nn.GELU() if not is_last else nn.Identity(),
            ))

    def forward(self, x, skips: List[torch.Tensor], target_lengths: List[int]):
        """
        x: (B, C, F, T) bottleneck features
        skips: encoder skip connections (reversed order)
        target_lengths: target frequency dimensions for each layer
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Match target size
            if i < len(target_lengths):
                target_f = target_lengths[i]
                if x.shape[2] != target_f:
                    x = F.interpolate(x, size=(target_f, x.shape[3]), mode='bilinear', align_corners=False)
            # Add skip connection if available
            if i < len(skips):
                skip = skips[i]
                # Project skip to match channels if needed
                if skip.shape[1] != x.shape[1]:
                    skip = skip[:, :x.shape[1]]  # Simple channel truncation
                if skip.shape[2:] != x.shape[2:]:
                    skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
                x = x + skip * 0.1  # Scaled residual
        return x


class TimeDecoder(nn.Module):
    """Time-domain decoder: outputs 1 source waveform."""

    def __init__(self, channels: List[int], kernel_size: int = 8, stride: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            is_last = (i == len(channels) - 2)

            self.layers.append(nn.Sequential(
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride, padding=kernel_size//4),
                nn.GroupNorm(1, out_ch) if not is_last else nn.Identity(),
                nn.GELU() if not is_last else nn.Identity(),
            ))

    def forward(self, x, skips: List[torch.Tensor], target_lengths: List[int]):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(target_lengths):
                target_t = target_lengths[i]
                if x.shape[2] != target_t:
                    x = F.interpolate(x, size=target_t, mode='linear', align_corners=False)
            if i < len(skips):
                skip = skips[i]
                if skip.shape[1] != x.shape[1]:
                    skip = skip[:, :x.shape[1]]
                if skip.shape[2] != x.shape[2]:
                    skip = F.interpolate(skip, size=x.shape[2], mode='linear', align_corners=False)
                x = x + skip * 0.1
        return x


class AudioTextHTDemucs(nn.Module):
    """
    Text-conditioned source separation.
    - HTDemucs encoder (frozen): extracts multi-scale audio features
    - CLAP (frozen): text embeddings
    - Cross-attention: conditions audio on text at bottleneck
    - Custom decoder (trainable): outputs single source
    """

    def __init__(
        self,
        htdemucs_model: HTDemucs,
        clap_encoder: ClapModel | ClapTextModelWithProjection,
        clap_tokenizer: RobertaTokenizerFast,
        model_dim: int = 384,
        text_dim: int = 512,
        num_heads: int = 8,
        sample_rate: int = 44100,
        segment: float = 7.8,
    ):
        super().__init__()

        self.htdemucs = htdemucs_model
        self.clap = clap_encoder
        self.tokenizer = clap_tokenizer
        self.sample_rate = sample_rate
        self.segment = segment

        # Freeze HTDemucs encoder
        for param in self.htdemucs.parameters():
            param.requires_grad = False

        # Freeze CLAP
        for param in self.clap.parameters():
            param.requires_grad = False

        # Text cross-attention at bottleneck
        self.text_attn = TextCrossAttention(model_dim, text_dim, num_heads)

        # Custom decoders (trainable) - output 1 source with 2 channels (stereo)
        # Channel progression: 384 -> 192 -> 96 -> 48 -> 4 (will be reshaped to 2 channels)
        self.freq_decoder = FreqDecoder([384, 192, 96, 48, 4])
        self.time_decoder = TimeDecoder([384, 192, 96, 48, 4])

        # Final projection to stereo
        self.freq_out = nn.Conv2d(4, 2, 1)
        self.time_out = nn.Conv1d(4, 2, 1)

    def _encode(self, x, xt):
        """Run HTDemucs encoder, save skip connections."""
        saved = []
        saved_t = []
        lengths = []
        lengths_t = []

        for idx, encode in enumerate(self.htdemucs.encoder):
            lengths.append(x.shape[-1])
            inject = None

            if idx < len(self.htdemucs.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = self.htdemucs.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt

            x = encode(x, inject)

            if idx == 0 and self.htdemucs.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.htdemucs.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.htdemucs.freq_emb_scale * emb

            saved.append(x)

        # Cross-transformer at bottleneck
        if self.htdemucs.crosstransformer:
            if self.htdemucs.bottom_channels:
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t -> b c (f t)")
                x = self.htdemucs.channel_upsampler(x)
                x = rearrange(x, "b c (f t) -> b c f t", f=f)
                xt = self.htdemucs.channel_upsampler_t(xt)

            x, xt = self.htdemucs.crosstransformer(x, xt)

            if self.htdemucs.bottom_channels:
                x = rearrange(x, "b c f t -> b c (f t)")
                x = self.htdemucs.channel_downsampler(x)
                x = rearrange(x, "b c (f t) -> b c f t", f=f)
                xt = self.htdemucs.channel_downsampler_t(xt)

        return x, xt, saved, saved_t, lengths, lengths_t

    def _get_clap_embeddings(self, text: List[str], device):
        inputs = self.tokenizer(text, padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if isinstance(self.clap, ClapModel):
            # Use get_text_features for ClapModel
            with torch.no_grad():
                return self.clap.get_text_features(**inputs)
        else:
            # Use forward pass for ClapTextModelWithProjection
            with torch.no_grad():
                return self.clap.forward(**inputs).text_embeds

    def forward(self, wav, text):
        """
        wav: (B, 2, T) stereo mixture
        text: List[str] prompts
        Returns: (B, 2, T) separated stereo source
        """
        device = wav.device
        B = wav.shape[0]
        original_length = wav.shape[-1]

        # Compute spectrogram
        z = self.htdemucs._spec(wav)
        mag = self.htdemucs._magnitude(z).to(device)
        x = mag

        B, C, Fq, T_spec = x.shape

        # Normalize
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        xt = wav
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        # Encode (frozen)
        with torch.no_grad():
            x_enc, xt_enc, saved, saved_t, lengths, lengths_t = self._encode(x, xt)

        # Text conditioning via cross-attention (trainable)
        text_emb = self._get_clap_embeddings(text, device)
        x_cond, xt_cond = self.text_attn(x_enc, xt_enc, text_emb)

        # Decode with custom decoder (trainable)
        # Reverse skips for decoder
        saved_rev = saved[::-1]
        saved_t_rev = saved_t[::-1]
        lengths_rev = lengths[::-1]
        lengths_t_rev = lengths_t[::-1]

        # Frequency decoder
        x_dec = self.freq_decoder(x_cond, saved_rev, lengths_rev)
        x_dec = self.freq_out(x_dec)  # (B, 2, F, T)

        # Interpolate to match original spectrogram size
        x_dec = F.interpolate(x_dec, size=(Fq, T_spec), mode='bilinear', align_corners=False)

        # Apply as mask and invert spectrogram
        mask = torch.sigmoid(x_dec)  # (B, 2, F, T) in [0, 1]

        # mag is (B, C, F, T) from htdemucs - take first 2 channels for stereo
        mag_stereo = mag[:, :2, :, :]  # (B, 2, F, T)
        masked_spec = mag_stereo * mask

        # z is complex (B, C, F, T) - take stereo channels
        z_stereo = z[:, :2, :, :]  # (B, 2, F, T)
        phase = z_stereo / (mag_stereo + 1e-8)  # Complex phase
        masked_z = masked_spec * phase  # Apply mask while preserving phase
        freq_wav = self.htdemucs._ispec(masked_z, original_length)

        # Time decoder
        xt_dec = self.time_decoder(xt_cond, saved_t_rev, lengths_t_rev)
        xt_dec = self.time_out(xt_dec)  # (B, 2, T)

        # Interpolate to original length
        if xt_dec.shape[-1] != original_length:
            xt_dec = F.interpolate(xt_dec, size=original_length, mode='linear', align_corners=False)

        # Denormalize time output
        xt_dec = xt_dec * stdt + meant

        # Combine frequency and time branches
        output = freq_wav + xt_dec

        return output


if __name__ == "__main__":
    from demucs import pretrained

    htdemucs = pretrained.get_model('htdemucs').models[0]
    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    tokenizer = __import__('transformers').AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

    model = AudioTextHTDemucs(htdemucs, clap, tokenizer)

    # Count params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")

    # Test forward
    wav = torch.randn(2, 2, 44100 * 3)
    prompts = ["drums", "bass"]
    out = model(wav, prompts)
    print(f"Input: {wav.shape} -> Output: {out.shape}")