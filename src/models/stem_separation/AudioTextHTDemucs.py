import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from demucs.htdemucs import HTDemucs
from transformers import AutoTokenizer, ClapModel, ClapTextModel

from typing import Optional, List, Tuple
from fractions import Fraction
from einops import rearrange


class TextCrossAttention(nn.Module):
    """
    Cross-attention module to condition frequency/time features on text embeddings.
    """

    def __init__(self, feat_dim, text_dim, n_heads=8, dropout=0.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.text_dim = text_dim
        self.n_heads = n_heads

        # project query from feat -> attn_dim
        self.q_proj = nn.Linear(feat_dim, feat_dim)
        # project text to key/value
        self.k_proj = nn.Linear(text_dim, feat_dim)
        self.v_proj = nn.Linear(text_dim, feat_dim)

        # multihead attention
        self.attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=dropout
        )

        # small MLP residual
        self.out_mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )
        self.norm_q = nn.LayerNorm(feat_dim)
        self.norm_out = nn.LayerNorm(feat_dim)

    def forward_attend(self, queries, text_emb):
        """
        queries: (B, Lq, feat_dim)
        text_emb: (B, Ltxt, text_dim) or pooled (B, text_dim)
        returns: (B, Lq, feat_dim)
        """
        q = self.norm_q(queries)

        # Prepare text key/value sequence
        if text_emb.dim() == 2:
            k_seq = text_emb.unsqueeze(1)
            v_seq = k_seq
        elif text_emb.dim() == 3:
            k_seq = text_emb
            v_seq = text_emb
        else:
            raise ValueError("text_emb must be (B, D) or (B, L, D)")

        k = self.k_proj(k_seq)
        v = self.v_proj(v_seq)
        q_proj = self.q_proj(q)

        attn_out, _ = self.attn(query=q_proj, key=k, value=v)

        # residual + MLP + norm
        out = queries + attn_out
        out = out + self.out_mlp(out)
        out = self.norm_out(out)
        return out

    def forward(self, x, xt, text_emb):
        """
        x  : (B, C, F, T)
        xt : (B, C, Tt)
        text_emb: (B, Ltxt, Dtxt) or (B, Dtxt)
        Returns updated (x, xt) with same shapes.
        """
        B, C, F, T = x.shape

        # flatten freq branch
        x_seq = rearrange(x, "b c f t -> b (f t) c")
        xt_seq = rearrange(xt, "b c t -> b t c")

        # attend
        x_seq_upd = self.forward_attend(x_seq, text_emb)
        xt_seq_upd = self.forward_attend(xt_seq, text_emb)

        # reshape back
        x_upd = rearrange(x_seq_upd, "b (f t) c -> b c f t", f=F, t=T)
        xt_upd = rearrange(xt_seq_upd, "b t c -> b c t")

        return x_upd, xt_upd


class AudioTextHTDemucs(nn.Module):
    """
    Wrapper around HTDemucs that adds text conditioning via CLAP.
    """

    def __init__(
            self,
            htdemucs_model: HTDemucs,
            clap_encoder: ClapModel,
            clap_tokenizer,
            model_dim=384,
            text_dim=512,
            n_fft=4096,
            hop_length=1024,
            num_heads=8,
            sample_rate=44100,
            segment=7.8  # Default segment length in seconds
    ):
        super().__init__()

        self.htdemucs = htdemucs_model
        self.clap = clap_encoder
        self.tokenizer = clap_tokenizer

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.segment = segment  # FIX: Initialize segment

        # Text cross attention
        self.text_attn = TextCrossAttention(model_dim, text_dim, num_heads)

        # Freeze CLAP
        for param in self.clap.parameters():
            param.requires_grad = False

    def _htdemucs_full_enc(self, x, xt):
        """Encode through HTDemucs encoder"""
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

        if self.htdemucs.crosstransformer:
            if self.htdemucs.bottom_channels:
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.htdemucs.channel_upsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = self.htdemucs.channel_upsampler_t(xt)

            x, xt = self.htdemucs.crosstransformer(x, xt)

            if self.htdemucs.bottom_channels:
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.htdemucs.channel_downsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = self.htdemucs.channel_downsampler_t(xt)

        return x, xt, saved, saved_t, lengths, lengths_t

    def _htdemucs_full_dec(self, length, length_pre_pad, dims, x, xt, z,
                           saved, saved_t, lengths, lengths_t,
                           mean, std, meant, stdt, Fq):
        """Decode through HTDemucs decoder"""
        for idx, decode in enumerate(self.htdemucs.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))

            offset = self.htdemucs.depth - len(self.htdemucs.tdecoder)
            if idx >= offset:
                tdec = self.htdemucs.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    assert pre.shape[2] == 1, pre.shape
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        assert len(saved) == 0
        assert len(lengths_t) == 0
        assert len(saved_t) == 0

        B, C, Fq, T = dims
        S = len(self.htdemucs.sources)

        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        # Handle MPS device
        x_is_mps = x.device.type == "mps"
        if x_is_mps:
            x = x.cpu()

        zout = self.htdemucs._mask(z, x)

        training_length = int(self.segment * self.sample_rate)
        if self.htdemucs.use_train_segment:
            use_length = length if self.training else training_length
        else:
            use_length = length

        x = self.htdemucs._ispec(zout, use_length)

        if x_is_mps:
            x = x.to("mps")

        xt = xt.view(B, S, -1, use_length)
        xt = xt * stdt[:, None] + meant[:, None]
        x = xt + x

        if length_pre_pad:
            x = x[..., :length_pre_pad]

        return x  # (B, S, C, T) - all sources

    def _get_clap_embeddings(self, text: List[str], device):
        """Get CLAP text embeddings - FIX: proper device handling"""
        with torch.no_grad():
            clap_inputs = self.tokenizer(text, padding=True, return_tensors="pt")
            # Move inputs to correct device
            clap_inputs = {k: v.to(device) for k, v in clap_inputs.items()}
            text_emb = self.clap.get_text_features(**clap_inputs)

        return text_emb  # (B, text_dim)

    def forward(self, wav, text):
        """
        wav  : (B, C, T)
        text : List[str]
        Returns: (B, C, T) - single extracted target
        """
        device = wav.device

        # Track lengths clearly
        original_length = wav.shape[-1]
        length_pre_pad = None  # Will store original length if we pad

        # Handle segment padding for inference
        if self.htdemucs.use_train_segment:
            if self.training:
                # During training, use input length (don't modify self.segment)
                current_segment = Fraction(wav.shape[-1], self.sample_rate)
            else:
                # During inference, pad to training segment length if needed
                current_segment = self.segment
                training_length = int(current_segment * self.sample_rate)
                if wav.shape[-1] < training_length:
                    length_pre_pad = original_length  # Save for cropping later
                    wav = F.pad(wav, (0, training_length - length_pre_pad))

        # Now length always reflects the current wav length (possibly padded)
        length = wav.shape[-1]

        # Compute spectrogram
        z = self.htdemucs._spec(wav)
        mag = self.htdemucs._magnitude(z).to(device)
        x = mag

        B, C, Fq, T = x.shape
        dims = tuple(x.shape)

        # Normalize frequency branch
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        # Normalize time branch
        xt = wav
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        # Encode through HTDemucs
        x, xt, saved, saved_t, lengths, lengths_t = self._htdemucs_full_enc(x, xt)

        # Get text embeddings from CLAP
        text_emb = self._get_clap_embeddings(text, device)

        # Apply text conditioning via cross-attention
        x_cond, xt_cond = self.text_attn(x, xt, text_emb)

        # Decode back to waveforms (returns all 4 sources)
        all_sources = self._htdemucs_full_dec(
            length, length_pre_pad, dims, x_cond, xt_cond, z,
            saved, saved_t, lengths, lengths_t, mean, std, meant, stdt, Fq
        )

        # Sum all sources (training will guide extraction of correct source)
        output = all_sources.sum(dim=1)  # (B, C, T)

        return output


if __name__ == "__main__":
    from demucs import pretrained

    # Load models
    htdemucs = pretrained.get_model('htdemucs').models[0]
    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

    model = AudioTextHTDemucs(htdemucs, clap, tokenizer)

    B = 2
    C = 2
    T = 44100 * 3
    waveform = torch.randn(B, C, T)
    prompts = ["drums", "bass"]

    out_waveform = model(waveform, prompts)
    print(f"Input shape: {waveform.shape}")
    print(f"Output shape: {out_waveform.shape}")