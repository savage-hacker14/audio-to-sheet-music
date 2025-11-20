import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# -----------------------
# Helper Blocks
# -----------------------

class PatchConv1d(nn.Module):
    """Turn 1D waveform into patch embeddings (Z-encoder)."""
    def __init__(self, in_ch=1, out_ch=256, kernel=16, stride=8):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride)

    def forward(self, x):
        # x: (B, 1, T)
        return self.conv(x).transpose(1, 2)  # → (B, Tokens, C)


class MLP(nn.Module):
    """Simple feedforward block."""
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class CrossAttention(nn.Module):
    """One cross-attention block: Q from x, K/V from context."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = MLP(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x, context):
        # x: (B, Tx, C) queries
        # context: (B, Ty, C) KV
        h, _ = self.attn(x, context, context)
        x = self.ln1(x + h)
        x = self.ln2(x + self.ff(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of self-attention layers."""
    def __init__(self, dim, depth=4, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 4,
                batch_first=True,
                activation='gelu'
            ) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------
#       MAIN MODEL — Waveform → STFT → CrossAttn → Spec
# ---------------------------------------------------------

class TextConditionedSeparator(nn.Module):
    def __init__(
        self,
        clap_text_encoder,
        clap_tokenizer,
        n_fft=512,
        hop_length=256,
        z_dim=256,
        t_dim=512,           # CLAP text dimension
        model_dim=256,
        depth=6,
    ):
        super().__init__()

        self.clap = clap_text_encoder  # frozen or trainable based on config
        self.tokenizer = tokenizer
        self.n_fft = n_fft
        self.hop = hop_length

        # ------------------------------
        # Waveform → latent (Z-encoder)
        # ------------------------------
        self.z_encoder = PatchConv1d(
            in_ch=1,
            out_ch=z_dim,
            kernel=16,
            stride=8
        )

        # Project CLAP embeddings to model_dim
        self.text_proj = nn.Linear(t_dim, model_dim)

        # Project waveform Z tokens to model_dim
        self.z_proj = nn.Linear(z_dim, model_dim)

        # ------------------------------
        # Cross Attention + Transformers
        # ------------------------------
        self.cross = CrossAttention(model_dim, num_heads=8)

        self.transformer = TransformerEncoder(
            dim=model_dim,
            depth=depth,
            num_heads=8
        )

        # ------------------------------
        # Decoder: latent tokens → spectrogram
        # ------------------------------
        # Final shape must match STFT magnitude shape (freq bins)
        self.spec_decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, n_fft // 2 + 1)  # output frequency bins
        )

    # ------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------
    def forward(self, waveform, text):
        """
        waveform: (B, 1, T)
        text: list[str]  (raw text prompts)
        returns: separated waveform (B, 1, T)
        """

        # ====================================================
        # 1. Convert waveform to STFT for reference length
        # ====================================================
        stft = torch.stft(
            waveform.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop,
            return_complex=True
        )   # (B, F, T')

        mag = stft.abs().transpose(1, 2)  # → (B, T', F)

        # ====================================================
        # 2. Encode waveform into latent tokens
        # ====================================================
        z_tokens = self.z_encoder(waveform)         # (B, Zt, z_dim)
        z_tokens = self.z_proj(z_tokens)            # (B, Zt, model_dim)

        # ====================================================
        # 3. Encode text via CLAP
        # ====================================================
        with torch.no_grad():
            clap_inputs = self.tokenizer(text, padding=True, return_tensors="pt")
            text_emb = self.clap.get_text_features(**clap_inputs)  # (B, t_dim)

        text_tokens = self.text_proj(text_emb).unsqueeze(1)  # (B, 1, model_dim)

        # ====================================================
        # 4. Cross Attention
        #    waveform tokens Q → text tokens K/V
        # ====================================================
        fused = self.cross(z_tokens, text_tokens)  # (B, Zt, model_dim)

        # ====================================================
        # 5. Transformer encoder (temporal modeling)
        # ====================================================
        h = self.transformer(fused)  # (B, Zt, model_dim)

        # ====================================================
        # 6. Decode latent tokens → magnitude spectrogram
        # ====================================================
        pred_mag = self.spec_decoder(h)             # (B, Zt, F)

        # Match temporal dimension to original STFT length
        pred_mag = F.interpolate(
            pred_mag.transpose(1, 2), 
            size=mag.size(1),
            mode='linear',
            align_corners=False
        ).transpose(1, 2)

        # ====================================================
        # 7. Restore phase from mixture (standard approach)
        # ====================================================
        phase = stft.angle().transpose(1, 2)  # (B, T', F)
        pred_stft = torch.polar(pred_mag, phase).transpose(1, 2)  # (B, F, T')

        # ====================================================
        # 8. iSTFT → separated waveform
        # ====================================================
        pred_wave = torch.istft(
            pred_stft,
            n_fft=self.n_fft,
            hop_length=self.hop,
            length=waveform.shape[-1]
        ).unsqueeze(1)

        return pred_wave, pred_mag
    


# Example usage template:
# THIS WORKS (on CPU) - but extremely memory hungry
if __name__ == "__main__":
    # Load pre-trained CLAP text encoder
    import torch
    from transformers import AutoTokenizer, ClapModel

    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

    model = TextConditionedSeparator(
        clap,
        tokenizer,
        n_fft=4096,
        hop_length=4        
    )
    print(f"Model:\n{model}")
    #model = model.to('cuda')

    B = 2
    C = 1
    T = 16000 * 3  # 3 seconds @ 16k
    waveform = torch.randn(B, C, T)
    prompts = ["extract viola", "extract drums"]
    out_waveform, out_spec = model(waveform, prompts)
    print(f"output waveform shape: {out_waveform.shape}")
    print(f"output spectrogram shape: {out_spec.shape}")
