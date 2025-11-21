import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from demucs.htdemucs import HTDemucs
from transformers import AutoTokenizer, ClapModel, ClapTextModel

from typing import Optional, List
from fractions import Fraction
from einops import rearrange


class TextCrossAttention(nn.Module):
    """
    Cross-attention module to condition frequency/time features on text embeddings.
    - Projects inputs to a common dim, performs multi-head attention from queries (x/xt)
      to keys/values coming from text embeddings.
    - Supports text_emb either (B, L, Dtxt) or pooled (B, Dtxt).
    """

    def __init__(self, feat_dim, text_dim, n_heads=8, dropout=0.0):
        """
        feat_dim : int  number of channels in x and xt (e.g. 384)
        text_dim : int  dimensionality of CLAP text embeddings
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.text_dim = text_dim
        self.n_heads = n_heads

        # project query from feat -> attn_dim
        self.q_proj = nn.Linear(feat_dim, feat_dim)
        # project text to key/value
        self.k_proj = nn.Linear(text_dim, feat_dim)
        self.v_proj = nn.Linear(text_dim, feat_dim)

        # multihead attention reusing PyTorch built-in for convenience
        self.attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=n_heads, batch_first=True, dropout=dropout)

        # small MLP residual to mix attended output with original features
        self.out_mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )
        self.norm_q = nn.LayerNorm(feat_dim)
        self.norm_out = nn.LayerNorm(feat_dim)

    def forward_attend(self, queries, text_emb):
        """
        queries: (B, Lq, feat_dim)  - sequence queries (flattened x or xt)
        text_emb: (B, Ltxt, text_dim) or pooled (B, text_dim)
        returns: (B, Lq, feat_dim) updated queries
        """
        B = queries.shape[0]

        # Normalize queries
        q = self.norm_q(queries)

        # Prepare text key/value sequence
        if text_emb.dim() == 2:  # pooled (B, Dtxt) -> expand to single token
            k_seq = text_emb.unsqueeze(1)    # (B, 1, Dtxt)
            v_seq = k_seq
        elif text_emb.dim() == 3:  # (B, Ltxt, Dtxt)
            k_seq = text_emb
            v_seq = text_emb
        else:
            raise ValueError("text_emb must be (B, D) or (B, L, D)")

        # linear project to feat_dim
        k = self.k_proj(k_seq)   # (B, Ltxt, feat_dim)
        v = self.v_proj(v_seq)   # (B, Ltxt, feat_dim)
        q_proj = self.q_proj(q)  # (B, Lq, feat_dim)

        # use PyTorch MultiheadAttention (batch_first=True)
        print(f"q_proj shape: {q_proj.shape}")
        print(f"k shape: {k.shape}")
        print(f"v shape: {v.shape}")
        attn_out, _ = self.attn(query=q_proj, key=k, value=v)  # (B, Lq, feat_dim)

        # residual + MLP + norm
        out = queries + attn_out
        out = out + self.out_mlp(out)
        out = self.norm_out(out)
        return out

    def forward(self, x, xt, text_emb):
        """
        x  : (B, C, F, T)  -> flatten to sequence Lx = F*T
        xt : (B, C, Tt)    -> sequence Lxt = Tt
        text_emb: (B, Ltxt, Dtxt) or (B, Dtxt)

        Returns updated (x, xt) with same shapes.
        """
        B, C, F, T = x.shape
        # flatten freq branch: (B, Lx, C) where Lx = F*T
        x_seq = rearrange(x, "b c f t -> b (f t) c")  # (B, Lx, C)

        # time branch is already (B, C, Tt) but needs seq last -> (B, Tt, C)
        xt_seq = rearrange(xt, "b c t -> b t c")       # (B, Lxt, C)

        # attend queries->text embeddings
        x_seq_upd = self.forward_attend(x_seq, text_emb)   # (B, Lx, C)
        xt_seq_upd = self.forward_attend(xt_seq, text_emb) # (B, Lxt, C)

        # reshape back
        x_upd = rearrange(x_seq_upd, "b (f t) c -> b c f t", f=F, t=T)
        xt_upd = rearrange(xt_seq_upd, "b t c -> b c t")

        return x_upd, xt_upd


class AudioTextHTDemucs(nn.Module):
    """
    Wrapper around HTDemucs that adds a single audio<->text cross attention layer.
    CLAP text encoder produces text embeddings that serve as keys/values.
    Audio encoder features (bottleneck) serve as queries.
    """

    def __init__(
        self,
        htdemucs_model: HTDemucs,           # Preloaded HTDemucs instance
        clap_encoder: ClapModel,            # Preloaded CLAP model
        clap_tokenizer,                     # Tokenizer for CLAP model
        model_dim=384,                      # dim of HTDemucs bottleneck
        text_dim=512,                       # CLAP text embedding dimension
        n_fft=4096,
        hop_length=1024,
        num_heads=8,
        sample_rate=44100                   # Sample rate [Hz]
    ):
        super().__init__()

        # Pre-trained models and tokenizers
        self.htdemucs = htdemucs_model
        self.clap = clap_encoder
        self.tokenizer = clap_tokenizer
        
        # FFT and audio parameters
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        
        # Text cross attention
        self.text_attn = TextCrossAttention(model_dim, text_dim)

    def _htdemucs_full_enc(self, wav):
        """
        Helper function to do forward pass for all the encoding layers + cross attention
        """
        length = wav.shape[-1]
        length_pre_pad = None
        if self.htdemucs.use_train_segment:
            if self.training:
                self.segment = Fraction(wav.shape[-1], self.sample_rate)
            else:
                training_length = int(self.segment * self.sample_rate)
                if wav.shape[-1] < training_length:
                    length_pre_pad = wav.shape[-1]
                    wav = F.pad(wav, (0, training_length - length_pre_pad))
                    
        z = self.htdemucs._spec(wav)
        mag = self.htdemucs._magnitude(z).to(wav.device)
        x = mag

        B, C, Fq, T = x.shape

        # unlike previous Demucs, we always normalize because it is easier.
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        # x will be the freq. branch input.

        # Prepare the time branch input.
        xt = wav
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        # Frequence and waveform encoder (with storage for skip connections) 
        saved = []  # skip connections, freq.
        saved_t = []  # skip connections, time.
        lengths = []  # saved lengths to properly remove padding, freq branch.
        lengths_t = []  # saved lengths for time branch.
        
        for idx, encode in enumerate(self.htdemucs.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(self.htdemucs.tencoder):
                # we have not yet merged branches.
                lengths_t.append(xt.shape[-1])
                tenc = self.htdemucs.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    # save for skip connection
                    saved_t.append(xt)
                else:
                    # tenc contains just the first conv., so that now time and freq.
                    # branches have the same shape and can be merged.
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and self.htdemucs.freq_emb is not None:
                # add frequency embedding to allow for non equivariant convolutions
                # over the frequency axis.
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
        
        return x, xt
    
    def _get_clap_embeddings(self, text: List[str]):
        clap_inputs = tokenizer(text, padding=True, return_tensors="pt")
        with torch.no_grad():       # or torch.inference_mode - Check this!
            clap_inputs = self.tokenizer(text, padding=True, return_tensors="pt")
            text_emb = self.clap.get_text_features(**clap_inputs)                   # (B, t_dim)
            
        return text_emb

    # ------------------------------------------------------------------
    # Forward: waveform + text → conditioned separated stems
    # ------------------------------------------------------------------
    def forward(self, wav, text):
        """
        wav  : (B, C, T)
        text : List[str]
        Returns: (B, 1, C, T)    # separated 1-track (4 channels) GOAL
        """
        # Encode mixed waveform in both frequency and time/waveform domain using HTDemucs
        x, xt = self._htdemucs_full_enc(wav)
        
        print(f"x shape: {x.shape}")
        print(f"xt shape: {xt.shape}")
        
        # Encode text conditioning using CLAP text model
        text_emb = self._get_clap_embeddings(text)
        
        print(f"text_emb shape: {text_emb.shape}")
        
        # Run the text-conditioned cross-attention
        x_cond, xt_cond = self.text_attn(x, xt, text_emb)
                    
        print(f"x_cond shape: {x_cond.shape}")
        print(f"xt_cond shape: {xt_cond.shape}")
        
        # Now reuse HTDemucs decoder
        
        return torch.rand((1, 2, 44100))


# Example usage template:
if __name__ == "__main__":
    # Load pre-trained CLAP text encoder
    import torch
    from demucs import pretrained

    # Load pre-trained HTDemucs model
    htdemucs = pretrained.get_model('htdemucs').models[0]

    # Load CLAP model
    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

    model = AudioTextHTDemucs(
        htdemucs,
        clap,
        tokenizer  
    )
    #print(f"Full Model:\n{model}")
    #model = model.to('cuda')

    B = 1
    C = 2
    T = 44100 * 3  # 3 seconds @ 16k
    waveform = torch.randn(B, C, T)
    prompts = ["extract viola"]
    out_waveform = model(waveform, prompts)
    print(f"output waveform shape: {out_waveform.shape}")