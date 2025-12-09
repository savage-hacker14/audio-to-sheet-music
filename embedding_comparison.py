"""
Embedding Comparison Script for CLAP vs Word2Vec

This script compares embeddings from:
1. CLAP (Contrastive Language-Audio Pre-training) - trained on audio-text pairs
2. Word2Vec - trained on general text corpora

Hypothesis: CLAP embeddings should cluster similar instruments together due to
contrastive learning on audio-text pairs, while Word2Vec embeddings may not
encode musical relationships as effectively.

Visualizations:
- 2D t-SNE/UMAP projections
- Cosine similarity matrices
- Distance matrices
- Optional TensorBoard projector
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from transformers import ClapModel, AutoTokenizer

# Try to import seaborn (optional, for better plot styling)
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Seaborn not available. Using matplotlib defaults. Install with: pip install seaborn")

# Try to import UMAP (optional, better than t-SNE for some cases)
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not available. Install with: pip install umap-learn")

# Try to import Word2Vec (gensim)
try:
    from gensim.models import KeyedVectors
    import gensim.downloader as api
    HAS_WORD2VEC = True
except ImportError:
    HAS_WORD2VEC = False
    print("Gensim not available. Install with: pip install gensim")

# Import stem prompts from dataloader
from src.dataloader import STEM_PROMPTS

# Set style for better plots
if HAS_SEABORN:
    sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class EmbeddingExtractor:
    """Extract embeddings from different models."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.clap_model = None
        self.clap_tokenizer = None
        self.word2vec_model = None
    
    def load_clap(self):
        """Load CLAP model and tokenizer."""
        print("Loading CLAP model...")
        self.clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        self.clap_tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")
        self.clap_model = self.clap_model.to(self.device)
        self.clap_model.eval()
        print("CLAP model loaded.")
    
    def load_word2vec(self, model_name: str = "word2vec-google-news-300"):
        """Load Word2Vec model from gensim."""
        if not HAS_WORD2VEC:
            raise ImportError("Gensim not available. Install with: pip install gensim")
        
        print(f"Loading Word2Vec model: {model_name}...")
        try:
            self.word2vec_model = api.load(model_name)
            print(f"Word2Vec model loaded. Vocabulary size: {len(self.word2vec_model.key_to_index)}")
        except Exception as e:
            print(f"Error loading Word2Vec model: {e}")
            print("Trying to load from local file or using alternative...")
            raise
    
    def get_clap_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get CLAP embeddings for a list of texts."""
        if self.clap_model is None:
            self.load_clap()
        
        with torch.no_grad():
            inputs = self.clap_tokenizer(texts, padding=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            embeddings = self.clap_model.get_text_features(**inputs)
            embeddings = embeddings.cpu().numpy()
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    def get_word2vec_embeddings(self, texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Get Word2Vec embeddings for a list of texts.
        
        Returns:
            embeddings: (N, dim) array of embeddings
            valid_texts: List of texts that were successfully embedded
        """
        if self.word2vec_model is None:
            self.load_word2vec()
        
        embeddings = []
        valid_texts = []
        
        for text in texts:
            # Split text into words and average their embeddings
            words = text.lower().split()
            word_embeddings = []
            
            for word in words:
                if word in self.word2vec_model:
                    word_embeddings.append(self.word2vec_model[word])
            
            if word_embeddings:
                # Average word embeddings
                text_embedding = np.mean(word_embeddings, axis=0)
                embeddings.append(text_embedding)
                valid_texts.append(text)
            else:
                print(f"Warning: Could not embed '{text}' - no words found in vocabulary")
        
        if not embeddings:
            raise ValueError("No valid embeddings could be generated")
        
        embeddings = np.array(embeddings)
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        return embeddings, valid_texts


def collect_all_prompts() -> Tuple[List[str], List[str]]:
    """Collect all stem prompts and their categories."""
    prompts = []
    categories = []
    
    # Add all prompts from STEM_PROMPTS
    for stem_name, prompt_list in STEM_PROMPTS.items():
        for prompt in prompt_list:
            prompts.append(prompt)
            categories.append(stem_name)
    
    # Add string instrument prompts as unique prompts
    string_instruments = ["violin", "viola", "cello"]
    for instrument in string_instruments:
        prompts.append(instrument)
        categories.append("strings")  # New category for string instruments
    
    return prompts, categories


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix."""
    return cosine_similarity(embeddings)


def compute_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance matrix."""
    return euclidean_distances(embeddings)


def reduce_dimensions(embeddings: np.ndarray, method: str = "tsne", n_components: int = 2, 
                     random_state: int = 42) -> np.ndarray:
    """Reduce embedding dimensions for visualization."""
    if method == "tsne":
        reducer = TSNE(n_components=n_components, random_state=random_state, 
                      perplexity=min(30, len(embeddings) - 1), max_iter=1000)
        reduced = reducer.fit_transform(embeddings)
    elif method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
        reduced = reducer.fit_transform(embeddings)
    elif method == "umap":
        if not HAS_UMAP:
            print("Warning: UMAP not available. Falling back to t-SNE.")
            print("Install UMAP with: pip install umap-learn")
            reducer = TSNE(n_components=n_components, random_state=random_state, 
                          perplexity=min(30, len(embeddings) - 1), max_iter=1000)
            reduced = reducer.fit_transform(embeddings)
        else:
            reducer = umap.UMAP(n_components=n_components, random_state=random_state)
            reduced = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return reduced


def plot_embeddings_2d(reduced_embeddings: np.ndarray, labels: List[str], 
                       categories: List[str], title: str, output_path: Optional[Path] = None):
    """Plot 2D embeddings with color coding by category."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color map for categories
    unique_categories = list(set(categories))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_categories)))
    category_to_color = {cat: colors[i] for i, cat in enumerate(unique_categories)}
    
    # Plot each category
    for category in unique_categories:
        mask = [cat == category for cat in categories]
        ax.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1],
                  c=[category_to_color[category]], label=category, s=100, alpha=0.7)
    
    # Add labels
    for i, label in enumerate(labels):
        ax.annotate(label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                   fontsize=8, alpha=0.7)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    return fig


def plot_similarity_matrix(similarity_matrix: np.ndarray, labels: List[str], 
                           title: str, output_path: Optional[Path] = None):
    """Plot cosine similarity matrix as heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if HAS_SEABORN:
        sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, vmin=-1, vmax=1, xticklabels=labels, yticklabels=labels,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    else:
        # Fallback to matplotlib imshow
        im = ax.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Cosine Similarity', rotation=270, labelpad=20)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        
        # Add text annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved similarity matrix to {output_path}")
    
    return fig


def convert_to_native_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_native_types(item) for item in obj)
    else:
        return obj


def analyze_clustering(embeddings: np.ndarray, categories: List[str]) -> Dict:
    """Analyze how well embeddings cluster by category."""
    # Compute intra-category and inter-category similarities
    unique_categories = list(set(categories))
    similarity_matrix = compute_similarity_matrix(embeddings)
    
    intra_similarities = []
    inter_similarities = []
    
    for i, cat_i in enumerate(categories):
        for j, cat_j in enumerate(categories):
            if i < j:  # Avoid duplicates
                sim = similarity_matrix[i, j]
                if cat_i == cat_j:
                    intra_similarities.append(sim)
                else:
                    inter_similarities.append(sim)
    
    results = {
        "intra_category_mean": float(np.mean(intra_similarities)),
        "intra_category_std": float(np.std(intra_similarities)),
        "inter_category_mean": float(np.mean(inter_similarities)),
        "inter_category_std": float(np.std(inter_similarities)),
        "separation": float(np.mean(intra_similarities) - np.mean(inter_similarities)),
    }
    
    return results


def save_to_tensorboard(embeddings: np.ndarray, labels: List[str], 
                       categories: List[str], log_dir: Path, model_name: str):
    """Save embeddings to TensorBoard format."""
    # Check if tensorboard package is installed
    try:
        import tensorboard
    except ImportError:
        print("TensorBoard package not installed. Install with: pip install tensorboard")
        return
    
    # Try PyTorch's TensorBoard integration (requires tensorboard package)
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as e:
        print(f"Could not import TensorBoard SummaryWriter: {e}")
        print("Make sure tensorboard is installed: pip install tensorboard")
        return
    
    log_dir = log_dir / f"{model_name}_embeddings"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(str(log_dir))
    
    # Convert to tensor
    embeddings_tensor = torch.from_numpy(embeddings).float()
    
    # Create metadata
    metadata = [f"{label} ({cat})" for label, cat in zip(labels, categories)]
    
    # Add embeddings to TensorBoard
    writer.add_embedding(
        embeddings_tensor,
        metadata=metadata,
        tag=f"{model_name}_embeddings",
        global_step=0
    )
    
    writer.close()
    print(f"Saved embeddings to TensorBoard: {log_dir}")
    print(f"Run: tensorboard --logdir {log_dir.parent}")


def main():
    parser = argparse.ArgumentParser(description="Compare CLAP vs Word2Vec embeddings")
    parser.add_argument("--output-dir", type=str, default="embedding_comparisons",
                       help="Directory to save outputs")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (auto-detect if not specified)")
    parser.add_argument("--method", type=str, default="tsne", choices=["tsne", "pca", "umap"],
                       help="Dimensionality reduction method")
    parser.add_argument("--word2vec-model", type=str, default="word2vec-google-news-300",
                       help="Word2Vec model to use")
    parser.add_argument("--tensorboard", action="store_true",
                       help="Export embeddings to TensorBoard format")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip generating plots")
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all prompts
    prompts, categories = collect_all_prompts()
    print(f"\nCollected {len(prompts)} prompts:")
    for prompt, cat in zip(prompts, categories):
        print(f"  {cat}: {prompt}")
    
    # Initialize extractor
    extractor = EmbeddingExtractor(device=device)
    
    # Extract CLAP embeddings
    print("\n" + "="*60)
    print("Extracting CLAP embeddings...")
    print("="*60)
    clap_embeddings = extractor.get_clap_embeddings(prompts)
    print(f"CLAP embeddings shape: {clap_embeddings.shape}")
    
    # Extract Word2Vec embeddings
    print("\n" + "="*60)
    print("Extracting Word2Vec embeddings...")
    print("="*60)
    try:
        word2vec_embeddings, valid_prompts_w2v = extractor.get_word2vec_embeddings(prompts)
        valid_categories_w2v = [cat for prompt, cat in zip(prompts, categories) 
                               if prompt in valid_prompts_w2v]
        print(f"Word2Vec embeddings shape: {word2vec_embeddings.shape}")
        print(f"Valid prompts: {len(valid_prompts_w2v)}/{len(prompts)}")
    except Exception as e:
        print(f"Error extracting Word2Vec embeddings: {e}")
        word2vec_embeddings = None
        valid_prompts_w2v = None
        valid_categories_w2v = None
    
    # Analyze CLAP embeddings
    print("\n" + "="*60)
    print("Analyzing CLAP embeddings...")
    print("="*60)
    clap_similarity = compute_similarity_matrix(clap_embeddings)
    clap_analysis = analyze_clustering(clap_embeddings, categories)
    
    print(f"Intra-category similarity: {clap_analysis['intra_category_mean']:.3f} ± {clap_analysis['intra_category_std']:.3f}")
    print(f"Inter-category similarity: {clap_analysis['inter_category_mean']:.3f} ± {clap_analysis['inter_category_std']:.3f}")
    print(f"Separation (intra - inter): {clap_analysis['separation']:.3f}")
    
    # Analyze Word2Vec embeddings if available
    if word2vec_embeddings is not None:
        print("\n" + "="*60)
        print("Analyzing Word2Vec embeddings...")
        print("="*60)
        w2v_similarity = compute_similarity_matrix(word2vec_embeddings)
        w2v_analysis = analyze_clustering(word2vec_embeddings, valid_categories_w2v)
        
        print(f"Intra-category similarity: {w2v_analysis['intra_category_mean']:.3f} ± {w2v_analysis['intra_category_std']:.3f}")
        print(f"Inter-category similarity: {w2v_analysis['inter_category_mean']:.3f} ± {w2v_analysis['inter_category_std']:.3f}")
        print(f"Separation (intra - inter): {w2v_analysis['separation']:.3f}")
    
    # Generate visualizations
    if not args.no_plots:
        print("\n" + "="*60)
        print("Generating visualizations...")
        print("="*60)
        
        # CLAP visualizations
        print("Generating CLAP visualizations...")
        clap_reduced = reduce_dimensions(clap_embeddings, method=args.method)
        plot_embeddings_2d(clap_reduced, prompts, categories,
                          f"CLAP Embeddings ({args.method.upper()})",
                          output_dir / "clap_embeddings_2d.png")
        plot_similarity_matrix(clap_similarity, prompts,
                              "CLAP Cosine Similarity Matrix",
                              output_dir / "clap_similarity_matrix.png")
        plt.close('all')
        
        # Word2Vec visualizations
        if word2vec_embeddings is not None:
            print("Generating Word2Vec visualizations...")
            w2v_reduced = reduce_dimensions(word2vec_embeddings, method=args.method)
            plot_embeddings_2d(w2v_reduced, valid_prompts_w2v, valid_categories_w2v,
                              f"Word2Vec Embeddings ({args.method.upper()})",
                              output_dir / "word2vec_embeddings_2d.png")
            plot_similarity_matrix(w2v_similarity, valid_prompts_w2v,
                                  "Word2Vec Cosine Similarity Matrix",
                                  output_dir / "word2vec_similarity_matrix.png")
            plt.close('all')
    
    # Save to TensorBoard if requested
    if args.tensorboard:
        print("\n" + "="*60)
        print("Exporting to TensorBoard...")
        print("="*60)
        save_to_tensorboard(clap_embeddings, prompts, categories, output_dir, "CLAP")
        if word2vec_embeddings is not None:
            save_to_tensorboard(word2vec_embeddings, valid_prompts_w2v, valid_categories_w2v,
                              output_dir, "Word2Vec")
    
    # Save analysis results
    results = {
        "clap": {
            "embeddings_shape": list(clap_embeddings.shape),
            "analysis": clap_analysis,
        },
        "word2vec": {
            "embeddings_shape": list(word2vec_embeddings.shape) if word2vec_embeddings is not None else None,
            "analysis": w2v_analysis if word2vec_embeddings is not None else None,
            "valid_prompts": valid_prompts_w2v if word2vec_embeddings is not None else None,
        },
        "prompts": prompts,
        "categories": categories,
    }
    
    # Convert numpy types to native Python types for JSON serialization
    results = convert_to_native_types(results)
    
    results_file = output_dir / "embedding_analysis.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print(f"\nAll outputs saved to {output_dir}")
    
    # Print summary comparison
    if word2vec_embeddings is not None:
        print("\n" + "="*60)
        print("SUMMARY COMPARISON")
        print("="*60)
        print(f"CLAP separation:     {clap_analysis['separation']:.3f}")
        print(f"Word2Vec separation: {w2v_analysis['separation']:.3f}")
        print(f"\nCLAP shows {'better' if clap_analysis['separation'] > w2v_analysis['separation'] else 'worse'} clustering by instrument category.")


if __name__ == "__main__":
    main()

