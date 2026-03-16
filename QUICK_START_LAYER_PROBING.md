# 🚀 Quick Start: Layer Probing for Anaphora Resolution

## What This Does

Analyzes **all 32 layers** of Llama-2-7b-chat-hf to identify **which layers encode anaphoric information**.

## 🎯 Answer: Which Layer?

This analysis will tell you at which depth (layer number) the model processes anaphora:
- **Early layers (0-10)**: Surface patterns
- **Middle layers (11-21)**: Semantic processing ← *Most likely*
- **Late layers (22-31)**: High-level reasoning

## ⚡ Run Everything (One Command)

```bash
cd /Users/duaaalshareif/AMMI/AnaphoraGym
bash scripts/mechanistic_analysis/run_layer_probing.sh
```

**That's it!** This will:
1. ✅ Probe all 32 layers
2. ✅ Generate statistics
3. ✅ Create 4+ visualizations
4. ✅ Display summary results

**Time:** 15-30 minutes with GPU

## 🧪 Quick Demo First (Recommended)

Test with a simple example before running full analysis:

```bash
python scripts/mechanistic_analysis/demo_layer_probing.py
```

**Time:** 5-10 minutes

## 📊 What You'll Get

### Visualizations (in `images/layer_probing/`)

1. **layer_performance.png** - Main 4-panel plot
   - Shows performance across all 32 layers
   - Highlights best layer
   - Displays accuracy and robustness

2. **layer_trajectory.png** - Gradient visualization
   - Beautiful color-coded evolution
   - Shows how anaphora encoding develops

3. **layer_comparison.png** - Multi-metric view
   - Compare different measurements
   - Normalized for easy interpretation

4. **layer_condition_heatmap.png** - Condition breakdown
   - Shows if different anaphora types use different layers

5. **analysis_report.txt** - Text summary
   - Key findings in plain language
   - Interpretation and insights

### Data (in `results/mechanistic_analysis/layer_probing/`)

- **layer_statistics.csv** - Main results per layer
- **detailed_layer_results.csv** - Per-example scores
- **summary.json** - Quick summary

## 🎨 View Results

```bash
# Open main visualization
open images/layer_probing/layer_performance.png

# Read text summary
cat images/layer_probing/analysis_report.txt

# View data
python3 -c "import pandas as pd; df = pd.read_csv('results/mechanistic_analysis/layer_probing/layer_statistics.csv'); print(df.head(10))"
```

## ⚙️ Configuration

Edit `scripts/mechanistic_analysis/run_layer_probing.sh`:

```bash
# Number of samples (increase for more comprehensive analysis)
MAX_SAMPLES=10  # Default
# MAX_SAMPLES=50  # More thorough

# Specific conditions (optional)
# CONDITIONS="stripping_VPE joins"  # Focus on specific types
```

## 📋 Requirements Checklist

- [x] Python 3.11+
- [x] GPU with CUDA (recommended) or CPU
- [x] 16GB+ RAM
- [x] Hugging Face account with Llama-2 access
- [x] HF token configured (`huggingface-cli login`)

## ⚠️ Common Issues

### Issue: "Model not found"
**Solution:**
```bash
huggingface-cli login
# Then request access at: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
```

### Issue: "Out of memory"
**Solution:**
```bash
# Use CPU (slower but works)
python scripts/mechanistic_analysis/layer_wise_probing.py --device cpu
```

### Issue: "Too slow"
**Solution:**
```bash
# Reduce samples in run_layer_probing.sh:
MAX_SAMPLES=3
```

## 📖 Documentation

- **Full Guide:** `scripts/mechanistic_analysis/LAYER_PROBING_README.md` (450 lines)
- **Complete Summary:** `LAYER_PROBING_SUMMARY.md`
- **Main README:** Updated with layer probing section

## 🔬 What You'll Learn

After running the analysis, you'll know:

1. **Best layer for anaphora encoding**
   - Example: "Layer 18 shows peak performance"

2. **Processing pattern**
   - Localized (one peak) or distributed (multiple peaks)?

3. **Region analysis**
   - Early, middle, or late layer processing?

4. **Condition differences**
   - Do different anaphora types use different layers?

## 📈 Expected Output Example

```
============================================
ANALYSIS COMPLETE!
============================================

Model: meta-llama/Llama-2-7b-chat-hf
Total layers analyzed: 32
Examples processed: 80

🎯 Best performing layer: Layer 18
   Score: 0.4234

Top 5 layers:
  1. Layer 18: Score = 0.4234, Accuracy = 78.5%
  2. Layer 19: Score = 0.4123, Accuracy = 76.2%
  3. Layer 17: Score = 0.3987, Accuracy = 75.8%
  4. Layer 20: Score = 0.3845, Accuracy = 74.1%
  5. Layer 16: Score = 0.3723, Accuracy = 72.9%
```

## 🎓 Research Insight

This tells you:
- **Where** anaphora resolution happens (layer 18 in middle region)
- **How well** each layer encodes this information (accuracy %)
- **Pattern** of processing (gradual vs. sudden)

## 🚀 Advanced Usage

### Compare Models

```bash
# Base model
python scripts/mechanistic_analysis/layer_wise_probing.py \
    --model "meta-llama/Llama-2-7b-hf" \
    --output_dir "results/layer_probing_base"

# Chat model
python scripts/mechanistic_analysis/layer_wise_probing.py \
    --model "meta-llama/Llama-2-7b-chat-hf" \
    --output_dir "results/layer_probing_chat"
```

### Focus on Specific Anaphora Types

```bash
python scripts/mechanistic_analysis/layer_wise_probing.py \
    --conditions stripping_VPE joins \
    --max_samples 20
```

## 💡 Tips

1. **Start with demo** - Verify everything works
2. **Use GPU** - Much faster (15 min vs 2 hours)
3. **Start small** - MAX_SAMPLES=10 for first run
4. **Check visualizations** - They're designed to be publication-ready
5. **Read the report** - Plain language interpretation included

## 📞 Help

- **Email:** dalshareif@aimsammi.org
- **Issues:** Run demo for quick testing
- **Docs:** See `LAYER_PROBING_README.md` for full details

---

**Ready to find out which layers encode anaphoric information?**

```bash
bash scripts/mechanistic_analysis/run_layer_probing.sh
```

**Let's go! 🚀**
