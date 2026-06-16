#!/bin/bash
# Push SocialLLM project files to github.com/erfanzabeh/LLMsocial
# Usage: bash push_to_github.sh

set -e
cd "$(dirname "$0")"

# Remove stale lock file
rm -f .git/index.lock 2>/dev/null || echo "Note: Could not remove index.lock — delete it manually if commit fails"

git config user.email "erfanzabeh1@gmail.com"
git config user.name "Erfan Zabeh"
git branch -M main

# Stage everything
git add .

git commit -m "Add SocialLLM framework

- Social interaction environment (agents, tasks, environment)
- Activation extraction with PyTorch forward hooks (HDF5 storage)
- Neuroscience-style selectivity analysis (ANOVA, SI, d-prime, sparseness)
- Population decoding (linear classifier, PCA, UMAP, RSA)
- Visualization suite (tuning curves, RDMs, geometry plots)
- Experiment runner scripts and config
- Same-model / persona-only interlocutor design"

git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/erfanzabeh/LLMsocial.git

echo ""
echo "Pushing to github.com/erfanzabeh/LLMsocial ..."
git push --force -u origin main
echo ""
echo "Done! https://github.com/erfanzabeh/LLMsocial"
