#!/bin/bash
# ============================================================
# Push learning-systems-from-scratch to github.com/mldsa
# Run once from inside this folder
# ============================================================

set -e

ORG="mldsa"
REPO="learning-systems-from-scratch"
REMOTE="https://github.com/$ORG/$REPO.git"

echo ""
echo "  From LeetCode to Learning Systems"
echo "  → github.com/$ORG/$REPO"
echo ""

# 1. Init git
git init
git checkout -b main

# 2. Stage + commit
git add .
git commit -m "Launch: Chapters 1–3, dual index, Jekyll book

- Part 0: The Asymmetric Candidate
- Ch 1: Hashing, Token Counting, and LLM Context (+ Colab notebook)
- Ch 2: Top-K, Vector Search, and RAG Retrieval (+ Colab notebook)
- Ch 3: Dynamic Programming — Beam Search, Viterbi, RL, ROUGE
- DSA → ML Index: every algorithm mapped to its ML system
- ML → DSA Index: every ML concept mapped to its algorithm
- Algo Depth Track: 4-stage reading path for engineers
- ML Depth Track: 7-stage reading path for ML practitioners
- Jekyll (Just the Docs) + GitHub Actions auto-deploy
- #MLDSA series Day 1"

# 3. Create the repo on GitHub first
echo ""
echo "  ⚠️  Before pushing:"
echo "  1. Go to https://github.com/organizations/mldsa/repositories/new"
echo "  2. Name: $REPO"
echo "  3. Visibility: Public"
echo "  4. Do NOT initialize (no README, no .gitignore)"
echo "  5. Click 'Create repository'"
echo ""
read -p "  Press Enter once the repo exists on GitHub..."

# 4. Push
git remote add origin "$REMOTE"
git push -u origin main

echo ""
echo "  ✅  Pushed to https://github.com/$ORG/$REPO"
echo ""
echo "  ── Enable GitHub Pages ───────────────────────────"
echo "  1. https://github.com/$ORG/$REPO/settings/pages"
echo "  2. Source: GitHub Actions"
echo "  3. Save"
echo ""
echo "  Book will be live at:"
echo "  https://$ORG.github.io/$REPO/"
echo ""
echo "  First deploy: ~2 min   Subsequent pushes: ~45 sec"
echo "  ──────────────────────────────────────────────────"
