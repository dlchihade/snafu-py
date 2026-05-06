#!/bin/bash
# Script to push all semantic fluency analysis code to GitHub

set -e

echo "🚀 Preparing to push all scripts to GitHub..."
echo "=============================================="

# Navigate to project root
cd "$(dirname "$0")/../.."

# Check git status
echo ""
echo "📋 Current git status:"
git status --short | head -20

# Show remotes
echo ""
echo "🔗 Available remotes:"
git remote -v

# Ask which remote to use
echo ""
echo "Which remote would you like to push to?"
echo "1) fork (dlchihade/snafu-py)"
echo "2) origin (AusterweilLab/snafu-py)"
read -p "Enter choice (1 or 2, default: 1): " remote_choice

if [ "$remote_choice" = "2" ]; then
    REMOTE="origin"
    BRANCH="main"
else
    REMOTE="fork"
    BRANCH="reorganized-structure"
fi

echo ""
echo "📦 Adding all Python scripts and documentation..."
git add research/semantic_fluency_analysis/*.py
git add research/semantic_fluency_analysis/*.md
git add research/semantic_fluency_analysis/src/*.py

# Add modified files
echo ""
echo "📝 Adding modified files..."
git add research/semantic_fluency_analysis/create_exploit_explore_bar.py
git add research/semantic_fluency_analysis/create_nature_quality_figures_real.py
git add research/semantic_fluency_analysis/mediation_disease_stage_working.py
git add research/semantic_fluency_analysis/mediation_figures_nature.py

# Check what will be committed
echo ""
echo "📋 Files to be committed:"
git status --short | grep "^A\|^M" | head -30

# Ask for confirmation
echo ""
read -p "Continue with commit? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "❌ Cancelled."
    exit 1
fi

# Commit
echo ""
echo "💾 Committing changes..."
git commit -m "Add semantic fluency analysis scripts and documentation

- Add all Python scripts for figure generation and analysis
- Add comprehensive code documentation (CODE_DOCUMENTATION.md)
- Add figures documentation (FIGURES_DOCUMENTATION.md)
- Add methods description (METHODS_DESCRIPTION.md)
- Add compilation scripts for PowerPoint presentations
- Update figure generation to exclude zero values (N=45)
- Add confidence interval and E-E index explanations"

# Push
echo ""
echo "⬆️  Pushing to $REMOTE/$BRANCH..."
git push $REMOTE $BRANCH

echo ""
echo "✅ Successfully pushed to GitHub!"
echo "   Repository: $REMOTE"
echo "   Branch: $BRANCH"
echo ""
echo "🔗 View at: https://github.com/$(git remote get-url $REMOTE | sed 's/.*github.com[:/]\(.*\)\.git/\1/')"

