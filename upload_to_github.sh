#!/bin/bash

# GitHub Upload Script for snafu-py Repository
# Quick upload script for pushing changes to GitHub

set -e  # Exit on any error

echo "🚀 Starting GitHub upload for snafu-py repository..."
echo "=================================================="

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: Not in a Git repository. Please run this script from the snafu-py directory."
    exit 1
fi

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "📍 Current branch: $CURRENT_BRANCH"

# Check for changes
if [ -z "$(git status --porcelain)" ]; then
    echo "✅ No changes to commit. Repository is up to date."
    exit 0
fi

# Show changes
echo "📋 Changes detected:"
git status --short

# Stage all changes
echo "🔄 Staging all changes..."
git add .

# Get commit message
if [ -z "$1" ]; then
    COMMIT_MSG="Update repository - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "📝 Using default commit message: $COMMIT_MSG"
else
    COMMIT_MSG="$1"
    echo "📝 Using provided commit message: $COMMIT_MSG"
fi

# Commit changes
echo "🔄 Committing changes..."
git commit -m "$COMMIT_MSG"

# Push to GitHub
echo "🚀 Pushing to GitHub..."
if git push origin "$CURRENT_BRANCH"; then
    echo ""
    echo "✅ Successfully uploaded to GitHub!"
    echo "📍 Repository: https://github.com/AusterweilLab/snafu-py"
    echo "📍 Branch: $CURRENT_BRANCH"
else
    echo ""
    echo "❌ Failed to push to GitHub. Please check your credentials and try again."
    echo "💡 You may need to:"
    echo "   1. Set up SSH keys or use a personal access token"
    echo "   2. Configure your Git credentials"
    echo "   3. Check if you have write access to the repository"
    exit 1
fi

