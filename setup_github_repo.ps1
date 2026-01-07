# PowerShell Script to Setup GitHub Repository and Create PR
# Run this after creating the repository on GitHub.com

param(
    [Parameter(Mandatory=$true)]
    [string]$GithubUsername
)

Write-Host "=== ChelatedAI GitHub Repository Setup ===" -ForegroundColor Cyan
Write-Host ""

$repoUrl = "https://github.com/$GithubUsername/CHELATEDAI.git"

Write-Host "Step 1: Verifying git repository..." -ForegroundColor Yellow
if (-not (Test-Path ".git")) {
    Write-Host "ERROR: Not a git repository!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Git repository found" -ForegroundColor Green

Write-Host ""
Write-Host "Step 2: Adding remote origin..." -ForegroundColor Yellow
$currentRemote = git remote get-url origin 2>$null
if ($currentRemote) {
    Write-Host "Remote already exists: $currentRemote" -ForegroundColor Yellow
    Write-Host "Updating remote URL..." -ForegroundColor Yellow
    git remote set-url origin $repoUrl
} else {
    git remote add origin $repoUrl
}
Write-Host "✓ Remote origin set to: $repoUrl" -ForegroundColor Green

Write-Host ""
Write-Host "Step 3: Creating and pushing main branch..." -ForegroundColor Yellow
$currentBranch = git branch --show-current

# Check if we need to create main branch
$mainExists = git branch --list main
if (-not $mainExists) {
    Write-Host "Creating main branch from current state..." -ForegroundColor Yellow

    # Commit current changes if there are any staged
    $staged = git diff --cached --name-only
    if ($staged) {
        Write-Host "Committing staged changes..." -ForegroundColor Yellow
        git commit -m "feat: Phase 1-3 production hardening complete

- Phase 1: Fixed critical bugs, error handling, cross-platform support
- Phase 2: Configuration management and checkpoint/rollback system
- Phase 3: Structured logging and 21 comprehensive unit tests

All changes are backward compatible. Zero breaking changes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
    }

    # Create main branch from current branch
    git branch main
}

# Switch to main
git checkout main

# Push main branch
Write-Host "Pushing main branch to GitHub..." -ForegroundColor Yellow
git push -u origin main

Write-Host "✓ Main branch pushed" -ForegroundColor Green

Write-Host ""
Write-Host "Step 4: Switching back to feature branch..." -ForegroundColor Yellow
git checkout refactor/phase-1-2-3-production-hardening

Write-Host ""
Write-Host "Step 5: Pushing feature branch..." -ForegroundColor Yellow
git push -u origin refactor/phase-1-2-3-production-hardening
Write-Host "✓ Feature branch pushed" -ForegroundColor Green

Write-Host ""
Write-Host "=== Setup Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Repository URL: https://github.com/$GithubUsername/CHELATEDAI" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "1. Go to: https://github.com/$GithubUsername/CHELATEDAI" -ForegroundColor White
Write-Host "2. Click 'Pull requests' tab" -ForegroundColor White
Write-Host "3. Click 'New pull request'" -ForegroundColor White
Write-Host "4. Set:" -ForegroundColor White
Write-Host "   - Base: main" -ForegroundColor White
Write-Host "   - Compare: refactor/phase-1-2-3-production-hardening" -ForegroundColor White
Write-Host "5. Copy title and description from PR_DESCRIPTION.md" -ForegroundColor White
Write-Host "6. Click 'Create pull request'" -ForegroundColor White
Write-Host ""
Write-Host "OR use GitHub Desktop:" -ForegroundColor Yellow
Write-Host "1. Open GitHub Desktop" -ForegroundColor White
Write-Host "2. It should now show your repository connected to GitHub" -ForegroundColor White
Write-Host "3. Click 'Create Pull Request'" -ForegroundColor White
Write-Host ""
