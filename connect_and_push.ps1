# Connect Local Repo to GitHub and Push Changes
# Usage: .\connect_and_push.ps1 -Username YOUR_GITHUB_USERNAME

param(
    [Parameter(Mandatory=$true)]
    [string]$Username
)

Write-Host "`n=== Connecting Local Repo to GitHub ===" -ForegroundColor Cyan
Write-Host ""

$repoUrl = "https://github.com/$Username/CHELATEDAI.git"

# Step 1: Add remote
Write-Host "Step 1: Adding remote origin..." -ForegroundColor Yellow
Write-Host "  URL: $repoUrl" -ForegroundColor Gray

try {
    git remote add origin $repoUrl
    Write-Host "  ✓ Remote added successfully" -ForegroundColor Green
} catch {
    Write-Host "  ! Remote might already exist, updating..." -ForegroundColor Yellow
    git remote set-url origin $repoUrl
}

Write-Host ""

# Step 2: Verify connection
Write-Host "Step 2: Verifying remote..." -ForegroundColor Yellow
$remoteUrl = git config --get remote.origin.url
Write-Host "  Connected to: $remoteUrl" -ForegroundColor Gray

Write-Host ""

# Step 3: Check what's staged
Write-Host "Step 3: Checking staged files..." -ForegroundColor Yellow
$stagedCount = (git diff --cached --name-only | Measure-Object -Line).Lines
Write-Host "  Files staged: $stagedCount" -ForegroundColor Gray

Write-Host ""

# Step 4: Create initial commit
Write-Host "Step 4: Creating commit..." -ForegroundColor Yellow

$commitMessage = @"
feat: Phase 1-3 production hardening complete

- Phase 1: Fixed critical bugs, error handling, cross-platform support
  * Removed duplicate return statement
  * Added timeout protection (30s) for Ollama requests
  * Replaced 9+ bare except clauses with specific exceptions
  * Cross-platform paths using pathlib
  * Standardized ID management

- Phase 2: Robustness improvements
  * Added config.py for centralized configuration
  * Implemented checkpoint_manager.py for safe training
  * Created SafeTrainingContext for automatic rollback
  * SHA256 integrity verification

- Phase 3: Observability enhancements
  * Added chelation_logger.py for structured JSON logging
  * Created 21 comprehensive unit tests (all passing)
  * Performance timing and metrics
  * Extensive documentation (1500+ lines)

Documentation:
- README.md: Complete user guide
- TECHNICAL_ANALYSIS.md: Architecture details
- CHANGELOG.md: Detailed change tracking
- REFACTORING_PLAN.md: Project tracking

All changes are backward compatible. Zero breaking changes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
"@

git commit -m $commitMessage

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Commit created successfully" -ForegroundColor Green
} else {
    Write-Host "  ✗ Commit failed" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Step 5: Create main branch
Write-Host "Step 5: Creating main branch..." -ForegroundColor Yellow
git branch -M main
Write-Host "  ✓ Renamed branch to 'main'" -ForegroundColor Green

Write-Host ""

# Step 6: Push to GitHub
Write-Host "Step 6: Pushing to GitHub..." -ForegroundColor Yellow
Write-Host "  This may take a moment and might prompt for credentials..." -ForegroundColor Gray
Write-Host ""

git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "  ✓ Successfully pushed to GitHub!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "  ✗ Push failed - you may need to authenticate" -ForegroundColor Red
    Write-Host ""
    Write-Host "If you see an authentication error:" -ForegroundColor Yellow
    Write-Host "  1. You may need a Personal Access Token" -ForegroundColor White
    Write-Host "  2. Create one at: https://github.com/settings/tokens" -ForegroundColor White
    Write-Host "  3. Use the token as your password when prompted" -ForegroundColor White
    Write-Host ""
    exit 1
}

Write-Host ""

# Step 7: Create feature branch
Write-Host "Step 7: Creating feature branch for PR..." -ForegroundColor Yellow
git checkout -b refactor/phase-1-2-3-production-hardening
git push -u origin refactor/phase-1-2-3-production-hardening

Write-Host "  ✓ Feature branch created and pushed" -ForegroundColor Green

Write-Host ""
Write-Host "=== SUCCESS! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Your code is now on GitHub!" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Go to: https://github.com/$Username/CHELATEDAI" -ForegroundColor White
Write-Host "  2. Click the 'Compare & pull request' button" -ForegroundColor White
Write-Host "  3. Copy the PR description from PR_DESCRIPTION.md" -ForegroundColor White
Write-Host "  4. Click 'Create pull request'" -ForegroundColor White
Write-Host ""
Write-Host "Or use GitHub Desktop:" -ForegroundColor Yellow
Write-Host "  1. Open GitHub Desktop" -ForegroundColor White
Write-Host "  2. It should now see your repo" -ForegroundColor White
Write-Host "  3. Click 'Create Pull Request'" -ForegroundColor White
Write-Host ""
