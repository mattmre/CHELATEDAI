# PowerShell Script to Prepare PR for Phase 1-3 Refactoring
# Run this script, then open GitHub Desktop to review and create the PR

Write-Host "=== ChelatedAI Phase 1-3 Refactoring PR Preparation ===" -ForegroundColor Cyan
Write-Host ""

# Check if we're in a git repository
if (-not (Test-Path ".git")) {
    Write-Host "ERROR: Not a git repository. Initializing..." -ForegroundColor Red
    git init
    Write-Host "Git repository initialized." -ForegroundColor Green
}

# Check current branch
$currentBranch = git branch --show-current
Write-Host "Current branch: $currentBranch" -ForegroundColor Yellow

# Create feature branch if not already on it
$featureBranch = "refactor/phase-1-2-3-production-hardening"
if ($currentBranch -ne $featureBranch) {
    Write-Host "Creating feature branch: $featureBranch" -ForegroundColor Yellow

    # Check if branch already exists
    $branchExists = git branch --list $featureBranch
    if ($branchExists) {
        Write-Host "Branch already exists, switching to it..." -ForegroundColor Yellow
        git checkout $featureBranch
    } else {
        git checkout -b $featureBranch
    }
} else {
    Write-Host "Already on feature branch: $featureBranch" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Staging New Files ===" -ForegroundColor Cyan

# Stage new files
$newFiles = @(
    "config.py",
    "checkpoint_manager.py",
    "chelation_logger.py",
    "test_unit_core.py",
    "README.md",
    "TECHNICAL_ANALYSIS.md",
    "REFACTORING_PLAN.md",
    "CHANGELOG.md",
    "COMPLETION_SUMMARY.md"
)

foreach ($file in $newFiles) {
    if (Test-Path $file) {
        git add $file
        Write-Host "✓ Staged: $file" -ForegroundColor Green
    } else {
        Write-Host "✗ Not found: $file" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=== Staging Modified Files ===" -ForegroundColor Cyan

# Stage modified files
$modifiedFiles = @(
    "antigravity_engine.py",
    "benchmark_evolution.py",
    "chelation_adapter.py"
)

foreach ($file in $modifiedFiles) {
    if (Test-Path $file) {
        git add $file
        Write-Host "✓ Staged: $file" -ForegroundColor Green
    } else {
        Write-Host "✗ Not found: $file" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=== Creating .gitignore (if needed) ===" -ForegroundColor Cyan

# Create .gitignore if it doesn't exist
if (-not (Test-Path ".gitignore")) {
    @"
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
*.egg-info/
dist/
build/

# Testing
.pytest_cache/
.coverage
htmlcov/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
*.backup
adapter_weights.pt
demo_log.jsonl
chelation_debug.jsonl
checkpoints/
db_*/
*.log

# Keep manual results for reference
!manual_results.txt

# Temp files
*.tmp
temp_*
"@ | Out-File -FilePath ".gitignore" -Encoding UTF8

    git add .gitignore
    Write-Host "✓ Created and staged .gitignore" -ForegroundColor Green
} else {
    Write-Host "✓ .gitignore already exists" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Git Status ===" -ForegroundColor Cyan
git status

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan
Write-Host "Branch: $featureBranch" -ForegroundColor Green
Write-Host "Files staged and ready for commit" -ForegroundColor Green
Write-Host ""
Write-Host "=== Next Steps ===" -ForegroundColor Yellow
Write-Host "1. Open GitHub Desktop" -ForegroundColor White
Write-Host "2. Review the changes in the 'Changes' tab" -ForegroundColor White
Write-Host "3. Commit with the message below:" -ForegroundColor White
Write-Host ""
Write-Host "--- COMMIT MESSAGE (copy this) ---" -ForegroundColor Cyan
Write-Host @"
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
"@ -ForegroundColor White

Write-Host "--- END COMMIT MESSAGE ---" -ForegroundColor Cyan
Write-Host ""
Write-Host "4. After committing, click 'Push origin' to push to GitHub" -ForegroundColor White
Write-Host "5. Click 'Create Pull Request' in GitHub Desktop" -ForegroundColor White
Write-Host "6. Use PR description from PR_DESCRIPTION.md (being created...)" -ForegroundColor White
Write-Host ""
Write-Host "Press any key to continue and create PR_DESCRIPTION.md..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
