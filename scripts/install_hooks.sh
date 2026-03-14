#!/bin/sh
# Install PhilanthroPy git hooks for this checkout.
# Run once after cloning: sh scripts/install_hooks.sh

set -e
HOOKS_DIR="$(git rev-parse --git-dir)/hooks"

cat > "$HOOKS_DIR/pre-push" << 'EOF'
#!/bin/sh
# PhilanthroPy pre-push hook
# Runs the full test suite before every push.
# To bypass in an emergency: git push --no-verify

set -e

echo "▶ Running pre-push checks..."

echo "  [1/2] Checking for collection errors..."
if ! python -m pytest tests/ --collect-only -q 2>/dev/null; then
    echo "✗ Collection errors found. Fix imports before pushing."
    python -m pytest tests/ --collect-only -q 2>&1 | tail -20
    exit 1
fi

echo "  [2/2] Running full test suite..."
python -m pytest tests/ -x --tb=short -q

echo "✓ All checks passed. Proceeding with push."
EOF

chmod +x "$HOOKS_DIR/pre-push"
echo "✓ pre-push hook installed."
