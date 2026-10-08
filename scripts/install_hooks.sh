#!/bin/sh
# Install PhilanthroPy git hooks for this checkout.
# Run once after cloning: sh scripts/install_hooks.sh

set -e
HOOKS_DIR="$(git rev-parse --git-dir)/hooks"

cat > "$HOOKS_DIR/pre-push" << 'EOF'
#!/bin/sh
# PhilanthroPy pre-push hook
# Fast checks only (seconds, not minutes). CI runs the full suite on every
# PR, so repeating it here only delays the push; run `make ci` before asking
# for review instead.

set -e

echo "▶ Running pre-push checks..."

echo "  [1/2] Checking for collection errors..."
if ! python -m pytest tests/ --collect-only -q >/dev/null 2>&1; then
    echo "✗ Collection errors found. Fix imports before pushing."
    python -m pytest tests/ --collect-only -q 2>&1 | tail -20
    exit 1
fi

echo "  [2/2] Linting (flake8)..."
python -m flake8 philanthropy tests examples

echo "✓ All checks passed. Proceeding with push."
EOF

chmod +x "$HOOKS_DIR/pre-push"
echo "✓ pre-push hook installed."
