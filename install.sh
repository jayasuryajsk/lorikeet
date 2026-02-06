#!/bin/bash
set -e

echo "Building lorikeet..."
cargo build --release

echo ""
echo "Downloading embedding model (~22MB)..."
# Run a quick initialization to download the model
./target/release/lorikeet index . 2>/dev/null || true

echo ""
echo "Installing to ~/bin..."
mkdir -p ~/bin
cp target/release/lorikeet ~/bin/
chmod +x ~/bin/lorikeet

# Add to PATH if not already there
if ! echo "$PATH" | grep -q "$HOME/bin"; then
    SHELL_RC=""
    if [ -f ~/.zshrc ]; then
        SHELL_RC=~/.zshrc
    elif [ -f ~/.bashrc ]; then
        SHELL_RC=~/.bashrc
    elif [ -f ~/.bash_profile ]; then
        SHELL_RC=~/.bash_profile
    fi

    if [ -n "$SHELL_RC" ]; then
        echo 'export PATH="$HOME/bin:$PATH"' >> "$SHELL_RC"
        echo "Added ~/bin to PATH in $SHELL_RC"
    fi
fi

echo ""
echo "Done! Run 'lorikeet' to start (open new terminal or run: source ~/.zshrc)"
