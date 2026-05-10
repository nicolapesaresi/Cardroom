# Cardroom

This project implements classic italian Card Games and AI agents to play against. The game env and agents are implemented in Rust, while the playing interface is made in Godot.
Work in progress.

## Installation and setup

To run the game you will need `rustc`, `cargo` and **Godot 4.x**, which can be installed following the instructions on [rustup.rs](https://rustup.rs) and [godotengine.org](https://godotengine.org/download).
To run the game with a `make` command, without opening the editor, you will need to expose Godot to the terminal in the following way:
**macOS** 
- make sure you moved the Godot app to `/Applications`.
- add the Godot binary to your PATH so the Makefile can find it, by
adding this line to your `~/.zshrc`:
```bash
export PATH="/Applications/Godot.app/Contents/MacOS:$PATH"
```
- reload the terminal:
```bash
source ~/.zshrc
```

**Linux** — symlink the binary:
```bash
sudo ln -s /path/to/Godot /usr/local/bin/godot
```
Clone the repository and enter the directory:
```bash
git clone https://github.com/nicolapesaresi/Cardroom.git
cd Cardroom
```
To run the game, enter:
```bash
make play
```

### Development

You can compile the rust codebase in development mode from the root folder with:
```bash
make build
```
Alternatively, you can enter the game subfolder and run the main script to play via text in the terminal or run other scripts.
```bash
cd cardroom/briscola
cargo build                            # compile in debug mode
cargo run                              # compile and run main.rs
cargo run --example bot_vs_random      # compile and run a script in /examples
```