CARDROOM_RUST_ROOT := ./cardroom
CARDROOM_GODOT_ROOT := ./cardroom-godot

# compile the rust codebase
build:
	cd $(CARDROOM_RUST_ROOT) && cargo build

# release the rust codebase
release:
	cd $(CARDROOM_RUST_ROOT) && cargo build --release

# run the godot game local without exporting
play: build
	godot --path $(CARDROOM_GODOT_ROOT)