CARDROOM_RUST_ROOT := ./cardroom
CARDROOM_GODOT_ROOT := ./cardroom-godot

# compile the rust codebase
build:
	cd $(CARDROOM_RUST_ROOT) && cargo build

# release the rust codebase
release:
	cd $(CARDROOM_RUST_ROOT) && cargo build --release

# import godot assets (textures, scenes) into the .godot cache
import: build
	mkdir -p $(CARDROOM_GODOT_ROOT)/.godot
	echo "res://cardroom.gdextension" > $(CARDROOM_GODOT_ROOT)/.godot/extension_list.cfg
	godot --headless --path $(CARDROOM_GODOT_ROOT) --import

# run the godot game local without exporting
play: import
	godot --path $(CARDROOM_GODOT_ROOT)