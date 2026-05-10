use godot::prelude::*;

mod briscola_game;
mod briscola_agents;

struct CardroomExtension;

#[gdextension]
unsafe impl ExtensionLibrary for CardroomExtension {}
