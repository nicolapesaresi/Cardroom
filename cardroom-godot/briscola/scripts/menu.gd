extends Control

@onready var p0_option: OptionButton = $CenterContainer/VBoxContainer/P0HBox/P0Option
@onready var p1_option: OptionButton = $CenterContainer/VBoxContainer/P1HBox/P1Option
@onready var start_button: Button = $CenterContainer/VBoxContainer/StartButton


func _ready() -> void:
	for agent in GameConfig.AGENTS:
		p0_option.add_item(agent)
		p1_option.add_item(agent)
	p0_option.select(GameConfig.AGENTS.find(GameConfig.p0_agent))
	p1_option.select(GameConfig.AGENTS.find(GameConfig.p1_agent))
	start_button.pressed.connect(_on_start_pressed)


func _on_start_pressed() -> void:
	GameConfig.p0_agent = GameConfig.AGENTS[p0_option.selected]
	GameConfig.p1_agent = GameConfig.AGENTS[p1_option.selected]
	get_tree().change_scene_to_file("res://briscola/scenes/game.tscn")
