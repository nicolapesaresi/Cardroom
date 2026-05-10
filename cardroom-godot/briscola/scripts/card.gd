extends Node

signal clicked(action_idx: int)

var face: int
var suit: String
var covered: bool
var playable: bool = false
var action_idx: int = -1

func init(_face: int = 1, _suit: String = "ori", _covered: bool = true) -> void:
	face = _face
	suit = _suit
	covered = _covered
	update_visuals()

func set_playable(_action_idx: int) -> void:
	playable = true
	action_idx = _action_idx

func update_visuals() -> void:
	if self.covered:
		$TextureRect.texture = load("res://assets/cards/bergamasche/retro.png")
	else:
		$TextureRect.texture = load("res://assets/cards/bergamasche/%s_%s.png" % [suit, face])

func _ready() -> void:
	$Area2D.input_pickable = true
	$Area2D.input_event.connect(_on_clicked)

func _on_clicked(_viewport, event, _shape_idx) -> void:
	if not playable:
		return
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
			playable = false
			clicked.emit(action_idx)
