extends Node2D

const Card = preload("res://briscola/scenes/card.tscn")

const AI_DELAY := 0.6
const PLAY_DUR := 0.35
const TAKE_DUR := 0.45
const DEAL_DUR := 0.30

signal human_action_selected(action_idx: int)

var env: BriscolaEnvNode
var agents: Array = [null, null]

# tracked visuals (persist across turns; updated incrementally)
var hand_visuals: Array = [[], []]
var table_visuals: Array = []
var deck_visuals: Array = []
var deck_label: Label = null
var taken_pile_visuals: Array = [[], []]
var points_labels: Array = [null, null]
var name_labels: Array = [null, null]
var current_dot: ColorRect = null
var dot_positions: Array = [Vector2.ZERO, Vector2.ZERO]
var result_panel: Node = null

var card_width: float = 0.0
var card_height: float = 0.0

var HAND_CENTER = DisplayServer.screen_get_size()[0] / 2
var HAND_P0_Y = DisplayServer.screen_get_size()[1] * 0.85
var HAND_P1_Y = DisplayServer.screen_get_size()[1] * 0.15
var HAND_CARDS_OFFSET = DisplayServer.screen_get_size()[1] * 0.05
var DECK_X = DisplayServer.screen_get_size()[0] * 0.10
var DECK_Y = DisplayServer.screen_get_size()[1] * 0.4
var DECK_Y_OFFSET = DisplayServer.screen_get_size()[1] * 0.001
var CARD_P0_X = DisplayServer.screen_get_size()[0] * 0.55
var CARD_P0_Y = DisplayServer.screen_get_size()[1] * 0.55
var CARD_P1_X = DisplayServer.screen_get_size()[0] * 0.45
var CARD_P1_Y = DisplayServer.screen_get_size()[1] * 0.45
var TAKEN_X = DisplayServer.screen_get_size()[0] * 0.82


func _ready() -> void:
	env = BriscolaEnvNode.new()
	add_child(env)
	env.init_with_names(PackedStringArray([GameConfig.p0_agent, GameConfig.p1_agent]))

	agents[0] = _make_agent_node(GameConfig.p0_agent)
	agents[1] = _make_agent_node(GameConfig.p1_agent)
	for a in agents:
		if a != null:
			add_child(a)

	await _initial_render()
	while not env.done():
		var state = env.get_state()
		var current: int = state["current_player_idx"]
		_setup_clickable(current)
		var action: int
		if agents[current] == null:
			action = await human_action_selected
		else:
			await get_tree().create_timer(AI_DELAY).timeout
			action = agents[current].select_action(env)
		_disable_clickable()
		await _animate_step(action)
	await _show_result()


func _make_agent_node(agent_name: String) -> Node:
	match agent_name:
		GameConfig.RANDOM:
			return RandomAgentNode.new()
		GameConfig.BOT:
			return BotAgentNode.new()
		_:
			return null


func _setup_clickable(human_player: int) -> void:
	if agents[human_player] != null:
		return
	var legal = env.get_state()["legal_actions"]
	for i in range(hand_visuals[human_player].size()):
		var c = hand_visuals[human_player][i]
		if not is_instance_valid(c):
			continue
		if i in legal:
			c.set_playable(i)
			if not c.clicked.is_connected(_on_card_clicked):
				c.clicked.connect(_on_card_clicked)


func _disable_clickable() -> void:
	for p in [0, 1]:
		for c in hand_visuals[p]:
			if is_instance_valid(c):
				c.playable = false


func _initial_render() -> void:
	await _measure_card()

	var state = env.get_state()
	var players: Array = state["players"]

	await _create_name_label(0, players[0]["name"])
	await _create_name_label(1, players[1]["name"])
	await _create_points_labels(players)
	await _draw_initial_deck(state["deck"])
	_create_current_dot(state["current_player_idx"])

	await _animate_initial_deal(state)


func _measure_card() -> void:
	var sample = Card.instantiate()
	add_child(sample)
	sample.position = Vector2(-10000, -10000)
	sample.init(1, "ori", true)
	await get_tree().process_frame
	card_width = sample.get_node("TextureRect").size.x
	card_height = sample.get_node("TextureRect").size.y
	sample.queue_free()


func _create_name_label(player_idx: int, name: String) -> void:
	var name_y = HAND_P0_Y - card_height / 1.25 if player_idx == 0 else HAND_P1_Y + card_height / 1.6
	var label = Label.new()
	add_child(label)
	label.text = "%s" % name
	label.add_theme_font_size_override("font_size", 22)
	await get_tree().process_frame
	label.position = Vector2(HAND_CENTER - label.size.x / 2.0, name_y)
	name_labels[player_idx] = label

	var dot_size = 10.0
	dot_positions[player_idx] = Vector2(label.position.x - dot_size * 2.0, name_y + label.size.y / 2.0 - dot_size / 2.0)


func _draw_initial_deck(deck) -> void:
	# inflate the deck by 6 placeholder cards to represent the to-be-dealt cards
	deck_visuals = []
	deck_label = null
	var n_extra = 6
	var total = deck.size() + n_extra
	if total == 0:
		return
	if deck.size() > 0:
		var spy = Card.instantiate()
		add_child(spy)
		spy.init(deck[0]["number"], deck[0]["suit"], false)
		spy.position = Vector2(DECK_X, DECK_Y + 75 * DECK_Y_OFFSET)
		deck_visuals.append(spy)
	for i in range(1, deck.size()):
		var card = Card.instantiate()
		add_child(card)
		card.init(deck[i]["number"], deck[i]["suit"], true)
		card.position = Vector2(DECK_X, DECK_Y - i * DECK_Y_OFFSET)
		card.rotation_degrees = 90.0
		deck_visuals.append(card)
	for j in range(n_extra):
		var card = Card.instantiate()
		add_child(card)
		card.init(1, "ori", true)
		var i = deck.size() + j
		card.position = Vector2(DECK_X, DECK_Y - i * DECK_Y_OFFSET)
		card.rotation_degrees = 90.0
		deck_visuals.append(card)

	deck_label = Label.new()
	add_child(deck_label)
	deck_label.text = "%d" % total
	await get_tree().process_frame
	deck_label.position = Vector2(DECK_X - 10, DECK_Y + 200 * DECK_Y_OFFSET)


func _animate_initial_deal(state) -> void:
	var deck_pos = Vector2(DECK_X, DECK_Y)
	var turn_order = state["turn_order"]
	var players: Array = state["players"]

	for slot_idx in range(3):
		var t = create_tween().set_parallel(true)
		var movers: Array = []
		var n_dealt = 0
		for player_id in turn_order:
			var hand: Array = players[player_id]["hand"]
			if slot_idx >= hand.size():
				continue
			var card_dict = hand[slot_idx]
			var moving = _spawn_card(card_dict, deck_pos, false)
			movers.append({"card": moving, "player": player_id})
			n_dealt += 1

			var hand_size_after = slot_idx + 1
			t.tween_property(moving, "position", _hand_card_pos(player_id, slot_idx, hand_size_after), DEAL_DUR)
			for i in range(hand_visuals[player_id].size()):
				var c = hand_visuals[player_id][i]
				if is_instance_valid(c):
					t.tween_property(c, "position", _hand_card_pos(player_id, i, hand_size_after), DEAL_DUR)

		if n_dealt == 0:
			continue

		for i in range(n_dealt):
			if deck_visuals.is_empty():
				break
			var top = deck_visuals.pop_back()
			if is_instance_valid(top):
				top.queue_free()
		if is_instance_valid(deck_label):
			deck_label.text = "%d" % max(0, deck_visuals.size())

		await t.finished
		for m in movers:
			hand_visuals[m["player"]].append(m["card"])


func _create_points_labels(players: Array) -> void:
	for p in [0, 1]:
		var y = HAND_P0_Y if p == 0 else HAND_P1_Y
		var label = Label.new()
		add_child(label)
		label.text = "%d pts" % players[p]["points"]
		label.add_theme_font_size_override("font_size", 18)
		await get_tree().process_frame
		var label_y = y - card_height / 2.0 - label.size.y - 20.0 if p == 0 else y + card_height / 2.0 + 20.0
		label.position = Vector2(TAKEN_X - label.size.x / 2.0, label_y)
		points_labels[p] = label


func _create_current_dot(current: int) -> void:
	var dot = ColorRect.new()
	dot.color = Color.YELLOW
	var dot_size = 10.0
	dot.size = Vector2(dot_size, dot_size)
	dot.position = dot_positions[current]
	add_child(dot)
	current_dot = dot


func _animate_step(action: int) -> void:
	var pre = env.get_state()
	var current: int = pre["current_player_idx"]
	var pre_size: int = pre["players"][current]["hand"].size()

	var card_visual = hand_visuals[current][action] if hand_visuals[current].size() > action else null
	if not is_instance_valid(card_visual):
		var card_dict = pre["players"][current]["hand"][action]
		var src = _hand_card_pos(current, action, pre_size)
		card_visual = _spawn_card(card_dict, src, false)
	hand_visuals[current][action] = null
	_compact_hand(current)

	var t = create_tween().set_parallel(true)
	t.tween_property(card_visual, "position", _table_pos(current), PLAY_DUR)
	for i in range(hand_visuals[current].size()):
		var c = hand_visuals[current][i]
		if is_instance_valid(c):
			t.tween_property(c, "position", _hand_card_pos(current, i, pre_size - 1), PLAY_DUR)
	await t.finished
	table_visuals.append(card_visual)

	env.step(action)
	var post = env.get_state()

	var trick_resolved: bool = pre["cards_on_table"].size() > 0 and post["cards_on_table"].size() == 0
	if trick_resolved:
		var winner: int = post["current_player_idx"]
		await _animate_take(winner)
		if is_instance_valid(points_labels[winner]):
			points_labels[winner].text = "%d pts" % post["players"][winner]["points"]

	var dealt: bool = pre["deck_size"] > post["deck_size"]
	if dealt:
		await _animate_deal(post)

	if is_instance_valid(current_dot) and not post["done"]:
		current_dot.position = dot_positions[post["current_player_idx"]]


func _animate_take(winner_idx: int) -> void:
	if table_visuals.is_empty():
		return
	var center = Vector2(TAKEN_X, HAND_P0_Y if winner_idx == 0 else HAND_P1_Y)
	var t = create_tween().set_parallel(true)
	for c in table_visuals:
		var jitter = Vector2(randf_range(-4.0, 4.0), randf_range(-4.0, 4.0))
		t.tween_property(c, "position", center + jitter, TAKE_DUR)
		t.tween_property(c, "rotation_degrees", randf_range(-20.0, 20.0), TAKE_DUR)
	await t.finished
	for c in table_visuals:
		if is_instance_valid(c):
			c.covered = true
			c.update_visuals()
			taken_pile_visuals[winner_idx].append(c)
	table_visuals = []


func _animate_deal(post) -> void:
	var deck_pos = Vector2(DECK_X, DECK_Y)
	var movers: Array = []
	var t = create_tween().set_parallel(true)
	var n_dealt = 0
	for p in [0, 1]:
		var post_hand: Array = post["players"][p]["hand"]
		if post_hand.is_empty():
			continue
		var post_size = post_hand.size()
		var slot_idx = post_size - 1
		var card_dict = post_hand[slot_idx]
		var moving = _spawn_card(card_dict, deck_pos, false)
		movers.append({"card": moving, "player": p})
		t.tween_property(moving, "position", _hand_card_pos(p, slot_idx, post_size), DEAL_DUR)
		n_dealt += 1
		for i in range(hand_visuals[p].size()):
			var c = hand_visuals[p][i]
			if is_instance_valid(c):
				t.tween_property(c, "position", _hand_card_pos(p, i, post_size), DEAL_DUR)
	if n_dealt == 0:
		return

	for i in range(n_dealt):
		if deck_visuals.is_empty():
			break
		var top = deck_visuals.pop_back()
		if is_instance_valid(top):
			top.queue_free()
	if is_instance_valid(deck_label):
		deck_label.text = "%d" % max(0, deck_visuals.size())

	await t.finished
	for m in movers:
		hand_visuals[m["player"]].append(m["card"])


func _show_result() -> void:
	if is_instance_valid(current_dot):
		current_dot.queue_free()
		current_dot = null

	var state = env.get_state()
	var players: Array = state["players"]
	var result: String = state["result"]
	var msg: String
	match result:
		"p0_win":
			msg = "Player 0 Wins"
		"p1_win":
			msg = "Player 1 Wins"
		"draw":
			msg = "Draw"
		_:
			msg = ""

	var box = VBoxContainer.new()
	box.add_theme_constant_override("separation", 16)
	add_child(box)

	var result_lbl = Label.new()
	result_lbl.text = msg
	result_lbl.add_theme_font_size_override("font_size", 48)
	result_lbl.add_theme_color_override("font_color", Color.WHITE)
	result_lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	result_lbl.size_flags_horizontal = Control.SIZE_SHRINK_CENTER
	box.add_child(result_lbl)

	var score_lbl = Label.new()
	score_lbl.text = "%d  -  %d" % [players[0]["points"], players[1]["points"]]
	score_lbl.add_theme_font_size_override("font_size", 28)
	score_lbl.add_theme_color_override("font_color", Color.WHITE)
	score_lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	score_lbl.size_flags_horizontal = Control.SIZE_SHRINK_CENTER
	box.add_child(score_lbl)

	var btn_row = HBoxContainer.new()
	btn_row.add_theme_constant_override("separation", 24)
	btn_row.alignment = BoxContainer.ALIGNMENT_CENTER
	btn_row.size_flags_horizontal = Control.SIZE_SHRINK_CENTER
	box.add_child(btn_row)

	var menu_btn = Button.new()
	menu_btn.text = "Menu"
	menu_btn.custom_minimum_size = Vector2(120, 40)
	menu_btn.pressed.connect(_on_menu_pressed)
	btn_row.add_child(menu_btn)

	var rematch_btn = Button.new()
	rematch_btn.text = "Rematch"
	rematch_btn.custom_minimum_size = Vector2(120, 40)
	rematch_btn.pressed.connect(_on_rematch_pressed)
	btn_row.add_child(rematch_btn)

	await get_tree().process_frame
	box.position = Vector2(
		HAND_CENTER - box.size.x / 2.0,
		DisplayServer.screen_get_size()[1] / 2.0 - box.size.y / 2.0)
	result_panel = box


func _on_menu_pressed() -> void:
	get_tree().change_scene_to_file("res://briscola/scenes/menu.tscn")


func _on_rematch_pressed() -> void:
	get_tree().change_scene_to_file("res://briscola/scenes/game.tscn")


func _compact_hand(player_idx: int) -> void:
	var compacted: Array = []
	for c in hand_visuals[player_idx]:
		if is_instance_valid(c):
			compacted.append(c)
	hand_visuals[player_idx] = compacted


func _hand_card_pos(player_idx: int, slot_idx: int, hand_size: int) -> Vector2:
	var y = HAND_P0_Y if player_idx == 0 else HAND_P1_Y
	var w = card_width if card_width > 0.0 else 96.0
	var total_width = hand_size * w + (hand_size - 1) * HAND_CARDS_OFFSET
	var x = HAND_CENTER - total_width / 2.0 + w / 2.0 + slot_idx * (w + HAND_CARDS_OFFSET)
	return Vector2(x, y)


func _table_pos(player_idx: int) -> Vector2:
	return Vector2(CARD_P0_X if player_idx == 0 else CARD_P1_X,
		CARD_P0_Y if player_idx == 0 else CARD_P1_Y)


func _spawn_card(card_dict, pos: Vector2, covered: bool) -> Node:
	var card = Card.instantiate()
	add_child(card)
	card.init(card_dict["number"], card_dict["suit"], covered)
	card.position = pos
	return card


func _on_card_clicked(action_idx: int) -> void:
	human_action_selected.emit(action_idx)
