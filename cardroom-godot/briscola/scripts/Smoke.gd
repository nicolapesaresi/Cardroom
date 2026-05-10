extends Node

func _ready() -> void:
	var env := BriscolaEnvNode.new()
	add_child(env)

	var s := env.snapshot()
	print("=== initial snapshot ===")
	print("trump suit: ", s["trump_suit"])
	print("trump card: ", s.get("trump_card", null))
	print("player 0 hand size: ", s["players"][0]["hand"].size())
	print("player 1 hand size: ", s["players"][1]["hand"].size())

	# play one card and verify on-table state
	env.step(env.legal_actions()[0])
	var s2 := env.snapshot()
	assert(s2["cards_on_table"].size() == 1, "expected 1 card on table after first step")
	var on_table_card: Dictionary = s2["cards_on_table"][0]["card"]
	assert(on_table_card["asset"].ends_with(".png"), "asset name should end with .png")
	print("first card played: ", on_table_card["asset"], " (", on_table_card["suit"], " ", on_table_card["face"], ")")

	# illegal action guard
	var ok := env.step(99)
	assert(not ok, "step(99) should return false")

	# run a full bot-vs-bot game
	env.reset()
	while not env.is_done():
		env.step(env.bot_action())

	print("=== game finished ===")
	print("result: ", env.result())
	var final_s := env.snapshot()
	print("player 0 points: ", final_s["players"][0]["points"])
	print("player 1 points: ", final_s["players"][1]["points"])
	assert(env.result() in ["p0_win", "p1_win", "draw"])
	print("all assertions passed")
