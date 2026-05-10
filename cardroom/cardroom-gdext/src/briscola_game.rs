use godot::prelude::*;
use briscola::env::env::{BriscolaEnv, BriscolaResult};
use briscola::env::player::{BriscolaAction, BriscolaPlayer};
use briscola::env::cards::BriscolaCard;

#[derive(GodotClass)]
#[class(base=Node)]
// godot node that runs briscola env
pub struct BriscolaEnvNode {
    pub inner: BriscolaEnv,
    base: Base<Node>,
}

#[godot_api]
impl INode for BriscolaEnvNode {
    fn init(base: Base<Node>) -> Self {
        Self {
            inner: BriscolaEnv::new(),
            base,
        }
    }
}

#[godot_api]
impl BriscolaEnvNode {
    #[func]
    fn init_with_names(&mut self, names: PackedStringArray) {
        let names_vec: Vec<String> = names.as_slice().iter().map(|s| s.to_string()).collect();
        self.inner = BriscolaEnv::new_with_names(names_vec);
    }

    #[func]
    fn reset(&mut self) {
        self.inner.reset();
    }

    #[func]
    fn step(&mut self, action_idx: i32) -> bool {
        self.inner.step(BriscolaAction::from_idx(action_idx as usize));
        true
    }

    #[func]
    fn done(&self) -> bool {
        self.inner.done
    }

    /// Full env state as a Dictionary — everything the UI needs to render the table.
    #[func]
    fn get_state(&self) -> VarDictionary {
        let env = &self.inner;
        let mut d = VarDictionary::new();

        let v = (env.current_player_id as i64).to_variant();
        d.set("current_player_idx", v);

        let v = (env.turn_counter as i64).to_variant();
        d.set("turn_counter", v);

        let v = (env.dealer.deck.len() as i64).to_variant();
        d.set("deck_size", v);

        let mut deck = VarArray::new();
        for card in &env.dealer.deck {
            let cv = card_to_dict(card).to_variant();
            deck.push(&cv);
        }
        d.set("deck", deck);

        let mut turn_order = PackedInt32Array::new();
        for &id in &env.turn_order {
            turn_order.push(id as i32);
        }
        let v = turn_order.to_variant();
        d.set("turn_order", v);

        let mut legal_actions = PackedInt32Array::new();
        for action in env.get_legal_actions() {
            legal_actions.push(action.idx() as i32);
        }
        let v = legal_actions.to_variant();
        d.set("legal_actions", v);

        let mut cards_on_table = VarArray::new();
        for (i, card) in env.cards_on_table.iter().enumerate() {
            let mut entry = VarDictionary::new();
            let sv = (env.turn_order[i] as i64).to_variant();
            entry.set("seat", sv);
            let cv = card_to_dict(card).to_variant();
            entry.set("card", cv);
            let ev = entry.to_variant();
            cards_on_table.push(&ev);
        }
        let v = cards_on_table.to_variant();
        d.set("cards_on_table", v);

        let mut turn_history = PackedInt32Array::new();
        for &id in &env.turn_history {
            turn_history.push(id as i32);
        }
        let v = turn_history.to_variant();
        d.set("turn_history", v);

        let mut played_cards_history = VarArray::new();
        for card in &env.played_cards_history {
            let cv = card_to_dict(card).to_variant();
            played_cards_history.push(&cv);
        }
        let v = played_cards_history.to_variant();
        d.set("played_cards_history", v);

        let mut players = VarArray::new();
        for player in &env.players {
            let pv = player_to_dict(player).to_variant();
            players.push(&pv);
        }
        let v = players.to_variant();
        d.set("players", v);

        let v = env.done.to_variant();
        d.set("done", v);

        let v = GString::from(result_to_str(env.result)).to_variant();
        d.set("result", v);

        d
    }
}

fn card_to_dict(card: &BriscolaCard) -> VarDictionary {
    let suit_lower = card.suit.name().to_lowercase();
    let mut d = VarDictionary::new();

    let v = GString::from(&suit_lower).to_variant();
    d.set("suit", v);

    let v = GString::from(card.face.name()).to_variant();
    d.set("face", v);

    let v = (card.face.number() as i64).to_variant();
    d.set("number", v);

    let v = (card.points as i64).to_variant();
    d.set("points", v);

    let v = (card.rank as i64).to_variant();
    d.set("rank", v);

    let v = card.is_briscola.to_variant();
    d.set("is_briscola", v);

    d
}

fn player_to_dict(player: &BriscolaPlayer) -> VarDictionary {
    let mut d = VarDictionary::new();

    let v = GString::from(player.name.as_str()).to_variant();
    d.set("name", v);

    let v = (player.points as i64).to_variant();
    d.set("points", v);

    let v = (player.taken_cards.len() as i64).to_variant();
    d.set("taken_count", v);

    let mut hand = VarArray::new();
    for card in &player.hand {
        let cv = card_to_dict(card).to_variant();
        hand.push(&cv);
    }
    let v = hand.to_variant();
    d.set("hand", v);

    d
}

fn result_to_str(result: BriscolaResult) -> &'static str {
    match result {
        BriscolaResult::P0Win => "p0_win",
        BriscolaResult::P1Win => "p1_win",
        BriscolaResult::Draw => "draw",
        BriscolaResult::InProgress => "in_progress",
    }
}
