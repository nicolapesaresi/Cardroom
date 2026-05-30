use super::generic::BriscolaAgent;
use crate::env::cards::{BriscolaCard, BriscolaSuit};
use crate::env::env::BriscolaObs;
use crate::env::player::BriscolaAction;
use crate::env::utils::env_from_obs;

pub struct BotAgent {
    name: String,
}

impl Default for BotAgent {
    fn default() -> Self {
        Self { name: format!("Bot") }
    }
}
impl BotAgent {
    pub fn new(name:String) -> Self {
        Self { name }
    }

    fn card_sort_key(card: &BriscolaCard, card_on_table: &BriscolaCard, briscola_suit: BriscolaSuit) -> (i32, i32) {
        // scenario 1: card on table is not briscola and has no points
        if card_on_table.suit != briscola_suit && card_on_table.points < 10 {
            if card.suit == card_on_table.suit && card.rank > card_on_table.rank && card.points > 0 {
                return (0, -card.rank);
            } else if card.suit != briscola_suit && card.points == 0 {
                return (1, -card.rank);
            } else if card.suit == briscola_suit {
                return (2, card.rank);
            } else {
                return (3, card.rank);
            }
        }
        // scenario 2: card on table is not briscola but has points
        if card_on_table.suit != briscola_suit && card_on_table.points >= 10 {
            if card.suit == card_on_table.suit && card.rank > card_on_table.rank {
                return (0, -card.rank);
            } else if card.suit == briscola_suit {
                return (1, card.rank);
            } else {
                return (2, card.rank);
            }
        }
        // scenario 3: card on table is briscola
        if card.points == 0 {
            return (0, card.rank);
        } else if card.rank > card_on_table.rank {
            return (1, card.rank);
        } else {
            return (2, card.rank);
        }
    }
}

impl BriscolaAgent for BotAgent {
    fn name(&self) -> &str { &self.name }

    fn select_action(&self, obs: BriscolaObs) -> BriscolaAction {
        let env = env_from_obs(&obs);
        let legal_actions = env.get_legal_actions();
        let hand = &env.players[env.current_player_id].hand;
        let briscola_suit = env.dealer.briscola;

        if env.cards_on_table.is_empty() {
            let best_idx = legal_actions.iter()
                .min_by_key(|a| {
                    let card = &hand[a.idx()];
                    (card.points, card.rank)
                })
                .unwrap();
            return *best_idx;
        }

        let card_on_table = &env.cards_on_table[0];
        let best_idx = legal_actions.iter()
            .min_by_key(|a| Self::card_sort_key(&hand[a.idx()], card_on_table, briscola_suit))
            .unwrap();
        *best_idx
    }
}