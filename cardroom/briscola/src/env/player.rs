use strum::EnumIter;
use super::cards::BriscolaCard;

#[derive(EnumIter, Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum BriscolaAction {
    Card0,
    Card1,
    Card2,
}


impl BriscolaAction {
    pub fn idx(&self) -> usize {
        match self {
            BriscolaAction::Card0 => 0,
            BriscolaAction::Card1 => 1,
            BriscolaAction::Card2 => 2,
        }
    }

    pub fn from_idx(action_idx: usize) -> Self {
        match action_idx {
            0 => BriscolaAction::Card0,
            1 => BriscolaAction::Card1,
            2 => BriscolaAction::Card2,
            _ => panic!("Invalid action index: {}", action_idx),
        }
    }
}

#[derive(Clone)]
pub struct BriscolaPlayer {
    pub name: String,
    pub points: i32,
    pub hand: Vec<BriscolaCard>,
    pub taken_cards: Vec<BriscolaCard>
}

impl BriscolaPlayer {
    pub fn new(name: String) -> Self {
        Self {
            name,
            points: 0,
            hand: vec![],
            taken_cards: vec![],
        }
    }

    pub fn reset(&mut self) {
        self.points = 0;
        self.hand = vec![];
        self.taken_cards = vec![]
    }

    pub fn play_card(&mut self, action: BriscolaAction) -> BriscolaCard {
        self.hand.remove(action.idx())
    }

    pub fn get_legal_actions(&self) -> Vec<BriscolaAction> {
        let mut legal_actions = vec![];
        for card_idx in 0..self.hand.len() {
            legal_actions.push(BriscolaAction::from_idx(card_idx));
        }
        legal_actions
    }
}


#[cfg(test)]
mod tests {
    use rstest::rstest;
    use crate::env::cards::{BriscolaCard, BriscolaFace, BriscolaSuit};

    use super::*;

    #[test]
    fn test_player() {
        let mut player = BriscolaPlayer::new(format!("Player"));

        assert_eq!(player.name, "Player");
        assert_eq!(player.hand, vec![]);
        assert_eq!(player.taken_cards, vec![]);

        player.hand.push(BriscolaCard::new(BriscolaFace::Ace, BriscolaSuit::Coppe));
        assert_eq!(player.hand.len(), 1);
    }

        #[test]
    fn test_reset() {
        let mut player = BriscolaPlayer {
            name: format!("Player"),
            points: 45,
            hand: vec![BriscolaCard::new(BriscolaFace::Ace, BriscolaSuit::Coppe)],
            taken_cards: vec![BriscolaCard::new(BriscolaFace::Ace, BriscolaSuit::Coppe)],
        };
        player.reset();

        assert_eq!(player.name, "Player");
        assert_eq!(player.hand, vec![]);
        assert_eq!(player.taken_cards, vec![]);
    }

    #[rstest]
    #[case(BriscolaAction::Card0, BriscolaCard::new(BriscolaFace::Ace, BriscolaSuit::Ori))]
    #[case(BriscolaAction::Card1, BriscolaCard::new(BriscolaFace::Five, BriscolaSuit::Coppe))]
    #[case(BriscolaAction::Card2, BriscolaCard::new(BriscolaFace::Two, BriscolaSuit::Spade))]
    fn test_play_card(#[case] action: BriscolaAction, #[case] expected: BriscolaCard) {
        let mut player = BriscolaPlayer {
            name: format!("Player"),
            points: 0,
            hand: vec![BriscolaCard::new(BriscolaFace::Ace, BriscolaSuit::Ori), BriscolaCard::new(BriscolaFace::Five, BriscolaSuit::Coppe), BriscolaCard::new(BriscolaFace::Two, BriscolaSuit::Spade)],
            taken_cards: vec![],
        };

        let played_card = player.play_card(action);
        assert_eq!(played_card, expected);
    }

    #[rstest]
    #[case(0, vec![])]
    #[case(1, vec![BriscolaAction::Card0])]
    #[case(2, vec![BriscolaAction::Card0, BriscolaAction::Card1])]
    #[case(3, vec![BriscolaAction::Card0, BriscolaAction::Card1, BriscolaAction::Card2])]
    fn test_legal_actions(#[case] n_cards: usize, #[case] expected: Vec<BriscolaAction>) {
        let mut player = BriscolaPlayer::new(format!("Player"));

        for i in 0..n_cards {
            player.hand.push(BriscolaCard::new(BriscolaFace::Ace, BriscolaSuit::Ori));
        }

        let legal_actions = player.get_legal_actions();
        assert_eq!(legal_actions, expected);
    }
}