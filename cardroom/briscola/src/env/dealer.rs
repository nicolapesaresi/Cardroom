use rand::seq::SliceRandom;
use strum::IntoEnumIterator;
use super::cards::{BriscolaCard, BriscolaFace, BriscolaSuit};

#[derive(Clone)]
pub struct BriscolaDealer {
    pub deck: Vec<BriscolaCard>,
    pub briscola: BriscolaSuit,
}

impl BriscolaDealer {
    pub fn new() -> Self {
        // instantiate a new ordered deck
        let mut deck = vec![];
        for s in BriscolaSuit::iter() {
            for f in BriscolaFace::iter() {
                deck.push(BriscolaCard::new(f, s));
            }
        }
        // shuffle and set briscola
        deck.shuffle(&mut rand::rng());
        let briscola = deck[0].suit;

        for card in &mut deck {
            if card.suit == briscola {
                card.set_briscola();
                card.rank += 100;
            }
        }

        Self {deck, briscola}
    }

    pub fn shuffle(&mut self) {
        self.deck.shuffle(&mut rand::rng());
    }

    pub fn from_deck(deck: Vec<BriscolaCard>) -> Self {
        let briscola = deck[0].suit;
        Self { deck, briscola }
    }

    pub fn deal(&mut self) -> BriscolaCard {
        self.deck.pop().expect("No cards left to deal.")
    }

}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dealer() {
        let dealer = BriscolaDealer::new();

        assert!(dealer.deck.len() == 40);
        assert!(dealer.briscola == dealer.deck[0].suit);

        for card in dealer.deck {
            if card.suit == dealer.briscola { assert!(card.rank >= 100); }
            else { assert!(card.rank < 100); }
        }
    }

    #[test]
    fn test_shuffle() {
        let mut dealer = BriscolaDealer::new();
        dealer.shuffle();

        assert!(dealer.deck.len() == 40);

        for card in dealer.deck {
            if card.suit == dealer.briscola { assert!(card.rank >= 100); }
            else { assert!(card.rank < 100); }
        }
    }

    #[test]
    fn test_deal() {
        let mut dealer = BriscolaDealer::new();

        for i in 0..40 {
            let last_card = dealer.deck.last().unwrap().clone();
            let dealt_card = dealer.deal();
            assert_eq!(last_card, dealt_card);
        }
    }

}