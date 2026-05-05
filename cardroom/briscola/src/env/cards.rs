use strum::IntoEnumIterator;
use strum::EnumIter;

#[derive(EnumIter, Clone, Copy, PartialEq, Eq, Debug)]
pub enum BriscolaFace {
    Ace,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
}

impl BriscolaFace {

    pub fn id(&self) -> u8 {
        match self {
            BriscolaFace::Ace => 0,
            BriscolaFace::Two => 1,
            BriscolaFace::Three => 2,
            BriscolaFace::Four => 3,
            BriscolaFace::Five => 4,
            BriscolaFace::Six => 5,
            BriscolaFace::Seven => 6,
            BriscolaFace::Eight => 7,
            BriscolaFace::Nine => 8,
            BriscolaFace::Ten => 9,
        }
    }

    pub fn number(&self) -> u8 {
        match self {
            BriscolaFace::Ace => 1,
            BriscolaFace::Two => 2,
            BriscolaFace::Three => 3,
            BriscolaFace::Four => 4,
            BriscolaFace::Five => 5,
            BriscolaFace::Six => 6,
            BriscolaFace::Seven => 7,
            BriscolaFace::Eight => 8,
            BriscolaFace::Nine => 9,
            BriscolaFace::Ten => 10,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            BriscolaFace::Ace => "A",
            BriscolaFace::Two => "2",
            BriscolaFace::Three => "3",
            BriscolaFace::Four => "4",
            BriscolaFace::Five => "5",
            BriscolaFace::Six => "6",
            BriscolaFace::Seven => "7",
            BriscolaFace::Eight => "8",
            BriscolaFace::Nine => "9",
            BriscolaFace::Ten => "10",
        }
    }

    pub fn rank(&self) -> i32 {
        match self {
            BriscolaFace::Ace => 10,
            BriscolaFace::Three => 9,
            BriscolaFace::Ten => 8,
            BriscolaFace::Nine => 7,
            BriscolaFace::Eight => 6,
            BriscolaFace::Seven => 5,
            BriscolaFace::Six => 4,
            BriscolaFace::Five => 3,
            BriscolaFace::Four => 2,
            BriscolaFace::Two => 1,
        }
    }

    pub fn points(&self)  -> i32 {
        match self {
            BriscolaFace::Ace => 11,
            BriscolaFace::Three => 10,
            BriscolaFace::Ten => 4,
            BriscolaFace::Nine => 3,
            BriscolaFace::Eight => 2,
            BriscolaFace::Seven => 0,
            BriscolaFace::Six => 0,
            BriscolaFace::Five => 0,
            BriscolaFace::Four => 0,
            BriscolaFace::Two => 0,
        }
    }
}


#[derive(EnumIter, Clone, Copy, PartialEq, Eq, Debug)]
pub enum BriscolaSuit {
    Ori,
    Coppe,
    Spade,
    Bastoni,
}

impl BriscolaSuit {
    pub fn id(&self) -> u8 {
        match self {
            BriscolaSuit::Ori => 0,
            BriscolaSuit::Coppe => 1,
            BriscolaSuit::Spade => 2,
            BriscolaSuit::Bastoni => 3,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            BriscolaSuit::Ori => "Ori",
            BriscolaSuit::Coppe => "Coppe",
            BriscolaSuit::Spade => "Spade",
            BriscolaSuit::Bastoni => "Bastoni",
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BriscolaCard {
    pub face: BriscolaFace,
    pub suit: BriscolaSuit,
    pub rank: i32,
    pub points: i32,
    pub name: String,
    pub is_briscola: bool,
}

impl BriscolaCard {
    pub fn new(face: BriscolaFace, suit: BriscolaSuit) -> Self {
        let name = format!("{} of {}", face.name(), suit.name());
        let rank = face.rank();
        let points = face.points();
        Self { face, suit, name, rank, points, is_briscola: false }
    }

    pub fn name(&self) -> &str { &self.name }
    pub fn points(&self) -> i32 { self.points }
    pub fn rank(&self) -> i32 { self.rank }
    pub fn is_briscola(&self) -> bool { self.is_briscola }
    pub fn set_briscola(&mut self) { self.is_briscola = true; }
}


#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    use crate::env::cards::BriscolaCard;

    #[rstest]
    #[case(BriscolaFace::Ace, 11)]
    #[case(BriscolaFace::Three, 10)]
    #[case(BriscolaFace::Ten, 4)]
    #[case(BriscolaFace::Nine, 3)]
    #[case(BriscolaFace::Eight, 2)]
    #[case(BriscolaFace::Seven, 0)]
    #[case(BriscolaFace::Six, 0)]
    #[case(BriscolaFace::Five, 0)]
    #[case(BriscolaFace::Four, 0)]
    #[case(BriscolaFace::Two, 0)]
    fn test_card_points(#[case] face:BriscolaFace, #[case] exp_points: i32) {
        for suit in BriscolaSuit::iter() {
            let card = BriscolaCard::new(face, suit);

            assert_eq!(card.points, exp_points);
        }
    }

    #[rstest]
    #[case(BriscolaCard::new(BriscolaFace::Ace, BriscolaSuit::Ori), BriscolaCard::new(BriscolaFace::Ace, BriscolaSuit::Bastoni), true, false, false)]
    #[case(BriscolaCard::new(BriscolaFace::Ace, BriscolaSuit::Coppe), BriscolaCard::new(BriscolaFace::Three, BriscolaSuit::Spade), false, true, false)]
    #[case(BriscolaCard::new(BriscolaFace::Four, BriscolaSuit::Ori), BriscolaCard::new(BriscolaFace::Three, BriscolaSuit::Coppe), false, false, true)]
    fn test_card_rank(#[case] card1: BriscolaCard, #[case] card2: BriscolaCard, #[case] exp_equal: bool, #[case] exp_bigger: bool, #[case] exp_smaller: bool) {
        assert_eq!(card1.rank == card2.rank, exp_equal);
        assert_eq!(card1.rank > card2.rank, exp_bigger);
        assert_eq!(card1.rank < card2.rank, exp_smaller);
    }

    #[rstest]
    #[case(BriscolaCard::new(BriscolaFace::Ace, BriscolaSuit::Ori), BriscolaCard::new(BriscolaFace::Ace, BriscolaSuit::Ori), true)]
    #[case(BriscolaCard::new(BriscolaFace::Ace, BriscolaSuit::Coppe), BriscolaCard::new(BriscolaFace::Three, BriscolaSuit::Spade), false)]
    #[case(BriscolaCard::new(BriscolaFace::Four, BriscolaSuit::Ori), BriscolaCard::new(BriscolaFace::Three, BriscolaSuit::Coppe), false)]
    fn test_card_equality(#[case] card1: BriscolaCard, #[case] card2: BriscolaCard, #[case] exp: bool) {
        assert_eq!(card1.rank == card2.rank, exp);
    }
}