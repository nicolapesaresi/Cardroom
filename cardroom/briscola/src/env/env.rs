use super::cards::BriscolaCard;
use super::dealer::BriscolaDealer;
use super::player::BriscolaPlayer;
use super::player::BriscolaAction;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BriscolaResult {
    P0Win,
    Draw,
    P1Win,
    InProgress,
}

impl BriscolaResult {
    pub fn winner_idx(&self) -> Option<usize> {
        match self {
            BriscolaResult::P0Win => Some(0),
            BriscolaResult::P1Win => Some(1),
            _ => None,
        }
    }
}

#[derive(Clone)]
pub struct BriscolaState {
    pub n_players: usize,
    pub players: Vec<BriscolaPlayer>,
    pub done: bool,
    pub played_cards_history: Vec<BriscolaCard>,
    pub turn_history: Vec<usize>,
    pub dealer: BriscolaDealer,
    pub cards_on_table: Vec<BriscolaCard>,
    pub current_player_id: usize,
    pub turn_order: Vec<usize>,
    pub turn_counter: u8,
    pub result: BriscolaResult,
}

pub struct BriscolaEnv {
    pub n_players: usize,
    pub players: Vec<BriscolaPlayer>,
    pub done: bool,
    pub played_cards_history: Vec<BriscolaCard>,
    pub turn_history: Vec<usize>,
    pub dealer: BriscolaDealer,
    pub cards_on_table: Vec<BriscolaCard>,
    pub current_player_id: usize,
    pub turn_order: Vec<usize>,
    pub turn_counter: u8,
    pub result: BriscolaResult,
}

impl BriscolaEnv {
    pub fn new() -> Self {
        let n_players = 2;
        let names: Vec<String> = (0..n_players).map(|i| format!("Player_{}", i)).collect();
        Self::new_with_names(names)
    }

    pub fn new_with_names(names: Vec<String>) -> Self {
        let n_players = names.len();
        let done = false;
        let mut players = vec![];
        let dealer = BriscolaDealer::new();
        for name in names {
            players.push(BriscolaPlayer::new(name));
        }
        let cards_on_table = vec![];
        let played_cards_history = vec![];
        let turn_history = vec![];
        let current_player_id = 0;
        let turn_order: Vec<usize> = (0..n_players).collect();
        let turn_counter = 1;
        let result = BriscolaResult::InProgress;

        let mut env = Self{n_players, players, done, played_cards_history, turn_history, dealer, cards_on_table, current_player_id, turn_order, turn_counter, result};
        env.reset();
        env
    }

    pub fn reset(&mut self) {
        self.done = false;
        self.played_cards_history = vec![];
        self.turn_history = vec![];
        self.dealer = BriscolaDealer::new();
        self.result = BriscolaResult::InProgress;

        for player in &mut self.players {
            player.reset();
        }
        // draw first player
        self.current_player_id = rand::random_range(0..self.n_players);
        self.set_turn_order(self.current_player_id);
        // deal first 3 cards
        for &id in &self.turn_order {
            let player = &mut self.players[id];
            for _ in 0..3 {
                player.hand.push(self.dealer.deal());
            }
        }

        self.turn_counter = 1;
    }

    pub fn set_turn_order(&mut self, first_id: usize) {
        let default_order: Vec<usize> = (0..self.n_players).collect();
        self.turn_order = [&default_order[first_id..], &default_order[..first_id]].concat();
    }

    pub fn pass_turn(&mut self) {
        self.current_player_id = (self.current_player_id + 1) % self.n_players
    }

    pub fn resolve_turn(&mut self) {
        //updates the game after a turn is played
        let first_suit_id = self.cards_on_table[0].suit;
        let mut played_rank = vec![];
        let mut played_points = 0;

        for card in &self.cards_on_table {
            let mut effective_rank = card.rank;
            if card.suit == first_suit_id {
                effective_rank += 50;
            }
            played_rank.push(effective_rank);
            played_points += card.points;
        }

        let winning_card_idx = played_rank.iter().enumerate().max_by_key(|&(_, x)| x).unwrap().0;
        let winning_player_id = self.turn_order[winning_card_idx];
        self.players[winning_player_id].points += played_points;
        self.players[winning_player_id].taken_cards.extend(self.cards_on_table.drain(..));
        self.set_turn_order(winning_player_id);
        self.current_player_id = winning_player_id;
    }

    pub fn step(&mut self, action: BriscolaAction) {
        // executes action of current player and updates the game
        let is_last = self.current_player_id == *self.turn_order.last().unwrap();
        
        let card = self.players[self.current_player_id].play_card(action);
        
        self.played_cards_history.push(card.clone());
        self.cards_on_table.push(card);
        self.turn_history.push(self.current_player_id);

        if !is_last {self.pass_turn();}
        else {
            self.resolve_turn();
            self.turn_counter += 1;

            // check if game is over, else deal new cards
            if self.turn_counter > 20 {
                let (result, points) = self.check_result();
                self.result = result;
                self.done = true;
            }
            else if self.dealer.deck.len() > 0 {
                for id in &self.turn_order {
                    self.players[*id].hand.push(self.dealer.deal());
                }
            }
        }
    }

    pub fn get_state(&self) -> BriscolaState {
        BriscolaState {
            n_players: self.n_players,
            players: self.players.clone(),
            done: self.done,
            played_cards_history: self.played_cards_history.clone(),
            turn_history: self.turn_history.clone(),
            dealer: self.dealer.clone(),
            cards_on_table: self.cards_on_table.clone(),
            current_player_id: self.current_player_id,
            turn_order: self.turn_order.clone(),
            turn_counter: self.turn_counter,
            result: self.result,
        }
    }

    pub fn check_result(&mut self) -> (BriscolaResult, Vec<i32>) {
        // checks result of the game
        let mut points = vec![];
        for player in &self.players {
            points.push(player.points);
        }
        let result = 
            if points[0] == points[1] {BriscolaResult::Draw}
            else if points[0] > points[1] {BriscolaResult::P0Win}
            else {BriscolaResult::P1Win};

        (result, points)
    }

    pub fn get_legal_actions(&self) -> Vec<BriscolaAction> {
        let player = &self.players[self.current_player_id];
        player.get_legal_actions()
    }

    pub fn render(&self) {
        // text rendering of the game
        // pre-game text
        if self.turn_counter == 1 && self.cards_on_table.len() == 0 {
            println!("==================");
            println!("Starting new game");
            println!("==================");
            println!("Players: {}, {}", self.players[0].name, self.players[1].name);
            println!("Starting player: {}", self.players[self.current_player_id].name);
            println!("Briscola suit: {} - spy: {}", self.dealer.briscola.name(), self.dealer.deck[0].name);
            return;
        }
        // pre-round text (render happens after step)
        if self.cards_on_table.len() == 1 {
            println!("** Turn {} - {} first of hand **", self.turn_counter, self.players[*self.turn_history.last().unwrap()].name);
            println!("{} plays {}", self.players[*self.turn_history.last().expect("No turns played.")].name, self.played_cards_history.last().unwrap().name);
        }
        //post-round text
        else {
            println!("{} plays {}", self.players[*self.turn_history.last().expect("No turns played.")].name, self.played_cards_history.last().unwrap().name);
        }

        //post-game text
        if self.done {
            println!("** Game finished **");
            for player in &self.players {
                println!("{}: {} points", player.name, player.points);
            }
            if self.result == BriscolaResult::Draw {
                println!("*** Draw ***");
            } else {
                println!("*** {} wins ***", self.players[self.result.winner_idx().unwrap()].name);
            }
        }

    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn play_game() -> BriscolaEnv {
        // plays simulated game where both players always play card 0
        let mut env = BriscolaEnv::new();

        while !env.done {
            env.step(BriscolaAction::Card0);
        }
        env
    }

    #[test]
    fn test_game() {
        for i in 0..100 {
            let env = play_game();

            assert!(env.done == true);
            assert!(env.players.iter().map(|p| p.points).sum::<i32>() == 120);
            assert!(env.turn_history.len() == 40);
            assert!(env.played_cards_history.len() == 40);
            assert!(vec![BriscolaResult::P0Win, BriscolaResult::Draw, BriscolaResult::P1Win].contains(&env.result));
        }
    }

    #[test]
    fn test_reset() {
        let mut env = BriscolaEnv::new();
        assert!(env.done == false);
        assert!(env.players.iter().map(|p| p.points).sum::<i32>() == 0);
        assert!(env.turn_counter == 1);
        assert!(env.turn_history.len() == 0);
        assert!(env.played_cards_history.len() == 0);
    
        env = play_game();
        env.reset();
        assert!(env.done == false);
        assert!(env.players.iter().map(|p| p.points).sum::<i32>() == 0);
        assert!(env.turn_counter == 1);
        assert!(env.turn_history.len() == 0);
        assert!(env.played_cards_history.len() == 0);
    }

    #[rstest]
    #[case(0, 1)]
    #[case(1, 0)]
    fn test_set_turn_order(#[case] current_player_id: usize, #[case] expected: usize) {
        let mut env = BriscolaEnv::new();
        env.current_player_id = current_player_id;
        env.pass_turn();

        assert_eq!(env.current_player_id, expected);
    }
}