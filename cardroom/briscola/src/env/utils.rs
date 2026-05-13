use std::collections::HashSet;
use rand::seq::SliceRandom;
use strum::IntoEnumIterator;
use super::cards::{BriscolaCard, BriscolaFace, BriscolaSuit};
use super::dealer::BriscolaDealer;
use super::player::BriscolaPlayer;
use super::env::{BriscolaEnv, BriscolaState, BriscolaObs};

pub fn env_from_state(state: BriscolaState) -> BriscolaEnv {
    BriscolaEnv {
        n_players: state.n_players,
        players: state.players,
        done: state.done,
        played_cards_history: state.played_cards_history,
        turn_history: state.turn_history,
        dealer: state.dealer,
        cards_on_table: state.cards_on_table,
        current_player_id: state.current_player_id,
        turn_order: state.turn_order,
        turn_counter: state.turn_counter,
        result: state.result,
    }
}


pub fn env_from_obs(obs: &BriscolaObs) -> BriscolaEnv {
    let briscola = obs.briscola;

    // Build the full 40-card set with correct briscola flags and rank bonuses.
    let all_cards: Vec<BriscolaCard> = BriscolaSuit::iter()
        .flat_map(|suit| {
            BriscolaFace::iter().map(move |face| {
                let mut card = BriscolaCard::new(face, suit);
                if suit == briscola {
                    card.set_briscola();
                    card.rank += 100;
                }
                card
            })
        })
        .collect();

    // Cards at known positions: already played (includes table) + current player's hand.
    let known: HashSet<(BriscolaFace, BriscolaSuit)> = obs.played_cards_history.iter()
        .chain(obs.player.hand.iter())
        .map(|c| (c.face, c.suit))
        .collect();

    let (opponent_hand, deck) = match &obs.briscola_spy {
        Some(spy) => {
            // Deck has cards: spy stays fixed at deck[0]; remaining unknowns are randomized.
            let spy_id = (spy.face, spy.suit);
            let mut unknown: Vec<BriscolaCard> = all_cards.into_iter()
                .filter(|c| {
                    let id = (c.face, c.suit);
                    id != spy_id && !known.contains(&id)
                })
                .collect();
            unknown.shuffle(&mut rand::rng());

            // 40 = history + cards_in_deck + current_hand + opponent_hand
            let opponent_hand_size = 40
                - obs.played_cards_history.len()
                - obs.player.hand.len()
                - obs.cards_in_deck;

            let opponent_hand: Vec<BriscolaCard> = unknown.drain(..opponent_hand_size).collect();
            let mut deck = vec![spy.clone()];
            deck.extend(unknown);
            (opponent_hand, deck)
        }
        None => {
            // Deck is empty: every unaccounted card must be in the opponent's hand.
            // No randomness remains — the opponent's hand is fully determined.
            let opponent_hand: Vec<BriscolaCard> = all_cards.into_iter()
                .filter(|c| !known.contains(&(c.face, c.suit)))
                .collect();
            (opponent_hand, vec![])
        }
    };

    let current_player_id = obs.current_player_id;
    let opponent_id = 1 - current_player_id;

    let mut players = vec![
        BriscolaPlayer { name: format!("Player_0"), points: obs.points[0], hand: vec![], taken_cards: vec![] },
        BriscolaPlayer { name: format!("Player_1"), points: obs.points[1], hand: vec![], taken_cards: vec![] },
    ];
    players[current_player_id] = BriscolaPlayer {
        name: obs.player.name.clone(),
        points: obs.player.points,
        hand: obs.player.hand.clone(),
        taken_cards: vec![],
    };
    players[opponent_id].hand = opponent_hand;

    BriscolaEnv {
        n_players: obs.n_players,
        players,
        done: obs.done,
        played_cards_history: obs.played_cards_history.clone(),
        turn_history: obs.turn_history.clone(),
        dealer: BriscolaDealer { deck, briscola },
        cards_on_table: obs.cards_on_table.clone(),
        current_player_id,
        turn_order: obs.turn_order.clone(),
        turn_counter: obs.turn_counter,
        result: obs.result,
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use crate::env::env::{BriscolaEnv, BriscolaResult};
    use crate::env::player::BriscolaAction;

    fn play_steps(env: &mut BriscolaEnv, n: usize) {
        for _ in 0..n {
            if !env.done {
                env.step(BriscolaAction::Card0);
            }
        }
    }

    #[rstest]
    #[case(0)]
    #[case(5)]
    #[case(10)]
    #[case(15)]
    fn test_env_from_state(#[case] steps: usize) {
        let mut env = BriscolaEnv::new();
        play_steps(&mut env, steps);

        let state = env.get_state();
        let mut clone = env_from_state(state);

        while !env.done {
            let action = BriscolaAction::Card0;
            env.step(action);
            clone.step(action);
            assert_eq!(env.turn_counter, clone.turn_counter);
            assert_eq!(env.current_player_id, clone.current_player_id);
            assert_eq!(env.cards_on_table, clone.cards_on_table);
            assert_eq!(env.players[0].points, clone.players[0].points);
            assert_eq!(env.players[1].points, clone.players[1].points);
        }
        assert_eq!(env.result, clone.result);
    }

    #[rstest]
    #[case(0)]
    #[case(5)]
    #[case(10)]
    #[case(15)]
    // Steps > 34: deck is empty (last 2 cards dealt at turn 17 resolution = step 34).
    #[case(35)]
    #[case(37)]
    #[case(39)]
    fn test_env_from_obs_observable_state(#[case] steps: usize) {
        let mut env = BriscolaEnv::new();
        play_steps(&mut env, steps);

        let obs = env.get_obs();
        let clone = env_from_obs(&obs);
        let clone_obs = clone.get_obs();

        assert_eq!(obs.player.hand, clone_obs.player.hand);
        assert_eq!(obs.cards_on_table, clone_obs.cards_on_table);
        assert_eq!(obs.played_cards_history, clone_obs.played_cards_history);
        assert_eq!(obs.points, clone_obs.points);
        assert_eq!(obs.briscola, clone_obs.briscola);
        assert_eq!(obs.briscola_spy, clone_obs.briscola_spy);
        assert_eq!(obs.cards_in_deck, clone_obs.cards_in_deck);
        assert_eq!(obs.current_player_id, clone_obs.current_player_id);
        assert_eq!(obs.turn_order, clone_obs.turn_order);
        assert_eq!(obs.turn_counter, clone_obs.turn_counter);
    }

    #[rstest]
    #[case(0)]
    #[case(5)]
    #[case(10)]
    #[case(15)]
    #[case(35)]
    #[case(37)]
    #[case(39)]
    fn test_env_from_obs_card_uniqueness(#[case] steps: usize) {
        let mut env = BriscolaEnv::new();
        play_steps(&mut env, steps);

        let obs = env.get_obs();
        let clone = env_from_obs(&obs);

        // history + deck + all hands = 40 (each card accounted for exactly once).
        let total = clone.played_cards_history.len()
            + clone.dealer.deck.len()
            + clone.players.iter().map(|p| p.hand.len()).sum::<usize>();
        assert_eq!(total, 40);

        // No card (face, suit) appears more than once across deck + hands + history.
        let mut seen = std::collections::HashSet::new();
        for card in &clone.played_cards_history {
            assert!(seen.insert((card.face, card.suit)), "duplicate in history");
        }
        for card in &clone.dealer.deck {
            assert!(seen.insert((card.face, card.suit)), "duplicate in deck");
        }
        for player in &clone.players {
            for card in &player.hand {
                assert!(seen.insert((card.face, card.suit)), "duplicate in hand");
            }
        }

        // When deck is non-empty, spy is preserved at deck[0].
        if let Some(spy) = &obs.briscola_spy {
            assert_eq!(&clone.dealer.deck[0], spy);
        } else {
            assert!(clone.dealer.deck.is_empty());
        }
    }

    /// When deck is empty, the opponent's hand is fully determined — same cards as in state.
    #[rstest]
    #[case(35)]
    #[case(37)]
    #[case(39)]
    fn test_env_from_obs_empty_deck_same_as_state(#[case] steps: usize) {
        let mut env = BriscolaEnv::new();
        play_steps(&mut env, steps);

        assert_eq!(env.dealer.deck.len(), 0);
        assert!(env.get_obs().briscola_spy.is_none());

        let state = env.get_state();
        let obs = env.get_obs();

        let from_state = env_from_state(state);
        let from_obs = env_from_obs(&obs);

        // Opponent's hand must contain the same set of cards (order may differ).
        let opponent_id = 1 - env.current_player_id;
        let mut state_hand: Vec<(BriscolaFace, BriscolaSuit)> = from_state.players[opponent_id].hand
            .iter().map(|c| (c.face, c.suit)).collect();
        let mut obs_hand: Vec<(BriscolaFace, BriscolaSuit)> = from_obs.players[opponent_id].hand
            .iter().map(|c| (c.face, c.suit)).collect();
        state_hand.sort_by_key(|&(f, s)| (f as u8, s as u8));
        obs_hand.sort_by_key(|&(f, s)| (f as u8, s as u8));
        assert_eq!(state_hand, obs_hand);
    }

    #[test]
    fn test_env_from_obs_game_completes() {
        for _ in 0..100 {
            let env = BriscolaEnv::new();
            let obs = env.get_obs();
            let mut clone = env_from_obs(&obs);

            while !clone.done {
                clone.step(BriscolaAction::Card0);
            }

            assert_eq!(clone.players.iter().map(|p| p.points).sum::<i32>(), 120);
            assert!(vec![BriscolaResult::P0Win, BriscolaResult::Draw, BriscolaResult::P1Win]
                .contains(&clone.result));
        }
    }
}
