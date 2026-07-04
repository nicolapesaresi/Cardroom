use crate::env::env::{BriscolaEnv, BriscolaObs};
use crate::env::player::BriscolaAction;
use crate::env::utils::env_from_obs;

/// Exhaustive minimax (with alpha-beta pruning) for the Briscola endgame.
///
/// Once the deck is empty there is no more drawing, so the observation maps to a
/// single, fully determined state: `env_from_obs` reconstructs the opponent's hand
/// exactly (see `BriscolaObs::briscola_spy == None`). From there the game is a
/// finite, perfect-information, zero-sum game over the remaining (at most three)
/// rounds, which we can solve optimally.
///
/// The value of a terminal node is the point margin from the perspective of the
/// player to move at the root (`root.points - opponent.points`). Maximising this
/// margin also maximises the chance of winning, since the 120 points are shared.
///
/// # Panics
/// Panics if called before the deck is empty (`obs.cards_in_deck != 0`), where the
/// state is not fully observable and minimax would be searching a guessed state.

pub fn minimax(obs: BriscolaObs) -> BriscolaAction {
    assert_eq!(
        obs.cards_in_deck, 0,
        "minimax is only valid in the endgame (empty deck), where the state is fully determined"
    );

    let env = env_from_obs(&obs);
    let root_player = env.current_player_id;

    let mut best_action = env.get_legal_actions()[0];
    let mut best_value = i32::MIN;

    for action in env.get_legal_actions() {
        let mut child = env.clone();
        child.step(action);
        let value = minimax_value(&child, root_player, best_value, i32::MAX);
        if value > best_value {
            best_value = value;
            best_action = action;
        }
    }

    best_action
}


fn minimax_value(env: &BriscolaEnv, root_player: usize, mut alpha: i32, mut beta: i32) -> i32 {
    if env.done {
        return env.players[root_player].points - env.players[1 - root_player].points;
    }

    let maximizing = env.current_player_id == root_player;
    let mut value = if maximizing { i32::MIN } else { i32::MAX };

    for action in env.get_legal_actions() {
        let mut child = env.clone();
        child.step(action);
        let child_value = minimax_value(&child, root_player, alpha, beta);

        if maximizing {
            value = value.max(child_value);
            alpha = alpha.max(value);
        } else {
            value = value.min(child_value);
            beta = beta.min(value);
        }
        if alpha >= beta {
            break; // remaining siblings cannot affect the result
        }
    }

    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::env::{BriscolaEnv, BriscolaResult};

    /// Plays a full game where both players always play `Card0` until only the
    /// last three rounds remain, then drives both players with minimax.
    /// The game must terminate with a valid result and the points must sum to 120.
    #[test]
    fn test_minimax_plays_endgame_to_completion() {
        for _ in 0..100 {
            let mut env = BriscolaEnv::new();
            // Play until the deck is empty (last three rounds).
            while env.dealer.deck.len() > 0 {
                env.step(BriscolaAction::Card0);
            }

            while !env.done {
                let action = minimax(env.get_obs());
                env.step(action);
            }

            assert_eq!(env.players.iter().map(|p| p.points).sum::<i32>(), 120);
            assert!([
                BriscolaResult::P0Win,
                BriscolaResult::Draw,
                BriscolaResult::P1Win
            ]
            .contains(&env.result));
        }
    }

    /// Minimax must never play worse than the value it claims: against any fixed
    /// opponent line (here, always `Card0`), the minimax player's realised margin
    /// must be at least the minimax value of the worst-case opponent.
    #[test]
    fn test_minimax_is_optimal_vs_worst_case() {
        for _ in 0..50 {
            let mut env = BriscolaEnv::new();
            while env.dealer.deck.len() > 0 {
                env.step(BriscolaAction::Card0);
            }

            let root_player = env.current_player_id;
            // Optimal value the root player can guarantee.
            let guaranteed = {
                let env_clone = env.clone();
                let mut best = i32::MIN;
                for action in env_clone.get_legal_actions() {
                    let mut child = env_clone.clone();
                    child.step(action);
                    best = best.max(minimax_value(&child, root_player, i32::MIN, i32::MAX));
                }
                best
            };

            // Play out: root uses minimax, opponent plays Card0.
            while !env.done {
                let action = if env.current_player_id == root_player {
                    minimax(env.get_obs())
                } else {
                    BriscolaAction::Card0
                };
                env.step(action);
            }

            let realised = env.players[root_player].points - env.players[1 - root_player].points;
            assert!(
                realised >= guaranteed,
                "realised margin {} below guaranteed minimax value {}",
                realised,
                guaranteed
            );
        }
    }

    #[test]
    #[should_panic(expected = "endgame")]
    fn test_minimax_panics_before_endgame() {
        let env = BriscolaEnv::new(); // full deck
        minimax(env.get_obs());
    }
}
