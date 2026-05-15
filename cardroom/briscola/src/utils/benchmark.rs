use crate::agents::generic::BriscolaAgent;
use crate::env::cards::{BriscolaCard, BriscolaSuit};
use crate::env::dealer::BriscolaDealer;
use crate::env::env::{BriscolaEnv, BriscolaResult};
use indicatif::{ProgressBar, ProgressStyle};

pub struct GameRecord {
    pub result: BriscolaResult,
    pub agent_points: [i32; 2],
    pub winning_agent: Option<usize>,
}

/// Cards dealt during game 1. Used by tests to verify the game-2 swap.
pub struct DealRecord {
    pub briscola: BriscolaSuit,
    pub initial: [Vec<BriscolaCard>; 2],
    pub per_trick: Vec<[BriscolaCard; 2]>,  // 16 elements; excludes the spy deal
}

pub struct BenchmarkRound {
    pub game1: GameRecord,
    pub game2: GameRecord,
    pub deals: DealRecord,
}

fn make_record(env: &BriscolaEnv, agent_order: [usize; 2]) -> GameRecord {
    let agent0_player = agent_order.iter().position(|&a| a == 0).unwrap();
    let agent1_player = agent_order.iter().position(|&a| a == 1).unwrap();
    let winning_agent = match env.result {
        BriscolaResult::P0Win => Some(agent_order[0]),
        BriscolaResult::P1Win => Some(agent_order[1]),
        _ => None,
    };
    GameRecord {
        result: env.result,
        agent_points: [
            env.players[agent0_player].points,
            env.players[agent1_player].points,
        ],
        winning_agent,
    }
}

pub fn benchmark_game(env: &mut BriscolaEnv, agents: &[Box<dyn BriscolaAgent>]) -> BenchmarkRound {
    let deck = BriscolaDealer::new().deck;

    // --- Game 1: agent 0 as player 0 ---
    env.reset_with_deck(deck.clone());

    let initial: [Vec<BriscolaCard>; 2] = [
        env.players[0].hand.clone(),
        env.players[1].hand.clone(),
    ];
    let briscola = env.dealer.briscola;
    let mut per_trick: Vec<[BriscolaCard; 2]> = Vec::new();

    while !env.done {
        let pre = env.dealer.deck.len();
        let action = agents[env.current_player_id].select_action(env.get_obs());
        env.step(action);
        // Record per-trick deals; skip the final spy deal (pre == 2 → post == 0)
        if env.dealer.deck.len() < pre && pre >= 4 {
            per_trick.push([
                *env.players[0].hand.last().unwrap(),
                *env.players[1].hand.last().unwrap(),
            ]);
        }
    }
    let game1 = make_record(env, [0, 1]);

    // --- Game 2: agent 1 as player 0, same deck so the spy is unchanged ---
    env.reset_with_deck(deck.clone());

    // Force-assign swapped initial hands regardless of what reset_with_deck dealt
    env.players[0].hand.clear();
    env.players[1].hand.clear();
    for &card in &initial[1] { env.players[0].hand.push(card); }
    for &card in &initial[0] { env.players[1].hand.push(card); }

    let mut trick_idx = 0;
    while !env.done {
        let pre = env.dealer.deck.len();
        let agent_idx = [1usize, 0][env.current_player_id];
        let action = agents[agent_idx].select_action(env.get_obs());
        env.step(action);
        // Overwrite the just-dealt cards with the swapped game-1 assignments
        if env.dealer.deck.len() < pre && pre >= 4 {
            *env.players[0].hand.last_mut().unwrap() = per_trick[trick_idx][1];
            *env.players[1].hand.last_mut().unwrap() = per_trick[trick_idx][0];
            trick_idx += 1;
        }
    }
    let game2 = make_record(env, [1, 0]);

    BenchmarkRound {
        game1,
        game2,
        deals: DealRecord { briscola, initial, per_trick },
    }
}

pub fn benchmark_n_games(agents: Vec<Box<dyn BriscolaAgent>>, n_rounds: usize) {
    let names: Vec<String> = agents.iter().map(|a| a.name().to_string()).collect();
    let mut env = BriscolaEnv::new_with_names(names.clone());
    let mut wins = [0usize; 2];
    let mut draws = 0usize;
    let mut points_total = [0i64; 2];

    let pb = ProgressBar::new(n_rounds as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len}  {msg}  ({per_sec}, eta {eta})",
        )
        .unwrap()
        .progress_chars("=>-"),
    );

    for _ in 0..n_rounds {
        let round = benchmark_game(&mut env, &agents);

        for record in [&round.game1, &round.game2] {
            match record.winning_agent {
                Some(0) => wins[0] += 1,
                Some(1) => wins[1] += 1,
                _ => draws += 1,
            }
            points_total[0] += record.agent_points[0] as i64;
            points_total[1] += record.agent_points[1] as i64;
        }

        let total_games = (wins[0] + wins[1] + draws) as f64;
        pb.set_message(format!(
            "{}: {:.0}%  {}: {:.0}%  D: {:.0}%",
            names[0],
            wins[0] as f64 / total_games * 100.0,
            names[1],
            wins[1] as f64 / total_games * 100.0,
            draws as f64 / total_games * 100.0,
        ));
        pb.inc(1);
    }

    pb.finish_and_clear();

    let n_games = (n_rounds * 2) as f64;
    println!(
        "Benchmark over {} rounds ({} games)  ({}=A0  vs  {}=A1)",
        n_rounds,
        n_rounds * 2,
        names[0],
        names[1]
    );
    println!("  {} wins: {:4}  ({:.1}%)", names[0], wins[0], wins[0] as f64 / n_games * 100.0);
    println!("  {} wins: {:4}  ({:.1}%)", names[1], wins[1], wins[1] as f64 / n_games * 100.0);
    println!("  Draws:   {:4}  ({:.1}%)", draws, draws as f64 / n_games * 100.0);
    println!("  Avg {} points/game: {:.1}", names[0], points_total[0] as f64 / n_games);
    println!("  Avg {} points/game: {:.1}", names[1], points_total[1] as f64 / n_games);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::random::RandomAgent;
    use crate::env::cards::{BriscolaFace, BriscolaSuit};

    fn make_agents() -> Vec<Box<dyn BriscolaAgent>> {
        vec![Box::new(RandomAgent::default()), Box::new(RandomAgent::default())]
    }

    fn make_env() -> BriscolaEnv {
        BriscolaEnv::new_with_names(vec!["A".into(), "B".into()])
    }

    #[test]
    fn benchmark_game_both_games_complete() {
        let mut env = make_env();
        let round = benchmark_game(&mut env, &make_agents());

        assert_ne!(round.game1.result, BriscolaResult::InProgress);
        assert_ne!(round.game2.result, BriscolaResult::InProgress);
    }

    #[test]
    fn benchmark_game_points_sum_to_120() {
        let mut env = make_env();
        let round = benchmark_game(&mut env, &make_agents());

        assert_eq!(round.game1.agent_points[0] + round.game1.agent_points[1], 120);
        assert_eq!(round.game2.agent_points[0] + round.game2.agent_points[1], 120);
    }

    #[test]
    fn benchmark_game_winning_agent_consistent_with_points() {
        for _ in 0..20 {
            let mut env = make_env();
            let round = benchmark_game(&mut env, &make_agents());

            for record in [&round.game1, &round.game2] {
                match record.winning_agent {
                    Some(0) => assert!(record.agent_points[0] > record.agent_points[1]),
                    Some(1) => assert!(record.agent_points[1] > record.agent_points[0]),
                    None => assert_eq!(record.agent_points[0], record.agent_points[1]),
                    _ => unreachable!(),
                }
            }
        }
    }

    #[test]
    fn benchmark_game_per_trick_deal_count() {
        let mut env = make_env();
        let round = benchmark_game(&mut env, &make_agents());

        assert_eq!(round.deals.per_trick.len(), 16);
    }

    #[test]
    fn benchmark_game_deal_cards_are_unique() {
        use std::collections::HashSet;

        let mut env = make_env();
        let round = benchmark_game(&mut env, &make_agents());

        let mut seen: HashSet<(BriscolaFace, BriscolaSuit)> = HashSet::new();
        for hand in &round.deals.initial {
            for card in hand {
                assert!(seen.insert((card.face, card.suit)), "duplicate in initial deal");
            }
        }
        for pair in &round.deals.per_trick {
            for card in pair {
                assert!(seen.insert((card.face, card.suit)), "duplicate in per-trick deal");
            }
        }
        assert_eq!(seen.len(), 38); // 40 cards minus the 2 spy-deal cards
    }

    #[test]
    fn benchmark_game_briscola_same_in_both_games() {
        let mut env = make_env();
        let round = benchmark_game(&mut env, &make_agents());

        // env is in game-2 state after benchmark_game returns
        assert_eq!(round.deals.briscola, env.dealer.briscola);
    }

    #[test]
    fn benchmark_n_games_does_not_panic() {
        benchmark_n_games(make_agents(), 3);
    }
}
