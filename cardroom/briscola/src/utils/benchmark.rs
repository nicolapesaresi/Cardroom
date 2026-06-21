use crate::agents::generic::BriscolaAgent;
use crate::env::cards::{BriscolaCard, BriscolaSuit};
use crate::env::dealer::BriscolaDealer;
use crate::env::env::{BriscolaEnv, BriscolaResult};
use indicatif::{ProgressBar, ProgressStyle};

pub struct TrickRecord {
    pub trick_num: u8,
    pub trick_points: i32,
    pub had_briscola: bool,
    pub agent0_cumpts: i32,
    pub agent1_cumpts: i32,
    pub agent0_took_trick: bool,
    pub agent0_card: BriscolaCard,
    pub agent1_card: BriscolaCard,
    pub first_agent: u8, // which agent (0/1) played first this trick
}

pub struct GameRecord {
    pub result: BriscolaResult,
    pub agent_points: [i32; 2],
    pub winning_agent: Option<usize>,
    pub tricks: Vec<TrickRecord>,
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

fn make_record(env: &BriscolaEnv, agent_order: [usize; 2], tricks: Vec<TrickRecord>) -> GameRecord {
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
        tricks,
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

    let mut tricks1: Vec<TrickRecord> = Vec::new();
    let mut prev_pts = [0i32; 2];
    let mut prev_turn = env.turn_counter;

    // agent_for_player[player_id] = agent_idx; game 1: identity mapping
    let agent_for_player_g1 = [0usize, 1usize];

    while !env.done {
        let pre = env.dealer.deck.len();
        let action = agents[env.current_player_id].select_action(env.get_obs());
        env.step(action);

        if env.dealer.deck.len() < pre && pre >= 4 {
            per_trick.push([
                *env.players[0].hand.last().unwrap(),
                *env.players[1].hand.last().unwrap(),
            ]);
        }

        if env.turn_counter != prev_turn {
            let new_pts = [env.players[0].points, env.players[1].points];
            let trick_pts = (new_pts[0] + new_pts[1]) - (prev_pts[0] + prev_pts[1]);
            let hist_len = env.played_cards_history.len();
            let had_briscola = env.played_cards_history[hist_len - 2..hist_len]
                .iter().any(|c| c.is_briscola);
            let trick_winner_agent = agent_for_player_g1[env.turn_order[0]];
            let p_first = env.turn_history[hist_len - 2];
            let c_first = env.played_cards_history[hist_len - 2];
            let c_second = env.played_cards_history[hist_len - 1];
            let card_of = |pid: usize| if pid == p_first { c_first } else { c_second };
            let agent0_player = agent_for_player_g1.iter().position(|&a| a == 0).unwrap();
            let agent1_player = agent_for_player_g1.iter().position(|&a| a == 1).unwrap();
            tricks1.push(TrickRecord {
                trick_num: tricks1.len() as u8 + 1,
                trick_points: trick_pts,
                had_briscola,
                agent0_cumpts: new_pts[0],
                agent1_cumpts: new_pts[1],
                agent0_took_trick: trick_winner_agent == 0,
                agent0_card: card_of(agent0_player),
                agent1_card: card_of(agent1_player),
                first_agent: agent_for_player_g1[p_first] as u8,
            });
            prev_pts = new_pts;
            prev_turn = env.turn_counter;
        }
    }
    let game1 = make_record(env, [0, 1], tricks1);

    // --- Game 2: same deck (spy unchanged); the agents swap seats so each one
    // now plays the OTHER agent's game-1 cards. The deal stays with the seat
    // (seat 0 = initial[0], seat 1 = initial[1]); only the agent controlling
    // each seat flips (agent_for_player_g2 below). Game 2 is therefore a
    // duplicate of game 1 played from the opposite side, so card luck cancels
    // across the pair.
    env.reset_with_deck(deck.clone());

    // reset_with_deck already deals seat 0 = initial[0], seat 1 = initial[1];
    // set them explicitly to document that the deal stays with the seat.
    env.players[0].hand.clear();
    env.players[1].hand.clear();
    for &card in &initial[0] { env.players[0].hand.push(card); }
    for &card in &initial[1] { env.players[1].hand.push(card); }

    let mut tricks2: Vec<TrickRecord> = Vec::new();
    let mut prev_pts = [0i32; 2];
    let mut prev_turn = env.turn_counter;
    let mut trick_idx = 0;

    // game 2: agent 1 = player 0, agent 0 = player 1
    let agent_for_player_g2 = [1usize, 0usize];

    while !env.done {
        let pre = env.dealer.deck.len();
        let agent_idx = agent_for_player_g2[env.current_player_id];
        let action = agents[agent_idx].select_action(env.get_obs());
        env.step(action);

        if env.dealer.deck.len() < pre && pre >= 4 {
            // Force each seat to draw the same card it drew in game 1, in the
            // same order, regardless of game-2 draw order (the trick winner
            // draws first, which can differ between games). The final draw
            // round — the briscola spy — is excluded (pre < 4 there), so the
            // spy may land with a different seat than in game 1.
            *env.players[0].hand.last_mut().unwrap() = per_trick[trick_idx][0];
            *env.players[1].hand.last_mut().unwrap() = per_trick[trick_idx][1];
            trick_idx += 1;
        }

        if env.turn_counter != prev_turn {
            let new_pts = [env.players[0].points, env.players[1].points];
            let trick_pts = (new_pts[0] + new_pts[1]) - (prev_pts[0] + prev_pts[1]);
            let hist_len = env.played_cards_history.len();
            let had_briscola = env.played_cards_history[hist_len - 2..hist_len]
                .iter().any(|c| c.is_briscola);
            // agent 0 = player 1 in game 2
            let trick_winner_agent = agent_for_player_g2[env.turn_order[0]];
            let p_first = env.turn_history[hist_len - 2];
            let c_first = env.played_cards_history[hist_len - 2];
            let c_second = env.played_cards_history[hist_len - 1];
            let card_of = |pid: usize| if pid == p_first { c_first } else { c_second };
            let agent0_player = agent_for_player_g2.iter().position(|&a| a == 0).unwrap();
            let agent1_player = agent_for_player_g2.iter().position(|&a| a == 1).unwrap();
            tricks2.push(TrickRecord {
                trick_num: tricks2.len() as u8 + 1,
                trick_points: trick_pts,
                had_briscola,
                agent0_cumpts: new_pts[1],
                agent1_cumpts: new_pts[0],
                agent0_took_trick: trick_winner_agent == 0,
                agent0_card: card_of(agent0_player),
                agent1_card: card_of(agent1_player),
                first_agent: agent_for_player_g2[p_first] as u8,
            });
            prev_pts = new_pts;
            prev_turn = env.turn_counter;
        }
    }
    let game2 = make_record(env, [1, 0], tricks2);

    BenchmarkRound {
        game1,
        game2,
        deals: DealRecord { briscola, initial, per_trick },
    }
}

pub fn benchmark_n_games(agents: Vec<Box<dyn BriscolaAgent>>, n_rounds: usize, csv: bool) {
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

    if csv {
        println!("round,game,trick,agent0,agent1,briscola_suit,had_briscola,trick_points,agent0_cumpts,agent1_cumpts,agent0_took_trick,agent0_card,agent1_card,first_agent,agent0_final,agent1_final,game_winner");
    }

    for round_idx in 1..=n_rounds {
        let round = benchmark_game(&mut env, &agents);

        for (record, game_num) in [(&round.game1, 1usize), (&round.game2, 2usize)] {
            match record.winning_agent {
                Some(0) => wins[0] += 1,
                Some(1) => wins[1] += 1,
                _ => draws += 1,
            }
            points_total[0] += record.agent_points[0] as i64;
            points_total[1] += record.agent_points[1] as i64;

            if csv {
                let winner_str = match record.winning_agent {
                    Some(0) => "agent0",
                    Some(1) => "agent1",
                    _ => "draw",
                };
                for trick in &record.tricks {
                    println!(
                        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                        round_idx,
                        game_num,
                        trick.trick_num,
                        names[0],
                        names[1],
                        round.deals.briscola.name(),
                        trick.had_briscola as u8,
                        trick.trick_points,
                        trick.agent0_cumpts,
                        trick.agent1_cumpts,
                        trick.agent0_took_trick as u8,
                        trick.agent0_card.name(),
                        trick.agent1_card.name(),
                        trick.first_agent,
                        record.agent_points[0],
                        record.agent_points[1],
                        winner_str,
                    );
                }
            }
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

    if !csv {
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

        assert_eq!(round.deals.briscola, env.dealer.briscola);
    }

    #[test]
    fn benchmark_game_agents_swap_cards_in_game2() {
        // Game 2 is game 1 played from the opposite side: each agent should
        // play the cards its opponent played in game 1. The whole 20-card set a
        // player holds gets played over the game, so comparing played-card sets
        // is exact — except the unforced briscola-spy draw, which can swap one
        // card between the two seats (symmetric difference up to 2).
        use std::collections::HashSet;

        for _ in 0..30 {
            let mut env = make_env();
            let round = benchmark_game(&mut env, &make_agents());

            let cards = |tricks: &[TrickRecord], agent0: bool| -> HashSet<(BriscolaFace, BriscolaSuit)> {
                tricks.iter().map(|t| {
                    let c = if agent0 { t.agent0_card } else { t.agent1_card };
                    (c.face, c.suit)
                }).collect()
            };

            // agent 0's game-2 cards vs agent 1's game-1 cards (and vice versa)
            let a0_g2 = cards(&round.game2.tricks, true);
            let a1_g1 = cards(&round.game1.tricks, false);
            let a1_g2 = cards(&round.game2.tricks, false);
            let a0_g1 = cards(&round.game1.tricks, true);

            assert_eq!(a0_g2.len(), 20);
            assert_eq!(a1_g2.len(), 20);
            assert!(
                a0_g2.symmetric_difference(&a1_g1).count() <= 2,
                "agent 0 should replay agent 1's game-1 cards (±spy)"
            );
            assert!(
                a1_g2.symmetric_difference(&a0_g1).count() <= 2,
                "agent 1 should replay agent 0's game-1 cards (±spy)"
            );
        }
    }

    #[test]
    fn benchmark_game_tricks_count() {
        let mut env = make_env();
        let round = benchmark_game(&mut env, &make_agents());

        assert_eq!(round.game1.tricks.len(), 20);
        assert_eq!(round.game2.tricks.len(), 20);
    }

    #[test]
    fn benchmark_game_tricks_cumpts_match_final() {
        for _ in 0..10 {
            let mut env = make_env();
            let round = benchmark_game(&mut env, &make_agents());

            for record in [&round.game1, &round.game2] {
                let last = record.tricks.last().unwrap();
                assert_eq!(last.agent0_cumpts, record.agent_points[0]);
                assert_eq!(last.agent1_cumpts, record.agent_points[1]);
            }
        }
    }

    #[test]
    fn benchmark_game_trick_points_sum_to_120() {
        for _ in 0..10 {
            let mut env = make_env();
            let round = benchmark_game(&mut env, &make_agents());

            for record in [&round.game1, &round.game2] {
                let total: i32 = record.tricks.iter().map(|t| t.trick_points).sum();
                assert_eq!(total, 120);
            }
        }
    }

    #[test]
    fn benchmark_n_games_does_not_panic() {
        benchmark_n_games(make_agents(), 3, false);
    }
}
