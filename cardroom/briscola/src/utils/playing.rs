use crate::agents::generic::BriscolaAgent;
use crate::env::env::{BriscolaEnv, BriscolaResult};
use indicatif::{ProgressBar, ProgressStyle};

pub fn play_game(env: &mut BriscolaEnv, agents: &[Box<dyn BriscolaAgent>], render: bool) {
    env.reset();
    if render {
        env.render();
    }
    while !env.done {
        let action = agents[env.current_player_id].select_action(env.get_obs());
        env.step(action);
        if render {
            env.render();
        }
    }
}

pub fn play_n_games(agents: Vec<Box<dyn BriscolaAgent>>, n_games: usize) {

    let names: Vec<String> = agents.iter().map(|a| a.name().to_string()).collect();
    let mut env = BriscolaEnv::new_with_names(names.clone());
    let mut wins = [0usize; 2];
    let mut draws = 0usize;
    let mut points_total = [0i64; 2];

    let pb = ProgressBar::new(n_games as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len}  {msg}  ({per_sec}, eta {eta})",
        )
        .unwrap()
        .progress_chars("=>-"),
    );

    for _ in 0..n_games {
        play_game(&mut env, &agents, false);
        match env.result {
            BriscolaResult::P0Win => wins[0] += 1,
            BriscolaResult::P1Win => wins[1] += 1,
            BriscolaResult::Draw => draws += 1,
            BriscolaResult::InProgress => unreachable!(),
        }
        points_total[0] += env.players[0].points as i64;
        points_total[1] += env.players[1].points as i64;

        let played = (wins[0] + wins[1] + draws) as f64;
        pb.set_message(format!(
            "{}: {:.0}%  {}: {:.0}%  D: {:.0}%",
            names[0],
            wins[0] as f64 / played * 100.0,
            names[1],
            wins[1] as f64 / played * 100.0,
            draws as f64 / played * 100.0,
        ));
        pb.inc(1);
    }

    pb.finish_and_clear();

    let n = n_games as f64;
    println!("Results over {} games  ({}=P0  vs  {}=P1)", n_games, names[0], names[1]);
    println!("  {} wins: {:4}  ({:.1}%)", names[0], wins[0], wins[0] as f64 / n * 100.0);
    println!("  {} wins: {:4}  ({:.1}%)", names[1], wins[1], wins[1] as f64 / n * 100.0);
    println!("  Draws:   {:4}  ({:.1}%)", draws, draws as f64 / n * 100.0);
    println!("  Avg {} points: {:.1}", names[0], points_total[0] as f64 / n);
    println!("  Avg {} points: {:.1}", names[1], points_total[1] as f64 / n);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::random::RandomAgent;

    fn make_agents() -> Vec<Box<dyn BriscolaAgent>> {
        vec![Box::new(RandomAgent::default()), Box::new(RandomAgent::default())]
    }

    #[test]
    fn play_game_completes() {
        let mut env = BriscolaEnv::new();
        play_game(&mut env, &make_agents(), false);

        assert!(env.done);
        assert_ne!(env.result, BriscolaResult::InProgress);
        assert_eq!(env.players.iter().map(|p| p.points).sum::<i32>(), 120);
        assert_eq!(env.turn_history.len(), 40);
    }

    #[test]
    fn play_game_resets_on_second_call() {
        let mut env = BriscolaEnv::new();
        let agents = make_agents();
        play_game(&mut env, &agents, false);
        play_game(&mut env, &agents, false);

        assert!(env.done);
        assert_eq!(env.players.iter().map(|p| p.points).sum::<i32>(), 120);
    }

    #[test]
    fn play_n_games_does_not_panic() {
        play_n_games(make_agents(), 5);
    }
}
