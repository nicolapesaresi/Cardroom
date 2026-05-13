use briscola::agents::bot::BotAgent;
use briscola::agents::generic::BriscolaAgent;
use briscola::agents::mcts::MCTSAgent;
use briscola::agents::random::RandomAgent;
use briscola::env::env::{BriscolaEnv, BriscolaResult};
use indicatif::{ProgressBar, ProgressStyle};

const DEFAULT_N_GAMES: usize = 100;
const DEFAULT_MCTS_SIMS: usize = 1000;

fn make_agent(name: &str) -> Box<dyn BriscolaAgent> {
    match name {
        "bot" => Box::new(BotAgent::new()),
        "random" => Box::new(RandomAgent::new()),
        "mcts" => Box::new(MCTSAgent::new(DEFAULT_MCTS_SIMS)),
        _ => {
            eprintln!("Unknown agent '{}'. Available: bot, random, mcts", name);
            std::process::exit(1);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: play_n_games <agent0> <agent1> [n_games]");
        eprintln!("Available agents: bot, random, mcts");
        std::process::exit(1);
    }

    let n_games: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_N_GAMES);

    let agents: Vec<Box<dyn BriscolaAgent>> = vec![make_agent(&args[1]), make_agent(&args[2])];
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
        env.reset();
        while !env.done {
            let action = agents[env.current_player_id].select_action(env.get_obs());
            env.step(action);
        }
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
