// Binary that monitors a folder for new JSON games added, to evaluate them with
// a powerful solver (no time limit). Used to estimate score upper bounds on
// games collected, while it is running.
//
// Note that this does not simulate the game -- it trusts the solver's solution
// to be correct and bug-free when reporting scores.

use std::collections::{HashSet};
use std::{thread, time};
use std::sync::Arc;

use lazy_static::lazy_static;
use regex::Regex;
use serde_json::Error as JSONError;
use serde_json::Value;
use thiserror::Error;

use blitz_bot::game_interface::{GameTick};
use blitz_bot::graph::{Graph};
use blitz_bot::solvers::{OptimalSolver, Solver};

#[derive(Debug, Clone)]
struct SavedGame {
    id: u32,
    path: String,
}

#[derive(Error, Debug)]
enum GameEvalError {
    #[error("Failed reading the saved game")]
    ReadError(#[from] std::io::Error),
    #[error("Failed parsing the saved game")]
    ParseError(#[from] JSONError),
}

fn read_saved_games(directory: &str) -> Vec<SavedGame> {
    lazy_static! {
        static ref GAME_PATH: Regex = Regex::new(r".*?(\d+).json").unwrap();
    }
    let paths = std::fs::read_dir(directory).unwrap();
    let mut out = Vec::new();
    for path in paths {
        let path = path.unwrap().path().to_str().unwrap().to_string();
        if let Some(game_id) = GAME_PATH.captures(&path)
            .and_then(|caps| caps.get(1))
            .and_then(|id| Some(id.as_str().parse::<u32>().unwrap())) {
            out.push(SavedGame { path: path.to_string(), id: game_id });
        }
    }
    out
}

fn evaluate_game(saved_game: &SavedGame) -> Result<i32, GameEvalError> {
    let game_data = std::fs::read_to_string(&saved_game.path)?;
    let parsed: Value = serde_json::from_str(&game_data)?;
    let tick: GameTick = serde_json::from_value(parsed)?;
    let tick = Arc::new(tick);

    if tick.map.ports.len() >= 20 {
        let graph: Arc<Graph> = Arc::new(Graph::new(&tick));
        let solution = OptimalSolver{}.solve(&graph)
            .expect("no solution possible on game");
        Ok(solution.score)
    } else {
        Ok(0)
    }
}

fn main() {
    let directory = std::env::args().nth(1).expect("no directory given");
    let mut seen_games: HashSet<u32> = HashSet::new();
    let mut scores: Vec<i32> = Vec::new();

    loop {
        let unseen_games: Vec<SavedGame> = read_saved_games(directory.as_str())
            .iter().filter(|game| !seen_games.contains(&game.id))
            .cloned().collect();
        if !unseen_games.is_empty() {
            println!("{} unseen game(s)", unseen_games.len());
            for game in unseen_games {
                println!("  evaluating game #{}...", game.id);
                let score = evaluate_game(&game);
                match score {
                    Ok(score) => {
                        if score > 0 {
                            println!("  score: {} points", score);
                            scores.push(score);
                        } else {
                            println!("  less than 20 ports. Ignoring.");
                        }
                        seen_games.insert(game.id);
                    },
                    Err(err) => println!("  error: {err:?} Will retry."),
                };
            }

            println!("Score stats:");
            println!("  #: {}", scores.len());
            println!("Min: {}", scores.iter().min().unwrap());
            println!("Max: {}", scores.iter().max().unwrap());
            let mut score_sum: i64 = 0;
            for score in &scores {
                score_sum += *score as i64;
            }
            let avg = (score_sum as f32) / (scores.len() as f32);
            println!("Avg: {:.1}", avg);
            println!();
        }

        thread::sleep(time::Duration::from_secs(3));
    }
}
