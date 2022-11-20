use criterion::{criterion_group, criterion_main, Criterion};
use serde_json::{Value};
use std::fs;
use std::sync::Arc;

use blitz_bot::game_interface::{GameTick};
use blitz_bot::graph::{Graph};
use blitz_bot::solvers::{AntColonyOptimizationSolver, ExactTspSolver, Solver};


fn make_game() -> Arc<GameTick> {
    // Note this isn't great, we ideally shouldn't read from disk here.
    let game_file = "./games/35334.json";
    let game_json = fs::read_to_string(game_file)
        .expect("Couldn't read game file");
    let parsed: Value = serde_json::from_str(&game_json)
        .expect("Couldn't parse JSON in game file");
    let game = serde_json::from_value(parsed)
        .expect("Couldn't parse game tick in game file");
    Arc::new(game)
}

fn bench_graph_creation(c: &mut Criterion) {
    let game = make_game();
    c.bench_function("graph_creation", |b| b.iter(|| {
        Graph::new(&game)
    }));
}

fn bench_ant_colony_optimization(c: &mut Criterion) {
    let game = make_game();
    let graph = Arc::new(Graph::new(&game));
    let mut solver = AntColonyOptimizationSolver::default();

    c.bench_function("ant_colony_optimization", |b| b.iter(|| {
        solver.do_solve(&graph)
    }));
}

fn bench_held_karp(c: &mut Criterion) {
    let game = make_game();
    let graph = Arc::new(Graph::new(&game));
    let mut solver = ExactTspSolver{};
    c.bench_function("held_karp", |b| b.iter(|| {
        solver.do_solve(&graph)
    }));
}

criterion_group!{
    name = benches;
    // Limit sample size given the slow processing. Results will be noisy.
    config = Criterion::default().sample_size(10);
    targets = bench_graph_creation, bench_ant_colony_optimization, bench_held_karp,
}
criterion_main!(benches);
