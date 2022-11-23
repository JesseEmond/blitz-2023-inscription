use criterion::{criterion_group, criterion_main, Criterion};
use serde_json::{Value};
use std::fs;
use std::sync::Arc;

use blitz_bot::game_interface::{GameTick};
use blitz_bot::graph::{Graph};
use blitz_bot::simple_ant_colony_optimization::{SimpleAntColonyOptimizationSolver};
use blitz_bot::simple_held_karp::{SimpleExactTspSolver};
use blitz_bot::simple_graph::{SimpleGraph};
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
    let mut group = c.benchmark_group("graph_creation");
    group.bench_function("simple pathfinding&graph", |b| b.iter(|| {
        SimpleGraph::new(&game)
    }));
    group.bench_function("optimized pathfinding&graph", |b| b.iter(|| {
        Graph::new(&game)
    }));
    group.finish();
}

fn bench_ant_colony_optimization(c: &mut Criterion) {
    let game = make_game();
    let simple_graph = Arc::new(SimpleGraph::new(&game));
    let graph = Arc::new(Graph::new(&game));
    let mut group = c.benchmark_group("ant_colony_optimization");
    group.bench_function("simple ACO", |b| b.iter(|| {
        let mut solver = SimpleAntColonyOptimizationSolver::default();
        solver.do_solve(&simple_graph)
    }));
    group.bench_function("optimized ACO", |b| b.iter(|| {
        let mut solver = AntColonyOptimizationSolver::default();
        solver.do_solve(&graph)
    }));
    group.finish();
}

fn bench_held_karp(c: &mut Criterion) {
    let game = make_game();
    let simple_graph = Arc::new(SimpleGraph::new(&game));
    let graph = Arc::new(Graph::new(&game));
    let mut group = c.benchmark_group("held_karp");
    group.bench_function("simple held_karp", |b| b.iter(|| {
        let mut solver = SimpleExactTspSolver{};
        solver.do_solve(&simple_graph)
    }));
    group.bench_function("optimized held_karp", |b| b.iter(|| {
        let mut solver = ExactTspSolver{};
        solver.do_solve(&graph)
    }));
    group.finish();
}

criterion_group!{
    name = benches;
    // Limit sample size given the slow processing. Results will be noisy.
    config = Criterion::default().sample_size(50);
    targets = bench_graph_creation, bench_ant_colony_optimization, bench_held_karp,
}
criterion_main!(benches);
