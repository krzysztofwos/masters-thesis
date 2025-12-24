use std::collections::{HashMap, HashSet};

use menace::{
    analysis::GameAnalysis,
    menace::{
        MenaceAgent, compute_optimal_policy, kl_divergence, kl_divergence_weighted,
        optimal_move_distribution,
    },
    tictactoe::{BoardState, Player, analyze_menace_positions, build_reduced_game_tree},
    workspace::StateFilter,
};

fn enumerate_board_strings() -> Vec<String> {
    let mut boards = Vec::with_capacity(3usize.pow(9));
    for index in 0..3usize.pow(9) {
        let mut n = index;
        let mut chars = ['.'; 9];
        for slot in (0..9).rev() {
            let digit = n % 3;
            n /= 3;
            chars[slot] = match digit {
                0 => '.',
                1 => 'X',
                2 => 'O',
                _ => unreachable!(),
            };
        }
        boards.push(chars.iter().collect());
    }
    boards
}

struct GameEnumerationStats {
    total_games: usize,
    length_histogram: HashMap<usize, usize>,
    x_wins: usize,
    o_wins: usize,
    draws: usize,
}

fn enumerate_all_games() -> GameEnumerationStats {
    fn traverse(state: &BoardState, history: &mut Vec<usize>, stats: &mut GameEnumerationStats) {
        if state.is_terminal() {
            stats.total_games += 1;
            *stats.length_histogram.entry(history.len()).or_insert(0) += 1;

            match state.winner() {
                Some(Player::X) => stats.x_wins += 1,
                Some(Player::O) => stats.o_wins += 1,
                None => stats.draws += 1,
            }
            return;
        }

        for pos in state.empty_positions() {
            let next = state.make_move(pos).expect("Expected legal move");
            history.push(pos);
            traverse(&next, history, stats);
            history.pop();
        }
    }

    let mut stats = GameEnumerationStats {
        total_games: 0,
        length_histogram: HashMap::new(),
        x_wins: 0,
        o_wins: 0,
        draws: 0,
    };

    let mut history = Vec::new();
    traverse(&BoardState::new(), &mut history, &mut stats);

    stats
}

#[test]
fn verify_state_space_counts() {
    const TOTAL_CONFIGURATIONS: usize = 19_683; // 3^9
    const TURN_VALID_STATES: usize = 6_046;
    const VALID_STATES: usize = 5_478;
    const INVALID_CONTINUATIONS: usize = 568;
    const CANONICAL_STATES: usize = 765;
    const DISTINCT_TERMINALS: usize = 958;
    const CANONICAL_TERMINALS: usize = 138;
    const CANONICAL_X_WINS: usize = 91;
    const CANONICAL_O_WINS: usize = 44;
    const CANONICAL_DRAWS: usize = 3;

    let mut turn_valid = 0usize;
    let mut valid_states = 0usize;
    let mut canonical_states = HashSet::new();
    let mut canonical_cells = HashSet::new();
    let mut canonical_per_ply: Vec<HashSet<String>> = (0..10).map(|_| HashSet::new()).collect();
    let mut terminal_boards = HashSet::new();
    let mut canonical_terminal_outcomes: HashMap<String, Option<Player>> = HashMap::new();

    for board in enumerate_board_strings() {
        if let Ok(state) = BoardState::from_string(&board) {
            turn_valid += 1;

            if state.is_valid() {
                valid_states += 1;

                let canonical = state.canonical();
                canonical_states.insert(canonical.encode());
                let canonical_cell_string: String =
                    canonical.cells.iter().map(|cell| cell.to_char()).collect();
                canonical_cells.insert(canonical_cell_string.clone());
                canonical_per_ply[state.occupied_count()].insert(canonical_cell_string.clone());

                if state.is_terminal() {
                    terminal_boards.insert(board.clone());
                    let winner = state.winner();
                    canonical_terminal_outcomes
                        .entry(canonical_cell_string)
                        .and_modify(|existing| {
                            assert_eq!(*existing, winner, "Canonical terminal outcome mismatch");
                        })
                        .or_insert(winner);
                }
            }
        }
    }

    assert_eq!(TOTAL_CONFIGURATIONS, 3usize.pow(9));
    assert_eq!(turn_valid, TURN_VALID_STATES);
    assert_eq!(valid_states, VALID_STATES);
    assert_eq!(turn_valid - valid_states, INVALID_CONTINUATIONS);
    assert_eq!(canonical_cells.len(), CANONICAL_STATES);
    assert_eq!(terminal_boards.len(), DISTINCT_TERMINALS);
    assert_eq!(canonical_terminal_outcomes.len(), CANONICAL_TERMINALS);

    const EXPECTED_PER_PLY: [usize; 10] = [
        1, // ply 0
        3, 12, 38, 108, 174, 204, 153, 57, 15,
    ];

    for (ply, &expected) in EXPECTED_PER_PLY.iter().enumerate() {
        assert_eq!(
            canonical_per_ply[ply].len(),
            expected,
            "Canonical ply count mismatch for ply {ply}"
        );
    }

    let mut x_wins = 0usize;
    let mut o_wins = 0usize;
    let mut draws = 0usize;
    for outcome in canonical_terminal_outcomes.values() {
        match outcome {
            Some(Player::X) => x_wins += 1,
            Some(Player::O) => o_wins += 1,
            None => draws += 1,
        }
    }

    assert_eq!(x_wins, CANONICAL_X_WINS);
    assert_eq!(o_wins, CANONICAL_O_WINS);
    assert_eq!(draws, CANONICAL_DRAWS);
}

#[test]
fn verify_game_and_trajectory_counts() {
    const TOTAL_GAMES: usize = 255_168;
    const LENGTH_DISTRIBUTION: &[(usize, usize)] = &[
        (5, 1_440),
        (6, 5_328),
        (7, 47_952),
        (8, 72_576),
        (9, 127_872),
    ];
    const X_WINS: usize = 131_184;
    const O_WINS: usize = 77_904;
    const DRAWS: usize = 46_080;
    const EXPECTED_AVG_LENGTH: f64 = 8.255;
    const CANONICAL_TRAJECTORIES: usize = 26_830;

    let enumeration = enumerate_all_games();

    assert_eq!(enumeration.total_games, TOTAL_GAMES);
    for &(length, expected_count) in LENGTH_DISTRIBUTION {
        let actual = enumeration
            .length_histogram
            .get(&length)
            .copied()
            .unwrap_or_default();
        assert_eq!(
            actual, expected_count,
            "Unexpected count for length {length}"
        );
    }

    assert_eq!(enumeration.x_wins, X_WINS);
    assert_eq!(enumeration.o_wins, O_WINS);
    assert_eq!(enumeration.draws, DRAWS);

    let total_moves: usize = enumeration
        .length_histogram
        .iter()
        .map(|(len, count)| len * count)
        .sum();
    let avg_length = total_moves as f64 / enumeration.total_games as f64;
    assert!(
        (avg_length - EXPECTED_AVG_LENGTH).abs() < 1e-3,
        "Average length mismatch: {avg_length}"
    );

    // Cross-check that the library analysis routine matches the primary counts.
    let analysis = GameAnalysis::analyze().unwrap();
    assert_eq!(analysis.total_games, enumeration.total_games);
    assert_eq!(analysis.outcome_distribution.x_wins, enumeration.x_wins);
    assert_eq!(analysis.outcome_distribution.o_wins, enumeration.o_wins);
    assert_eq!(analysis.outcome_distribution.draws, enumeration.draws);

    eprintln!(
        "Canonical trajectories: {} (expected {})",
        analysis.canonical_trajectories, CANONICAL_TRAJECTORIES
    );
    eprintln!(
        "Symmetry reduction: {:.2}x",
        analysis.total_games as f64 / analysis.canonical_trajectories as f64
    );

    assert_eq!(analysis.canonical_trajectories, CANONICAL_TRAJECTORIES);
}

#[test]
fn verify_optimal_policy_empty_board() {
    let policies = compute_optimal_policy();
    let empty_key = BoardState::new().encode();
    let policy = policies
        .get(&empty_key)
        .expect("Policy for empty board should exist");

    assert_eq!(policy.value, 0, "Optimal value should indicate a draw");
    assert_eq!(
        policy.optimal_moves.len(),
        9,
        "All first moves should be evaluated as drawing options"
    );

    let distributions = optimal_move_distribution();
    let dist = distributions
        .get(&empty_key)
        .expect("Distribution for empty board should exist");
    let prob_sum: f64 = dist.values().sum();
    assert!((prob_sum - 1.0).abs() < 1e-9);
    for (&mv, &prob) in dist {
        assert!(
            (prob - 1.0 / dist.len() as f64).abs() < 1e-9,
            "Expected uniform probability for move {mv}"
        );
    }
}

#[test]
fn verify_menace_pipeline_counts() {
    const MENACE_X_TO_MOVE: usize = 338;
    const MENACE_DECISION_POINTS: usize = 304;
    const MENACE_MATCHBOXES: usize = 287;
    const FORCED_POSITIONS: usize = 34;
    const DOUBLE_THREAT_POSITIONS: usize = 17;

    let tree = build_reduced_game_tree(true, false);
    let stats_all = analyze_menace_positions(&tree);
    assert_eq!(stats_all.total_x_positions, MENACE_X_TO_MOVE);
    assert_eq!(stats_all.forced_positions.len(), FORCED_POSITIONS);
    assert_eq!(
        stats_all.double_threat_positions.len(),
        DOUBLE_THREAT_POSITIONS
    );
    assert_eq!(
        stats_all.total_x_positions - stats_all.forced_positions.len(),
        MENACE_DECISION_POINTS
    );

    let menace_tree = build_reduced_game_tree(true, true);
    let menace_stats = analyze_menace_positions(&menace_tree);
    assert_eq!(menace_stats.total_x_positions, MENACE_MATCHBOXES);
    assert_eq!(menace_stats.forced_positions.len(), 0);
    assert_eq!(menace_stats.double_threat_positions.len(), 0);
}

#[test]
fn kl_divergence_matches_optimal_agent() {
    let optimal = optimal_move_distribution();
    let mut agent = MenaceAgent::builder()
        .seed(123)
        .filter(StateFilter::All)
        .build()
        .expect("agent construction should succeed");

    for (state_key, distribution) in &optimal {
        let label = menace::CanonicalLabel::parse(state_key).expect("valid canonical label");
        let mut weights = HashMap::new();
        for mv in distribution.keys() {
            weights.insert(*mv, 100.0);
        }
        agent.set_canonical_move_weights(&label, &weights);
    }

    let kl = kl_divergence(&agent, &optimal);
    assert!(
        kl.abs() < 1e-9,
        "KL divergence should be zero for optimal agent: {kl}"
    );

    let weights: HashMap<_, _> = optimal.keys().map(|k| (k.clone(), 1.0)).collect();
    let weighted = kl_divergence_weighted(&agent, &optimal, &weights);
    assert!(
        weighted.abs() < 1e-9,
        "Weighted KL should be zero for optimal agent: {weighted}"
    );
}
