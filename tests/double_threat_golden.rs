use menace::tictactoe::{analyze_menace_positions, build_reduced_game_tree};

#[test]
fn double_threat_positions_match_golden_list() {
    let tree = build_reduced_game_tree(true, false);
    let stats = analyze_menace_positions(&tree);
    let mut observed = stats.double_threat_positions.clone();
    let mut golden: Vec<String> = include_str!("../resources/double_threat_positions.txt")
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.trim().to_string())
        .collect();

    observed.sort();
    golden.sort();

    assert_eq!(observed.len(), 17, "expected 17 double-threat positions");
    assert_eq!(
        observed, golden,
        "double-threat positions differ from golden set"
    );
}
