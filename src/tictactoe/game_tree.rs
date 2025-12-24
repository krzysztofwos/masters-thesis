//! Game tree construction and MENACE-specific analysis utilities

use std::collections::{HashMap, HashSet, VecDeque};

use super::{Player, board::BoardState, lines::LineAnalyzer};
use crate::identifiers::{MoveId, StateId};

/// Game tree representation for Tic-Tac-Toe decision-making.
///
/// This structure captures the complete state space of Tic-Tac-Toe with
/// symmetry reduction applied via the D4 dihedral group.
#[derive(Debug, Clone)]
pub struct GameTree {
    /// All states in the game tree
    pub states: HashSet<StateId>,
    /// Moves (transitions) from each state: state -> [(move_id, target_state)]
    pub moves: HashMap<StateId, Vec<(MoveId, StateId)>>,
    /// Canonical board states by label
    pub canonical_states: HashMap<String, BoardState>,
    /// Mapping from all states to their canonical representatives
    pub state_to_canonical: HashMap<String, String>,
}

impl GameTree {
    /// Create a new empty game tree
    pub fn new() -> Self {
        Self {
            states: HashSet::new(),
            moves: HashMap::new(),
            canonical_states: HashMap::new(),
            state_to_canonical: HashMap::new(),
        }
    }
}

impl Default for GameTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Build the complete Tic-Tac-Toe game tree with D4 symmetry reduction
pub fn build_reduced_game_tree(x_only: bool, menace_rules: bool) -> GameTree {
    let mut tree = GameTree::new();

    let mut to_explore = vec![BoardState::new()];
    if !x_only {
        to_explore.push(BoardState::new_with_player(Player::O));
    }
    let mut explored = HashSet::new();

    while let Some(state) = to_explore.pop() {
        let state_label = state.encode();
        if explored.contains(&state_label) {
            continue;
        }
        explored.insert(state_label.clone());

        let canonical = state.canonical();
        let canonical_label = canonical.encode();
        let canonical_state_id = StateId::from(canonical_label.as_str());
        let include_current = should_include_state(&canonical, x_only, menace_rules);

        tree.state_to_canonical
            .insert(state_label.clone(), canonical_label.clone());

        if !tree.canonical_states.contains_key(&canonical_label) {
            tree.canonical_states
                .insert(canonical_label.clone(), canonical);
        }

        if include_current && !tree.states.contains(&canonical_state_id) {
            tree.states.insert(canonical_state_id.clone());
        }

        for move_pos in canonical.legal_moves() {
            let next_state = canonical
                .make_move(move_pos)
                .expect("move positions always legal during construction");
            to_explore.push(next_state);

            let next_canonical = next_state.canonical();
            let next_canonical_label = next_canonical.encode();
            let next_canonical_state_id = StateId::from(next_canonical_label.as_str());
            let include_target = should_include_state(&next_canonical, x_only, menace_rules);

            if include_target && !tree.states.contains(&next_canonical_state_id) {
                tree.states.insert(next_canonical_state_id.clone());
            }

            let move_id = MoveId::from(format!("{canonical_label}_{move_pos}"));

            if include_current && include_target {
                tree.moves
                    .entry(canonical_state_id.clone())
                    .or_default()
                    .push((move_id, next_canonical_state_id));
            }
        }
    }

    tree
}

/// Collect canonical state labels reachable from the standard starting position
/// (empty board with X to move), under D4 symmetry reduction.
///
/// This corresponds to the classic 765-canonical-state enumeration.
pub fn collect_reachable_canonical_labels() -> Vec<String> {
    let mut canonical_states = HashSet::new();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    let root = BoardState::new();
    queue.push_back(root);
    visited.insert(root.encode());

    while let Some(state) = queue.pop_front() {
        canonical_states.insert(state.canonical().encode());

        if state.is_terminal() {
            continue;
        }

        for pos in state.empty_positions() {
            let Ok(next) = state.make_move(pos) else {
                continue;
            };
            let key = next.encode();
            if visited.insert(key) {
                queue.push_back(next);
            }
        }
    }

    let mut labels: Vec<String> = canonical_states.into_iter().collect();
    labels.sort();
    labels
}

fn should_include_state(state: &BoardState, x_only: bool, menace_rules: bool) -> bool {
    if menace_rules && state.to_move == Player::X && !state.is_terminal() {
        let piece_count = state.occupied_count();
        let is_forced = state.has_forced_move();
        let is_double_threat = state.opponent_has_double_threat();

        !(piece_count == 8 && is_forced || piece_count == 6 && is_double_threat)
    } else {
        !x_only || state.to_move == Player::X || state.is_terminal()
    }
}

/// Statistics about MENACE matchbox filtering pipeline
#[derive(Debug, Clone)]
pub struct MenacePositionStats {
    pub total_x_positions: usize,
    pub ply_counts: [usize; 5],
    pub forced_positions: Vec<String>,
    pub double_threat_positions: Vec<String>,
}

/// Analyze MENACE matchbox selection criteria on game tree states
pub fn analyze_menace_positions(tree: &GameTree) -> MenacePositionStats {
    let mut states: Vec<_> = tree.states.iter().map(|id| id.to_string()).collect();
    states.sort();

    let mut ply_counts = [0usize; 5];
    let mut forced = Vec::new();
    let mut double_threats = Vec::new();

    for label in states.iter() {
        if !label.ends_with("_X") {
            continue;
        }

        let Some(state) = BoardState::from_label(label).ok() else {
            continue;
        };

        if state.is_terminal() {
            continue;
        }

        let piece_count = state.occupied_count();
        let ply_index = piece_count / 2;
        if ply_index < ply_counts.len() {
            ply_counts[ply_index] += 1;
        }

        if piece_count == 8 {
            forced.push(label.clone());
            continue;
        }

        if piece_count != 6 {
            continue;
        }

        if LineAnalyzer::has_immediate_win(&state.cells, Player::X) {
            continue;
        }

        let o_winning_moves = LineAnalyzer::winning_moves(&state.cells, Player::O);
        if o_winning_moves.len() >= 2 {
            double_threats.push(label.clone());
        }
    }

    forced.sort();
    double_threats.sort();

    MenacePositionStats {
        total_x_positions: ply_counts.iter().sum(),
        ply_counts,
        forced_positions: forced,
        double_threat_positions: double_threats,
    }
}

/// Render board as "XXX / OO. / ..." helper string
pub fn format_board(label: &str) -> Option<String> {
    let mut parts = label.split('_');
    let board = parts.next()?;
    if board.len() != 9 {
        return None;
    }
    let chars: Vec<char> = board.chars().collect();
    Some(format!(
        "{}{}{} / {}{}{} / {}{}{}",
        chars[0], chars[1], chars[2], chars[3], chars[4], chars[5], chars[6], chars[7], chars[8]
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_moves_map_back_to_legal_positions() {
        let tree = build_reduced_game_tree(true, false);
        for state in tree.canonical_states.values() {
            if state.is_terminal() {
                continue;
            }
            let ctx = state.canonical_context();
            let canonical_moves = ctx.state.legal_moves();
            let legal_moves = state.legal_moves();
            for canonical_move in canonical_moves {
                let actual_move = ctx.transform.apply_inverse_to_pos(canonical_move);
                assert!(
                    legal_moves.contains(&actual_move),
                    "inverse mapped move not legal"
                );
                let round_trip = ctx.transform.transform_position(actual_move);
                assert_eq!(canonical_move, round_trip);
            }
        }
    }

    #[test]
    fn menace_state_counts_match_expected() {
        let tree = build_reduced_game_tree(true, false);
        let stats_all = analyze_menace_positions(&tree);
        assert_eq!(stats_all.total_x_positions, 338);
        assert_eq!(stats_all.ply_counts, [1, 12, 108, 183, 34]);
        assert_eq!(stats_all.forced_positions.len(), 34);
        assert_eq!(stats_all.double_threat_positions.len(), 17);

        let menace_tree = build_reduced_game_tree(true, true);
        let menace_stats = analyze_menace_positions(&menace_tree);
        assert_eq!(menace_stats.total_x_positions, 287);
        assert_eq!(menace_stats.ply_counts[4], 0);
        assert_eq!(menace_stats.forced_positions.len(), 0);
        assert_eq!(menace_stats.double_threat_positions.len(), 0);
    }

    #[test]
    fn reachable_canonical_labels_match_expected() {
        assert_eq!(collect_reachable_canonical_labels().len(), 765);
    }
}
