use std::collections::{HashMap, HashSet};

use menace::{
    Result,
    pipeline::OptimalLearner,
    types::CanonicalLabel,
    workspace::{InitialBeadSchedule, MenaceWorkspace, RestockMode, StateFilter},
};

pub mod workspaces {
    use super::*;

    /// Controls how the optimal policy is encoded.
    #[derive(Clone, Copy, Debug)]
    #[allow(dead_code)]
    pub enum OptimalPolicyMode {
        /// Store a single canonical minimax action per state (legacy behaviour).
        SingleBest,
        /// Distribute weight uniformly across all minimax-optimal moves.
        FullDistribution,
    }

    /// Build a MENACE workspace whose move weights are replaced with the minimax-optimal policy.
    ///
    /// `filter` picks which canonical states are included (e.g., `StateFilter::Michie` for the
    /// historical 287 matchboxes or `StateFilter::Both` for player-agnostic analyses).
    /// `mode` chooses whether to keep only one canonical move or a distribution over all optimal moves.
    pub fn build_optimal_workspace(
        filter: StateFilter,
        log_progress: bool,
        mode: OptimalPolicyMode,
    ) -> Result<MenaceWorkspace> {
        let mut workspace = MenaceWorkspace::with_config(
            filter,
            RestockMode::None,
            InitialBeadSchedule::default(),
        )?;

        let mut optimal = OptimalLearner::new("Optimal".to_string());
        let decision_labels: Vec<String> = workspace.decision_labels().cloned().collect();

        if log_progress {
            println!(
                "  Extracting optimal policy for {} states...",
                decision_labels.len()
            );
        }

        for (i, label_str) in decision_labels.iter().enumerate() {
            let label = CanonicalLabel::parse(label_str)?;

            if let Some(state) = workspace.state(&label)
                && !state.is_terminal()
            {
                let move_values = optimal.evaluate_moves(state);
                if move_values.is_empty() {
                    continue;
                }

                let is_x = state.to_move == menace::tictactoe::Player::X;
                let best_value = if is_x {
                    move_values
                        .iter()
                        .map(|(_, value)| *value)
                        .max()
                        .unwrap_or(0)
                } else {
                    move_values
                        .iter()
                        .map(|(_, value)| *value)
                        .min()
                        .unwrap_or(0)
                };

                let mut new_weights = HashMap::new();
                match mode {
                    OptimalPolicyMode::SingleBest => {
                        if let Some((selected_move, _)) =
                            move_values.iter().find(|(_, value)| *value == best_value)
                        {
                            for &(mv, _) in &move_values {
                                let weight = if mv == *selected_move { 100.0 } else { 1.0 };
                                new_weights.insert(mv, weight);
                            }
                        }
                    }
                    OptimalPolicyMode::FullDistribution => {
                        let optimal_moves: HashSet<usize> = move_values
                            .iter()
                            .filter(|(_, value)| *value == best_value)
                            .map(|(mv, _)| *mv)
                            .collect();
                        let count = optimal_moves.len().max(1);
                        let optimal_weight = 100.0 / count as f64;

                        for &(mv, _) in &move_values {
                            let weight = if optimal_moves.contains(&mv) {
                                optimal_weight
                            } else {
                                1.0
                            };
                            new_weights.insert(mv, weight);
                        }
                    }
                }

                if !new_weights.is_empty() {
                    workspace.set_move_weights(&label, &new_weights);
                }
            }

            if log_progress && (i + 1).is_multiple_of(50) {
                println!("    Processed {}/{} states", i + 1, decision_labels.len());
            }
        }

        if log_progress {
            println!(
                "    Processed {}/{} states",
                decision_labels.len(),
                decision_labels.len()
            );
        }

        Ok(workspace)
    }
}
