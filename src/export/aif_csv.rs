//! CSV export for Active Inference decompositions
//!
//! This module provides functionality to export detailed Active Inference evaluations
//! for game states to CSV format for analysis and research.

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use crate::{
    Result,
    active_inference::{GenerativeModel, Opponent, PreferenceModel},
    beliefs::Beliefs,
    tictactoe::{BoardState, Player},
    workspace::StateFilter,
};

/// Configuration for Active Inference CSV export
pub struct AifExportConfig<'a> {
    /// State filter to apply (determines which states to export)
    pub state_filter: StateFilter,
    /// Optional list of specific state labels to export (if None, exports all matching filter)
    pub selected_states: Option<Vec<String>>,
    /// Whether to include O-to-move states (default: false, only X states)
    pub include_o_states: bool,
    /// The generative model for evaluation
    pub generative_model: &'a GenerativeModel,
    /// Preference model with all AIF parameters
    pub preferences: &'a PreferenceModel,
    /// Opponent model
    pub opponent: &'a dyn Opponent,
    /// Current beliefs about opponent
    pub beliefs: &'a Beliefs,
}

/// A single row in the Active Inference CSV export
#[derive(Debug, Clone)]
pub struct AifExportRecord {
    // State identification
    pub state_label: String,
    pub player: Player,
    pub action: usize,
    pub next_state: String,

    // Optimal policy
    pub q_star: f64,

    // Action evaluation components
    pub action_risk: f64,
    pub action_epistemic: f64,
    pub action_f_approx: f64,
    pub opponent_eig: f64,

    // Outcome probabilities
    pub p_win: f64,
    pub p_draw: f64,
    pub p_loss: f64,

    // Policy-level metrics (same for all actions in a state)
    pub f_exact: f64,
    pub policy_kl: f64,
    pub policy_prior: f64,

    // Configuration metadata
    pub beta: f64,
    pub lambda: f64,
    pub opponent_alpha: f64,
    pub risk_model: String,
    pub opponent_model: String,
    pub filter: String,
}

/// Exporter for Active Inference CSV files
pub struct AifCsvExporter;

impl AifCsvExporter {
    /// Collect state labels to export based on filter
    ///
    /// # Arguments
    /// * `filter` - State filter to apply
    /// * `selected_states` - Optional explicit list of states (overrides filter)
    /// * `include_o_states` - Whether to include O-to-move states
    ///
    /// # Returns
    /// Sorted, deduplicated list of state labels to export
    pub fn collect_states(
        filter: StateFilter,
        selected_states: Option<Vec<String>>,
        include_o_states: bool,
    ) -> Result<Vec<String>> {
        use crate::tictactoe::game_tree::build_reduced_game_tree;

        // Build game tree to get all canonical states
        let tree = build_reduced_game_tree(true, false);

        // Get forced/double-threat positions for filtering
        let (forced_set, double_set) = Self::get_position_sets()?;

        let mut state_labels: Vec<String> = if let Some(list) = selected_states {
            // Use explicit list
            list
        } else {
            // Collect based on filter
            tree.canonical_states
                .iter()
                .filter_map(|(label, state)| {
                    if state.is_terminal() {
                        return None;
                    }

                    // Only include X-to-move states (O states handled separately)
                    if state.to_move != Player::X {
                        return None;
                    }

                    // Apply filter
                    let include = match filter {
                        StateFilter::All => true,
                        StateFilter::DecisionOnly => !forced_set.contains(label),
                        StateFilter::Michie => {
                            !forced_set.contains(label) && !double_set.contains(label)
                        }
                        StateFilter::Both => {
                            !forced_set.contains(label) && !double_set.contains(label)
                        }
                    };

                    if include { Some(label.clone()) } else { None }
                })
                .collect()
        };

        // Add O-to-move states if requested
        if include_o_states {
            for (label, state) in &tree.canonical_states {
                if state.to_move == Player::O && !state.is_terminal() {
                    state_labels.push(label.clone());
                }
            }
        }

        state_labels.sort();
        state_labels.dedup();

        Ok(state_labels)
    }

    /// Get forced and double-threat position sets for filtering
    fn get_position_sets() -> Result<(
        std::collections::HashSet<String>,
        std::collections::HashSet<String>,
    )> {
        use std::collections::HashSet;

        use crate::tictactoe::game_tree::{analyze_menace_positions, build_reduced_game_tree};

        let tree = build_reduced_game_tree(true, false);
        let base_stats = analyze_menace_positions(&tree);

        let forced_set: HashSet<String> = base_stats.forced_positions.into_iter().collect();
        let double_set: HashSet<String> = base_stats.double_threat_positions.into_iter().collect();

        Ok((forced_set, double_set))
    }

    /// Export Active Inference decompositions to CSV
    ///
    /// # Arguments
    /// * `config` - Export configuration with all AIF parameters
    /// * `path` - Output CSV file path
    /// * `states_to_export` - List of state labels to evaluate and export
    ///
    /// # Returns
    /// Number of states successfully exported
    pub fn export(
        config: &AifExportConfig,
        path: &Path,
        states_to_export: Vec<String>,
    ) -> Result<usize> {
        let mut writer = BufWriter::new(File::create(path)?);

        // Write CSV header
        Self::write_header(&mut writer)?;

        let mut exported_count = 0;

        for state_label in states_to_export {
            // Parse the state
            let Some(board) = Self::parse_state(&state_label) else {
                eprintln!("[export] Skipping unparseable state: {state_label}");
                continue;
            };

            if board.is_terminal() {
                continue;
            }

            // Export based on player to move
            match board.to_move {
                Player::X => {
                    if Self::export_x_state(config, &mut writer, &state_label)? {
                        exported_count += 1;
                    }
                }
                Player::O => {
                    if config.include_o_states
                        && Self::export_o_state(config, &mut writer, &state_label, &board)?
                    {
                        exported_count += 1;
                    }
                }
            }
        }

        writer.flush()?;
        Ok(exported_count)
    }

    /// Write CSV header
    fn write_header<W: Write>(writer: &mut W) -> Result<()> {
        writeln!(
            writer,
            "state,player,action,next_state,q_star,action_risk,action_epistemic,action_F_approx,opponent_eig,p_win,p_draw,p_loss,F_exact,policy_KL,policy_prior,beta,lambda,opponent_alpha,opponent_model,filter,risk_model,units"
        )?;
        Ok(())
    }

    /// Export an X-to-move state
    fn export_x_state<W: Write>(
        config: &AifExportConfig,
        writer: &mut W,
        state_label: &str,
    ) -> Result<bool> {
        // Evaluate the state using exact policy computation
        let summary = config.generative_model.exact_state_summary(
            state_label,
            config.preferences,
            config.opponent,
            config.beliefs,
            Player::X,
        );

        if summary.actions.is_empty() {
            return Ok(false);
        }

        // Normalize policy priors
        let prior_weights: Vec<f64> = summary.actions.iter().map(|a| a.policy_prior).collect();
        let prior_norm = Self::normalize(&prior_weights);

        // Write a row for each action
        for (idx, action_eval) in summary.actions.iter().enumerate() {
            let q_star = summary.policy.q.get(idx).copied().unwrap_or(0.0);
            let approx_f =
                action_eval.risk - config.preferences.epistemic_weight * action_eval.epistemic;
            let prior = *prior_norm.get(idx).unwrap_or(&0.0);

            let record = AifExportRecord {
                state_label: state_label.to_string(),
                player: Player::X,
                action: action_eval.action,
                next_state: action_eval.next_state.clone(),
                q_star,
                action_risk: action_eval.risk,
                action_epistemic: action_eval.epistemic,
                action_f_approx: approx_f,
                opponent_eig: action_eval.opponent_eig,
                p_win: action_eval.outcome_distribution.x_win,
                p_draw: action_eval.outcome_distribution.draw,
                p_loss: action_eval.outcome_distribution.o_win,
                f_exact: summary.policy.f_exact,
                policy_kl: summary.policy.policy_kl,
                policy_prior: prior,
                beta: config.preferences.epistemic_weight,
                lambda: config.preferences.policy_lambda,
                opponent_alpha: config.preferences.opponent_dirichlet_alpha,
                risk_model: Self::risk_model_label(config.preferences),
                opponent_model: format!("{:?}", config.opponent.kind()),
                filter: Self::filter_label(config.state_filter),
            };

            Self::write_record(writer, &record)?;
        }

        Ok(true)
    }

    /// Export an O-to-move state (by swapping players)
    fn export_o_state<W: Write>(
        config: &AifExportConfig,
        writer: &mut W,
        state_label: &str,
        board: &BoardState,
    ) -> Result<bool> {
        // Swap players to evaluate from O's perspective
        let swapped = board.swap_players();
        let swapped_label = swapped.encode();

        let summary = config.generative_model.exact_state_summary(
            &swapped_label,
            config.preferences,
            config.opponent,
            config.beliefs,
            Player::O,
        );

        if summary.actions.is_empty() {
            return Ok(false);
        }

        let prior_weights: Vec<f64> = summary.actions.iter().map(|a| a.policy_prior).collect();
        let prior_norm = Self::normalize(&prior_weights);

        for (idx, action_eval) in summary.actions.iter().enumerate() {
            let q_star = summary.policy.q.get(idx).copied().unwrap_or(0.0);
            let approx_f =
                action_eval.risk - config.preferences.epistemic_weight * action_eval.epistemic;
            let prior = *prior_norm.get(idx).unwrap_or(&0.0);

            let record = AifExportRecord {
                state_label: state_label.to_string(),
                player: Player::O,
                action: action_eval.action,
                next_state: action_eval.next_state.clone(),
                q_star,
                action_risk: action_eval.risk,
                action_epistemic: action_eval.epistemic,
                action_f_approx: approx_f,
                opponent_eig: action_eval.opponent_eig,
                // Swap win/loss probabilities for O's perspective
                p_win: action_eval.outcome_distribution.o_win,
                p_draw: action_eval.outcome_distribution.draw,
                p_loss: action_eval.outcome_distribution.x_win,
                f_exact: summary.policy.f_exact,
                policy_kl: summary.policy.policy_kl,
                policy_prior: prior,
                beta: config.preferences.epistemic_weight,
                lambda: config.preferences.policy_lambda,
                opponent_alpha: config.preferences.opponent_dirichlet_alpha,
                risk_model: Self::risk_model_label(config.preferences),
                opponent_model: format!("{:?}", config.opponent.kind()),
                filter: Self::filter_label(config.state_filter),
            };

            Self::write_record(writer, &record)?;
        }

        Ok(true)
    }

    /// Write a single record to CSV
    fn write_record<W: Write>(writer: &mut W, record: &AifExportRecord) -> Result<()> {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},nats",
            record.state_label,
            match record.player {
                Player::X => "X",
                Player::O => "O",
            },
            record.action,
            record.next_state,
            Self::fmt_float(record.q_star),
            Self::fmt_float(record.action_risk),
            Self::fmt_float(record.action_epistemic),
            Self::fmt_float(record.action_f_approx),
            Self::fmt_float(record.opponent_eig),
            Self::fmt_float(record.p_win),
            Self::fmt_float(record.p_draw),
            Self::fmt_float(record.p_loss),
            Self::fmt_float(record.f_exact),
            Self::fmt_float(record.policy_kl),
            Self::fmt_float(record.policy_prior),
            Self::fmt_float(record.beta),
            Self::fmt_float(record.lambda),
            Self::fmt_float(record.opponent_alpha),
            record.opponent_model,
            record.filter,
            record.risk_model,
        )?;
        Ok(())
    }

    /// Format float for CSV (handles NaN/Inf)
    fn fmt_float(value: f64) -> String {
        if value.is_nan() {
            "nan".to_string()
        } else if value.is_infinite() {
            if value.is_sign_positive() {
                "inf".to_string()
            } else {
                "-inf".to_string()
            }
        } else {
            format!("{value:.6}")
        }
    }

    /// Normalize a distribution
    fn normalize(weights: &[f64]) -> Vec<f64> {
        let sum: f64 = weights.iter().sum();
        if sum > 0.0 {
            weights.iter().map(|&w| w / sum).collect()
        } else {
            vec![0.0; weights.len()]
        }
    }

    /// Get risk model label
    fn risk_model_label(prefs: &PreferenceModel) -> String {
        use crate::active_inference::RiskModel;
        match prefs.risk_model() {
            RiskModel::NegativeLogPreference => "log_pref",
            RiskModel::NegativeUtility => "neg_utility",
            RiskModel::KLDivergence => "kl_div",
        }
        .to_string()
    }

    /// Get filter label
    fn filter_label(filter: StateFilter) -> String {
        match filter {
            StateFilter::All => "all",
            StateFilter::DecisionOnly => "decision",
            StateFilter::Michie => "michie",
            StateFilter::Both => "both",
        }
        .to_string()
    }

    /// Parse board state from canonical label
    fn parse_state(label: &str) -> Option<BoardState> {
        use crate::tictactoe::Cell;

        let mut parts = label.split('_');
        let cells_part = parts.next()?;
        let to_move_part = parts.next()?;

        if parts.next().is_some() || cells_part.len() != 9 {
            return None;
        }

        let mut state = BoardState::new();
        for (idx, ch) in cells_part.chars().enumerate() {
            state.cells[idx] = match ch {
                'X' => Cell::X,
                'O' => Cell::O,
                '.' => Cell::Empty,
                _ => return None,
            };
        }

        state.to_move = match to_move_part {
            "X" => Player::X,
            "O" => Player::O,
            _ => return None,
        };

        Some(state)
    }
}
