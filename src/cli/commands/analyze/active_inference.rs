//! Active Inference analysis
//!
//! This module provides analysis of Expected Free Energy (EFE) decomposition
//! for game states using Active Inference principles.

use std::path::PathBuf;

use anyhow::{Result, anyhow};

use crate::{
    active_inference::{GenerativeModel, OpponentKind, PreferenceModel},
    beliefs::Beliefs,
    export::{AifCsvExporter, AifExportConfig},
    tictactoe::{BoardState, Player},
    workspace::StateFilter,
};

/// Analyze Active Inference EFE decomposition
pub fn analyze(
    state_opt: Option<String>,
    opponent_str: String,
    beta: f64,
    export_path: Option<PathBuf>,
    states_str: Option<String>,
) -> Result<()> {
    println!("=== Active Inference EFE Analysis ===\n");

    // Parse opponent kind
    let opponent_kind = match opponent_str.to_lowercase().as_str() {
        "uniform" => OpponentKind::Uniform,
        "adversarial" => OpponentKind::Adversarial,
        "minimax" => OpponentKind::Minimax,
        other => {
            return Err(anyhow!(
                "Unknown opponent model: '{other}'. Use: uniform, adversarial, minimax"
            ));
        }
    };

    println!("Configuration:");
    println!("  Opponent model: {opponent_kind:?}");
    println!("  Epistemic weight (β): {beta:.2}");
    println!("  EFE mode: Exact (with full decomposition)");

    // Create preferences with exact EFE mode
    let (win_pref, draw_pref, loss_pref) =
        crate::active_inference::preferences::CANONICAL_PREFERENCE_PROBS;
    let mut preferences = PreferenceModel::from_probabilities(win_pref, draw_pref, loss_pref);
    preferences.epistemic_weight = beta;
    preferences.efe_mode = crate::active_inference::EFEMode::Exact;

    // Create generative model and beliefs
    let model = GenerativeModel::new();
    let opponent = opponent_kind.into_boxed_opponent();
    let beliefs = Beliefs::symmetric(1.0);

    // If a specific state is provided, analyze just that state
    if let Some(state_str) = state_opt {
        let state = BoardState::from_string(&state_str)?;

        println!("\nAnalyzing state:");
        println!("{state}");
        println!();

        analyze_efe_for_state(&state, &model, &preferences, opponent.as_ref(), &beliefs)?;
        return Ok(());
    }

    // If export is requested, export EFE decompositions
    if let Some(export_path) = export_path {
        println!("\n=== Exporting EFE Decompositions ===");

        // Parse selected states if provided
        let selected_states = states_str.as_ref().map(|s| {
            s.split(',')
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
                .collect::<Vec<String>>()
        });

        // Use DecisionOnly filter for comprehensive analysis
        let state_filter = StateFilter::DecisionOnly;
        let states = AifCsvExporter::collect_states(state_filter, selected_states, false)?;

        println!("Exporting {} state(s)...", states.len());

        let config = AifExportConfig {
            state_filter,
            selected_states: None,
            include_o_states: false,
            generative_model: &model,
            preferences: &preferences,
            opponent: opponent.as_ref(),
            beliefs: &beliefs,
        };

        let exported_count = AifCsvExporter::export(&config, &export_path, states)?;

        println!(
            "✓ Exported {} state(s) to {}",
            exported_count,
            export_path.display()
        );
        return Ok(());
    }

    // Otherwise, show EFE for some key positions
    println!("\n=== Key Position EFE Analysis ===");

    let key_positions = vec![
        ("........._X", "Empty board (X to move)"),
        ("X...O...._X", "Corner vs center (X to move)"),
    ];

    for (state_str, description) in key_positions {
        let state = BoardState::from_string(state_str)?;

        println!("\n{description}:");
        println!("{state}");

        analyze_efe_for_state(&state, &model, &preferences, opponent.as_ref(), &beliefs)?;
    }

    println!("\nTip: Use --state <state> to analyze a specific state");
    println!("     Use --export <path> to export full EFE decompositions");

    Ok(())
}

/// Analyze EFE decomposition for a single state
fn analyze_efe_for_state(
    state: &BoardState,
    model: &GenerativeModel,
    preferences: &PreferenceModel,
    opponent: &dyn crate::active_inference::Opponent,
    beliefs: &Beliefs,
) -> Result<()> {
    // Convert to canonical form (generative model uses canonical states)
    let canonical_state = state.canonical();
    let state_label = canonical_state.encode();

    // Compute exact EFE decomposition for this state
    let summary =
        model.exact_state_summary(&state_label, preferences, opponent, beliefs, Player::X);

    if summary.actions.is_empty() {
        println!("  No legal moves (terminal state)");
        return Ok(());
    }

    println!("  Legal moves: {}", summary.actions.len());
    println!();

    // Display results
    println!("  EFE Decomposition (sorted by total EFE):");
    println!(
        "  {:<8} {:<12} {:<12} {:<12} {:<12}",
        "Move", "Risk", "Epistemic", "F_approx", "EIG"
    );
    println!("  {}", "-".repeat(60));

    for action_eval in summary.actions.iter() {
        let (row, col) = (action_eval.action / 3, action_eval.action % 3);
        let approx_f = action_eval.risk - preferences.epistemic_weight * action_eval.epistemic;

        println!(
            "  {} ({}:{}) {:<12.3} {:<12.3} {:<12.3} {:<12.3}",
            action_eval.action,
            row,
            col,
            action_eval.risk,
            action_eval.epistemic,
            approx_f,
            action_eval.opponent_eig
        );
    }

    println!();
    println!("  Policy-level metrics:");
    println!("    F_exact (total EFE): {:.3}", summary.policy.f_exact);
    println!("    Policy KL: {:.3}", summary.policy.policy_kl);

    // Highlight best move
    if let Some(first_action) = summary.actions.first() {
        let (row, col) = (first_action.action / 3, first_action.action % 3);
        println!();
        println!(
            "  → Best move (lowest EFE): position {} (row {}, col {})",
            first_action.action, row, col
        );
        println!("    Risk: {:.3}", first_action.risk);
        println!("    Epistemic: {:.3}", first_action.epistemic);
    }

    Ok(())
}
