//! Learned strategy analysis
//!
//! This module analyzes trained MENACE agents including opening moves,
//! policy statistics, and key position analysis.

use std::path::PathBuf;

use anyhow::Result;

use crate::{
    efe::dirichlet_categorical_mi,
    menace::{MenaceAgent, SavedMenaceAgent},
    tictactoe::BoardState,
    types::CanonicalLabel,
};

/// Analyze a learned strategy
pub fn analyze(agent_path: PathBuf, export: Option<PathBuf>) -> Result<()> {
    println!("Loading trained agent from: {}", agent_path.display());
    let saved = SavedMenaceAgent::load_from_file(&agent_path)?;
    let agent = saved.to_agent()?;

    println!("\n=== Trained Agent Info ===");
    println!("Filter: {:?}", saved.state_filter);
    println!("Algorithm: {:?}", saved.algorithm_type);
    println!("Restock mode: {:?}", saved.restock_mode);

    if let Some(games) = saved.metadata.games_trained {
        println!("Games trained: {games}");
    }
    if !saved.metadata.opponents.is_empty() {
        println!("Opponents: {}", saved.metadata.opponents.join(", "));
    }

    // Opening move analysis (like menace_complete.rs)
    println!("\n=== Opening Move Analysis ===");
    analyze_opening_moves(&agent)?;

    // Policy statistics
    println!("\n=== Policy Statistics ===");
    analyze_policy_statistics(&agent)?;

    // Key position analysis
    println!("\n=== Key Position Analysis ===");
    println!("Analyzing move weights for important positions...\n");

    let key_positions = vec![
        ("........._X", "Empty board"),
        ("....X...._O", "Center taken by X"),
        ("X........_O", "Corner taken by X"),
        ("X...O...._X", "Corner vs center"),
    ];

    for (state_str, description) in key_positions {
        if let Ok(state) = BoardState::from_string(state_str) {
            let ctx = state.canonical_context();
            let label = CanonicalLabel::from(&ctx);

            if let Some(weights) = agent.workspace.move_weights(&label) {
                println!("{description}:");
                println!("{state}");
                println!("Move weights:");

                let total: f64 = weights.iter().map(|(_, w)| w).sum();
                let mut sorted_weights: Vec<_> = weights.iter().collect();
                sorted_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                for (move_pos, weight) in sorted_weights {
                    let prob = if total > 0.0 {
                        weight / total * 100.0
                    } else {
                        0.0
                    };
                    let (row, col) = (move_pos / 3, move_pos % 3);
                    println!(
                        "  Position {move_pos} (row {row}, col {col}): {weight:.1} beads ({prob:.1}%)"
                    );
                }

                // Calculate policy entropy
                let entropy = crate::utils::entropy_from_weights(weights.iter().map(|(_, w)| *w));
                println!("  Policy entropy: {entropy:.3} (lower = more decisive)");
                println!();
            }
        }
    }

    // Export full strategy if requested
    if let Some(path) = export {
        export_learned_strategy(&agent, &path)?;
        println!("\nFull strategy exported to: {}", path.display());
    }

    if let Some(pure) = agent.pure_active_inference() {
        println!("\n=== Pure Active Inference Belief Diagnostics ===");
        let base_state = BoardState::new();
        let (ctx, label) = base_state.canonical_context_and_label();
        let legal_moves = ctx.state.legal_moves();

        if legal_moves.is_empty() {
            println!("No legal moves from empty board (unexpected).");
        } else {
            let preferences = pure.preference_model();
            let agent_player = pure.agent_player();
            let beliefs = pure.action_beliefs();

            println!("Root state analysis (empty board):");
            println!("{base_state}");
            println!(
                "  λ_policy: {:.3}, ambiguity weight: {:.3}, epistemic weight: {:.3}",
                preferences.policy_lambda,
                preferences.ambiguity_weight,
                preferences.epistemic_weight
            );

            let mut rows = Vec::new();
            for &canonical_move in &legal_moves {
                let alpha = beliefs.alpha_for(label.as_str(), canonical_move);
                let predictive = beliefs.predictive(label.as_str(), canonical_move);
                let q_win = predictive[0];
                let q_draw = predictive[1];
                let q_loss = predictive[2];

                let p_win = preferences.agent_win_preference_for(agent_player);
                let p_loss = preferences.agent_loss_preference_for(agent_player);
                let p_draw = preferences.draw_preference();

                let mut risk = 0.0;
                if q_win > 0.0 && p_win > 0.0 {
                    risk += q_win * (q_win / p_win).ln();
                }
                if q_draw > 0.0 && p_draw > 0.0 {
                    risk += q_draw * (q_draw / p_draw).ln();
                }
                if q_loss > 0.0 && p_loss > 0.0 {
                    risk += q_loss * (q_loss / p_loss).ln();
                }

                let ambiguity = -predictive
                    .iter()
                    .filter(|&&p| p > 0.0)
                    .map(|&p| p * p.ln())
                    .sum::<f64>();
                let epistemic = dirichlet_categorical_mi(&alpha);
                let efe = risk + preferences.ambiguity_weight * ambiguity
                    - preferences.epistemic_weight * epistemic;

                let original_move = ctx.map_canonical_to_original(canonical_move);
                rows.push((
                    canonical_move,
                    original_move,
                    alpha,
                    predictive,
                    risk,
                    ambiguity,
                    epistemic,
                    efe,
                ));
            }

            rows.sort_by(|a, b| a.7.partial_cmp(&b.7).unwrap());

            println!(
                "{:<8} {:<8} {:<22} {:<22} {:<12} {:<12} {:<12} {:<12}",
                "Canon",
                "Actual",
                "Alpha [W,D,L]",
                "Predictive [W,D,L]",
                "Risk",
                "Ambiguity",
                "EIG",
                "EFE"
            );
            for (canon, actual, alpha, pred, risk, amb, eig, efe) in rows {
                println!(
                    "{:<8} {:<8} [{:>5.1},{:>5.1},{:>5.1}] [{:>5.2},{:>5.2},{:>5.2}] {:<12.3} {:<12.3} {:<12.3} {:<12.3}",
                    canon,
                    actual,
                    alpha[0],
                    alpha[1],
                    alpha[2],
                    pred[0],
                    pred[1],
                    pred[2],
                    risk,
                    amb,
                    eig,
                    efe
                );
            }
        }
    }

    Ok(())
}

/// Analyze opening move preferences (Center/Corner/Edge distribution)
fn analyze_opening_moves(agent: &MenaceAgent) -> Result<()> {
    let empty_state = BoardState::new();
    let ctx = empty_state.canonical_context();
    let label = CanonicalLabel::from(&ctx);

    if let Some(weights) = agent.workspace.move_weights(&label) {
        let total: f64 = weights.iter().map(|(_, w)| w).sum();

        println!("Opening move preferences from empty board:");

        // Categorize moves
        let mut center_weight = 0.0;
        let mut corner_weight = 0.0;
        let mut edge_weight = 0.0;
        let mut sorted_moves: Vec<_> = weights.iter().collect();
        sorted_moves.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (pos, weight) in &sorted_moves {
            let move_type = match pos {
                4 => {
                    center_weight += weight;
                    "Center"
                }
                0 | 2 | 6 | 8 => {
                    corner_weight += weight;
                    "Corner"
                }
                _ => {
                    edge_weight += weight;
                    "Edge"
                }
            };
            let prob = if total > 0.0 {
                weight / total * 100.0
            } else {
                0.0
            };
            println!("  Position {pos} ({move_type}): {weight:.1} beads ({prob:.1}%)");
        }

        println!("\nStrategy summary:");
        if total > 0.0 {
            println!("  Center preference: {:.1}%", center_weight / total * 100.0);
            println!("  Corner preference: {:.1}%", corner_weight / total * 100.0);
            println!("  Edge preference: {:.1}%", edge_weight / total * 100.0);
        }
    } else {
        println!("  No policy found for empty board");
    }

    Ok(())
}

/// Calculate policy statistics across all learned states
fn analyze_policy_statistics(agent: &MenaceAgent) -> Result<()> {
    // Build game tree to get all decision states
    let tree = crate::tictactoe::build_reduced_game_tree(true, false);

    let states: Vec<_> = tree.states.iter().collect();

    let mut total_states = 0;
    let mut states_with_policy = 0;
    let mut empty_matchboxes = 0;
    let mut all_weights = Vec::new();
    let mut all_entropies = Vec::new();

    for state_label in &states {
        let state_str = state_label.to_string();
        if let Ok(state) = BoardState::from_string(&state_str)
            && !state.is_terminal()
        {
            total_states += 1;
            let ctx = state.canonical_context();
            let label = CanonicalLabel::from(&ctx);

            if let Some(weights) = agent.workspace.move_weights(&label) {
                let total: f64 = weights.iter().map(|(_, w)| w).sum();

                if total <= 0.0 {
                    empty_matchboxes += 1;
                } else {
                    states_with_policy += 1;
                    all_weights.extend(weights.iter().map(|(_, w)| *w));

                    let entropy =
                        crate::utils::entropy_from_weights(weights.iter().map(|(_, w)| *w));
                    all_entropies.push(entropy);
                }
            }
        }
    }

    println!("Total decision states: {total_states}");
    println!("States with learned policy: {states_with_policy}");
    println!("Empty matchboxes: {empty_matchboxes}");

    if !all_weights.is_empty() {
        let min_weight = all_weights
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max_weight = all_weights
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let avg_weight: f64 = all_weights.iter().sum::<f64>() / all_weights.len() as f64;

        println!("\nWeight statistics:");
        println!("  Min weight: {min_weight:.2}");
        println!("  Max weight: {max_weight:.2}");
        println!("  Avg weight: {avg_weight:.2}");
    }

    if !all_entropies.is_empty() {
        let avg_entropy: f64 = all_entropies.iter().sum::<f64>() / all_entropies.len() as f64;
        let min_entropy = all_entropies
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max_entropy = all_entropies
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        println!("\nPolicy decisiveness (entropy):");
        println!("  Avg entropy: {avg_entropy:.3} (lower = more decisive)");
        println!("  Min entropy: {min_entropy:.3} (most decisive)");
        println!("  Max entropy: {max_entropy:.3} (least decisive)");
    }

    Ok(())
}

/// Export learned strategy to JSON file
fn export_learned_strategy(agent: &MenaceAgent, path: &PathBuf) -> Result<()> {
    use std::{fs::File, io::Write};

    println!("\nExporting learned strategy for all decision states...");

    // Build game tree to get all X player states
    let tree = crate::tictactoe::build_reduced_game_tree(true, false);

    let states: Vec<_> = tree.states.iter().collect();
    let mut strategy = std::collections::HashMap::new();

    // Get move weights for each state
    for (i, state_label) in states.iter().enumerate() {
        let state_str = state_label.to_string();
        if let Ok(state) = BoardState::from_string(&state_str)
            && !state.is_terminal()
        {
            let ctx = state.canonical_context();
            let label = CanonicalLabel::from(&ctx);

            if let Some(weights) = agent.workspace.move_weights(&label) {
                let total: f64 = weights.iter().map(|(_, w)| w).sum();
                let probs: Vec<_> = weights
                    .iter()
                    .map(|(pos, w)| (*pos, if total > 0.0 { w / total } else { 0.0 }))
                    .collect();
                strategy.insert(state_str, probs);
            }
        }

        if (i + 1).is_multiple_of(100) {
            println!("  Processed {}/{} states...", i + 1, states.len());
        }
    }

    println!("  Processed {}/{} states", states.len(), states.len());
    println!("  Total strategy entries: {}", strategy.len());

    // Write as JSON
    let mut file = File::create(path)?;
    writeln!(file, "{{")?;
    writeln!(
        file,
        "  \"description\": \"Learned MENACE strategy with move probabilities\","
    )?;
    writeln!(file, "  \"player\": \"X\",")?;
    writeln!(file, "  \"total_states\": {},", strategy.len())?;
    writeln!(file, "  \"strategy\": {{")?;

    let mut entries: Vec<_> = strategy.iter().collect();
    entries.sort_by_key(|(k, _)| *k);

    for (i, (state_str, moves)) in entries.iter().enumerate() {
        let comma = if i < entries.len() - 1 { "," } else { "" };
        write!(file, "    \"{state_str}\": {{")?;

        for (j, (pos, prob)) in moves.iter().enumerate() {
            let move_comma = if j < moves.len() - 1 { ", " } else { "" };
            write!(file, "\"{pos}\": {prob:.4}{move_comma}")?;
        }

        writeln!(file, "}}{comma}")?;
    }

    writeln!(file, "  }}")?;
    writeln!(file, "}}")?;

    Ok(())
}
