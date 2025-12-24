//! Symmetry analysis
//!
//! This module analyzes D4 symmetry transformations and stabilizer subgroups.

use std::collections::HashMap;

use anyhow::Result;

use crate::tictactoe::BoardState;

/// Analyze board symmetries
pub fn analyze(state_str: Option<String>, visualize: bool, stabilizers: bool) -> Result<()> {
    let state = if let Some(s) = state_str {
        BoardState::from_string(&s)?
    } else {
        BoardState::new()
    };

    println!("=== Symmetry Analysis ===");
    println!("Original state:");
    println!("{state}");

    let canonical = state.canonical();
    let is_canonical = state.encode() == canonical.encode();

    println!("\nCanonical form:");
    println!("{canonical}");

    if is_canonical {
        println!("\n✓ This state is already in canonical form");
    } else {
        println!("\n→ State can be reduced to canonical form via symmetry");
    }

    if visualize {
        println!("\n=== All D4 Symmetry Transformations ===");
        println!("The D4 group has 8 elements (4 rotations × 2 reflections):\n");

        let transforms = crate::tictactoe::symmetry::D4Transform::all();
        for (i, transform) in transforms.iter().enumerate() {
            let transformed = state.transform(transform);
            let is_canon = transformed.encode() == canonical.encode();

            let reflect_str = if transform.reflection {
                "reflected"
            } else {
                "no reflection"
            };
            println!(
                "{}. Rotation {}°, {} {}",
                i + 1,
                transform.rotation,
                reflect_str,
                if is_canon { "(canonical)" } else { "" }
            );
            println!("{transformed}");
            println!();
        }
    } else {
        println!("\nTip: Use --visualize to see all 8 D4 symmetry transformations");
    }

    if stabilizers {
        analyze_stabilizer_subgroups()?;
    }

    Ok(())
}

/// Analyze stabilizer subgroups across all states
fn analyze_stabilizer_subgroups() -> Result<()> {
    println!("\n=== Stabilizer Subgroup Analysis ===");
    println!("Computing stabilizer sizes for all valid states...\n");

    let transforms = crate::tictactoe::symmetry::D4Transform::all();
    let mut stabilizer_counts = HashMap::new();
    let mut visited = std::collections::HashSet::new();
    let mut queue = std::collections::VecDeque::new();

    queue.push_back(BoardState::new());
    visited.insert(BoardState::new().encode());

    while let Some(state) = queue.pop_front() {
        let mut stabilizer_size = 0;
        for t in &transforms {
            let transformed = state.transform(t);
            if transformed.encode() == state.encode() {
                stabilizer_size += 1;
            }
        }
        *stabilizer_counts.entry(stabilizer_size).or_insert(0) += 1;

        if state.is_terminal() {
            continue;
        }

        for pos in state.empty_positions() {
            if let Ok(next) = state.make_move(pos) {
                let key = next.encode();
                if !visited.contains(&key) {
                    visited.insert(key);
                    queue.push_back(next);
                }
            }
        }
    }

    println!("Stabilizer subgroup size distribution:");
    for size in &[1, 2, 4, 8] {
        if let Some(count) = stabilizer_counts.get(size) {
            println!(
                "  |Stab(s)| = {}: {} positions",
                size,
                super::format_number(*count)
            );
        }
    }

    println!("\nExample positions:");

    // Empty board has full D4 symmetry
    println!("\n  |Stab| = 8 (full D4 symmetry):");
    let empty = BoardState::new();
    println!("{empty}");

    // Center-only has 8-fold symmetry
    println!("\n  |Stab| = 8 (center symmetry):");
    let mut center = BoardState::new();
    center = center.make_move(4)?;
    println!("{center}");

    // Corner has 2-fold symmetry
    println!("\n  |Stab| = 2 (corner symmetry):");
    let mut corner = BoardState::new();
    corner = corner.make_move(0)?;
    println!("{corner}");

    // Edge has 2-fold symmetry
    println!("\n  |Stab| = 2 (edge symmetry):");
    let mut edge = BoardState::new();
    edge = edge.make_move(1)?;
    println!("{edge}");

    // Asymmetric position
    println!("\n  |Stab| = 1 (no symmetry):");
    let mut asym = BoardState::new();
    asym = asym.make_move(0)?;
    asym = asym.make_move(1)?;
    println!("{asym}");

    Ok(())
}
