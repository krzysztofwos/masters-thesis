//! Mathematical structures analysis
//!
//! This module demonstrates the mathematical structures underlying Tic-Tac-Toe:
//! - Magic square isomorphism
//! - Positional value hierarchy
//! - Categorical representation

use anyhow::{Result, anyhow};

use crate::tictactoe::BoardState;

/// Analyze mathematical structures
pub fn analyze(structure_opt: Option<String>) -> Result<()> {
    let show_all = structure_opt.is_none();
    let structure = structure_opt.as_deref().unwrap_or("all");

    if show_all || structure == "magic-square" {
        demonstrate_magic_square()?;
    }

    if show_all || structure == "positional-values" {
        demonstrate_positional_values()?;
    }

    if show_all || structure == "categorical" {
        demonstrate_categorical_representation()?;
    }

    if !show_all
        && structure != "magic-square"
        && structure != "positional-values"
        && structure != "categorical"
    {
        return Err(anyhow!(
            "Unknown structure: '{structure}'. Use: magic-square, positional-values, categorical"
        ));
    }

    Ok(())
}

/// Demonstrate magic square isomorphism
fn demonstrate_magic_square() -> Result<()> {
    println!("=== Magic Square Isomorphism ===\n");

    println!("Board positions map to magic square numbers:");
    println!("  2 7 6");
    println!("  9 5 1");
    println!("  4 3 8");

    println!("\nWinning lines correspond to subsets summing to 15:");
    let winning_sets = vec![
        vec![2, 7, 6],
        vec![9, 5, 1],
        vec![4, 3, 8], // rows
        vec![2, 9, 4],
        vec![7, 5, 3],
        vec![6, 1, 8], // columns
        vec![2, 5, 8],
        vec![6, 5, 4], // diagonals
    ];

    for set in winning_sets {
        println!("  {:?} = {}", set, set.iter().sum::<u8>());
    }

    // Test with an example
    let mut state = BoardState::new();
    state = state.make_move(0)?; // Position 0 = number 2
    state = state.make_move(4)?; // Position 4 = number 5
    state = state.make_move(8)?; // Position 8 = number 8

    println!("\nExample: X at positions 0, 4, 8 maps to numbers 2, 5, 8");
    println!("Sum: 2 + 5 + 8 = 15 ✓ (winning diagonal)");
    println!("{state}");

    Ok(())
}

/// Demonstrate positional value hierarchy
fn demonstrate_positional_values() -> Result<()> {
    println!("\n=== Positional Value Hierarchy ===\n");

    println!("Number of winning lines through each position:");
    println!("  3 2 3");
    println!("  2 4 2");
    println!("  3 2 3");

    println!("\nHierarchy:");
    println!("  Center (4): 4 lines - MOST VALUABLE");
    println!("  Corners (0,2,6,8): 3 lines each");
    println!("  Edges (1,3,5,7): 2 lines each - LEAST VALUABLE");

    println!("\nStrategic implications:");
    println!("  1. Always take center if available");
    println!("  2. Prefer corners over edges");
    println!("  3. Edges are defensive positions");

    Ok(())
}

/// Demonstrate categorical representation
fn demonstrate_categorical_representation() -> Result<()> {
    println!("\n=== Categorical Representation ===\n");

    println!("Building categorical model...");
    let tree = crate::tictactoe::build_reduced_game_tree(true, true);

    let total_moves = tree.moves.values().map(|v| v.len()).sum::<usize>();

    println!("\nCategory structure:");
    println!("  Objects (states): {}", tree.states.len());
    println!("  Morphisms (moves): {total_moves}");

    println!("\nCategorical interpretation:");
    println!("  - Objects: Game positions");
    println!("  - Morphisms: Legal moves");
    println!("  - Composition: Move sequences");
    println!("  - Identity: Staying in same state");

    println!("\nThis structure enables:");
    println!("  - Formal analysis of game dynamics");
    println!("  - Path finding and strategy optimization");
    println!("  - Historical learning with category evolution");

    Ok(())
}
