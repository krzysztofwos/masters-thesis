//! State space validation
//!
//! This module validates the game state space and verifies mathematical properties
//! against known theoretical values.

use std::collections::HashSet;

use anyhow::Result;

use crate::tictactoe::{BoardState, Cell};

/// Validate state space and verify mathematical properties
pub fn analyze(show_invalid: bool, verify: bool) -> Result<()> {
    println!("=== State Space Validation ===\n");

    println!("Step 1: Generating all possible configurations...");

    // Generate all 3^9 = 19,683 board configurations
    let mut all_configs = HashSet::new();
    fn generate_all(cells: &mut [Cell; 9], pos: usize, configs: &mut HashSet<u64>) {
        if pos == 9 {
            // Encode board configuration
            let mut code = 0u64;
            for &cell in cells.iter() {
                code = code * 3
                    + match cell {
                        Cell::Empty => 0,
                        Cell::X => 1,
                        Cell::O => 2,
                    };
            }
            configs.insert(code);
            return;
        }

        for cell in [Cell::Empty, Cell::X, Cell::O] {
            cells[pos] = cell;
            generate_all(cells, pos + 1, configs);
        }
    }

    let mut cells = [Cell::Empty; 9];
    generate_all(&mut cells, 0, &mut all_configs);

    println!(
        "  Total configurations: {} (3^9)",
        super::format_number(all_configs.len())
    );

    println!("\nStep 2: Filtering to valid game states...");

    // Count valid states via BFS (reachable through legal play)
    let mut valid_states = HashSet::new();
    let mut queue = std::collections::VecDeque::new();

    queue.push_back(BoardState::new());
    valid_states.insert(BoardState::new().encode());

    while let Some(state) = queue.pop_front() {
        if state.is_terminal() {
            continue;
        }

        for pos in state.empty_positions() {
            if let Ok(next) = state.make_move(pos) {
                let key = next.encode();
                if !valid_states.contains(&key) {
                    valid_states.insert(key);
                    queue.push_back(next);
                }
            }
        }
    }

    println!(
        "  Valid game states: {}",
        super::format_number(valid_states.len())
    );

    // Count states with valid turn counts but invalid continuations
    let invalid_continuations =
        all_configs.len() - valid_states.len() - count_invalid_turns(&all_configs);

    println!(
        "  Invalid continuations: {}",
        super::format_number(invalid_continuations)
    );
    println!("  (Games that continued after a win)");

    if show_invalid {
        show_invalid_continuation_examples()?;
    }

    // Compute canonical states
    println!("\nStep 3: Computing canonical forms under D4 symmetry...");
    let mut canonical_states = HashSet::new();
    let mut visited_for_canonical = HashSet::new();
    let mut queue_canonical = std::collections::VecDeque::new();

    queue_canonical.push_back(BoardState::new());
    visited_for_canonical.insert(BoardState::new().encode());

    while let Some(state) = queue_canonical.pop_front() {
        canonical_states.insert(state.canonical().encode());

        if state.is_terminal() {
            continue;
        }

        for pos in state.empty_positions() {
            if let Ok(next) = state.make_move(pos) {
                let key = next.encode();
                if !visited_for_canonical.contains(&key) {
                    visited_for_canonical.insert(key);
                    queue_canonical.push_back(next);
                }
            }
        }
    }

    println!(
        "  Canonical states: {}",
        super::format_number(canonical_states.len())
    );

    if verify {
        show_verification_table(
            all_configs.len(),
            valid_states.len(),
            canonical_states.len(),
        )?;
    }

    println!("\n=== Summary ===");
    println!("State space hierarchy:");
    println!(
        "  {} configurations (all possible boards)",
        super::format_number(all_configs.len())
    );
    println!(
        "  {} valid states (reachable through legal play)",
        super::format_number(valid_states.len())
    );
    println!(
        "  {} canonical states (under D4 symmetry)",
        super::format_number(canonical_states.len())
    );
    println!("\nReduction factors:");
    println!(
        "  Validity filter: {:.2}x reduction",
        all_configs.len() as f64 / valid_states.len() as f64
    );
    println!(
        "  Symmetry filter: {:.2}x reduction",
        valid_states.len() as f64 / canonical_states.len() as f64
    );
    println!(
        "  Total: {:.2}x reduction",
        all_configs.len() as f64 / canonical_states.len() as f64
    );

    Ok(())
}

/// Count configurations with invalid turn counts
fn count_invalid_turns(all_configs: &HashSet<u64>) -> usize {
    let mut count = 0;
    for &code in all_configs {
        // Decode configuration
        let mut cells = [Cell::Empty; 9];
        let mut temp_code = code;
        for i in (0..9).rev() {
            cells[i] = match temp_code % 3 {
                0 => Cell::Empty,
                1 => Cell::X,
                2 => Cell::O,
                _ => unreachable!(),
            };
            temp_code /= 3;
        }

        let x_count = cells.iter().filter(|&&c| c == Cell::X).count();
        let o_count = cells.iter().filter(|&&c| c == Cell::O).count();

        // Invalid if turn counts are wrong (X plays first, so x_count == o_count or x_count == o_count + 1)
        if !(x_count == o_count || x_count == o_count + 1) {
            count += 1;
        }
    }
    count
}

/// Show examples of invalid continuations
fn show_invalid_continuation_examples() -> Result<()> {
    println!("\n=== Invalid Continuation Examples ===");
    println!("These are positions where the game continued after a win:\n");

    // Example 1: O wins but X makes another move
    let mut state1 = BoardState::new();
    state1 = state1.make_move(0)?; // X
    state1 = state1.make_move(3)?; // O
    state1 = state1.make_move(1)?; // X
    state1 = state1.make_move(4)?; // O
    state1 = state1.make_move(6)?; // X
    state1 = state1.make_move(5)?; // O wins (3,4,5)
    // Game should stop here, but continuing would be invalid

    println!("Example 1: O completes a line (3,4,5):");
    println!("{state1}");
    println!("✓ Game ends here (valid)");
    println!("✗ Any further moves would be invalid continuations\n");

    // Example 2: X wins but O makes another move
    let mut state2 = BoardState::new();
    state2 = state2.make_move(0)?; // X
    state2 = state2.make_move(3)?; // O
    state2 = state2.make_move(1)?; // X
    state2 = state2.make_move(4)?; // O
    state2 = state2.make_move(2)?; // X wins (0,1,2)

    println!("Example 2: X completes a line (0,1,2):");
    println!("{state2}");
    println!("✓ Game ends here (valid)");
    println!("✗ Any further moves would be invalid continuations");

    Ok(())
}

/// Show verification table comparing computed vs expected values
fn show_verification_table(
    total_configs: usize,
    valid_states: usize,
    canonical_states: usize,
) -> Result<()> {
    println!("\n=== Verification Against Mathematical Theory ===\n");

    println!("{:<45} {:>12} {:>12}", "Metric", "Expected", "Computed");
    println!("{}", "─".repeat(70));

    // Known mathematical values for Tic-Tac-Toe
    let expected_configs = 19683;
    let expected_valid = 5478;
    let expected_canonical = 765;

    let check = |computed: usize, expected: usize| {
        if computed == expected { "✓" } else { "✗" }
    };

    println!(
        "{:<45} {:>12} {:>12} {}",
        "Total configurations (3^9)",
        super::format_number(expected_configs),
        super::format_number(total_configs),
        check(total_configs, expected_configs)
    );

    println!(
        "{:<45} {:>12} {:>12} {}",
        "Valid game states",
        super::format_number(expected_valid),
        super::format_number(valid_states),
        check(valid_states, expected_valid)
    );

    println!(
        "{:<45} {:>12} {:>12} {}",
        "Canonical states (D4 symmetry)",
        super::format_number(expected_canonical),
        super::format_number(canonical_states),
        check(canonical_states, expected_canonical)
    );

    println!("\nNote: These values are from Crowley & Siegler (1993)");
    println!("and Schubert's game tree enumeration.");

    Ok(())
}
