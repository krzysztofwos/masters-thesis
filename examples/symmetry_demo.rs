//! Demonstration of D4 symmetry group operations on Tic-tac-toe boards
//!
//! This example shows:
//! - All 8 D4 transformations
//! - Board canonicalization
//! - Symmetry reduction in practice
//! - Stabilizer subgroups

use menace::tictactoe::{BoardState, Cell, symmetry::D4Transform};

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║             D4 SYMMETRY GROUP DEMONSTRATION              ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Part 1: Show all D4 transformations
    println!("PART 1: THE 8 ELEMENTS OF D4");
    println!("────────────────────────────");
    demonstrate_d4_elements();
    println!();

    // Part 2: Demonstrate canonicalization
    println!("PART 2: BOARD CANONICALIZATION");
    println!("──────────────────────────────");
    demonstrate_canonicalization();
    println!();

    // Part 3: Show symmetry reduction
    println!("PART 3: SYMMETRY REDUCTION");
    println!("──────────────────────────");
    demonstrate_reduction();
    println!();

    // Part 4: Stabilizer subgroups
    println!("PART 4: STABILIZER SUBGROUPS");
    println!("────────────────────────────");
    demonstrate_stabilizers();
    println!();

    // Part 5: Practical application
    println!("PART 5: PRACTICAL APPLICATION");
    println!("─────────────────────────────");
    demonstrate_practical_use();
}

fn demonstrate_d4_elements() {
    // Create a sample board to show transformations
    let mut board = BoardState::new();
    board = board.make_move(0).unwrap(); // X at top-left
    board = board.make_move(4).unwrap(); // O at center
    board = board.make_move(8).unwrap(); // X at bottom-right

    println!("Original board:");
    print_board(&board);
    println!();

    let transforms = D4Transform::all();
    println!("All 8 D4 transformations:");

    for (i, t) in transforms.iter().enumerate() {
        let transformed = board.transform(t);
        println!(
            "\n{}. Rotation: {}°, Reflection: {}",
            i + 1,
            t.rotation,
            if t.reflection { "Yes" } else { "No" }
        );
        print_board(&transformed);
    }
}

fn demonstrate_canonicalization() {
    // Create several equivalent boards
    let mut boards = Vec::new();

    // Board 1: X at corner (0)
    let mut b1 = BoardState::new();
    b1 = b1.make_move(0).unwrap();
    boards.push(("X at top-left corner", b1));

    // Board 2: X at different corner (2) - should be equivalent
    let mut b2 = BoardState::new();
    b2 = b2.make_move(2).unwrap();
    boards.push(("X at top-right corner", b2));

    // Board 3: X at another corner (6)
    let mut b3 = BoardState::new();
    b3 = b3.make_move(6).unwrap();
    boards.push(("X at bottom-left corner", b3));

    // Board 4: X at last corner (8)
    let mut b4 = BoardState::new();
    b4 = b4.make_move(8).unwrap();
    boards.push(("X at bottom-right corner", b4));

    println!("Four equivalent boards:");
    for (desc, board) in &boards {
        println!("\n{desc}:");
        print_board(board);
        let canonical = board.canonical();
        println!("Canonical form key: {}", canonical.encode());
    }

    println!("\nAll four boards have the same canonical form!");

    // Show a non-equivalent board
    let mut different = BoardState::new();
    different = different.make_move(4).unwrap(); // X at center
    println!("\nNon-equivalent board (X at center):");
    print_board(&different);
    println!("Canonical form key: {}", different.canonical().encode());
}

fn demonstrate_reduction() {
    println!("Counting unique positions at each ply:\n");

    for ply in 0..=2 {
        let (total, canonical) = count_positions_at_ply(ply);
        let reduction = if total > 0 {
            (1.0 - canonical as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        println!("Ply {ply} (after {ply} moves):");
        println!("  Total positions: {total}");
        println!("  Canonical positions: {canonical}");
        println!("  Reduction: {reduction:.1}%");
        println!();
    }
}

fn demonstrate_stabilizers() {
    println!("Positions with different stabilizer subgroup sizes:\n");

    // Empty board - full D4 symmetry
    let empty = BoardState::new();
    let empty_stab = count_stabilizer(&empty);
    println!("Empty board:");
    print_board(&empty);
    println!("Stabilizer size: {empty_stab} (full D4 symmetry)\n");

    // Center only - full D4 symmetry
    let mut center = BoardState::new();
    center = center.make_move(4).unwrap();
    let center_stab = count_stabilizer(&center);
    println!("X at center:");
    print_board(&center);
    println!("Stabilizer size: {center_stab} (full rotational and reflection symmetry)\n");

    // Corner only - 2-fold symmetry
    let mut corner = BoardState::new();
    corner = corner.make_move(0).unwrap();
    let corner_stab = count_stabilizer(&corner);
    println!("X at corner:");
    print_board(&corner);
    println!("Stabilizer size: {corner_stab} (one reflection symmetry)\n");

    // Edge only - 2-fold symmetry
    let mut edge = BoardState::new();
    edge = edge.make_move(1).unwrap();
    let edge_stab = count_stabilizer(&edge);
    println!("X at edge:");
    print_board(&edge);
    println!("Stabilizer size: {edge_stab} (one reflection symmetry)\n");

    // Asymmetric position
    let mut asymmetric = BoardState::new();
    asymmetric = asymmetric.make_move(0).unwrap(); // X at corner
    asymmetric = asymmetric.make_move(1).unwrap(); // O at edge
    let asym_stab = count_stabilizer(&asymmetric);
    println!("Asymmetric position:");
    print_board(&asymmetric);
    println!("Stabilizer size: {asym_stab} (no symmetry)\n");
}

fn demonstrate_practical_use() {
    println!("Practical benefits of symmetry reduction:\n");

    // Count total vs canonical states
    let mut total_states = 0;
    let mut canonical_states = std::collections::HashSet::new();
    let mut queue = std::collections::VecDeque::new();
    let mut visited = std::collections::HashSet::new();

    queue.push_back(BoardState::new());

    while let Some(state) = queue.pop_front() {
        let key = state.encode();
        if visited.contains(&key) {
            continue;
        }
        visited.insert(key);

        total_states += 1;
        canonical_states.insert(state.canonical().encode());

        if state.is_terminal() || total_states > 500 {
            continue;
        }

        for pos in state.empty_positions() {
            if let Ok(next) = state.make_move(pos) {
                queue.push_back(next);
            }
        }
    }

    println!("First 500 game states:");
    println!("  Without symmetry: {total_states} states");
    println!(
        "  With symmetry: {} canonical states",
        canonical_states.len()
    );
    let reduction = (1.0 - canonical_states.len() as f64 / total_states as f64) * 100.0;
    println!("  Memory savings: {reduction:.1}%");
    println!();

    println!("For MENACE, this means:");
    println!(
        "  - {} matchboxes needed instead of {total_states}",
        canonical_states.len(),
    );
    let beads_per_box = 20; // Approximate
    println!(
        "  - {} beads instead of {}",
        canonical_states.len() * beads_per_box,
        total_states * beads_per_box
    );
}

// Helper functions

fn print_board(board: &BoardState) {
    for row in 0..3 {
        if row > 0 {
            println!("───┼───┼───");
        }
        for col in 0..3 {
            let pos = row * 3 + col;
            let cell = board.get(pos);
            let symbol = match cell {
                Cell::Empty => ' ',
                Cell::X => 'X',
                Cell::O => 'O',
            };
            if col > 0 {
                print!("│");
            }
            print!(" {symbol} ");
        }
        println!();
    }
}

fn count_stabilizer(board: &BoardState) -> usize {
    let transforms = D4Transform::all();
    let mut count = 0;
    let original_key = board.encode();

    for t in &transforms {
        let transformed = board.transform(t);
        if transformed.encode() == original_key {
            count += 1;
        }
    }

    count
}

fn count_positions_at_ply(target_ply: usize) -> (usize, usize) {
    let mut total = 0;
    let mut canonical_set = std::collections::HashSet::new();
    let mut queue = std::collections::VecDeque::new();
    let mut visited = std::collections::HashSet::new();

    queue.push_back((BoardState::new(), 0));

    while let Some((state, ply)) = queue.pop_front() {
        if ply == target_ply {
            let key = state.encode();
            if !visited.contains(&key) {
                visited.insert(key);
                total += 1;
                canonical_set.insert(state.canonical().encode());
            }
        } else if ply < target_ply && !state.is_terminal() {
            for pos in state.empty_positions() {
                if let Ok(next) = state.make_move(pos) {
                    queue.push_back((next, ply + 1));
                }
            }
        }
    }

    (total, canonical_set.len())
}
