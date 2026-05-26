use hashbrown::HashMap;
use plank_core::LoopLimit;

use crate::analyses::{AnalysesMask, ControlFlowGraphInOutBundling, InOutGroupId};
use sir_data::{
    BasicBlockId, Control, EthIRProgram, IndexVec, LocalId, LocalIdx, Operation, Span,
    operation::InlineOperands,
};

use crate::{AnalysesStore, Pass};

#[derive(Default)]
pub struct CopyPropagation {
    copy_map: HashMap<LocalId, LocalId>,
    io_groups: IndexVec<InOutGroupId, InOutGroupBlocks>,
    function_entries: IndexVec<BasicBlockId, bool>,
}

impl Pass for CopyPropagation {
    fn run(&mut self, program: &mut EthIRProgram, store: &AnalysesStore) {
        self.build_function_entries(program);
        self.build_in_out_groups(program, store);

        let mut limit = LoopLimit::new();
        loop {
            limit.tick();
            self.propagate_operation_copies(program);
            if !self.propagate_input_output_copies(program) {
                break;
            }
        }

        debug_assert_eq!(crate::Legalizer::default().run(program, store), Ok(()));
    }

    fn preserves(&self) -> AnalysesMask {
        AnalysesMask::Predecessors
            | AnalysesMask::Dominators
            | AnalysesMask::DominanceFrontiers
            | AnalysesMask::BasicBlockOwnership
            | AnalysesMask::Reachability
    }
}

impl CopyPropagation {
    fn propagate_operation_copies(&mut self, program: &mut EthIRProgram) {
        for bb in program.basic_blocks.iter_mut() {
            self.copy_map.clear();

            let ops_span = bb.operations;
            for op in &program.operations[ops_span] {
                if let Operation::SetCopy(InlineOperands { ins: [src], outs: [dst] }) = op {
                    let resolved_src = self.copy_map.get(src).unwrap_or(src);
                    let prev = self.copy_map.insert(*dst, *resolved_src);
                    debug_assert!(prev.is_none(), "SSA violation: {:?} defined twice", dst);
                }
            }

            for op_idx in ops_span.iter() {
                let mut op = program.operations[op_idx];
                for input in op.inputs_mut(&mut program.locals) {
                    replace_if_copied(input, &self.copy_map);
                }
                program.operations[op_idx] = op;
            }

            for local in &mut program.locals[bb.outputs] {
                replace_if_copied(local, &self.copy_map);
            }

            replace_control_uses(&mut bb.control, &self.copy_map);
        }
    }

    fn propagate_input_output_copies(&mut self, program: &mut EthIRProgram) -> bool {
        let Some(candidate) = self.find_removable_input_output_slot(program) else {
            return false;
        };

        self.copy_map.clear();
        add_slot_replacements(
            program,
            &self.io_groups[candidate.group],
            candidate.slot,
            candidate.replacement,
            &mut self.copy_map,
        );
        debug_assert!(!self.copy_map.is_empty());

        replace_uses(program, &self.copy_map);
        compact_input_output_slot(program, &self.io_groups[candidate.group], candidate.slot);

        true
    }

    fn find_removable_input_output_slot(&self, program: &EthIRProgram) -> Option<RemovalCandidate> {
        for group_id in self.io_groups.iter_idx() {
            let group = &self.io_groups[group_id];
            if group.inputs.is_empty() || group.outputs.is_empty() {
                continue;
            }

            if group.inputs.iter().any(|&bb_id| self.function_entries[bb_id]) {
                continue;
            }

            let slots = program.basic_blocks[group.outputs[0]].outputs.len() as usize;
            debug_assert!(
                group
                    .outputs
                    .iter()
                    .all(|&bb_id| program.basic_blocks[bb_id].outputs.len() as usize == slots)
            );
            debug_assert!(
                group
                    .inputs
                    .iter()
                    .all(|&bb_id| program.basic_blocks[bb_id].inputs.len() as usize == slots)
            );

            for slot in 0..slots {
                let Some(replacement) = common_output_at_slot(program, group, slot) else {
                    continue;
                };

                if !group.inputs.iter().any(|&bb_id| {
                    local_at_slot(program, program.basic_blocks[bb_id].inputs, slot) != replacement
                }) {
                    continue;
                }

                return Some(RemovalCandidate { group: group_id, slot, replacement });
            }
        }

        None
    }

    fn build_function_entries(&mut self, program: &EthIRProgram) {
        self.function_entries.clear();
        self.function_entries.resize(program.basic_blocks.len(), false);

        for function in program.functions.iter() {
            self.function_entries[function.entry()] = true;
        }
    }

    fn build_in_out_groups(&mut self, program: &EthIRProgram, store: &AnalysesStore) {
        let bundling = ControlFlowGraphInOutBundling::new(program, store);

        let total_groups = bundling.total_groups() as usize;
        if self.io_groups.len() < total_groups {
            self.io_groups.resize_with(total_groups, InOutGroupBlocks::default);
        }
        for group in self.io_groups.iter_mut() {
            group.clear();
        }

        for bb_id in program.basic_blocks.iter_idx() {
            if let Some(group) = bundling.get_in_group(bb_id) {
                self.group_mut(group).inputs.push(bb_id);
            }
            if let Some(group) = bundling.get_out_group(bb_id) {
                self.group_mut(group).outputs.push(bb_id);
            }
        }
    }

    fn group_mut(&mut self, group: InOutGroupId) -> &mut InOutGroupBlocks {
        &mut self.io_groups[group]
    }
}

fn replace_if_copied(input: &mut LocalId, copy_map: &HashMap<LocalId, LocalId>) {
    if let Some(replacement) = copy_map.get(input) {
        *input = *replacement;
    }
}

#[derive(Default)]
struct InOutGroupBlocks {
    inputs: Vec<BasicBlockId>,
    outputs: Vec<BasicBlockId>,
}

impl InOutGroupBlocks {
    fn clear(&mut self) {
        self.inputs.clear();
        self.outputs.clear();
    }
}

#[derive(Clone, Copy)]
struct RemovalCandidate {
    group: InOutGroupId,
    slot: usize,
    replacement: LocalId,
}

fn common_output_at_slot(
    program: &EthIRProgram,
    group: &InOutGroupBlocks,
    slot: usize,
) -> Option<LocalId> {
    let mut outputs = group
        .outputs
        .iter()
        .map(|&bb_id| local_at_slot(program, program.basic_blocks[bb_id].outputs, slot));

    let first = outputs.next()?;
    outputs.all(|local| local == first).then_some(first)
}

fn local_at_slot(program: &EthIRProgram, span: Span<LocalIdx>, slot: usize) -> LocalId {
    program.locals[span.start + slot as u32]
}

fn add_slot_replacements(
    program: &EthIRProgram,
    group: &InOutGroupBlocks,
    slot: usize,
    replacement: LocalId,
    copy_map: &mut HashMap<LocalId, LocalId>,
) {
    for &bb_id in &group.inputs {
        let input = local_at_slot(program, program.basic_blocks[bb_id].inputs, slot);
        if input == replacement {
            continue;
        }

        let prev = copy_map.insert(input, replacement);
        debug_assert!(prev.is_none(), "SSA violation: ${input} used as two block inputs");
    }
}

fn compact_input_output_slot(program: &mut EthIRProgram, group: &InOutGroupBlocks, slot: usize) {
    for &bb_id in &group.inputs {
        let inputs = program.basic_blocks[bb_id].inputs;
        program.basic_blocks[bb_id].inputs = compact_span_slot(&mut program.locals, inputs, slot);
    }
    for &bb_id in &group.outputs {
        let outputs = program.basic_blocks[bb_id].outputs;
        program.basic_blocks[bb_id].outputs = compact_span_slot(&mut program.locals, outputs, slot);
    }
}

fn replace_uses(program: &mut EthIRProgram, copy_map: &HashMap<LocalId, LocalId>) {
    for bb in program.basic_blocks.iter_mut() {
        let ops_span = bb.operations;
        for op_idx in ops_span.iter() {
            let mut op = program.operations[op_idx];
            for input in op.inputs_mut(&mut program.locals) {
                replace_if_copied(input, copy_map);
            }
            program.operations[op_idx] = op;
        }

        for local in &mut program.locals[bb.outputs] {
            replace_if_copied(local, copy_map);
        }

        replace_control_uses(&mut bb.control, copy_map);
    }
}

fn replace_control_uses(control: &mut Control, copy_map: &HashMap<LocalId, LocalId>) {
    match control {
        Control::Branches(branch) => {
            replace_if_copied(&mut branch.condition, copy_map);
        }
        Control::Switch(switch) => {
            replace_if_copied(&mut switch.condition, copy_map);
        }
        _ => {}
    }
}

fn compact_span_slot(
    locals: &mut IndexVec<LocalIdx, LocalId>,
    span: Span<LocalIdx>,
    remove_slot: usize,
) -> Span<LocalIdx> {
    debug_assert!(remove_slot < span.len() as usize);

    let mut write = span.start;
    for (slot, read) in span.iter().enumerate() {
        if slot == remove_slot {
            continue;
        }
        locals[write] = locals[read];
        write += 1;
    }

    Span::new(span.start, write)
}

#[cfg(test)]
mod tests {
    use super::CopyPropagation;
    use crate::{AnalysesStore, Legalizer, run_pass};
    use sir_data::{EthIRProgram, assert_ir_display};
    use sir_parser::{EmitConfig, parse_or_panic};

    fn run_copy_propagation_and_display(source: &str) -> EthIRProgram {
        let mut program = parse_or_panic(source, EmitConfig::init_only());
        let store = AnalysesStore::default();
        run_pass(&mut CopyPropagation::default(), &mut program, &store);
        assert_eq!(Legalizer::default().run(&program, &store), Ok(()));
        program
    }

    #[test]
    fn test_copy_chains_and_inline_operands() {
        let input = r#"
            fn init:
                entry {
                    stop
                }
            fn test:
                entry b {
                    a1 = copy b
                    a2 = copy b
                    c1 = copy a1
                    c2 = copy a2
                    d = add c1 c2
                    stop
                }
        "#;

        let actual = run_copy_propagation_and_display(input);
        assert_ir_display(
            &actual,
            r#"
            Init: @0
            Functions:
                fn @0 -> entry @0  (outputs: 0)
                fn @1 -> entry @1  (outputs: 0)

            Basic Blocks:
                @0 {
                    stop
                }

                @1 $0 {
                    $1 = copy $0
                    $2 = copy $0
                    $3 = copy $0
                    $4 = copy $0
                    $5 = add $0 $0
                    stop
                }
            "#,
        );
    }

    #[test]
    fn test_input_output_copy_eliminates_block_argument() {
        let input = r#"
            fn init:
                entry {
                    stop
                }
            fn test:
                entry b -> a_out {
                    a = copy b
                    a_out = copy a
                    => @next
                }
                next a_in {
                    c = add a_in a_in
                    stop
                }
        "#;

        let actual = run_copy_propagation_and_display(input);
        assert_ir_display(
            &actual,
            r#"
            Init: @0
            Functions:
                fn @0 -> entry @0  (outputs: 0)
                fn @1 -> entry @1  (outputs: 0)

            Basic Blocks:
                @0 {
                    stop
                }

                @1 $0 {
                    $1 = copy $0
                    $2 = copy $0
                    => @2
                }

                @2 {
                    $4 = add $0 $0
                    stop
                }
            "#,
        );
    }

    #[test]
    fn test_branch_condition_propagation() {
        let input = r#"
            fn init:
                entry {
                    stop
                }
            fn test:
                entry x {
                    cond = copy x
                    => cond ? @nonzero : @zero
                }
                nonzero {
                    stop
                }
                zero {
                    stop
                }
        "#;

        let actual = run_copy_propagation_and_display(input);
        assert_ir_display(
            &actual,
            r#"
            Init: @0
            Functions:
                fn @0 -> entry @0  (outputs: 0)
                fn @1 -> entry @1  (outputs: 0)

            Basic Blocks:
                @0 {
                    stop
                }

                @1 $0 {
                    $1 = copy $0
                    => $0 ? @2 : @3
                }

                @2 {
                    stop
                }

                @3 {
                    stop
                }
            "#,
        );
    }

    #[test]
    fn test_switch_condition_propagation() {
        let input = r#"
            fn init:
                entry {
                    stop
                }
            fn test:
                entry x {
                    cond = copy x
                    switch cond {
                        0 => @case_zero
                        default => @case_default
                    }
                }
                case_zero {
                    stop
                }
                case_default {
                    stop
                }
        "#;

        let actual = run_copy_propagation_and_display(input);
        assert_ir_display(
            &actual,
            r#"
            Init: @0
            Functions:
                fn @0 -> entry @0  (outputs: 0)
                fn @1 -> entry @1  (outputs: 0)

            Basic Blocks:
                @0 {
                    stop
                }

                @1 $0 {
                    $1 = copy $0
                    switch $0 {
                        0x0 => @2,
                        else => @3
                    }

                }

                @2 {
                    stop
                }

                @3 {
                    stop
                }
            "#,
        );
    }

    #[test]
    fn test_icall_argument_propagation() {
        let input = r#"
            fn init:
                entry {
                    stop
                }
            fn callee:
                entry x -> result {
                    result = add x x
                    iret
                }
            fn caller:
                entry b {
                    a = copy b
                    sum = icall @callee a
                    stop
                }
        "#;

        let actual = run_copy_propagation_and_display(input);
        assert_ir_display(
            &actual,
            r#"
            Init: @0
            Functions:
                fn @0 -> entry @0  (outputs: 0)
                fn @1 -> entry @1  (outputs: 1)
                fn @2 -> entry @2  (outputs: 0)

            Basic Blocks:
                @0 {
                    stop
                }

                @1 $0 -> $1 {
                    $1 = add $0 $0
                    iret
                }

                @2 $2 {
                    $3 = copy $2
                    $4 = icall @1 $2
                    stop
                }
            "#,
        );
    }

    #[test]
    fn test_input_output_copy_chain_across_blocks() {
        let input = r#"
            fn init:
                entry {
                    stop
                }
            fn test:
                entry b -> a b {
                    a = copy b
                    => @next
                }
                next c d {
                    e = add c d
                    stop
                }
        "#;

        let actual = run_copy_propagation_and_display(input);
        assert_ir_display(
            &actual,
            r#"
            Init: @0
            Functions:
                fn @0 -> entry @0  (outputs: 0)
                fn @1 -> entry @1  (outputs: 0)

            Basic Blocks:
                @0 {
                    stop
                }

                @1 $0 {
                    $1 = copy $0
                    => @2
                }

                @2 {
                    $4 = add $0 $0
                    stop
                }
            "#,
        );
    }

    #[test]
    fn test_input_output_copy_chain_across_three_blocks() {
        let input = r#"
            fn init:
                entry {
                    stop
                }
            fn test:
                entry x -> x {
                    => @middle
                }
                middle y -> y {
                    => @next
                }
                next z {
                    w = add z z
                    stop
                }
        "#;

        let actual = run_copy_propagation_and_display(input);
        assert_ir_display(
            &actual,
            r#"
            Init: @0
            Functions:
                fn @0 -> entry @0  (outputs: 0)
                fn @1 -> entry @1  (outputs: 0)

            Basic Blocks:
                @0 {
                    stop
                }

                @1 $0 {
                    => @2
                }

                @2 {
                    => @3
                }

                @3 {
                    $3 = add $0 $0
                    stop
                }
            "#,
        );
    }

    #[test]
    fn test_function_entry_input_is_preserved() {
        let input = r#"
            fn init:
                entry {
                    stop
                }
            fn callee:
                entry x -> x {
                    => @return
                }
                return y -> y {
                    iret
                }
            fn caller:
                entry a {
                    out = icall @callee a
                    stop
                }
        "#;

        let actual = run_copy_propagation_and_display(input);
        assert_ir_display(
            &actual,
            r#"
            Init: @0
            Functions:
                fn @0 -> entry @0  (outputs: 0)
                fn @1 -> entry @1  (outputs: 1)
                fn @2 -> entry @3  (outputs: 0)

            Basic Blocks:
                @0 {
                    stop
                }

                @1 $0 {
                    => @2
                }

                @2 -> $0 {
                    iret
                }

                @3 $2 {
                    $3 = icall @1 $2
                    stop
                }
            "#,
        );
    }

    #[test]
    fn test_branch_inputs_outputs_same_local_eliminated() {
        let input = r#"
            fn init:
                entry {
                    stop
                }
            fn test:
                entry x {
                    => x ? @left : @right
                }
                left -> x {
                    => @merge
                }
                right -> x {
                    => @merge
                }
                merge y {
                    z = add y y
                    stop
                }
        "#;

        let actual = run_copy_propagation_and_display(input);
        assert_ir_display(
            &actual,
            r#"
            Init: @0
            Functions:
                fn @0 -> entry @0  (outputs: 0)
                fn @1 -> entry @1  (outputs: 0)

            Basic Blocks:
                @0 {
                    stop
                }

                @1 $0 {
                    => $0 ? @2 : @3
                }

                @2 {
                    => @4
                }

                @3 {
                    => @4
                }

                @4 {
                    $2 = add $0 $0
                    stop
                }
            "#,
        );
    }

    #[test]
    fn test_branch_inputs_outputs_different_locals_preserved() {
        let input = r#"
            fn init:
                entry {
                    stop
                }
            fn test:
                entry cond {
                    => cond ? @left : @right
                }
                left -> a {
                    a = const 1
                    => @merge
                }
                right -> b {
                    b = const 2
                    => @merge
                }
                merge y {
                    z = add y y
                    stop
                }
        "#;

        let actual = run_copy_propagation_and_display(input);
        assert_ir_display(
            &actual,
            r#"
            Init: @0
            Functions:
                fn @0 -> entry @0  (outputs: 0)
                fn @1 -> entry @1  (outputs: 0)

            Basic Blocks:
                @0 {
                    stop
                }

                @1 $0 {
                    => $0 ? @2 : @3
                }

                @2 -> $1 {
                    $1 = const 0x1
                    => @4
                }

                @3 -> $2 {
                    $2 = const 0x2
                    => @4
                }

                @4 $3 {
                    $4 = add $3 $3
                    stop
                }
            "#,
        );
    }

    #[test]
    fn test_switch_inputs_outputs_same_local_eliminated() {
        let input = r#"
            fn init:
                entry {
                    stop
                }
            fn test:
                entry selector x {
                    switch selector {
                        0 => @left
                        1 => @right
                        default => @fallback
                    }
                }
                left -> x {
                    => @merge
                }
                right -> x {
                    => @merge
                }
                fallback -> x {
                    => @merge
                }
                merge y {
                    z = add y y
                    stop
                }
        "#;

        let actual = run_copy_propagation_and_display(input);
        assert_ir_display(
            &actual,
            r#"
            Init: @0
            Functions:
                fn @0 -> entry @0  (outputs: 0)
                fn @1 -> entry @1  (outputs: 0)

            Basic Blocks:
                @0 {
                    stop
                }

                @1 $0 $1 {
                    switch $0 {
                        0x0 => @2,
                        0x1 => @3,
                        else => @4
                    }

                }

                @2 {
                    => @5
                }

                @3 {
                    => @5
                }

                @4 {
                    => @5
                }

                @5 {
                    $3 = add $1 $1
                    stop
                }
            "#,
        );
    }

    #[test]
    fn test_self_loop_input_output_is_preserved() {
        let input = r#"
            fn init:
                entry {
                    stop
                }
            fn test:
                loop i -> next {
                    one = const 1
                    next = add i one
                    => @loop
                }
        "#;

        let actual = run_copy_propagation_and_display(input);
        assert_ir_display(
            &actual,
            r#"
            Init: @0
            Functions:
                fn @0 -> entry @0  (outputs: 0)
                fn @1 -> entry @1  (outputs: 0)

            Basic Blocks:
                @0 {
                    stop
                }

                @1 $0 -> $2 {
                    $1 = const 0x1
                    $2 = add $0 $1
                    => @1
                }
            "#,
        );
    }

    #[test]
    fn test_self_loop_same_input_output_is_preserved() {
        let input = r#"
            fn init:
                entry {
                    stop
                }
            fn test:
                loop i -> i {
                    => @loop
                }
        "#;

        let actual = run_copy_propagation_and_display(input);
        assert_ir_display(
            &actual,
            r#"
            Init: @0
            Functions:
                fn @0 -> entry @0  (outputs: 0)
                fn @1 -> entry @1  (outputs: 0)

            Basic Blocks:
                @0 {
                    stop
                }

                @1 $0 -> $0 {
                    => @1
                }
            "#,
        );
    }

    #[test]
    fn test_two_block_loop_does_not_replace_header_input_with_backedge_input() {
        let input = r#"
            fn init:
                entry {
                    stop
                }
            fn test:
                entry i -> i {
                    => @tail
                }
                tail j -> j {
                    => @entry
                }
        "#;

        let actual = run_copy_propagation_and_display(input);
        assert_ir_display(
            &actual,
            r#"
            Init: @0
            Functions:
                fn @0 -> entry @0  (outputs: 0)
                fn @1 -> entry @1  (outputs: 0)

            Basic Blocks:
                @0 {
                    stop
                }

                @1 $0 {
                    => @2
                }

                @2 -> $0 {
                    => @1
                }
            "#,
        );
    }
}
