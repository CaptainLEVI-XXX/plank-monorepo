use hashbrown::HashMap;

use crate::analyses::{AnalysesMask, ControlFlowGraphInOutBundling, Dominators, InOutGroupId};
use sir_data::{
    BasicBlockId, Control, EthIRProgram, Idx, IndexVec, LocalId, LocalIdx, Operation, Span,
    operation::InlineOperands,
};

use crate::{AnalysesStore, Pass};

#[derive(Default)]
pub struct CopyPropagation {
    copy_map: HashMap<LocalId, LocalId>,
    io_groups: Vec<InOutGroupBlocks>,
    def_blocks: IndexVec<LocalId, Option<BasicBlockId>>,
    function_entries: IndexVec<BasicBlockId, bool>,
    slots_to_remove: Vec<Vec<bool>>,
}

impl Pass for CopyPropagation {
    fn run(&mut self, program: &mut EthIRProgram, store: &AnalysesStore) {
        loop {
            self.propagate_operation_copies(program);
            if !self.propagate_input_output_copies(program, store) {
                break;
            }
        }
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

    fn propagate_input_output_copies(
        &mut self,
        program: &mut EthIRProgram,
        store: &AnalysesStore,
    ) -> bool {
        self.build_def_blocks(program);
        self.build_function_entries(program);
        self.build_in_out_groups(program, store);
        let dominators = store.dominators(program);

        let mut raw_map = HashMap::new();
        let mut candidates = Vec::new();

        for (group_idx, group) in self.io_groups.iter().enumerate() {
            if group.inputs.is_empty() || group.outputs.is_empty() {
                continue;
            }

            if group.inputs.iter().any(|&bb_id| self.function_entries[bb_id]) {
                continue;
            }

            let slots = program.basic_blocks[group.outputs[0]].outputs.len() as usize;
            if !group_has_matching_arity(program, group, slots) {
                continue;
            }

            for slot in 0..slots {
                let Some(replacement) = common_output_at_slot(program, group, slot) else {
                    continue;
                };

                if !self.replacement_is_available_at_inputs(&dominators, replacement, group) {
                    continue;
                }

                let Some(input_locals) =
                    try_add_slot_replacements(program, group, slot, replacement, &mut raw_map)
                else {
                    continue;
                };

                if !input_locals.is_empty() {
                    candidates.push(RemovalCandidate { group_idx, slot, input_locals });
                }
            }
        }

        if raw_map.is_empty() {
            return false;
        }

        let canonical_map = canonicalized_copy_map(&raw_map);
        if canonical_map.is_empty() {
            return false;
        }

        self.copy_map.clear();
        if self.slots_to_remove.len() < self.io_groups.len() {
            self.slots_to_remove.resize_with(self.io_groups.len(), Vec::new);
        }
        for remove in &mut self.slots_to_remove {
            remove.clear();
        }

        let mut changed = false;
        for candidate in &candidates {
            let Some(replacement) =
                self.candidate_final_replacement(&dominators, candidate, &canonical_map)
            else {
                continue;
            };

            for &input in &candidate.input_locals {
                let prev = self.copy_map.insert(input, replacement);
                debug_assert!(
                    prev.is_none_or(|prev| prev == replacement),
                    "conflicting replacement for ${input}"
                );
            }

            let group = &self.io_groups[candidate.group_idx];
            let slots = program.basic_blocks[group.outputs[0]].outputs.len() as usize;
            let remove = &mut self.slots_to_remove[candidate.group_idx];
            if remove.len() != slots {
                remove.clear();
                remove.resize(slots, false);
            }
            remove[candidate.slot] = true;
            changed = true;
        }

        if !changed || self.copy_map.is_empty() {
            return false;
        }

        replace_uses(program, &self.copy_map);
        self.compact_removed_slots(program);

        true
    }

    fn build_def_blocks(&mut self, program: &EthIRProgram) {
        self.def_blocks.clear();
        self.def_blocks.resize(program.next_free_local_id.idx(), None);

        for bb_id in program.basic_blocks.iter_idx() {
            let bb = &program.basic_blocks[bb_id];
            for &local in &program.locals[bb.inputs] {
                let prev = self.def_blocks[local].replace(bb_id);
                debug_assert!(prev.is_none(), "SSA violation: {local:?} defined twice");
            }
            for op_idx in bb.operations.iter() {
                for &local in program.operations[op_idx].outputs(program) {
                    let prev = self.def_blocks[local].replace(bb_id);
                    debug_assert!(prev.is_none(), "SSA violation: {local:?} defined twice");
                }
            }
        }
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

        self.io_groups.clear();
        self.io_groups.resize_with(bundling.total_groups() as usize, InOutGroupBlocks::default);

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
        &mut self.io_groups[group.idx()]
    }

    fn replacement_is_available_at_inputs(
        &self,
        dominators: &Dominators,
        replacement: LocalId,
        group: &InOutGroupBlocks,
    ) -> bool {
        let Some(Some(def_bb)) = self.def_blocks.get(replacement).copied() else {
            return false;
        };

        group.inputs.iter().all(|&input_bb| strictly_dominates(dominators, def_bb, input_bb))
    }

    fn candidate_final_replacement(
        &self,
        dominators: &Dominators,
        candidate: &RemovalCandidate,
        canonical_map: &HashMap<LocalId, LocalId>,
    ) -> Option<LocalId> {
        let group = &self.io_groups[candidate.group_idx];
        let mut final_replacement = None;

        for &input in &candidate.input_locals {
            let &replacement = canonical_map.get(&input)?;
            match final_replacement {
                None => final_replacement = Some(replacement),
                Some(existing) if existing == replacement => {}
                Some(_) => return None,
            }
        }

        let replacement = final_replacement?;
        self.replacement_is_available_at_inputs(dominators, replacement, group)
            .then_some(replacement)
    }

    fn compact_removed_slots(&mut self, program: &mut EthIRProgram) {
        for (group, remove) in self.io_groups.iter().zip(&self.slots_to_remove) {
            if !remove.iter().any(|&remove| remove) {
                continue;
            }

            for &bb_id in &group.inputs {
                let inputs = program.basic_blocks[bb_id].inputs;
                program.basic_blocks[bb_id].inputs =
                    compact_span(&mut program.locals, inputs, remove);
            }
            for &bb_id in &group.outputs {
                let outputs = program.basic_blocks[bb_id].outputs;
                program.basic_blocks[bb_id].outputs =
                    compact_span(&mut program.locals, outputs, remove);
            }
        }
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

struct RemovalCandidate {
    group_idx: usize,
    slot: usize,
    input_locals: Vec<LocalId>,
}

fn strictly_dominates(
    dominators: &Dominators,
    dominator: BasicBlockId,
    mut block: BasicBlockId,
) -> bool {
    if dominator == block {
        return false;
    }

    while let Some(idom) = dominators.of(block) {
        if idom == dominator {
            return true;
        }
        if idom == block {
            return false;
        }
        block = idom;
    }

    false
}

fn group_has_matching_arity(
    program: &EthIRProgram,
    group: &InOutGroupBlocks,
    slots: usize,
) -> bool {
    group.outputs.iter().all(|&bb_id| program.basic_blocks[bb_id].outputs.len() as usize == slots)
        && group
            .inputs
            .iter()
            .all(|&bb_id| program.basic_blocks[bb_id].inputs.len() as usize == slots)
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

fn try_add_slot_replacements(
    program: &EthIRProgram,
    group: &InOutGroupBlocks,
    slot: usize,
    replacement: LocalId,
    raw_map: &mut HashMap<LocalId, LocalId>,
) -> Option<Vec<LocalId>> {
    let mut inserted = Vec::new();
    let mut input_locals = Vec::new();

    for &bb_id in &group.inputs {
        let input = local_at_slot(program, program.basic_blocks[bb_id].inputs, slot);
        if input == replacement {
            continue;
        }

        input_locals.push(input);
        match raw_map.get(&input).copied() {
            None => {
                raw_map.insert(input, replacement);
                inserted.push(input);
            }
            Some(existing) if existing == replacement => {}
            Some(_) => {
                for input in inserted {
                    raw_map.remove(&input);
                }
                return None;
            }
        }
    }

    Some(input_locals)
}

fn canonicalized_copy_map(raw_map: &HashMap<LocalId, LocalId>) -> HashMap<LocalId, LocalId> {
    let mut canonical = HashMap::with_capacity(raw_map.len());

    for &key in raw_map.keys() {
        let Some(resolved) = resolve_copy_target_from_snapshot(key, raw_map) else {
            continue;
        };

        if resolved != key {
            canonical.insert(key, resolved);
        }
    }

    canonical
}

fn resolve_copy_target_from_snapshot(
    local: LocalId,
    snapshot: &HashMap<LocalId, LocalId>,
) -> Option<LocalId> {
    let mut current = local;
    let mut seen = Vec::new();

    while let Some(&next) = snapshot.get(&current) {
        if seen.contains(&current) {
            return None;
        }

        seen.push(current);
        current = next;
    }

    Some(current)
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

fn compact_span(
    locals: &mut IndexVec<LocalIdx, LocalId>,
    span: Span<LocalIdx>,
    remove: &[bool],
) -> Span<LocalIdx> {
    debug_assert_eq!(span.len() as usize, remove.len());

    let mut write = span.start;
    for (slot, read) in span.iter().enumerate() {
        if remove[slot] {
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
    use crate::run_pass_and_display;
    use sir_data::assert_ir_display;

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

        let actual = run_pass_and_display::<CopyPropagation>(input);
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
    fn test_phi_nodes_block_propagation() {
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

        let actual = run_pass_and_display::<CopyPropagation>(input);
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
                    $1 = copy $0
                    $2 = copy $0
                    => @2
                }

                @2 $3 {
                    $4 = add $3 $3
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

        let actual = run_pass_and_display::<CopyPropagation>(input);
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

        let actual = run_pass_and_display::<CopyPropagation>(input);
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

        let actual = run_pass_and_display::<CopyPropagation>(input);
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
    fn test_copy_map_does_not_leak_between_blocks() {
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

        let actual = run_pass_and_display::<CopyPropagation>(input);
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

                @1 $0 -> $0 $0 {
                    $1 = copy $0
                    => @2
                }

                @2 $2 $3 {
                    $4 = add $2 $3
                    stop
                }
            "#,
        );
    }
}
