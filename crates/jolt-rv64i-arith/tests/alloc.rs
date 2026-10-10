use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use jolt_program::image::decode::decode_instruction;
use jolt_riscv::RV64IMAC_JOLT;
use jolt_rv64i_arith::{
    BaseWords, BitsBuilder, Bytecode, CycleFacts, Layout, RowSystem, WitnessRow,
};

struct CountingAllocator;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every allocation operation forwards its original pointer/layout to
// System; the additional atomic counter does not affect allocator ownership.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: AllocLayout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            let _previous = ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: the caller supplies the valid allocation layout required by System.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: AllocLayout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            let _previous = ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: the caller supplies the valid allocation layout required by System.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: AllocLayout) {
        // SAFETY: ptr and layout are forwarded unchanged from the allocator caller.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: AllocLayout, size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            let _previous = ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: ptr, layout and size are forwarded unchanged from the allocator caller.
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test fixtures have fixed valid shapes"
)]
fn per_cycle_paths_allocate_nothing_for_4096_cycles() {
    let layout = Layout::new(4, 4, 0).unwrap();
    let instructions: Vec<_> = [
        0x0020_81b3,
        0x0020_f1b3,
        0x0020_b1b3,
        0x0010_9193,
        0x0030_8183,
        0x0020_9323,
        0x0020_8463,
        0x0010_8067,
    ]
    .into_iter()
    .enumerate()
    .map(|(i, word)| decode_instruction(word, 0x1000 + 4 * i as u64, false, RV64IMAC_JOLT).unwrap())
    .collect();
    let bytecode = Bytecode::preprocess(&instructions, &layout).unwrap();
    let builder = BitsBuilder::new(&layout, &bytecode).unwrap();
    let rows = RowSystem::new(&layout);
    let small = [
        CycleFacts {
            rs1_value: 1,
            rs2_value: 1,
            rd_post_value: 2,
            next_pc: 0x1004,
            ..CycleFacts::default()
        },
        CycleFacts {
            bytecode_index: 1,
            rs1_value: 3,
            rs2_value: 5,
            rd_post_value: 1,
            next_pc: 0x1008,
            ..CycleFacts::default()
        },
        CycleFacts {
            bytecode_index: 2,
            rs1_value: 1,
            rs2_value: 2,
            rd_post_value: 1,
            next_pc: 0x100c,
            ..CycleFacts::default()
        },
        CycleFacts {
            bytecode_index: 3,
            rs1_value: 1,
            rd_post_value: 2,
            next_pc: 0x1010,
            ..CycleFacts::default()
        },
        CycleFacts {
            bytecode_index: 4,
            rs1_value: 8,
            rd_post_value: 0xffff_ffff_ffff_ff80,
            ram_word_index: 1,
            ram_pre_value: 0x8000_0000,
            next_pc: 0x1014,
            ..CycleFacts::default()
        },
        CycleFacts {
            bytecode_index: 5,
            rs1_value: 8,
            rs2_value: 0xabcd,
            ram_word_index: 1,
            ram_pre_value: 0x1122_3344_5566_7788,
            ram_post_value: 0xabcd_3344_5566_7788,
            next_pc: 0x1018,
            ..CycleFacts::default()
        },
        CycleFacts {
            bytecode_index: 6,
            rs1_value: 1,
            rs2_value: 2,
            next_pc: 0x101c,
            ..CycleFacts::default()
        },
        CycleFacts {
            bytecode_index: 7,
            rs1_value: 0x1000,
            next_pc: 0x1000,
            ..CycleFacts::default()
        },
    ];
    let facts: Vec<_> = (0..4096).map(|i| small[i % small.len()]).collect();
    let bases: Vec<_> = facts.iter().map(BaseWords::from_facts).collect();
    let mut bits = vec![[0; 4]; facts.len()];
    builder.fill(&facts, &mut bits).unwrap();
    for (fact, (base, bits)) in facts.iter().zip(bases.iter().zip(&bits)) {
        let z = WitnessRow::compute(
            &layout,
            &bytecode.rows()[fact.bytecode_index as usize],
            base,
            bits,
        );
        assert!(rows.failing_rows(&z).is_empty());
    }
    ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::SeqCst);
    let result = builder.fill(black_box(&facts), black_box(&mut bits));
    for (fact, (base, bits)) in facts.iter().zip(bases.iter().zip(&bits)) {
        let z = WitnessRow::compute(
            black_box(&layout),
            black_box(&bytecode.rows()[fact.bytecode_index as usize]),
            black_box(base),
            black_box(bits),
        );
        for row in rows.lane_rows() {
            let _values = black_box(row.values(black_box(&z)));
        }
        for row in rows.packed_rows() {
            let _values = black_box(row.values(black_box(&z)));
        }
    }
    COUNTING.store(false, Ordering::SeqCst);
    assert!(result.is_ok());
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
}
