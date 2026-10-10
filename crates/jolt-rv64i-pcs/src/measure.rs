//! Phase and release events consumed by the documented measurement example.

/// Timed operations from the specification's Performance table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    LevelZeroEncode,
    LevelZeroTree,
    CommitSample,
    BridgeTables,
    BridgeFirstPass,
    BridgeFirstFold,
    LaterRounds,
    InducedWeights,
    EqualityAndSamples,
    LaterEncodes,
    LaterTrees,
    Queries,
}

/// Joined-worker snapshots at the allocation lifetimes of specification §9.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReleasePoint {
    Commit,
    FirstPass,
    FirstFold,
    LevelFold(usize),
    NextOracle(usize),
    NextSample(usize),
    OldOracle(usize),
    Induction(usize),
}

/// Timers and allocator snapshots are outside the protocol and transcript.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    Start(Phase),
    End(Phase),
    Released(ReleasePoint),
}

pub(crate) fn run<R>(
    observer: &mut impl FnMut(Event),
    phase: Phase,
    operation: impl FnOnce() -> R,
) -> R {
    observer(Event::Start(phase));
    let result = operation();
    observer(Event::End(phase));
    result
}
