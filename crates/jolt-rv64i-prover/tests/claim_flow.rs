//! Canonical claim identifiers, aliases and terminal projections across all eight batches.
#![expect(
    clippy::unwrap_used,
    reason = "invalid fixtures fail the enclosing test"
)]
use jolt_claims::{InputClaims, OutputClaims, SymbolicSumcheck};
use jolt_field::{Zero, F128};
use jolt_rv64i_arith::Layout;
use jolt_rv64i_verifier::{
    claims::{router_short::RouterShortSymbolic, spartan_inner::SpartanInnerSymbolic},
    ids::{CommittedPolynomial, OpeningId, PolynomialId, RelationId, VirtualPolynomial},
    stages::{
        stage1::{SpartanOuterF128, SpartanOuterF2, Stage1Sumchecks},
        stage2::SpartanInner,
        stage3a::RouterShort,
        stage3b::{
            RouterCycleBranch, RouterCycleCompare, RouterCycleMemory, RouterCycleShift,
            RouterCycleVariant, Stage3bSumchecks,
        },
        stage4::{RamOutputCheck, RamReadChecking, RegistersReadChecking, Stage4Sumchecks},
        stage5::{RamValEvaluation, RegistersValEvaluation, Stage5Sumchecks},
        stage6a::{BytecodeReadAddress, Stage6aSumchecks},
        stage6b::{BitsReduction, BytecodeReadCycle, RamRaProduct, Stage6bSumchecks},
    },
};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, SumcheckInputClaims, SumcheckOutputClaims,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

struct Member {
    relation: RelationId,
    inputs: Vec<OpeningId>,
    outputs: Vec<OpeningId>,
    aliases: Vec<(OpeningId, OpeningId)>,
}
fn member<R>(symbolic: &R::Symbolic) -> Member
where
    R: ConcreteSumcheck<F128>,
    R::Symbolic: SymbolicSumcheck<RelationId = RelationId, OpeningId = OpeningId>,
    SumcheckInputClaims<F128, R>: Default + InputClaims<F128, OpeningId>,
    SumcheckOutputClaims<F128, R>: OutputClaims<F128, OpeningId>,
{
    let expected = symbolic.expected_output_openings::<F128>();
    let outputs = SumcheckOutputClaims::<F128, R>::from_opening_values(|id| {
        expected.contains(id).then(F128::zero)
    })
    .unwrap();
    Member {
        relation: R::Symbolic::id(),
        inputs: SumcheckInputClaims::<F128, R>::default().canonical_order(),
        outputs: outputs.canonical_order(),
        aliases: R::aliased_output_openings(),
    }
}
macro_rules! snapshot {
    ($function:ident, batch=$batch:ident,label=$label:literal,aggregates={$($aggregates:tt)*},shape=$shape:ident,
     members=[$({name:$field:ident,relation:$relation:ident,presence:required},)+]) => {
        fn $function(batch:&$batch<F128>)->Vec<Member> { vec![$(member::<$relation<F128>>(batch.$field.symbolic())),+] }
    };
}
jolt_rv64i_verifier::stage1_sumchecks_members!(snapshot stage1,);
jolt_rv64i_verifier::stage3b_sumchecks_members!(snapshot stage3b,);
jolt_rv64i_verifier::stage4_sumchecks_members!(snapshot stage4,);
jolt_rv64i_verifier::stage5_sumchecks_members!(snapshot stage5,);
jolt_rv64i_verifier::stage6a_sumchecks_members!(snapshot stage6a,);
jolt_rv64i_verifier::stage6b_sumchecks_members!(snapshot stage6b,);

fn is_projection(id: &OpeningId) -> bool {
    matches!(
        id.polynomial,
        PolynomialId::Virtual(
            VirtualPolynomial::BytecodeRaChunk(_) | VirtualPolynomial::RamRaChunk(_)
        )
    )
}
fn is_column(id: &OpeningId) -> bool {
    matches!(
        id.polynomial,
        PolynomialId::Committed(CommittedPolynomial::Column(_))
    )
}

const FLOW: &str = "1 SpartanOuterF2.V:Az -> SpartanInner\n1 SpartanOuterF2.V:Bz -> SpartanInner\n1 SpartanOuterF2.V:Cz -> SpartanInner\n1 SpartanOuterF128.V:Az -> SpartanInner\n1 SpartanOuterF128.V:Bz -> SpartanInner\n1 SpartanOuterF128.V:Cz -> SpartanInner\n2 SpartanInner.V:WitnessRouted -> RouterShort\n2 SpartanInner.C:DirectColumns -> BitsReduction\n3a RouterShort.V:RouterFold(Variant) -> RouterCycleVariant\n3a RouterShort.V:RouterFold(Shift) -> RouterCycleShift\n3a RouterShort.V:RouterFold(Memory) -> RouterCycleMemory\n3a RouterShort.V:RouterFold(Compare) -> RouterCycleCompare\n3a RouterShort.V:RouterFold(Branch) -> RouterCycleBranch\n3b RouterCycleVariant.V:Rs1Value -> RegistersReadChecking\n3b RouterCycleVariant.V:Rs2Value -> RegistersReadChecking\n3b RouterCycleVariant.V:RdPreValue -> RegistersReadChecking\n3b RouterCycleVariant.V:Imm -> BytecodeReadAddress\n3b RouterCycleVariant.V:FallThroughPC -> BytecodeReadAddress\n3b RouterCycleVariant.V:PCPlusImm -> BytecodeReadAddress\n3b RouterCycleVariant.V:PC -> BytecodeReadAddress\n3b RouterCycleVariant.V:NextPC -> BytecodeReadAddress\n3b RouterCycleVariant.C:VariantBits -> BitsReduction\n3b RouterCycleVariant.V:Variant -> BytecodeReadAddress\n3b RouterCycleShift.V:ShiftKind -> BytecodeReadAddress\n3b RouterCycleShift.C:PosRa0 -> BitsReduction\n3b RouterCycleShift.C:PosRa1 -> BitsReduction\n3b RouterCycleMemory.V:RamReadValue -> RamReadChecking\n3b RouterCycleMemory.V:AccessKind -> BytecodeReadAddress\n3b RouterCycleCompare.V:KeyKind -> BytecodeReadAddress\n3b RouterCycleBranch.V:Branch -> BytecodeReadAddress\n3b RouterCycleBranch.C:ShouldBranch -> BitsReduction\n4 RegistersReadChecking.V:Rs1Ra -> BytecodeReadAddress\n4 RegistersReadChecking.V:Rs2Ra -> BytecodeReadAddress\n4 RegistersReadChecking.V:RdWa -> BytecodeReadAddress\n4 RegistersReadChecking.V:RegistersVal -> RegistersValEvaluation\n4 RamReadChecking.V:RamRa -> RamRaProduct\n4 RamReadChecking.V:RamVal -> RamValEvaluation\n4 RamOutputCheck.V:RamValFinal -> RamValEvaluation\n5 RegistersValEvaluation.V:RdWa -> BytecodeReadAddress\n5 RegistersValEvaluation.V:Store -> BytecodeReadAddress\n5 RegistersValEvaluation.C:Inc -> BitsReduction\n5 RamValEvaluation.V:RamRa -> RamRaProduct\n6a BytecodeReadAddress.V:BytecodeAddressClaim -> BytecodeReadCycle\n6b BitsReduction.C:Column(0..256) -> Opening\n";

#[test]
#[expect(
    clippy::print_stdout,
    reason = "the claim-flow acceptance criterion prints its regenerated table"
)]
fn canonical_claim_flow_is_the_protocol_table_at_two_layouts() {
    for (b, a) in [(20, 20), (20, 23)] {
        let layout = Layout::new(b, a, 0).unwrap();
        let batches = [
            stage1(&Stage1Sumchecks::for_geometry(22, &layout).unwrap()),
            vec![member::<SpartanInner<F128>>(&SpartanInnerSymbolic::new(()))],
            vec![member::<RouterShort<F128>>(&RouterShortSymbolic::new(()))],
            stage3b(&Stage3bSumchecks::for_geometry(22, &layout).unwrap()),
            stage4(&Stage4Sumchecks::<F128>::for_geometry(22, &layout).unwrap()),
            stage5(&Stage5Sumchecks::<F128>::for_geometry(22, &layout).unwrap()),
            stage6a(&Stage6aSumchecks::<F128>::for_geometry(22, &layout).unwrap()),
            stage6b(&Stage6bSumchecks::<F128>::for_geometry(22, &layout).unwrap()),
        ];
        let mut produced = BTreeMap::new();
        let mut consumed: BTreeMap<OpeningId, Vec<RelationId>> = BTreeMap::new();
        let mut wire_order = Vec::new();
        let labels = ["1", "2", "3a", "3b", "4", "5", "6a", "6b"];
        let mut aliases = 0;
        let mut projections = 0;
        for (batch, members) in batches.iter().enumerate() {
            for relation in members {
                for &id in &relation.inputs {
                    let source = *produced.get(&id).unwrap();
                    assert!(source < batch, "{id:?} consumed before its producing batch");
                    consumed.entry(id).or_default().push(relation.relation);
                }
            }
            let batch_wires: BTreeSet<_> = members
                .iter()
                .flat_map(|member| {
                    member
                        .outputs
                        .iter()
                        .filter(|id| {
                            !is_projection(id)
                                && !member.aliases.iter().any(|(alias, _)| alias == *id)
                        })
                        .copied()
                })
                .collect();
            for relation in members {
                for &(alias, source) in &relation.aliases {
                    assert!(relation.outputs.contains(&alias));
                    assert!(batch_wires.contains(&source));
                    assert_eq!(alias.polynomial, source.polynomial);
                    aliases += 1;
                }
                for &id in &relation.outputs {
                    if relation.aliases.iter().any(|(alias, _)| *alias == id) {
                        continue;
                    }
                    if is_projection(&id) {
                        assert_eq!(batch, 7);
                        match id.polynomial {
                            PolynomialId::Virtual(VirtualPolynomial::BytecodeRaChunk(chunk)) => {
                                assert!(chunk < layout.bytecode_ra().len());
                            }
                            PolynomialId::Virtual(VirtualPolynomial::RamRaChunk(chunk)) => {
                                assert!(chunk < layout.ram_ra().len());
                            }
                            _ => unreachable!(),
                        }
                        projections += 1;
                        continue;
                    }
                    assert!(produced.insert(id, batch).is_none());
                    wire_order.push((batch, id));
                }
            }
        }
        assert_eq!(aliases, 12);
        assert_eq!(
            projections,
            layout.bytecode_ra().len() + layout.ram_ra().len()
        );
        assert_eq!(wire_order.len(), 299);
        let columns: Vec<_> = wire_order
            .iter()
            .filter(|(_, id)| is_column(id))
            .map(|(_, id)| *id)
            .collect();
        assert_eq!(
            columns,
            (0..256)
                .map(|column| OpeningId::committed(
                    CommittedPolynomial::Column(column),
                    RelationId::BitsReduction
                ))
                .collect::<Vec<_>>()
        );
        let mut table = String::new();
        for &(batch, id) in &wire_order {
            if is_column(&id) {
                assert!(!consumed.contains_key(&id));
                continue;
            }
            let consumers = consumed.get(&id).unwrap();
            assert_eq!(consumers.len(), 1, "{id:?} has several consuming cells");
            let polynomial = match id.polynomial {
                PolynomialId::Virtual(p) => format!("V:{p:?}"),
                PolynomialId::Committed(p) => format!("C:{p:?}"),
            };
            writeln!(
                table,
                "{} {:?}.{} -> {:?}",
                labels[batch], id.relation, polynomial, consumers[0]
            )
            .unwrap();
        }
        table.push_str("6b BitsReduction.C:Column(0..256) -> Opening\n");
        print!("{table}");
        assert_eq!(
            table, FLOW,
            "canonical ids differ from the claim-flow table"
        );
    }
}
