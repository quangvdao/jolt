//! Deterministic combine DAG (balanced fold) for homomorphic GT commitment combination.
//!
//! This is **not** Dory's symbolic AST. It is a small, recursion-specific DAG describing the
//! deterministic balanced binary-tree used by `GTCombineWitness`:
//! - inputs (leaves) are the `exp_witnesses[i].result` values
//! - internal nodes are GT muls pairing adjacent elements left-to-right per level
//! - if a level has an odd number of nodes, the last node is carried forward unchanged
//!
//! The shape is fully determined by `num_leaves` and must match:
//! - prover-side witness generation (`DoryCommitmentScheme::generate_combine_witness`)
//! - verifier-side plan derivation / wiring constraints (future work)

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CombineDag {
    pub num_leaves: usize,
    pub layers: Vec<CombineLayer>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CombineLayer {
    /// Multiplications performed at this level, in left-to-right order.
    pub muls: Vec<CombineMul>,
    /// Node ids (leaf or internal) that form the next level’s input list.
    pub next_nodes: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CombineMul {
    pub lhs: usize,
    pub rhs: usize,
    pub out: usize,
}

impl CombineDag {
    pub fn new(num_leaves: usize) -> Self {
        assert!(num_leaves > 0, "combine DAG requires at least one leaf");

        // Leaves are node ids 0..num_leaves-1.
        let mut current: Vec<usize> = (0..num_leaves).collect();
        let mut next_node_id = num_leaves;

        let mut layers = Vec::new();
        while current.len() > 1 {
            let mut muls = Vec::with_capacity(current.len() / 2);
            let mut next = Vec::with_capacity(current.len().div_ceil(2));

            for chunk in current.chunks(2) {
                if let [a, b] = chunk {
                    let out = next_node_id;
                    next_node_id += 1;
                    muls.push(CombineMul {
                        lhs: *a,
                        rhs: *b,
                        out,
                    });
                    next.push(out);
                } else {
                    // Odd tail: carry forward the last node unchanged.
                    next.push(chunk[0]);
                }
            }

            layers.push(CombineLayer {
                muls,
                next_nodes: next.clone(),
            });
            current = next;
        }

        Self { num_leaves, layers }
    }

    pub fn num_muls_total(&self) -> usize {
        self.layers.iter().map(|l| l.muls.len()).sum()
    }

    pub fn root(&self) -> usize {
        // Root is always the only node remaining after all layers.
        if self.num_leaves == 1 {
            return 0;
        }
        *self
            .layers
            .last()
            .expect("num_leaves>1 implies at least one layer")
            .next_nodes
            .last()
            .expect("non-empty")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combine_dag_has_expected_mul_count() {
        for n in 1..64 {
            let dag = CombineDag::new(n);
            assert_eq!(dag.num_leaves, n);
            assert_eq!(dag.num_muls_total(), n.saturating_sub(1));
        }
    }

    #[test]
    fn combine_dag_levels_follow_pair_and_carry_rule() {
        // Spot-check a non-power-of-two size.
        let dag = CombineDag::new(5);

        // Level 0: (0,1)->5, (2,3)->6 carry 4 => next [5,6,4]
        assert_eq!(
            dag.layers[0].muls,
            vec![
                CombineMul {
                    lhs: 0,
                    rhs: 1,
                    out: 5
                },
                CombineMul {
                    lhs: 2,
                    rhs: 3,
                    out: 6
                }
            ]
        );
        assert_eq!(dag.layers[0].next_nodes, vec![5, 6, 4]);

        // Level 1: (5,6)->7 carry 4 => next [7,4]
        assert_eq!(
            dag.layers[1].muls,
            vec![CombineMul {
                lhs: 5,
                rhs: 6,
                out: 7
            }]
        );
        assert_eq!(dag.layers[1].next_nodes, vec![7, 4]);

        // Level 2: (7,4)->8 => next [8]
        assert_eq!(
            dag.layers[2].muls,
            vec![CombineMul {
                lhs: 7,
                rhs: 4,
                out: 8
            }]
        );
        assert_eq!(dag.layers[2].next_nodes, vec![8]);

        assert_eq!(dag.root(), 8);
    }
}
