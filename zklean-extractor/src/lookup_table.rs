use jolt_core::zkvm::lookup_table::{JoltLookupTable, LookupTables};
use strum::IntoEnumIterator as _;

use crate::{
    constants::JoltParameterSet,
    modules::{AsModule, Module},
    util::{indent, ZkLeanReprField},
};

/// Wrapper around a JoltLookupTable
#[derive(Debug)]
pub struct ZkLeanLookupTable<F: ZkLeanReprField, J, const WORD_SIZE: usize> {
    table: LookupTables<WORD_SIZE>,
    phantom: std::marker::PhantomData<(F, J)>,
}

impl<F: ZkLeanReprField, J, const WORD_SIZE: usize> From<LookupTables<WORD_SIZE>> for ZkLeanLookupTable<F, J, WORD_SIZE> {
    fn from(value: LookupTables<WORD_SIZE>) -> Self {
        Self {
            table: value,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<F: ZkLeanReprField, J, const WORD_SIZE: usize> From<JoltLookupTable<F, WORD_SIZE>> for ZkLeanLookupTable<F, J, WORD_SIZE> {
    fn from(value: JoltLookupTable<F, WORD_SIZE>) -> Self {
        Self {
            table: value.into(),
            phantom: std::marker::PhantomData,
        }
    }
}

impl<F: ZkLeanReprField, J: JoltParameterSet, const WORD_SIZE: usize> ZkLeanLookupTable<F, J, WORD_SIZE> {
    pub fn name(&self) -> String {
        let name = <&'static str>::from(&self.table);
        let log_m = J::LOG_M;

        format!("{name}_{log_m}")
    }

    pub fn evaluate_mle(&self, reg_name: char) -> F {
        let reg = F::register(reg_name, J::LOG_M);
        self.subtables.evaluate_mle(&reg)
    }

    pub fn iter() -> impl Iterator<Item = Self> {
        LookupTables::<WORD_SIZE>::iter().map(Self::from)
    }

    /// Pretty print a subtable as a ZkLean `Subtable`.
    pub fn zklean_pretty_print(
        &self,
        f: &mut impl std::io::Write,
        mut indent_level: usize,
    ) -> std::io::Result<()> {
        let name = self.name();
        let log_m = J::LOG_M;
        let mle = self.evaluate_mle('x').as_computation();

        f.write_fmt(format_args!(
            "{}def {name} [Field f] : LookupTable f {log_m} :=\n",
            indent(indent_level),
        ))?;
        indent_level += 1;
        f.write_fmt(format_args!(
            "{}lookupTableFromMLE (fun x => {mle})\n",
            indent(indent_level),
        ))?;

        Ok(())
    }
}

pub struct ZkLeanLookupTables<F: ZkLeanReprField, J> {
    tables: Vec<ZkLeanLookupTable<F, J>>,
    phantom: std::marker::PhantomData<J>,
}

impl<F: ZkLeanReprField, J: JoltParameterSet> ZkLeanLookupTables<F, J> {
    pub fn extract() -> Self {
        Self {
            tables: ZkLeanLookupTable::<F, J>::iter().collect(),
            phantom: std::marker::PhantomData,
        }
    }

    pub fn zklean_pretty_print(
        &self,
        f: &mut impl std::io::Write,
        indent_level: usize,
    ) -> std::io::Result<()> {
        for table in &self.tables {
            table.zklean_pretty_print(f, indent_level)?;
        }
        Ok(())
    }

    pub fn zklean_imports(&self) -> Vec<String> {
        vec![String::from("ZkLean")]
    }
}

impl<F: ZkLeanReprField, J: JoltParameterSet> AsModule for ZkLeanLookupTables<F, J> {
    fn as_module(&self) -> std::io::Result<Module> {
        let mut contents: Vec<u8> = vec![];
        self.zklean_pretty_print(&mut contents, 0)?;

        Ok(Module {
            name: String::from("LookupTables"),
            imports: self.zklean_imports(),
            contents,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::util::arb_field_elem;

    use jolt_core::{field::JoltField, zkvm::lookup_table::LookupTables};

    use proptest::{collection::vec, prelude::*};
    use strum::EnumCount as _;

    type RefField = ark_bn254::Fr;
    type TestField = crate::mle_ast::MleAst<4096>;
    type ParamSet = crate::constants::RV32IParameterSet;

    struct TestableLookupTable<R: JoltField, T: ZkLeanReprField, J: JoltParameterSet> {
        reference: LookupTables,
        test: ZkLeanLookupTable<T, J>,
    }

    impl<R: JoltField, T: ZkLeanReprField, J: JoltParameterSet> std::fmt::Debug
        for TestableLookupTable<R, T, J>
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_fmt(format_args!("{}", self.test.name()))
        }
    }

    impl<R: JoltField, T: ZkLeanReprField, J: JoltParameterSet> TestableLookupTable<R, T, J> {
        fn iter() -> impl Iterator<Item = Self> {
            LookupTables::iter()
                .zip(ZkLeanLookupTable::iter())
                .map(|(reference, test)| Self {
                    reference,
                    test,
                })
        }

        fn reference_evaluate_mle(&self, inputs: &[R]) -> R {
            assert_eq!(inputs.len(), J::LOG_M);

            self.reference.evaluate_mle(inputs)
        }

        fn test_evaluate_mle(&self, inputs: &[R]) -> R {
            assert_eq!(inputs.len(), J::LOG_M);

            let ast = self.test.evaluate_mle('x');
            ast.evaluate(inputs)
        }
    }

    fn arb_lookup_table<R: JoltField, T: ZkLeanReprField, J: JoltParameterSet>(
    ) -> impl Strategy<Value = TestableLookupTable<R, T, J>> {
        (0..LookupTables::<R>::COUNT).prop_map(|n| TestableLookupTable::iter().nth(n).unwrap())
    }

    proptest! {
        #[test]
        fn evaluate_mle(
            lookup_table in arb_lookup_table::<RefField, TestField, ParamSet>(),
            inputs in vec(arb_field_elem::<RefField>(), ParamSet::LOG_M),
        ) {
            prop_assert_eq!(
                lookup_table.test_evaluate_mle(&inputs),
                lookup_table.reference_evaluate_mle(&inputs),
            );
        }
    }
}
