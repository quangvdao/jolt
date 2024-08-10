# Anatomy of a Jolt proof

We give a full specification on the format of a Jolt proof, how it is generated, and the constraints it imposes on the Jolt witness.

![Jolt Program Overview](../imgs/jolt-program.png)

![Jolt Preprocessing](../imgs/jolt-preprocessing.png)

![Jolt Trace](../imgs/jolt-trace.png)

(prove: program's IO => Jolt preprocessing => Jolt witness => Jolt proof)

(verify: program's IO => Jolt preprocessing => Jolt proof => accept/reject)

## Preprocessing for Jolt

## Generating a Jolt witness

## Format of a Jolt proof

## Constraints on the Jolt witness

Jolt imposes a set of constraints on the witness above, which is of three forms:
- Read-only memory constraints (for bytecode and table lookups)
- Read-write memory constraints (for registers and memory)
- R1CS constraints (for linking the above constraints together)