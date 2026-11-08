# EGV public expression DSL v1

Status: frozen candidate contract for learnable-corpus Gate A.

Identifier: `egv-public-expression-dsl-v1`.

This document is the normative, model-visible meaning of the `program` field in
an `egv-learnable-public-spec-v2` task. A conforming solver needs only the
serialized public task package, this operator contract, and the runtime input.
It must not use an evaluator seed, private manifest, hidden case, expected
answer, reference implementation, or source fingerprint.

## Data model

Values use the JSON data model: null, Boolean, integer, text, list, and object.
Objects have text keys. Implementations must preserve list order and must reject
an expression node containing an unknown or extra field.

Evaluation is deterministic and side-effect free. There is no file, clock,
randomness, environment, process, network, import, or mutable-state operator.
An invalid operand or unavailable path is an evaluation error, not a value.

Paths are non-empty lists of non-empty text keys. A path walks objects only.

## Closed expression nodes

Each node is an object with exactly the fields shown below.

| `op` | Other fields | Result |
| --- | --- | --- |
| `literal` | `value` | The JSON value in `value`. |
| `input` | `path` | The value reached by walking `path` from the runtime input. |
| `variable` | `name`, `path` | The value reached by walking `path` from the lexically bound variable `name`. |
| `object` | `fields` | An object made from the ordered field records. |
| `add_integers` | `items` | The integer sum of a non-empty list of evaluated integer expressions. Booleans are not integers. |
| `split_text` | `text`, `delimiter` | `text` split at every exact, non-empty `delimiter`; leading, trailing, and adjacent empty tokens are preserved. |
| `if_equal` | `left`, `right`, `then`, `else` | Evaluate `left` and `right`; evaluate only `then` when equal, otherwise only `else`. |
| `sort_objects` | `items`, `key` | A stable ascending sort of a list of objects by the value at text key `key`. |
| `map_list` | `items`, `variable`, `body` | Evaluate `body` once per list item with that item lexically bound to `variable`; preserve list order. |
| `list_length` | `items` | The number of items in an evaluated list. |
| `sum_integers` | `items` | The integer sum of an evaluated list of integers. Booleans are not integers. |
| `mapping_get` | `mapping`, `key`, `default` | The object value for the evaluated key, or the evaluated default when absent. |

An `object` node's `fields` value is a list of closed records with exactly
`name` and `value`. Names must be non-empty, unique text. `value` is an
expression node.

`map_list.variable` must be non-empty text. Its binding is visible only while
evaluating that node's `body`; an inner binding with the same name shadows the
outer binding.

## Runtime contract

The runtime input is a closed object with exactly `case_nonce`, `payload`, and
`task_scope`. `task_scope` must equal the public specification's task scope.
The public `payload_contract` defines the accepted payload shape and bound.
The root program must evaluate to an object whose fields exactly match
`output_contract`.

## Conformance boundary

A Gate-A blind solver is conforming only when it:

1. parses the serialized model-visible package independently;
2. rejects unknown schemas, operators, fields, paths, and invalid operands;
3. derives source or output exclusively from the public program and runtime
   input; and
4. passes the frozen blind slice through the production sandbox/controller
   with externally verifiable receipts.

Passing a same-module oracle or a local helper alone is not Gate-A acceptance.
