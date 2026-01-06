# olm.core.registry

### Classes

| [`Registry`](#olm.core.registry.Registry)()   |    |
|-----------------------------------------------|----|

### *class* olm.core.registry.Registry

Bases: `object`

#### get(name: str) → Callable[[...], T]

#### register(name: str)

### *class* olm.core.registry.TypeVar

Bases: `object`

Type variable.

The preferred way to construct a type variable is via the dedicated
syntax for generic functions, classes, and type aliases:

```default
class Sequence[T]:  # T is a TypeVar
    ...
```

This syntax can also be used to create bound and constrained type
variables:

```default
# S is a TypeVar bound to str
class StrSequence[S: str]:
    ...

# A is a TypeVar constrained to str or bytes
class StrOrBytesSequence[A: (str, bytes)]:
    ...
```

Type variables can also have defaults:

> class IntDefault[T = int]:
> : …

However, if desired, reusable type variables can also be constructed
manually, like so:

```default
T = TypeVar('T')  # Can be anything
S = TypeVar('S', bound=str)  # Can be any subtype of str
A = TypeVar('A', str, bytes)  # Must be exactly str or bytes
D = TypeVar('D', default=int)  # Defaults to int
```

Type variables exist primarily for the benefit of static type
checkers.  They serve as the parameters for generic types as well
as for generic function and type alias definitions.

The variance of type variables is inferred by type checkers when they
are created through the type parameter syntax and when
`infer_variance=True` is passed. Manually created type variables may
be explicitly marked covariant or contravariant by passing
`covariant=True` or `contravariant=True`. By default, manually
created type variables are invariant. See PEP 484 and PEP 695 for more
details.

#### evaluate_bound

#### evaluate_constraints

#### evaluate_default

#### has_default()
