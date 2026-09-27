namespace ManagedCode.Synapse.Contracts.Features.GraphExecution;

/// <summary>Stable node identity inside one model graph version.</summary>
/// <param name="Value">Unsigned identity value.</param>
public readonly record struct NodeId(uint Value);

/// <summary>Stable SSA value identity inside one model graph version.</summary>
/// <param name="Value">Unsigned identity value.</param>
public readonly record struct ValueId(uint Value);

/// <summary>Stable immutable tensor identity.</summary>
/// <param name="Value">Unsigned identity value.</param>
public readonly record struct TensorId(uint Value);

/// <summary>Stable executable-region identity.</summary>
/// <param name="Value">Unsigned identity value.</param>
public readonly record struct RegionId(uint Value);

/// <summary>Stable named graph entry-point identity.</summary>
/// <param name="Value">Unsigned identity value.</param>
public readonly record struct EntryPointId(uint Value);

/// <summary>Stable mutable-state slot identity.</summary>
/// <param name="Value">Unsigned identity value.</param>
public readonly record struct StateSlotId(uint Value);

/// <summary>SSA token that serializes observable state effects.</summary>
/// <param name="Value">Unsigned identity value.</param>
public readonly record struct EffectToken(uint Value);

/// <summary>Version of the portable graph schema.</summary>
/// <param name="Major">Breaking schema version.</param>
/// <param name="Minor">Backward-compatible schema revision.</param>
public readonly record struct GraphVersion(ushort Major, ushort Minor);

/// <summary>Version of the operation semantics used by a graph.</summary>
/// <param name="Major">Breaking operation-set version.</param>
/// <param name="Minor">Backward-compatible operation-set revision.</param>
public readonly record struct OpSetVersion(ushort Major, ushort Minor);

/// <summary>Lower-case SHA-256 content identity.</summary>
/// <param name="Value">Hexadecimal digest.</param>
public readonly record struct ContentHash(string Value);
