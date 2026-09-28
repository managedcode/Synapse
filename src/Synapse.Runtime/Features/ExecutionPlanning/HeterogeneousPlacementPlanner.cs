using ManagedCode.Synapse.Runtime.Features.Quantization;

namespace ManagedCode.Synapse.Runtime.Features.ExecutionPlanning;

/// <summary>
/// Bounded search (spec §10.3) that splits layer regions into contiguous stages over up to
/// four heterogeneous devices. Every device order is tried; for each order a dynamic
/// program chooses the split. Each stage reserves KV for the requested context and gets
/// the highest importance-aware precision that fits the device's remaining memory and
/// kernels. Ties prefer lower distortion, then the earlier device order. Costs are
/// planning estimates from measured profiles, not measurements.
/// </summary>
public static class HeterogeneousPlacementPlanner
{
    /// <summary>Largest device count searched exhaustively over orders.</summary>
    public const int MaximumDevices = 4;

    private const double RelativeTolerance = 1e-12;

    /// <summary>Returns the best feasible plan or throws a typed failure.</summary>
    public static PlacementPlan Plan(PlacementRequest request)
    {
        Validate(request);
        var plan = Search(request);
        if (plan is not null)
        {
            return plan;
        }

        var failure = !request.AllowApproximation && Search(request with { AllowApproximation = true }) is not null
            ? PlacementFailure.ApproximationNotAllowed
            : PlacementFailure.InsufficientMemory;
        throw new PlacementException(failure, $"No placement of {request.Layers.Count} layers fits the devices ({failure}).");
    }

    private static PlacementPlan? Search(PlacementRequest request)
    {
        var stages = new StageCostCache(request);
        var order = Enumerable.Range(0, request.Devices.Count).ToArray();
        Solution? best = null;
        do
        {
            var candidate = SolveOrder(request, stages, order);
            if (candidate is not null && IsBetter(candidate.Score, best?.Score))
            {
                best = candidate;
            }
        }
        while (NextPermutation(order));

        return best is null ? null : CreatePlan(request, best);
    }

    private static Solution? SolveOrder(PlacementRequest request, StageCostCache stages, int[] order)
    {
        var layerCount = request.Layers.Count;
        var link = request.Link.LatencySeconds + (request.ActivationBytesPerToken / request.Link.BytesPerSecond);
        var previous = new Solution?[layerCount + 1];
        previous[0] = Solution.Empty;
        foreach (var device in order)
        {
            var current = new Solution?[layerCount + 1];
            for (var end = 0; end <= layerCount; end++)
            {
                current[end] = previous[end];
                for (var start = 0; start < end; start++)
                {
                    var stage = previous[start] is null ? null : stages.Get(device, start, end - 1);
                    if (stage is null)
                    {
                        continue;
                    }

                    var extended = previous[start]!.Extend(stage, start > 0 ? link : 0, request.Objective);
                    if (IsBetter(extended.Score, current[end]?.Score))
                    {
                        current[end] = extended;
                    }
                }
            }

            previous = current;
        }

        return previous[layerCount];
    }

    private static bool IsBetter(Score candidate, Score? incumbent)
    {
        if (incumbent is null)
        {
            return true;
        }

        var tolerance = RelativeTolerance * Math.Max(1, Math.Abs(incumbent.Value.Seconds));
        return candidate.Seconds < incumbent.Value.Seconds - tolerance ||
            (Math.Abs(candidate.Seconds - incumbent.Value.Seconds) <= tolerance &&
                candidate.Distortion < incumbent.Value.Distortion - RelativeTolerance);
    }

    private static PlacementPlan CreatePlan(PlacementRequest request, Solution solution)
    {
        var stages = solution.Stages
            .Select(stage => new PlacementStage(
                request.Devices[stage.Device].Id,
                stage.FirstLayer,
                stage.LastLayer,
                stage.Precision,
                stage.ReservedBytes,
                request.Devices[stage.Device].MemoryBudgetBytes,
                stage.Seconds))
            .ToArray();
        return new PlacementPlan(
            Array.AsReadOnly(stages),
            solution.Score.Seconds,
            solution.Score.Distortion,
            stages.Any(stage => stage.Precision.IsApproximate),
            request.Objective);
    }

    private static bool NextPermutation(int[] order)
    {
        var pivot = order.Length - 2;
        while (pivot >= 0 && order[pivot] >= order[pivot + 1])
        {
            pivot--;
        }

        if (pivot < 0)
        {
            return false;
        }

        var successor = order.Length - 1;
        while (order[successor] <= order[pivot])
        {
            successor--;
        }

        (order[pivot], order[successor]) = (order[successor], order[pivot]);
        Array.Reverse(order, pivot + 1, order.Length - pivot - 1);
        return true;
    }

    private static void Validate(PlacementRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(request.Layers);
        ArgumentNullException.ThrowIfNull(request.Devices);
        ArgumentNullException.ThrowIfNull(request.Link);
        if (request.Layers.Count == 0 || request.Layers.Where((layer, index) => layer is null || layer.Index != index || layer.KvBytesPerToken < 0).Any())
        {
            throw new ArgumentException("Layers must be non-empty, ordered by index from zero, with non-negative KV growth.");
        }

        if (request.Devices.Count is 0 or > MaximumDevices ||
            request.Devices.Select(device => device.Id).Distinct(StringComparer.Ordinal).Count() != request.Devices.Count ||
            request.Devices.Any(device => device.MemoryBudgetBytes <= 0 || !(device.WeightBytesPerSecond > 0) ||
                !double.IsFinite(device.WeightBytesPerSecond) || !(device.LayerOverheadSeconds >= 0)))
        {
            throw new ArgumentException($"Placement needs 1..{MaximumDevices} unique devices with positive memory and measured rates.");
        }

        if (!(request.Link.LatencySeconds >= 0) || !(request.Link.BytesPerSecond > 0) ||
            request.ActivationBytesPerToken < 0 || request.ContextTokens < 0)
        {
            throw new ArgumentException("Link, activation size, and context must be non-negative with a positive link rate.");
        }
    }

    private readonly record struct Score(double Seconds, double Distortion);

    private sealed record StageCost(
        int Device,
        int FirstLayer,
        int LastLayer,
        PrecisionPlan Precision,
        long ReservedBytes,
        double Seconds);

    private sealed record Solution(Score Score, IReadOnlyList<StageCost> Stages)
    {
        public static Solution Empty { get; } = new(new Score(0, 0), []);

        public Solution Extend(StageCost stage, double linkSeconds, PlacementObjective objective)
        {
            var seconds = objective == PlacementObjective.Latency
                ? Score.Seconds + stage.Seconds + linkSeconds
                : Math.Max(Score.Seconds, Math.Max(stage.Seconds, linkSeconds));
            return new Solution(
                new Score(seconds, Score.Distortion + stage.Precision.TotalDistortion),
                [.. Stages, stage]);
        }
    }

    private sealed class StageCostCache(PlacementRequest request)
    {
        private readonly Dictionary<(int Device, int First, int Last), StageCost?> _costs = [];

        public StageCost? Get(int device, int first, int last)
        {
            if (!_costs.TryGetValue((device, first, last), out var cost))
            {
                cost = Compute(device, first, last);
                _costs[(device, first, last)] = cost;
            }

            return cost;
        }

        private StageCost? Compute(int deviceIndex, int first, int last)
        {
            var device = request.Devices[deviceIndex];
            var layers = request.Layers.Skip(first).Take(last - first + 1).ToArray();
            var kvBytes = checked(layers.Sum(layer => layer.KvBytesPerToken) * request.ContextTokens);
            var weightBudget = device.MemoryBudgetBytes - kvBytes;
            if (weightBudget <= 0)
            {
                return null;
            }

            try
            {
                var precision = PrecisionBudgetSelector.Select(
                    [.. layers.SelectMany(layer => layer.Tensors)],
                    new PrecisionPolicy(weightBudget, request.AllowApproximation, device.SupportedEncodings));
                var seconds = (precision.TotalBytes / device.WeightBytesPerSecond) +
                    (layers.Length * device.LayerOverheadSeconds);
                return new StageCost(deviceIndex, first, last, precision, precision.TotalBytes + kvBytes, seconds);
            }
            catch (PrecisionBudgetException)
            {
                return null;
            }
        }
    }
}
