#include "mcts/params.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <mutex>

#include "chess/board.h"
#include "config.h"
#include "neural/encoder.h"
#include "neural/factory.h"
#include "neural/shared_params.h" // Already included
#include "utils/exception.h"
#include "utils/string.h"

// Conditional include for overrides (keep if present - it is in uwuplant)
#if __has_include("params_override.h")
#include "params_override.h"
#endif

#ifndef DEFAULT_MAX_PREFETCH
#define DEFAULT_MAX_PREFETCH 32
#endif
#ifndef DEFAULT_TASK_WORKERS
#define DEFAULT_TASK_WORKERS -1 // Use diff's default
#endif

namespace lczero {

namespace {

using namespace lczero::utils;

// Define constants for the OptionId objects.
const OptionId SearchParams::kMiniBatchSizeId{
    "minibatch-size", "MiniBatchSize",
    "Size of the minibatch used for NN inference."};
const OptionId SearchParams::kMaxPrefetchBatchId{ // Already exists
    "max-prefetch", "MaxPrefetch",
    "Maximum number of batches prefetched in the batching queue. Set to 0 to "
    "disable prefetching."};
const OptionId SearchParams::kRootHasOwnCpuctParamsId{
    "root-has-own-cpuct-params", "RootHasOwnCpuctParams",
    "If true, use dedicated root cpuct params. If false, use regular cpuct "
    "params."};
const OptionId SearchParams::kCpuctId{
    "cpuct", "CPuct",
    "cpuct constant from \"UCT search\" algorithm."};
const OptionId SearchParams::kCpuctAtRootId{
    "cpuct-at-root", "CPuctAtRoot",
    "cpuct constant from \"UCT search\" algorithm, for root node."};
const OptionId SearchParams::kCpuctExponentId{ // Already exists
    "cpuct-exponent", "CPuctExponent",
    "cpuct_exponent constant from \"UCT search\" algorithm."};
const OptionId SearchParams::kCpuctExponentAtRootId{ // Already exists
    "cpuct-exponent-at-root", "CPuctExponentAtRoot",
    "cpuct_exponent constant from \"UCT search\" algorithm, for root node."};
const OptionId SearchParams::kCpuctBaseId{
    "cpuct-base", "CPuctBase",
    "cpuct_base constant from \"UCT search\" algorithm."};
const OptionId SearchParams::kCpuctBaseAtRootId{
    "cpuct-base-at-root", "CPuctBaseAtRoot",
    "cpuct_base constant from \"UCT search\" algorithm, for root node."};
const OptionId SearchParams::kCpuctFactorId{
    "cpuct-factor", "CPuctFactor",
    "cpuct_factor constant from \"UCT search\" algorithm."};
const OptionId SearchParams::kCpuctFactorAtRootId{
    "cpuct-factor-at-root", "CPuctFactorAtRoot",
    "cpuct_factor constant from \"UCT search\" algorithm, for root node."};
const OptionId SearchParams::kUseUncertaintyWeightingId{
    "use-uncertainty-weighting", "UseUncertaintyWeighting",
    "Use uncertainty weighting to scale the cpuct."};
const OptionId SearchParams::kUncertaintyWeightingCoefficientId{
    "uncertainty-weighting-coefficient", "UncertaintyWeightingCoefficient",
    "Uncertainty weighting coefficient."};
const OptionId SearchParams::kUncertaintyWeightingExponentId{
    "uncertainty-weighting-exponent", "UncertaintyWeightingExponent",
    "Uncertainty weighting exponent."};
const OptionId SearchParams::kUncertaintyWeightingCapId{
    "uncertainty-weighting-cap", "UncertaintyWeightingCap",
    "Uncertainty weighting cap."};
const OptionId SearchParams::kMoveRuleBucketingId{
    "move-rule-bucketing", "MoveRuleBucketing",
    "Apply uncertainty weighting based on move rule type (captures, checks, promotions, etc)."};
const OptionId SearchParams::kTemperatureId{
    "temperature", "Temperature",
    "Temperature for the first N moves (see temperature-visit-offset)."};
const OptionId SearchParams::kTemperatureRootId{
    "temperature-root", "TemperatureRoot",
    "Temperature for the root node, overrides regular temperature. Set to 0 to disable."};
const OptionId SearchParams::kTemperatureColdId{
    "temperature-cold", "TemperatureCold",
    "Temperature applied after N moves (see temperature-visit-offset)."};
const OptionId SearchParams::kTemperatureWarmupScaleId{
    "temperature-warmup-scale", "TemperatureWarmupScale",
    "Scale factor applied to temperature based on visit count (0=no effect)."};
const OptionId SearchParams::kTemperatureVisitOffsetId{
    "temperature-visit-offset", "TemperatureVisitOffset",
    "Number of moves after which cold temperature is applied (halfmoves)."};
const OptionId SearchParams::kQvalueTempIsEnabledId{
    "qvalue-temp-enabled", "QvalueTempEnabled",
    "Enable temperature scaling of Q-values."};
const OptionId SearchParams::kQvalueZeroTempId{
    "qvalue-temp-zero", "QvalueTempZero",
    "Temperature for Q-values when the node value is close to 0."};
const OptionId SearchParams::kQvalueOneTempId{
    "qvalue-temp-one", "QvalueTempOne",
    "Temperature for Q-values when the node value is close to 1."};
// const OptionId SearchParams::kPolicyTemperatureId; // Defined in SharedParams
const OptionId SearchParams::kUsePolicyBoostingId{
    "use-policy-boosting", "UsePolicyBoosting",
    "Enable policy boosting for top N moves."};
const OptionId SearchParams::kTopPolicyBoostId{
    "top-policy-boost", "TopPolicyBoost",
    "Policy boost factor for the top N moves."};
const OptionId SearchParams::kTopPolicyNumBoostId{
    "top-policy-num-boost", "TopPolicyNumBoost",
    "Number of top moves to apply policy boost."};
const OptionId SearchParams::kTopPolicyTierTwoBoostId{
    "top-policy-tier-two-boost", "TopPolicyTierTwoBoost",
    "Policy boost factor for the tier two moves (after top N)."};
const OptionId SearchParams::kTopPolicyTierTwoNumBoostId{
    "top-policy-tier-two-num-boost", "TopPolicyTierTwoNumBoost",
    "Number of tier two moves to apply policy boost."};
const OptionId SearchParams::kDirichletAlphaId{
    "dirichlet-alpha", "DirichletAlpha",
    "Alpha parameter for dirichlet noise."};
const OptionId SearchParams::kNoiseEpsilonId{
    "noise-epsilon", "NoiseEpsilon",
    "Epsilon parameter for dirichlet noise."};
const OptionId SearchParams::kPbCInitId{
    "pbc-init", "PbCInit",
    "Initial value for the polynomial bound."};
const OptionId SearchParams::kPbCInitAtRootId{
    "pbc-init-at-root", "PbCInitAtRoot",
    "Initial value for the polynomial bound at the root node."};
const OptionId SearchParams::kPbCFactorId{
    "pbc-factor", "PbCFactor",
    "Factor for the polynomial bound."};
const OptionId SearchParams::kPbCFactorAtRootId{
    "pbc-factor-at-root", "PbCFactorAtRoot",
    "Factor for the polynomial bound at the root node."};
const OptionId SearchParams::kFpuStrategyId{
    "fpu-strategy", "FpuStrategy",
    "First Play Urgency (FPU) strategy: zero (0), parent (parent q), "
    "absolute (user defined value)."};
const OptionId SearchParams::kFpuStrategyAtRootId{
    "fpu-strategy-at-root", "FpuStrategyAtRoot",
    "FPU strategy for the root node."};
const OptionId SearchParams::kFpuValueId{
    "fpu-value", "FpuValue",
    "FPU value when using 'absolute' strategy."};
const OptionId SearchParams::kFpuValueAtRootId{
    "fpu-value-at-root", "FpuValueAtRoot",
    "FPU value for the root node when using 'absolute' strategy."};
const OptionId SearchParams::kCacheHistoryLengthId{
    "cache-history-length", "CacheHistoryLength",
    "Length of history, in half-moves, to include into the cache key."};
const OptionId SearchParams::kPolicyDecayExponentId{ // Exists in uwuplant
    "policy-decay-exponent", "PolicyDecayExponent",
    "Policy decay exponent. Sets the exponent of the visit based policy decay "
    "term."};
const OptionId SearchParams::kPolicyDecayFactorId{ // Exists in uwuplant
    "policy-decay-factor", "PolicyDecayFactor",
    "Policy decay factor. Scales the visit count for the visit based policy "
    "decay term."};
const OptionId SearchParams::kMaxCollisionEventsId{
    "max-collision-events", "MaxCollisionEvents",
    "Maximum number of hash collisions allowed before resizing the cache."};
const OptionId SearchParams::kMaxCollisionVisitsId{
    "max-collision-visits", "MaxCollisionVisits",
    "Maximum visits on a node before treating collisions as critical."};
const OptionId SearchParams::kUseCorrectionHistoryId{
    "use-correction-history", "UseCorrectionHistory",
    "Use correction history to adjust policy based on past performance."};
const OptionId SearchParams::kCorrectionHistoryAlphaId{
    "correction-history-alpha", "CorrectionHistoryAlpha",
    "Alpha parameter for correction history (learning rate)."};
const OptionId SearchParams::kCorrectionHistoryLambdaId{
    "correction-history-lambda", "CorrectionHistoryLambda",
    "Lambda parameter for correction history (decay rate)."};
const OptionId SearchParams::kUseDesperationId{
    "use-desperation", "UseDesperation",
    "Enable desperation logic to boost policies leading to complex positions "
    "when behind."};
const OptionId SearchParams::kDesperationLowId{
    "desperation-low", "DesperationLow",
    "Lower Q-value threshold to start applying desperation."};
const OptionId SearchParams::kDesperationHighId{
    "desperation-high", "DesperationHigh",
    "Upper Q-value threshold where desperation effect is maximized."};
const OptionId SearchParams::kDesperationMultiplierId{
    "desperation-multiplier", "DesperationMultiplier",
    "Maximum policy multiplier applied by desperation logic."};
const OptionId SearchParams::kDesperationPriorWeightId{
    "desperation-prior-weight", "DesperationPriorWeight",
    "Weight given to the prior policy when calculating desperation boost."};
const OptionId SearchParams::kScoreTypeId{
    "score-type", "ScoreType",
    "Score type to use for node evaluation: Q (raw Q), WDL_W (win %), WDL_L "
    "(loss %), WDL_mu (combined WDL)."};
// const OptionId SearchParams::kHistoryFillId; // Defined in SharedParams
const OptionId SearchParams::kMovesLeftMaxEffectId{
    "moves-left-max-effect", "MovesLeftMaxEffect",
    "Maximum effect of moves left scaling on Q-value."};
const OptionId SearchParams::kMovesLeftThresholdId{
    "moves-left-threshold", "MovesLeftThreshold",
    "Threshold (proportion of total moves) below which moves left scaling "
    "starts."};
const OptionId SearchParams::kMovesLeftLinearFactorId{
    "moves-left-linear-factor", "MovesLeftLinearFactor",
    "Linear factor for moves left scaling."};
const OptionId SearchParams::kMovesLeftScaledFactorId{
    "moves-left-scaled-factor", "MovesLeftScaledFactor",
    "Scaled factor for moves left scaling."};
const OptionId SearchParams::kMovesLeftQuadraticFactorId{
    "moves-left-quadratic-factor", "MovesLeftQuadraticFactor",
    "Quadratic factor for moves left scaling."};
const OptionId SearchParams::kDisplayCacheUsageId{
    "display-cache-usage", "DisplayCacheUsage",
    "Display cache usage statistics periodically."};
const OptionId SearchParams::kMaxConcurrentSearchersId{
    "max-concurrent-searchers", "MaxConcurrentSearchers",
    "Maximum number of searchers allowed to run concurrently."};
const OptionId SearchParams::kDrawScoreId{
    "draw-score", "DrawScore",
    "Score assigned to draw positions (used for WDL rescaling)."};
const OptionId SearchParams::kContemptModeId{
    "contempt-mode", "ContemptMode",
    "Contempt mode: off, white_side_analysis, black_side_analysis, play."};
const OptionId SearchParams::kContemptId{
    "contempt", "Contempt",
    "Contempt value (positive favors avoiding draws). Can be a list based on "
    "opponent."};
const OptionId SearchParams::kContemptMaxValueId{
    "contempt-max-value", "ContemptMaxValue",
    "Maximum absolute value contempt can reach after Elo calibration."};
const OptionId SearchParams::kWDLCalibrationEloId{
    "wdl-calibration-elo", "WDLCalibrationElo",
    "Elo rating used for WDL calibration (adjusts contempt based on rating "
    "difference). 0 disables."};
const OptionId SearchParams::kWDLContemptAttenuationId{
    "wdl-contempt-attenuation", "WDLContemptAttenuation",
    "Factor controlling how quickly contempt attenuates with Elo difference."};
const OptionId SearchParams::kWDLDrawRateTargetId{
    "wdl-draw-rate-target", "WDLDrawRateTarget",
    "Target draw rate used for WDL calibration."};
const OptionId SearchParams::kWDLBookExitBiasId{
    "wdl-book-exit-bias", "WDLBookExitBias",
    "Bias added to WDL score when considering leaving the opening book."};
const OptionId SearchParams::kWDLRescaleRatioId{
    "wdl-rescale-ratio", "WDLRescaleRatio",
    "Ratio used for rescaling WDL values (internal, usually not set by "
    "user)."};
const OptionId SearchParams::kWDLRescaleDiffId{
    "wdl-rescale-diff", "WDLRescaleDiff",
    "Difference used for rescaling WDL values (internal, usually not set by "
    "user)."};
const OptionId SearchParams::kWDLMaxSId{ // Added Definition
    "wdl-max-s", "WDLMaxS",
    "Limits the WDL derived sharpness s to a reasonable value to avoid "
    "erratic behavior at high contempt values. Default recommended for "
    "regular chess, increase value for more volatile positions like DFRC "
    "or piece odds."};
const OptionId SearchParams::kWDLEvalObjectivityId{
    "wdl-eval-objectivity", "WDLEvalObjectivity",
    "When calculating the centipawn eval output, decides how objective/"
    "subjective the eval should be (0=fully subjective, 1=fully objective)."};
const OptionId SearchParams::kMaxOutOfOrderEvalsFactorId{ // Renamed Definition
    "max-out-of-order-evals-factor", "MaxOutOfOrderEvalsFactor",
    "Maximum number of NN evaluations allowed to be processed out of order, "
    "as a factor of batch size. Can improve performance but might affect search "
    "accuracy. Zero to disable."};
const OptionId SearchParams::kNpsLimitId{
    "nps-limit", "NpsLimit",
    "Limit the search speed to approximately this many nodes per second. 0 "
    "disables."};
const OptionId SearchParams::kSolidTreeThresholdId{ // Already exists
    "solid-tree-threshold", "SolidTreeThreshold",
    "Only nodes with at least this number of visits will be considered for "
    "solidification for improved cache locality."};
const OptionId SearchParams::kTaskWorkersPerSearchWorkerId{
    "task-workers", "TaskWorkers",
    "The number of task workers to use to help the search worker. Setting to "
    "-1 lets the engine choose heuristically."};
const OptionId SearchParams::kMinimumWorkSizeForProcessingId{
    "minimum-work-size-for-processing", "MinimumWorkSizeForProcessing",
    "Minimum number of tasks required before processing starts."};
const OptionId SearchParams::kMinimumWorkSizeForPickingId{
    "minimum-work-size-for-picking", "MinimumWorkSizeForPicking",
    "Minimum number of tasks required before picking work."};
const OptionId SearchParams::kMinimumRemainingWorkSizeForPickingId{
    "minimum-remaining-work-size-for-picking",
    "MinimumRemainingWorkSizeForPicking",
    "Minimum remaining tasks after picking work."};
const OptionId SearchParams::kMinimumWorkPerTaskForProcessingId{
    "minimum-work-per-task-for-processing",
    "MinimumWorkPerTaskForProcessing",
    "Minimum work items per task required for processing."};
const OptionId SearchParams::kMaxCollisionVisitsScalingStartId{
    "max-collision-visits-scaling-start", "MaxCollisionVisitsScalingStart",
    "Visit count where scaling of max collision visits begins."};
const OptionId SearchParams::kMaxCollisionVisitsScalingEndId{
    "max-collision-visits-scaling-end", "MaxCollisionVisitsScalingEnd",
    "Visit count where scaling of max collision visits ends."};
const OptionId SearchParams::kMaxCollisionVisitsScalingPowerId{
    "max-collision-visits-scaling-power", "MaxCollisionVisitsScalingPower",
    "Power factor for scaling max collision visits."};
const OptionId SearchParams::kThreadIdlingThresholdId{
    "thread-idling-threshold", "ThreadIdlingThreshold",
    "Number of idle threads allowed before reducing concurrency."};
const OptionId SearchParams::kUCIOpponentId{
    "uci-opponent", "UCIOpponent",
    "Information about the UCI opponent (e.g., name, rating)."};
const OptionId SearchParams::kUCIRatingAdvId{
    "uci-rating-adv", "UCIRatingAdv",
    "Rating advantage over the UCI opponent."};
const OptionId SearchParams::kSearchSpinBackoffId{ // Already exists
    "search-spin-backoff", "SearchSpinBackoff",
    "Enable backoff for the spin lock that acquires available searcher."};

// --- Root Beam Search ADDED ---
const OptionId SearchParams::kRootBeamMinWidthId{ // Added Definition
    "root-beam-min-width", "RootBeamMinWidth",
    "Minimum beam width when using dynamic width (based on score gap). Set to 0 or >= MaxWidth to disable dynamic width."};
const OptionId SearchParams::kRootBeamMaxWidthId{ // Added Definition
    "root-beam-max-width", "RootBeamMaxWidth",
    "Maximum beam width (or fixed width if MinWidth=0/disabled). 0 disables beam."};
const OptionId SearchParams::kRootBeamUpdateThresholdId{ // Added Definition
    "root-beam-update-threshold", "RootBeamUpdateThreshold",
    "Number of root visits after which the root beam is calculated and activated."};
const OptionId SearchParams::kRootBeamUpdateIntervalFactorId{ // Added Definition (renamed)
    "root-beam-update-interval-factor", "RootBeamUpdateIntervalFactor",
    "Geometric factor to increase update interval (>1.0 enables geometric). 1.0 means fixed interval."};
// --- END Root Beam Search ADDED ---

// --- Variance Scaling IDs (Already present in uwuplant/lc0) ---
const OptionId SearchParams::kCpuctUtilityStdevPriorId{
    "cpuct-utility-stdev-prior", "CpuctUtilityStdevPrior",
    "Prior standard deviation for utility."};
const OptionId SearchParams::kCpuctUtilityStdevScaleId{
    "cpuct-utility-stdev-scale", "CpuctUtilityStdevScale",
    "Scale factor for standard deviation based on policy."};
const OptionId SearchParams::kCpuctUtilityStdevPriorWeightId{
    "cpuct-utility-stdev-prior-weight", "CpuctUtilityStdevPriorWeight",
    "Weight given to the prior standard deviation."};
const OptionId SearchParams::kUseVarianceScalingId{
    "use-variance-scaling", "UseVarianceScaling",
    "Enable variance scaling for PUCT."};

// --- Uncertainty IDs (Already present in uwuplant/lc0) ---
const OptionId SearchParams::kCpuctUncertaintyMinFactorId{
    "cpuct-uncertainty-min-factor", "CpuctUncertaintyMinFactor",
    "Minimum PUCT factor applied based on uncertainty."};
const OptionId SearchParams::kCpuctUncertaintyMaxFactorId{
    "cpuct-uncertainty-max-factor", "CpuctUncertaintyMaxFactor",
    "Maximum PUCT factor applied based on uncertainty."};
const OptionId SearchParams::kCpuctUncertaintyMinUncertaintyId{
    "cpuct-uncertainty-min-uncertainty", "CpuctUncertaintyMinUncertainty",
    "Minimum uncertainty value considered for scaling."};
const OptionId SearchParams::kCpuctUncertaintyMaxUncertaintyId{
    "cpuct-uncertainty-max-uncertainty", "CpuctUncertaintyMaxUncertainty",
    "Maximum uncertainty value considered for scaling."};
const OptionId SearchParams::kUseCpuctUncertaintyId{
    "use-cpuct-uncertainty", "UseCpuctUncertainty",
    "Enable PUCT scaling based on NN uncertainty."};
const OptionId SearchParams::kJustFpuUncertaintyId{
    "just-fpu-uncertainty", "JustFpuUncertainty",
    "Apply uncertainty scaling only during FPU evaluation."};

// --- EasyEval IDs (Already present in uwuplant/lc0) ---
const OptionId SearchParams::kEasyEvalWeightDecayId{
    "easy-eval-weight-decay", "EasyEvalWeightDecay",
    "Decay factor for the weight given to easy evaluations."};
const OptionId SearchParams::kEasyEvalValueThresholdId{
    "easy-eval-value-threshold", "EasyEvalValueThreshold",
    "Q-value threshold below which evaluations are considered 'easy'."};

} // namespace

void SearchParams::PopulateOptions(OptionsParser* options) {
  // Add options for shared parameters first.
  SharedBackendParams::PopulateOptions(options);

  // Add options specific to search.
  options->Add<IntOption>(kMiniBatchSizeId, 1, 1024) = 16; // uwuplant default
  options->Add<IntOption>(kMaxPrefetchBatchId, 0, 1024) = DEFAULT_MAX_PREFETCH;
  options->Add<BoolOption>(kRootHasOwnCpuctParamsId) = true;
  options->Add<FloatOption>(kCpuctId, 0.0f, 100.0f) = 1.7f; // uwuplant default
  options->Add<FloatOption>(kCpuctAtRootId, 0.0f, 100.0f) = 1.7f; // uwuplant default
  options->Add<FloatOption>(kCpuctExponentId, 0.0f, 1.0f) = 0.5f; // Added/Match diff
  options->Add<FloatOption>(kCpuctExponentAtRootId, 0.0f, 1.0f) = 0.5f; // Added/Match diff
  options->Add<FloatOption>(kCpuctBaseId, 1.0f, 1000000000.0f) = 39000.0f; // uwuplant default
  options->Add<FloatOption>(kCpuctBaseAtRootId, 1.0f, 1000000000.0f) = 39000.0f; // uwuplant default
  options->Add<FloatOption>(kCpuctFactorId, 0.0f, 1000.0f) = 3.9f; // uwuplant default
  options->Add<FloatOption>(kCpuctFactorAtRootId, 0.0f, 1000.0f) = 3.9f; // uwuplant default
  options->Add<BoolOption>(kUseUncertaintyWeightingId) = true;
  options->Add<FloatOption>(kUncertaintyWeightingCoefficientId, 0.0f, 100.0f) =
      0.5f;
  options->Add<FloatOption>(kUncertaintyWeightingExponentId, 0.0f, 10.0f) = 1.0f;
  options->Add<FloatOption>(kUncertaintyWeightingCapId, 0.0f, 100.0f) = 1.5f;
  options->Add<BoolOption>(kMoveRuleBucketingId) = false;
  options->Add<FloatOption>(kTemperatureId, 0.0f, 10.0f) = 1.0f;
  options->Add<FloatOption>(kTemperatureRootId, 0.0f, 10.0f) = 0.0f; // Default disable
  options->Add<FloatOption>(kTemperatureColdId, 0.0f, 10.0f) = 0.0f;
  options->Add<FloatOption>(kTemperatureWarmupScaleId, 0.0f, 1.0f) = 0.0f;
  options->Add<IntOption>(kTemperatureVisitOffsetId, 0, 100) = 30;
  options->Add<BoolOption>(kQvalueTempIsEnabledId) = false;
  options->Add<FloatOption>(kQvalueZeroTempId, 0.01f, 10.0f) = 1.0f;
  options->Add<FloatOption>(kQvalueOneTempId, 0.01f, 10.0f) = 1.0f;
  // kPolicyTemperatureId populated by SharedBackendParams::Populate
  options->Add<BoolOption>(kUsePolicyBoostingId) = true; // uwuplant default
  options->Add<FloatOption>(kTopPolicyBoostId, 0.0f, 100.0f) = 0.25f; // uwuplant default
  options->Add<IntOption>(kTopPolicyNumBoostId, 0, 20) = 3; // uwuplant default
  options->Add<FloatOption>(kTopPolicyTierTwoBoostId, 0.0f, 100.0f) = 0.1f; // uwuplant default
  options->Add<IntOption>(kTopPolicyTierTwoNumBoostId, 0, 20) = 10; // uwuplant default
  options->Add<FloatOption>(kDirichletAlphaId, 0.0f, 10.0f) = 0.3f;
  options->Add<FloatOption>(kNoiseEpsilonId, 0.0f, 1.0f) = 0.25f;
  options->Add<FloatOption>(kPbCInitId, 0.0f, 10.0f) = 1.25f;
  options->Add<FloatOption>(kPbCInitAtRootId, 0.0f, 10.0f) = 1.25f;
  options->Add<FloatOption>(kPbCFactorId, 0.0f, 10.0f) = 0.0f;
  options->Add<FloatOption>(kPbCFactorAtRootId, 0.0f, 10.0f) = 0.0f;
  std::vector<std::string> fpu_strategy = {"zero", "parent", "absolute"};
  options->Add<ChoiceOption>(kFpuStrategyId, fpu_strategy) = "parent";
  options->Add<ChoiceOption>(kFpuStrategyAtRootId, fpu_strategy) = "parent";
  options->Add<FloatOption>(kFpuValueId, -100.0f, 100.0f) = 1.0f;
  options->Add<FloatOption>(kFpuValueAtRootId, -100.0f, 100.0f) = 1.0f;
  options->Add<IntOption>(kCacheHistoryLengthId, 0, 7) = 0;
  options->Add<FloatOption>(kPolicyDecayExponentId, 0.0f, 10.0f) = 0.5f; // Exists in uwuplant
  options->Add<FloatOption>(kPolicyDecayFactorId, 0.0f, 1.0f) = 0.0001f; // Exists in uwuplant
  options->Add<IntOption>(kMaxCollisionEventsId, 1, 65536) = 917;
  options->Add<IntOption>(kMaxCollisionVisitsId, 1, 100000000) = 80000;
  options->Add<BoolOption>(kUseCorrectionHistoryId) = false;
  options->Add<FloatOption>(kCorrectionHistoryAlphaId, 0, 1) = 0.5;
  options->Add<FloatOption>(kCorrectionHistoryLambdaId, 0, 1) = 0.3;
  options->Add<BoolOption>(kUseDesperationId) = false;
  options->Add<FloatOption>(kDesperationLowId, -1, 1) = 0.3;
  options->Add<FloatOption>(kDesperationHighId, -1, 1) = 0.7;
  options->Add<FloatOption>(kDesperationMultiplierId, 1, 100) = 5.0;
  options->Add<FloatOption>(kDesperationPriorWeightId, 0, 1) = 0.1;
  std::vector<std::string> score_type = {"Q", "WDL_W", "WDL_L", "WDL_mu"};
  options->Add<ChoiceOption>(kScoreTypeId, score_type) = "WDL_mu";
  // kHistoryFillId populated by SharedBackendParams::Populate
  options->Add<FloatOption>(kMovesLeftMaxEffectId, 0.0f, 1.0f) = 0.0345f;
  options->Add<FloatOption>(kMovesLeftThresholdId, 0.0f, 1.0f) = 0.8f;
  options->Add<FloatOption>(kMovesLeftLinearFactorId, -2.0f, 2.0f) = 0.0f;
  options->Add<FloatOption>(kMovesLeftScaledFactorId, -2.0f, 2.0f) = 1.6521f;
  options->Add<FloatOption>(kMovesLeftQuadraticFactorId, -1.0f, 1.0f) =
      -0.6521f;
  options->Add<BoolOption>(kDisplayCacheUsageId) = false; // Keep Add line
  options->Add<IntOption>(kMaxConcurrentSearchersId, 0, 128) = 1;
  options->Add<FloatOption>(kDrawScoreId, -1.0f, 1.0f) = 0.0f;
  std::vector<std::string> mode = {"play", "white_side_analysis",
                                   "black_side_analysis", "off"};
  options->Add<ChoiceOption>(kContemptModeId, mode) = "play";
  // Don't add default value for contempt. If UCI doesn't send it, the empty
  // separated kContemptId list will override this.
  options->Add<StringOption>(kContemptId) = "";
  options->Add<FloatOption>(kContemptMaxValueId, 0, 10000.0f) = 420.0f;
  options->Add<FloatOption>(kWDLCalibrationEloId, 0, 10000.0f) = 0.0f; // Default 0 disables
  options->Add<FloatOption>(kWDLContemptAttenuationId, -10.0f, 10.0f) = 1.0f;
  options->Add<FloatOption>(kWDLDrawRateTargetId, 0.001f, 0.999f) = 0.5f;
  options->Add<FloatOption>(kWDLBookExitBiasId, -2.0f, 2.0f) = 0.65f;
  options->Add<FloatOption>(kWDLRescaleRatioId, 0.0f, 10.0f) = 1.0f;
  options->Add<FloatOption>(kWDLRescaleDiffId, -1.0f, 1.0f) = 0.0f;
  options->Add<FloatOption>(kWDLMaxSId, 0.0f, 10.0f) = 1.4f; // Added option
  options->Add<FloatOption>(kWDLEvalObjectivityId, 0.0f, 1.0f) = 1.0f;
  options->Add<FloatOption>(kMaxOutOfOrderEvalsFactorId, 0.0f, 16.0f) = 0.0f; // Renamed option
  options->Add<FloatOption>(kNpsLimitId, 0.0f, 1e6f) = 0.0f;
  options->Add<IntOption>(kSolidTreeThresholdId, 1, 2000000000) = 100; // Exists
  options->Add<IntOption>(kTaskWorkersPerSearchWorkerId, -1, 128) = DEFAULT_TASK_WORKERS;
  options->Add<IntOption>(kMinimumWorkSizeForProcessingId, 2, 100000) = 20;
  options->Add<IntOption>(kMinimumWorkSizeForPickingId, 1, 100000) = 1;
  options->Add<IntOption>(kMinimumRemainingWorkSizeForPickingId, 1, 100000) = 1;
  options->Add<IntOption>(kMinimumWorkPerTaskForProcessingId, 1, 100000) = 1;
  options->Add<IntOption>(kMaxCollisionVisitsScalingStartId, 0, 100000000) = 0;
  options->Add<IntOption>(kMaxCollisionVisitsScalingEndId, 0, 100000000) = 0;
  options->Add<FloatOption>(kMaxCollisionVisitsScalingPowerId, 0.0f, 10.0f) =
      1.0f;
  options->Add<IntOption>(kThreadIdlingThresholdId, 0, 128) = 1;
  options->Add<StringOption>(kUCIOpponentId);
  options->Add<FloatOption>(kUCIRatingAdvId, -10000.0f, 10000.0f) = 0.0f;
  options->Add<BoolOption>(kSearchSpinBackoffId) = false; // Exists

  // --- Root Beam Search ADDED ---
  options->Add<IntOption>(kRootBeamMinWidthId, 0, 500) = 0; // Dynamic width disabled
  options->Add<IntOption>(kRootBeamMaxWidthId, 0, 500) = 0; // Beam disabled
  options->Add<IntOption>(kRootBeamUpdateThresholdId, 0, 1000000) = 100;
  options->Add<FloatOption>(kRootBeamUpdateIntervalFactorId, 1.0f, 10.0f) = 1.0f; // Default 1.0 (fixed interval)
  // --- END Root Beam Search ADDED ---

  // --- Variance Scaling options (Already present in uwuplant/lc0) ---
  options->Add<FloatOption>(kCpuctUtilityStdevPriorId, 0.0f, 10.0f) = 0.0f;
  options->Add<FloatOption>(kCpuctUtilityStdevScaleId, 0.0f, 10.0f) = 0.0f;
  options->Add<FloatOption>(kCpuctUtilityStdevPriorWeightId, 0.0f, 1.0f) = 0.0f;
  options->Add<BoolOption>(kUseVarianceScalingId) = false;

  // --- Uncertainty options (Already present in uwuplant/lc0) ---
  options->Add<FloatOption>(kCpuctUncertaintyMinFactorId, 0.0f, 1.0f) = 0.0f;
  options->Add<FloatOption>(kCpuctUncertaintyMaxFactorId, 1.0f, 100.0f) = 1.0f;
  options->Add<FloatOption>(kCpuctUncertaintyMinUncertaintyId, 0.0f, 1.0f) =
      0.0f;
  options->Add<FloatOption>(kCpuctUncertaintyMaxUncertaintyId, 0.0f, 1.0f) =
      1.0f;
  options->Add<BoolOption>(kUseCpuctUncertaintyId) = false;
  options->Add<BoolOption>(kJustFpuUncertaintyId) = false;

  // --- EasyEval options (Already present in uwuplant/lc0) ---
  options->Add<FloatOption>(kEasyEvalWeightDecayId, 0.0f, 1.0f) = 0.0f;
  options->Add<FloatOption>(kEasyEvalValueThresholdId, 0.0f, 1.0f) = 0.0f;


  options->HideOption(kNoiseEpsilonId);
  options->HideOption(kDirichletAlphaId);
  options->HideOption(kPbCInitId);
  options->HideOption(kPbCFactorId);
  options->HideOption(kPbCInitAtRootId);
  options->HideOption(kPbCFactorAtRootId);
  options->HideOption(kTemperatureId);
  options->HideOption(kTemperatureColdId);
  options->HideOption(kTemperatureVisitOffsetId);
  options->HideOption(kContemptMaxValueId);
  options->HideOption(kWDLContemptAttenuationId);
  options->HideOption(kWDLMaxSId); // Added hide
  options->HideOption(kWDLDrawRateTargetId); // Already hidden
  options->HideOption(kWDLBookExitBiasId);
  options->HideOption(kWDLRescaleRatioId);
  options->HideOption(kWDLRescaleDiffId);
  // --- Hide Beam options if desired ---
  // options->HideOption(kRootBeamMinWidthId);
  // options->HideOption(kRootBeamMaxWidthId);
  // options->HideOption(kRootBeamUpdateThresholdId);
  // options->HideOption(kRootBeamUpdateIntervalFactorId);
}

SearchParams::SearchParams(const OptionsDict& options)
    : // Initialize shared parameters first.
      kMiniBatchSize(options.Get<int>(kMiniBatchSizeId)),
      kMaxPrefetch(options.Get<int>(kMaxPrefetchBatchId)),
      kRootHasOwnCpuctParams(options.Get<bool>(kRootHasOwnCpuctParamsId)),
      kCpuct(options.Get<float>(kCpuctId)),
      kCpuctAtRoot(options.Get<float>(
          options.Get<bool>(kRootHasOwnCpuctParamsId) ? kCpuctAtRootId
                                                      : kCpuctId)),
      kCpuctExponent(options.Get<float>(kCpuctExponentId)), // Match diff
      kCpuctExponentAtRoot(options.Get<float>(
          options.Get<bool>(kRootHasOwnCpuctParamsId) ? kCpuctExponentAtRootId // Match diff
                                                      : kCpuctExponentId)), // Match diff
      kCpuctBase(options.Get<float>(kCpuctBaseId)),
      kCpuctBaseAtRoot(options.Get<float>(
          options.Get<bool>(kRootHasOwnCpuctParamsId) ? kCpuctBaseAtRootId
                                                      : kCpuctBaseId)),
      kCpuctFactor(options.Get<float>(kCpuctFactorId)),
      kCpuctFactorAtRoot(options.Get<float>(
          options.Get<bool>(kRootHasOwnCpuctParamsId) ? kCpuctFactorAtRootId
                                                      : kCpuctFactorId)),
      kUseUncertaintyWeighting(options.Get<bool>(kUseUncertaintyWeightingId)),
      kUncertaintyWeightingCoefficient(
          options.Get<float>(kUncertaintyWeightingCoefficientId)),
      kUncertaintyWeightingExponent(
          options.Get<float>(kUncertaintyWeightingExponentId)),
      kUncertaintyWeightingCap(options.Get<float>(kUncertaintyWeightingCapId)),
      kMoveRuleBucketing(options.Get<bool>(kMoveRuleBucketingId)),
      kTemperature(options.Get<float>(kTemperatureId)),
      kTemperatureRoot(options.Get<float>(kTemperatureRootId)),
      kTemperatureCold(options.Get<float>(kTemperatureColdId)),
      kTemperatureWarmupScale(options.Get<float>(kTemperatureWarmupScaleId)),
      kTemperatureVisitOffset(options.Get<int>(kTemperatureVisitOffsetId)),
      kQvalueTempIsEnabled(options.Get<bool>(kQvalueTempIsEnabledId)),
      kQvalueZeroTemp(options.Get<float>(kQvalueZeroTempId)),
      kQvalueOneTemp(options.Get<float>(kQvalueOneTempId)),
      kPolicyTemperature(options.Get<float>(kPolicyTemperatureId)),
      kUsePolicyBoosting(options.Get<bool>(kUsePolicyBoostingId)),
      kTopPolicyBoost(options.Get<float>(kTopPolicyBoostId)),
      kTopPolicyNumBoost(options.Get<int>(kTopPolicyNumBoostId)),
      kTopPolicyTierTwoBoost(options.Get<float>(kTopPolicyTierTwoBoostId)),
      kTopPolicyTierTwoNumBoost(options.Get<int>(kTopPolicyTierTwoNumBoostId)),
      kDirichletAlpha(options.Get<float>(kDirichletAlphaId)),
      kNoiseEpsilon(options.Get<float>(kNoiseEpsilonId)),
      kPbCInit(options.Get<float>(kPbCInitId)),
      kPbCInitAtRoot(options.Get<float>(
          options.Get<bool>(kRootHasOwnCpuctParamsId) ? kPbCInitAtRootId
                                                      : kPbCInitId)),
      kPbCFactor(options.Get<float>(kPbCFactorId)),
      kPbCFactorAtRoot(options.Get<float>(
          options.Get<bool>(kRootHasOwnCpuctParamsId) ? kPbCFactorAtRootId
                                                      : kPbCFactorId)),
      kFpuStrategy(
          options.GetEnum<FpuStrategy>(kFpuStrategyId, {{"zero", FpuStrategy::ZERO},
                                       {"parent", FpuStrategy::PARENT},
                                       {"absolute", FpuStrategy::ABSOLUTE}})),
      kFpuStrategyAtRoot(
          options.GetEnum<FpuStrategy>(kFpuStrategyAtRootId,
                                       {{"zero", FpuStrategy::ZERO},
                                        {"parent", FpuStrategy::PARENT},
                                        {"absolute", FpuStrategy::ABSOLUTE}})),
      kFpuValue(options.Get<float>(kFpuValueId)),
      kFpuValueAtRoot(options.Get<float>(kFpuValueAtRootId)),
      kCacheHistoryLength(options.Get<int>(kCacheHistoryLengthId)),
      kPolicyDecayExponent(options.Get<float>(kPolicyDecayExponentId)), // Exists
      kPolicyDecayFactor(options.Get<float>(kPolicyDecayFactorId)), // Exists
      kMaxCollisionEvents(options.Get<int>(kMaxCollisionEventsId)),
      kMaxCollisionVisits(options.Get<int>(kMaxCollisionVisitsId)),
      kUseCorrectionHistory(options.Get<bool>(kUseCorrectionHistoryId)),
      kCorrectionHistoryAlpha(options.Get<float>(kCorrectionHistoryAlphaId)),
      kCorrectionHistoryLambda(options.Get<float>(kCorrectionHistoryLambdaId)),
      kUseDesperation(options.Get<bool>(kUseDesperationId)),
      kDesperationLow(options.Get<float>(kDesperationLowId)),
      kDesperationHigh(options.Get<float>(kDesperationHighId)),
      kDesperationMultiplier(options.Get<float>(kDesperationMultiplierId)),
      kDesperationPriorWeight(options.Get<float>(kDesperationPriorWeightId)),
      kScoreType(options.GetEnum<ScoreType>(
          kScoreTypeId, {{"Q", ScoreType::Q},
                         {"WDL_W", ScoreType::WDL_W},
                         {"WDL_L", ScoreType::WDL_L},
                         {"WDL_mu", ScoreType::WDL_MU}})),
      kHistoryFill(options.GetEnum<HistoryFill>(
          kHistoryFillId, {{"no", HistoryFill::NO},
                           {"fen_only", HistoryFill::FEN_ONLY},
                           {"always", HistoryFill::ALWAYS}})),
      kMovesLeftMaxEffect(options.Get<float>(kMovesLeftMaxEffectId)),
      kMovesLeftThreshold(options.Get<float>(kMovesLeftThresholdId)),
      kMovesLeftLinearFactor(options.Get<float>(kMovesLeftLinearFactorId)),
      kMovesLeftScaledFactor(options.Get<float>(kMovesLeftScaledFactorId)),
      kMovesLeftQuadraticFactor(
          options.Get<float>(kMovesLeftQuadraticFactorId)),
      kDisplayCacheUsage(options.Get<bool>(kDisplayCacheUsageId)),
      kMaxConcurrentSearchers(options.Get<int>(kMaxConcurrentSearchersId)),
      kDrawScore(options.Get<float>(kDrawScoreId)),
      kContemptMode(options.GetEnum<ContemptMode>(
          kContemptModeId,
          {{"off", ContemptMode::OFF},
           {"white_side_analysis", ContemptMode::WHITE_SIDE_ANALYSIS},
           {"black_side_analysis", ContemptMode::BLACK_SIDE_ANALYSIS},
           {"play", ContemptMode::PLAY}})),
      kContempt(GetContempt(options.Get<std::string>(kUCIOpponentId),
                            options.Get<std::string>(kContemptId),
                            options.Get<float>(kUCIRatingAdvId), kContemptMode)),
      kWDLRescaleParams(CalculateWDLRescaleParams(
          kDrawScore, kContempt, options.Get<float>(kWDLCalibrationEloId),
          options.Get<float>(kContemptMaxValueId),
          options.Get<float>(kWDLContemptAttenuationId))),
      kWDLMaxS(options.Get<float>(kWDLMaxSId)), // Added init
      kWDLEvalObjectivity(options.Get<float>(kWDLEvalObjectivityId)),
      kMaxOutOfOrderEvalsFactor(options.Get<float>(kMaxOutOfOrderEvalsFactorId)), // Renamed & type changed
      kNpsLimit(options.Get<float>(kNpsLimitId)),
      kSolidTreeThreshold(options.Get<int>(kSolidTreeThresholdId)), // Exists
      kTaskWorkersPerSearchWorker(
          options.Get<int>(kTaskWorkersPerSearchWorkerId)),
      kMinimumWorkSizeForProcessing(
          options.Get<int>(kMinimumWorkSizeForProcessingId)),
      kMinimumWorkSizeForPicking(
          options.Get<int>(kMinimumWorkSizeForPickingId)),
      kMinimumRemainingWorkSizeForPicking(
          options.Get<int>(kMinimumRemainingWorkSizeForPickingId)),
      kMinimumWorkPerTaskForProcessing(
          options.Get<int>(kMinimumWorkPerTaskForProcessingId)),
      kMaxCollisionVisitsScalingStart(
          options.Get<int>(kMaxCollisionVisitsScalingStartId)),
      kMaxCollisionVisitsScalingEnd(
          options.Get<int>(kMaxCollisionVisitsScalingEndId)),
      kMaxCollisionVisitsScalingPower(
          options.Get<float>(kMaxCollisionVisitsScalingPowerId)),
      kThreadIdlingThreshold(options.Get<int>(kThreadIdlingThresholdId)),
      kUciOpponent(options.Get<std::string>(kUCIOpponentId)),
      kUciRatingAdv(options.Get<float>(kUCIRatingAdvId)),
      kSearchSpinBackoff(options.Get<bool>(kSearchSpinBackoffId)), // Exists

      // --- Root Beam Search ADDED ---
      kRootBeamMinWidth(options.Get<int>(kRootBeamMinWidthId)),
      kRootBeamMaxWidth(options.Get<int>(kRootBeamMaxWidthId)),
      kRootBeamUpdateThreshold(options.Get<int>(kRootBeamUpdateThresholdId)),
      kRootBeamUpdateIntervalFactor(options.Get<float>(kRootBeamUpdateIntervalFactorId)), // Changed type
      // --- END Root Beam Search ADDED ---

      // --- Variance Scaling initializers (Already present in uwuplant/lc0) ---
      kCpuctUtilityStdevPrior(options.Get<float>(kCpuctUtilityStdevPriorId)),
      kCpuctUtilityStdevScale(options.Get<float>(kCpuctUtilityStdevScaleId)),
      kCpuctUtilityStdevPriorWeight(
          options.Get<float>(kCpuctUtilityStdevPriorWeightId)),
      kUseVarianceScaling(options.Get<bool>(kUseVarianceScalingId)),

      // --- Uncertainty initializers (Already present in uwuplant/lc0) ---
      kCpuctUncertaintyMinFactor(options.Get<float>(kCpuctUncertaintyMinFactorId)),
      kCpuctUncertaintyMaxFactor(options.Get<float>(kCpuctUncertaintyMaxFactorId)),
      kCpuctUncertaintyMinUncertainty(
          options.Get<float>(kCpuctUncertaintyMinUncertaintyId)),
      kCpuctUncertaintyMaxUncertainty(
          options.Get<float>(kCpuctUncertaintyMaxUncertaintyId)),
      kUseCpuctUncertainty(options.Get<bool>(kUseCpuctUncertaintyId)),
      kJustFpuUncertainty(options.Get<bool>(kJustFpuUncertaintyId)),

      // --- EasyEval initializers (Already present in uwuplant/lc0) ---
      kEasyEvalWeightDecay(options.Get<float>(kEasyEvalWeightDecayId)),
      kEasyEvalValueThreshold(options.Get<float>(kEasyEvalValueThresholdId))

  {
    // --- Added calculation for kMaxOutOfOrderEvals ---
    // Use the effective batch size from SharedBackendParams, default if not found.
    // TODO: Ideally, SearchParams should have direct access to SharedBackendParams
    // or this calculation should happen elsewhere. For now, we re-fetch.
    int effective_batch_size = kMiniBatchSize;
    if (options.Exists(SharedBackendParams::kEffectiveBatchSizeId)) {
        effective_batch_size = options.Get<int>(SharedBackendParams::kEffectiveBatchSizeId);
    }
    kMaxOutOfOrderEvals = std::max(1, static_cast<int>(kMaxOutOfOrderEvalsFactor * effective_batch_size));
    // --- End Added Calculation ---
  }

// Rest of the file (CalculateWDLRescaleParams, GetContempt) remains unchanged...
// ... (Make sure no closing brace was accidentally removed)

}  // namespace lczero
