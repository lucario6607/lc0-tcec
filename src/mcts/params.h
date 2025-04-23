#pragma once

#include <array>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "move.h"
#include "neural/encoder.h"
#include "utils/optionsdict.h"
#include "utils/optionsparser.h"
#include "neural/shared_params.h" // Already present in uwuplant/lc0

namespace lczero {

class SearchParams {
 public:
  // Explicitly defined default constructor/destructor.
  SearchParams() = default;
  virtual ~SearchParams() = default;

  // Build a SearchParams from the global options.
  explicit SearchParams(const OptionsDict& options);

  // FPU strategy.
  enum class FpuStrategy { ZERO, PARENT, ABSOLUTE };

  // Score type.
  enum class ScoreType { Q, WDL_W, WDL_L, WDL_MU };

  // History fill.
  enum class HistoryFill { NO, FEN_ONLY, ALWAYS };

  // Contempt mode.
  enum class ContemptMode { OFF, WHITE_SIDE_ANALYSIS, BLACK_SIDE_ANALYSIS, PLAY };

  // Parameter getters.
  int GetMiniBatchSize() const { return kMiniBatchSize; }
  int GetMaxPrefetch() const { return kMaxPrefetch; }
  bool GetRootHasOwnCpuctParams() const { return kRootHasOwnCpuctParams; }
  float GetCpuct(bool at_root) const { return at_root ? kCpuctAtRoot : kCpuct; }
  float GetCpuctExponent(bool at_root) const { // Getter already exists, matches diff
    return at_root ? kCpuctExponentAtRoot : kCpuctExponent;
  }
  float GetCpuctBase(bool at_root) const {
    return at_root ? kCpuctBaseAtRoot : kCpuctBase;
  }
  float GetCpuctFactor(bool at_root) const {
    return at_root ? kCpuctFactorAtRoot : kCpuctFactor;
  }
  bool GetUseUncertaintyWeighting() const {
    return kUseUncertaintyWeighting;
  }
  float GetUncertaintyWeightingCoefficient() const {
    return kUncertaintyWeightingCoefficient;
  }
  float GetUncertaintyWeightingExponent() const {
    return kUncertaintyWeightingExponent;
  }
  float GetUncertaintyWeightingCap() const {
    return kUncertaintyWeightingCap;
  }
  bool GetMoveRuleBucketing() const { return kMoveRuleBucketing; }
  float GetTemperature() const { return kTemperature; }
  float GetTemperatureRoot() const { return kTemperatureRoot; }
  float GetTemperatureCold() const { return kTemperatureCold; }
  float GetTemperatureWarmupScale() const { return kTemperatureWarmupScale; }
  int GetTemperatureVisitOffset() const { return kTemperatureVisitOffset; }
  bool GetQvalueTempIsEnabled() const { return kQvalueTempIsEnabled; }
  float GetQvalueZeroTemp() const { return kQvalueZeroTemp; }
  float GetQvalueOneTemp() const { return kQvalueOneTemp; }
  float GetPolicyTemperature() const { return kPolicyTemperature; }
  bool GetUsePolicyBoosting() const { return kUsePolicyBoosting; }
  float GetTopPolicyBoost() const { return kTopPolicyBoost; }
  int GetTopPolicyNumBoost() const { return kTopPolicyNumBoost; }
  float GetTopPolicyTierTwoBoost() const { return kTopPolicyTierTwoBoost; }
  int GetTopPolicyTierTwoNumBoost() const {
    return kTopPolicyTierTwoNumBoost;
  }
  float GetDirichletAlpha() const { return kDirichletAlpha; }
  float GetNoiseEpsilon() const { return kNoiseEpsilon; }
  float GetPbCInit(bool at_root) const {
    return at_root ? kPbCInitAtRoot : kPbCInit;
  }
  float GetPbCFactor(bool at_root) const {
    return at_root ? kPbCFactorAtRoot : kPbCFactor;
  }
  FpuStrategy GetFpuStrategy(bool at_root) const {
    return at_root ? kFpuStrategyAtRoot : kFpuStrategy;
  }
  float GetFpuValue(bool at_root) const {
    return at_root ? kFpuValueAtRoot : kFpuValue;
  }
  int GetCacheHistoryLength() const { return kCacheHistoryLength; }
  float GetPolicyDecayExponent() const { return kPolicyDecayExponent; } // Exists in uwuplant
  float GetPolicyDecayFactor() const { return kPolicyDecayFactor; } // Exists in uwuplant
  int GetMaxCollisionEvents() const { return kMaxCollisionEvents; }
  int GetMaxCollisionVisits() const { return kMaxCollisionVisits; }
  bool GetUseCorrectionHistory() const { return kUseCorrectionHistory; }
  float GetCorrectionHistoryAlpha() const { return kCorrectionHistoryAlpha; }
  float GetCorrectionHistoryLambda() const { return kCorrectionHistoryLambda; }
  bool GetUseDesperation() const { return kUseDesperation; }
  float GetDesperationLow() const { return kDesperationLow; }
  float GetDesperationHigh() const { return kDesperationHigh; }
  float GetDesperationMultiplier() const { return kDesperationMultiplier; }
  float GetDesperationPriorWeight() const { return kDesperationPriorWeight; }
  ScoreType GetScoreType() const { return kScoreType; }
  HistoryFill GetHistoryFill() const { return kHistoryFill; }
  float GetMovesLeftMaxEffect() const { return kMovesLeftMaxEffect; }
  float GetMovesLeftThreshold() const { return kMovesLeftThreshold; }
  float GetMovesLeftLinearFactor() const { return kMovesLeftLinearFactor; }
  float GetMovesLeftScaledFactor() const { return kMovesLeftScaledFactor; }
  float GetMovesLeftQuadraticFactor() const {
    return kMovesLeftQuadraticFactor;
  }
  bool GetDisplayCacheUsage() const { return kDisplayCacheUsage; }
  int GetMaxConcurrentSearchers() const { return kMaxConcurrentSearchers; }
  float GetDrawScore() const { return kDrawScore; }
  ContemptMode GetContemptMode() const { return kContemptMode; }
  float GetContempt() const { return kContempt; }
  float GetWDLRescaleRatio() const { return kWDLRescaleParams.ratio; }
  float GetWDLRescaleDiff() const { return kWDLRescaleParams.diff; }
  float GetWDLMaxS() const { return kWDLMaxS; } // Added getter
  float GetWDLEvalObjectivity() const { return kWDLEvalObjectivity; }
  uint32_t GetMaxOutOfOrderEvals() const { return kMaxOutOfOrderEvals; } // Type changed later
  float GetNpsLimit() const { return kNpsLimit; }
  int GetSolidTreeThreshold() const { return kSolidTreeThreshold; } // Exists in uwuplant
  int GetTaskWorkersPerSearchWorker() const {
    return kTaskWorkersPerSearchWorker;
  }
  int GetMinimumWorkSizeForProcessing() const {
    return kMinimumWorkSizeForProcessing;
  }
  int GetMinimumWorkSizeForPicking() const {
    return kMinimumWorkSizeForPicking;
  }
  int GetMinimumRemainingWorkSizeForPicking() const {
    return kMinimumRemainingWorkSizeForPicking;
  }
  int GetMinimumWorkPerTaskForProcessing() const {
    return kMinimumWorkPerTaskForProcessing;
  }
  int GetMaxCollisionVisitsScalingStart() const {
    return kMaxCollisionVisitsScalingStart;
  }
  int GetMaxCollisionVisitsScalingEnd() const {
    return kMaxCollisionVisitsScalingEnd;
  }
  float GetMaxCollisionVisitsScalingPower() const {
    return kMaxCollisionVisitsScalingPower;
  }
  int GetThreadIdlingThreshold() const { return kThreadIdlingThreshold; }
  std::string GetUciOpponent() const { return kUciOpponent; }
  float GetUciRatingAdv() const { return kUciRatingAdv; }
  bool GetSearchSpinBackoff() const { return kSearchSpinBackoff; } // Exists in uwuplant

  // --- Root Beam Search Getters ---
  // int GetRootBeamWidth() const; // Removed if it existed (it doesn't in uwuplant)
  int GetRootBeamMinWidth() const { return kRootBeamMinWidth; }         // Added
  int GetRootBeamMaxWidth() const { return kRootBeamMaxWidth; }         // Added
  int GetRootBeamUpdateThreshold() const { return kRootBeamUpdateThreshold; } // Added
  float GetRootBeamUpdateIntervalFactor() const { return kRootBeamUpdateIntervalFactor; } // Added (renamed from Interval)
  // --- END Root Beam Search Getters ---

  // --- Variance Scaling getters (Already present in uwuplant/lc0) ---
  float GetCpuctUtilityStdevPrior() const {
    return kCpuctUtilityStdevPrior;
  }
  float GetCpuctUtilityStdevScale() const {
    return kCpuctUtilityStdevScale;
  }
  float GetCpuctUtilityStdevPriorWeight() const {
    return kCpuctUtilityStdevPriorWeight;
  }
  bool GetUseVarianceScaling() const { return kUseVarianceScaling; }

  // --- Uncertainty getters (Already present in uwuplant/lc0) ---
  float GetCpuctUncertaintyMinFactor() const {
    return kCpuctUncertaintyMinFactor;
  }
  float GetCpuctUncertaintyMaxFactor() const {
    return kCpuctUncertaintyMaxFactor;
  }
  float GetCpuctUncertaintyMinUncertainty() const {
    return kCpuctUncertaintyMinUncertainty;
  }
  float GetCpuctUncertaintyMaxUncertainty() const {
    return kCpuctUncertaintyMaxUncertainty;
  }
  bool GetUseCpuctUncertainty() const { return kUseCpuctUncertainty; }
  bool GetJustFpuUncertainty() const { return kJustFpuUncertainty; }

  // --- EasyEval getters (Already present in uwuplant/lc0) ---
  float GetEasyEvalWeightDecay() const { return kEasyEvalWeightDecay; }
  float GetEasyEvalValueThreshold() const { return kEasyEvalValueThreshold; }


  // Search parameter IDs.
  static const OptionId kMiniBatchSizeId;
  static const OptionId kMaxPrefetchBatchId; // Already exists in uwuplant
  static const OptionId kRootHasOwnCpuctParamsId;
  static const OptionId kCpuctId;
  static const OptionId kCpuctAtRootId;
  static const OptionId kCpuctExponentId; // Already exists in uwuplant
  static const OptionId kCpuctExponentAtRootId; // Already exists in uwuplant
  static const OptionId kCpuctBaseId;
  static const OptionId kCpuctBaseAtRootId;
  static const OptionId kCpuctFactorId;
  static const OptionId kCpuctFactorAtRootId;
  static const OptionId kUseUncertaintyWeightingId;
  static const OptionId kUncertaintyWeightingCoefficientId;
  static const OptionId kUncertaintyWeightingExponentId;
  static const OptionId kUncertaintyWeightingCapId;
  static const OptionId kMoveRuleBucketingId;
  static const OptionId kTemperatureId;
  static const OptionId kTemperatureRootId;
  static const OptionId kTemperatureColdId;
  static const OptionId kTemperatureWarmupScaleId;
  static const OptionId kTemperatureVisitOffsetId;
  static const OptionId kQvalueTempIsEnabledId;
  static const OptionId kQvalueZeroTempId;
  static const OptionId kQvalueOneTempId;
  static const OptionId kPolicyTemperatureId; // Shared with SharedBackendParams
  static const OptionId kUsePolicyBoostingId;
  static const OptionId kTopPolicyBoostId;
  static const OptionId kTopPolicyNumBoostId;
  static const OptionId kTopPolicyTierTwoBoostId;
  static const OptionId kTopPolicyTierTwoNumBoostId;
  static const OptionId kDirichletAlphaId;
  static const OptionId kNoiseEpsilonId;
  static const OptionId kPbCInitId;
  static const OptionId kPbCInitAtRootId;
  static const OptionId kPbCFactorId;
  static const OptionId kPbCFactorAtRootId;
  static const OptionId kFpuStrategyId;
  static const OptionId kFpuStrategyAtRootId;
  static const OptionId kFpuValueId;
  static const OptionId kFpuValueAtRootId;
  static const OptionId kCacheHistoryLengthId;
  static const OptionId kPolicyDecayExponentId; // Exists in uwuplant
  static const OptionId kPolicyDecayFactorId; // Exists in uwuplant
  static const OptionId kMaxCollisionEventsId;
  static const OptionId kMaxCollisionVisitsId;
  static const OptionId kUseCorrectionHistoryId;
  static const OptionId kCorrectionHistoryAlphaId;
  static const OptionId kCorrectionHistoryLambdaId;
  static const OptionId kUseDesperationId;
  static const OptionId kDesperationLowId;
  static const OptionId kDesperationHighId;
  static const OptionId kDesperationMultiplierId;
  static const OptionId kDesperationPriorWeightId;
  static const OptionId kScoreTypeId;
  static const OptionId kHistoryFillId; // Shared with SharedBackendParams
  static const OptionId kMovesLeftMaxEffectId;
  static const OptionId kMovesLeftThresholdId;
  static const OptionId kMovesLeftLinearFactorId;
  static const OptionId kMovesLeftScaledFactorId;
  static const OptionId kMovesLeftQuadraticFactorId;
  static const OptionId kDisplayCacheUsageId;
  static const OptionId kMaxConcurrentSearchersId;
  static const OptionId kDrawScoreId;
  static const OptionId kContemptModeId;
  static const OptionId kContemptId;
  static const OptionId kContemptMaxValueId;
  static const OptionId kWDLCalibrationEloId;
  static const OptionId kWDLContemptAttenuationId;
  static const OptionId kWDLDrawRateTargetId;
  static const OptionId kWDLBookExitBiasId;
  static const OptionId kWDLRescaleRatioId;
  static const OptionId kWDLRescaleDiffId;
  static const OptionId kWDLMaxSId; // Added ID
  static const OptionId kWDLEvalObjectivityId;
  static const OptionId kMaxOutOfOrderEvalsFactorId; // Renamed ID
  static const OptionId kNpsLimitId;
  static const OptionId kSolidTreeThresholdId; // Exists in uwuplant
  static const OptionId kTaskWorkersPerSearchWorkerId;
  static const OptionId kMinimumWorkSizeForProcessingId;
  static const OptionId kMinimumWorkSizeForPickingId;
  static const OptionId kMinimumRemainingWorkSizeForPickingId;
  static const OptionId kMinimumWorkPerTaskForProcessingId;
  static const OptionId kMaxCollisionVisitsScalingStartId;
  static const OptionId kMaxCollisionVisitsScalingEndId;
  static const OptionId kMaxCollisionVisitsScalingPowerId;
  static const OptionId kThreadIdlingThresholdId;
  static const OptionId kUCIOpponentId;
  static const OptionId kUCIRatingAdvId;
  static const OptionId kSearchSpinBackoffId; // Exists in uwuplant

  // --- Root Beam Search Parameter IDs ADDED ---
  // static const OptionId kRootBeamWidthId; // Remove if exists (doesn't)
  static const OptionId kRootBeamMinWidthId;           // Added ID
  static const OptionId kRootBeamMaxWidthId;           // Added ID
  static const OptionId kRootBeamUpdateThresholdId;    // Added ID
  static const OptionId kRootBeamUpdateIntervalFactorId; // Added ID (renamed from IntervalId)
  // --- END Root Beam Search Parameter IDs ADDED ---

  // --- Variance Scaling IDs (Already present in uwuplant/lc0) ---
  static const OptionId kCpuctUtilityStdevPriorId;
  static const OptionId kCpuctUtilityStdevScaleId;
  static const OptionId kCpuctUtilityStdevPriorWeightId;
  static const OptionId kUseVarianceScalingId;

  // --- Uncertainty IDs (Already present in uwuplant/lc0) ---
  static const OptionId kCpuctUncertaintyMinFactorId;
  static const OptionId kCpuctUncertaintyMaxFactorId;
  static const OptionId kCpuctUncertaintyMinUncertaintyId;
  static const OptionId kCpuctUncertaintyMaxUncertaintyId;
  static const OptionId kUseCpuctUncertaintyId;
  static const OptionId kJustFpuUncertaintyId;

  // --- EasyEval IDs (Already present in uwuplant/lc0) ---
  static const OptionId kEasyEvalWeightDecayId;
  static const OptionId kEasyEvalValueThresholdId;


  // Setup the global options related to search.
  static void PopulateOptions(OptionsParser* options);

 private:
  // WDL rescaling parameters.
  struct WDLRescaleParams {
    float ratio = 1.0f;
    float diff = 0.0f;
  };
  WDLRescaleParams CalculateWDLRescaleParams(float draw_score,
                                             float contempt,
                                             float calibration_elo,
                                             float contempt_max_value,
                                             float contempt_attenuation) const;
  float GetContempt(const std::string& uci_opponent,
                    const std::string& contempt_list,
                    float rating_adv, ContemptMode mode) const;

  // Search parameters.
  const int kMiniBatchSize;
  const int kMaxPrefetch;
  const bool kRootHasOwnCpuctParams;
  const float kCpuct;
  const float kCpuctAtRoot;
  const float kCpuctExponent;         // Exists in uwuplant
  const float kCpuctExponentAtRoot;   // Exists in uwuplant
  const float kCpuctBase;
  const float kCpuctBaseAtRoot;
  const float kCpuctFactor;
  const float kCpuctFactorAtRoot;
  const bool kUseUncertaintyWeighting;
  const float kUncertaintyWeightingCoefficient;
  const float kUncertaintyWeightingExponent;
  const float kUncertaintyWeightingCap;
  const bool kMoveRuleBucketing;
  const float kTemperature;
  const float kTemperatureRoot;
  const float kTemperatureCold;
  const float kTemperatureWarmupScale;
  const int kTemperatureVisitOffset;
  const bool kQvalueTempIsEnabled;
  const float kQvalueZeroTemp;
  const float kQvalueOneTemp;
  const float kPolicyTemperature; // From SharedParams
  const bool kUsePolicyBoosting;
  const float kTopPolicyBoost;
  const int kTopPolicyNumBoost;
  const float kTopPolicyTierTwoBoost;
  const int kTopPolicyTierTwoNumBoost;
  const float kDirichletAlpha;
  const float kNoiseEpsilon;
  const float kPbCInit;
  const float kPbCInitAtRoot;
  const float kPbCFactor;
  const float kPbCFactorAtRoot;
  const FpuStrategy kFpuStrategy;
  const FpuStrategy kFpuStrategyAtRoot;
  const float kFpuValue;
  const float kFpuValueAtRoot;
  const int kCacheHistoryLength;
  const float kPolicyDecayExponent; // Exists in uwuplant
  const float kPolicyDecayFactor;   // Exists in uwuplant
  const int kMaxCollisionEvents;
  const int kMaxCollisionVisits;
  const bool kUseCorrectionHistory;
  const float kCorrectionHistoryAlpha;
  const float kCorrectionHistoryLambda;
  const bool kUseDesperation;
  const float kDesperationLow;
  const float kDesperationHigh;
  const float kDesperationMultiplier;
  const float kDesperationPriorWeight;
  const ScoreType kScoreType;
  const HistoryFill kHistoryFill; // From SharedParams
  const float kMovesLeftMaxEffect;
  const float kMovesLeftThreshold;
  const float kMovesLeftLinearFactor;
  const float kMovesLeftScaledFactor;
  const float kMovesLeftQuadraticFactor;
  const bool kDisplayCacheUsage;
  const int kMaxConcurrentSearchers;
  const float kDrawScore;
  const ContemptMode kContemptMode;
  const float kContempt;
  const WDLRescaleParams kWDLRescaleParams;
  const float kWDLMaxS; // Added member variable (based on .cc init)
  const float kWDLEvalObjectivity;
  const float kMaxOutOfOrderEvalsFactor; // Changed type and name
  uint32_t kMaxOutOfOrderEvals; // Made non-const, calculated in constructor
  const float kNpsLimit;
  const int kSolidTreeThreshold; // Exists in uwuplant
  const int kTaskWorkersPerSearchWorker;
  const int kMinimumWorkSizeForProcessing;
  const int kMinimumWorkSizeForPicking;
  const int kMinimumRemainingWorkSizeForPicking;
  const int kMinimumWorkPerTaskForProcessing;
  const int kMaxCollisionVisitsScalingStart;
  const int kMaxCollisionVisitsScalingEnd;
  const float kMaxCollisionVisitsScalingPower;
  const int kThreadIdlingThreshold;
  const std::string kUciOpponent;
  const float kUciRatingAdv;
  const bool kSearchSpinBackoff; // Exists in uwuplant

  // --- Root Beam Search ADDED ---
  const int kRootBeamMinWidth;             // Added member
  const int kRootBeamMaxWidth;             // Added member
  const int kRootBeamUpdateThreshold;      // Added member
  const float kRootBeamUpdateIntervalFactor; // Added member (float type)
  // --- END Root Beam Search ADDED ---

  // --- Variance Scaling members (Already present in uwuplant/lc0) ---
  const float kCpuctUtilityStdevPrior;
  const float kCpuctUtilityStdevScale;
  const float kCpuctUtilityStdevPriorWeight;
  const bool kUseVarianceScaling;

  // --- Uncertainty members (Already present in uwuplant/lc0) ---
  const float kCpuctUncertaintyMinFactor;
  const float kCpuctUncertaintyMaxFactor;
  const float kCpuctUncertaintyMinUncertainty;
  const float kCpuctUncertaintyMaxUncertainty;
  const bool kUseCpuctUncertainty;
  const bool kJustFpuUncertainty;

  // --- EasyEval members (Already present in uwuplant/lc0) ---
  const float kEasyEvalWeightDecay;
  const float kEasyEvalValueThreshold;
};

} // namespace lczero
