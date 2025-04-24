/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#pragma once

#include <array>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "chess/bitboard.h" // Includes Move, MoveList definitions
#include "neural/encoder.h" // Often defines necessary enums/types
#include "utils/optionsdict.h"
#include "utils/optionsparser.h"
// #include "neural/shared_params.h" // REMOVED - Does not exist

namespace lczero {

// Define HistoryFill locally if not brought in by other headers
enum class HistoryFill { NO, FEN_ONLY, ALWAYS };

enum class ContemptMode { PLAY, WHITE, BLACK, NONE };

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

  // Contempt mode enum was duplicated, removed second one

  // --- Parameter Getters (Keep all getters from previous correct version) ---
  int GetMiniBatchSize() const { return kMiniBatchSize; }
  int GetMaxPrefetch() const { return kMaxPrefetch; }
  bool GetRootHasOwnCpuctParams() const { return kRootHasOwnCpuctParams; }
  float GetCpuct(bool at_root) const { return at_root ? kCpuctAtRoot : kCpuct; }
  float GetCpuctExponent(bool at_root) const { return at_root ? kCpuctExponentAtRoot : kCpuctExponent; }
  float GetCpuctBase(bool at_root) const { return at_root ? kCpuctBaseAtRoot : kCpuctBase; }
  float GetCpuctFactor(bool at_root) const { return at_root ? kCpuctFactorAtRoot : kCpuctFactor; }
  bool GetUseUncertaintyWeighting() const { return kUseUncertaintyWeighting; }
  float GetUncertaintyWeightingCoefficient() const { return kUncertaintyWeightingCoefficient; }
  float GetUncertaintyWeightingExponent() const { return kUncertaintyWeightingExponent; }
  float GetUncertaintyWeightingCap() const { return kUncertaintyWeightingCap; }
  bool GetMoveRuleBucketing() const { return kMoveRuleBucketing; }
  float GetTemperature() const { return kTemperature; }
  float GetTemperatureRoot() const { return kTemperatureRoot; }
  float GetTemperatureCold() const { return kTemperatureCold; }
  float GetTemperatureWarmupScale() const { return kTemperatureWarmupScale; }
  int GetTemperatureVisitOffset() const { return kTemperatureVisitOffset; }
  bool GetQvalueTempIsEnabled() const { return kQvalueTempIsEnabled; }
  float GetQvalueZeroTemp() const { return kQvalueZeroTemp; }
  float GetQvalueOneTemp() const { return kQvalueOneTemp; }
  float GetPolicyTemperature() const { return kPolicyTemperature; } // Use member kPolicyTemperature
  bool GetUsePolicyBoosting() const { return kUsePolicyBoosting; }
  float GetTopPolicyBoost() const { return kTopPolicyBoost; }
  int GetTopPolicyNumBoost() const { return kTopPolicyNumBoost; }
  float GetTopPolicyTierTwoBoost() const { return kTopPolicyTierTwoBoost; }
  int GetTopPolicyTierTwoNumBoost() const { return kTopPolicyTierTwoNumBoost; }
  float GetDirichletAlpha() const { return kDirichletAlpha; }
  float GetNoiseEpsilon() const { return kNoiseEpsilon; }
  float GetPbCInit(bool at_root) const { return at_root ? kPbCInitAtRoot : kPbCInit; }
  float GetPbCFactor(bool at_root) const { return at_root ? kPbCFactorAtRoot : kPbCFactor; }
  FpuStrategy GetFpuStrategy(bool at_root) const { return at_root ? kFpuStrategyAtRoot : kFpuStrategy; }
  float GetFpuValue(bool at_root) const { return at_root ? kFpuValueAtRoot : kFpuValue; }
  int GetCacheHistoryLength() const { return kCacheHistoryLength; }
  float GetPolicyDecayExponent() const { return kPolicyDecayExponent; }
  float GetPolicyDecayFactor() const { return kPolicyDecayFactor; }
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
  HistoryFill GetHistoryFill() const { return kHistoryFill; } // Use member kHistoryFill
  float GetMovesLeftMaxEffect() const { return kMovesLeftMaxEffect; }
  float GetMovesLeftThreshold() const { return kMovesLeftThreshold; }
  float GetMovesLeftLinearFactor() const { return kMovesLeftLinearFactor; }
  float GetMovesLeftScaledFactor() const { return kMovesLeftScaledFactor; }
  float GetMovesLeftQuadraticFactor() const { return kMovesLeftQuadraticFactor; }
  bool GetDisplayCacheUsage() const { return kDisplayCacheUsage; }
  int GetMaxConcurrentSearchers() const { return kMaxConcurrentSearchers; }
  float GetDrawScore() const { return kDrawScore; }
  ContemptMode GetContemptMode() const { return kContemptMode; }
  float GetContempt() const { return kContempt; }
  float GetWDLRescaleRatio() const { return kWDLRescaleParams.ratio; }
  float GetWDLRescaleDiff() const { return kWDLRescaleParams.diff; }
  float GetWDLMaxS() const { return kWDLMaxS; }
  float GetWDLEvalObjectivity() const { return kWDLEvalObjectivity; }
  uint32_t GetMaxOutOfOrderEvals() const { return kMaxOutOfOrderEvals; }
  float GetNpsLimit() const { return kNpsLimit; }
  int GetSolidTreeThreshold() const { return kSolidTreeThreshold; }
  int GetTaskWorkersPerSearchWorker() const { return kTaskWorkersPerSearchWorker; }
  int GetMinimumWorkSizeForProcessing() const { return kMinimumWorkSizeForProcessing; }
  int GetMinimumWorkSizeForPicking() const { return kMinimumWorkSizeForPicking; }
  int GetMinimumRemainingWorkSizeForPicking() const { return kMinimumRemainingWorkSizeForPicking; }
  int GetMinimumWorkPerTaskForProcessing() const { return kMinimumWorkPerTaskForProcessing; }
  int GetMaxCollisionVisitsScalingStart() const { return kMaxCollisionVisitsScalingStart; }
  int GetMaxCollisionVisitsScalingEnd() const { return kMaxCollisionVisitsScalingEnd; }
  float GetMaxCollisionVisitsScalingPower() const { return kMaxCollisionVisitsScalingPower; }
  int GetThreadIdlingThreshold() const { return kThreadIdlingThreshold; }
  std::string GetUciOpponent() const { return kUciOpponent; }
  float GetUciRatingAdv() const { return kUciRatingAdv; }
  bool GetSearchSpinBackoff() const { return kSearchSpinBackoff; }
  int GetRootBeamMinWidth() const { return kRootBeamMinWidth; }
  int GetRootBeamMaxWidth() const { return kRootBeamMaxWidth; }
  int GetRootBeamUpdateThreshold() const { return kRootBeamUpdateThreshold; }
  float GetRootBeamUpdateIntervalFactor() const { return kRootBeamUpdateIntervalFactor; }
  float GetRootBeamScoreMargin() const { return kRootBeamScoreMargin; }
  float GetCpuctUtilityStdevPrior() const { return kCpuctUtilityStdevPrior; }
  float GetCpuctUtilityStdevScale() const { return kCpuctUtilityStdevScale; }
  float GetCpuctUtilityStdevPriorWeight() const { return kCpuctUtilityStdevPriorWeight; }
  bool GetUseVarianceScaling() const { return kUseVarianceScaling; }
  float GetCpuctUncertaintyMinFactor() const { return kCpuctUncertaintyMinFactor; }
  float GetCpuctUncertaintyMaxFactor() const { return kCpuctUncertaintyMaxFactor; }
  float GetCpuctUncertaintyMinUncertainty() const { return kCpuctUncertaintyMinUncertainty; }
  float GetCpuctUncertaintyMaxUncertainty() const { return kCpuctUncertaintyMaxUncertainty; }
  bool GetUseCpuctUncertainty() const { return kUseCpuctUncertainty; }
  bool GetJustFpuUncertainty() const { return kJustFpuUncertainty; }
  float GetEasyEvalWeightDecay() const { return kEasyEvalWeightDecay; }
  float GetEasyEvalValueThreshold() const { return kEasyEvalValueThreshold; }
  // --- END Parameter Getters ---


  // --- Search parameter IDs (Keep all IDs from previous correct version) ---
  static const OptionId kMiniBatchSizeId;
  static const OptionId kMaxPrefetchBatchId;
  static const OptionId kRootHasOwnCpuctParamsId;
  static const OptionId kCpuctId;
  static const OptionId kCpuctAtRootId;
  static const OptionId kCpuctExponentId;
  static const OptionId kCpuctExponentAtRootId;
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
  static const OptionId kPolicyTemperatureId; // Declare shared ID here
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
  static const OptionId kPolicyDecayExponentId;
  static const OptionId kPolicyDecayFactorId;
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
  static const OptionId kHistoryFillId; // Declare shared ID here
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
  static const OptionId kWDLMaxSId;
  static const OptionId kWDLEvalObjectivityId;
  static const OptionId kMaxOutOfOrderEvalsFactorId;
  static const OptionId kNpsLimitId;
  static const OptionId kSolidTreeThresholdId;
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
  static const OptionId kSearchSpinBackoffId;
  static const OptionId kRootBeamMinWidthId;
  static const OptionId kRootBeamMaxWidthId;
  static const OptionId kRootBeamUpdateThresholdId;
  static const OptionId kRootBeamUpdateIntervalFactorId;
  static const OptionId kRootBeamScoreMarginId;
  static const OptionId kCpuctUtilityStdevPriorId;
  static const OptionId kCpuctUtilityStdevScaleId;
  static const OptionId kCpuctUtilityStdevPriorWeightId;
  static const OptionId kUseVarianceScalingId;
  static const OptionId kCpuctUncertaintyMinFactorId;
  static const OptionId kCpuctUncertaintyMaxFactorId;
  static const OptionId kCpuctUncertaintyMinUncertaintyId;
  static const OptionId kCpuctUncertaintyMaxUncertaintyId;
  static const OptionId kUseCpuctUncertaintyId;
  static const OptionId kJustFpuUncertaintyId;
  static const OptionId kEasyEvalWeightDecayId;
  static const OptionId kEasyEvalValueThresholdId;
  // --- END Search parameter IDs ---


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

  // --- Search parameters Members (Keep all from previous correct version) ---
  const int kMiniBatchSize;
  const int kMaxPrefetch;
  const bool kRootHasOwnCpuctParams;
  const float kCpuct;
  const float kCpuctAtRoot;
  const float kCpuctExponent;
  const float kCpuctExponentAtRoot;
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
  const float kPolicyTemperature; // Add member for shared param
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
  const float kPolicyDecayExponent;
  const float kPolicyDecayFactor;
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
  const HistoryFill kHistoryFill; // Add member for shared param
  const float kMovesLeftMaxEffect;
  const float kMovesLeftThreshold;
  const float kMovesLeftLinearFactor; // Missing in original uwuplant?
  const float kMovesLeftScaledFactor;
  const float kMovesLeftQuadraticFactor;
  const bool kDisplayCacheUsage;
  const int kMaxConcurrentSearchers;
  const float kDrawScore;
  const ContemptMode kContemptMode;
  const float kContempt;
  const WDLRescaleParams kWDLRescaleParams;
  const float kWDLMaxS;
  const float kWDLEvalObjectivity;
  const float kMaxOutOfOrderEvalsFactor;
  uint32_t kMaxOutOfOrderEvals;
  const float kNpsLimit;
  const int kSolidTreeThreshold;
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
  const bool kSearchSpinBackoff;
  const int kRootBeamMinWidth;
  const int kRootBeamMaxWidth;
  const int kRootBeamUpdateThreshold;
  const float kRootBeamUpdateIntervalFactor;
  const float kRootBeamScoreMargin;
  const float kCpuctUtilityStdevPrior;
  const float kCpuctUtilityStdevScale;
  const float kCpuctUtilityStdevPriorWeight;
  const bool kUseVarianceScaling;
  const float kCpuctUncertaintyMinFactor;
  const float kCpuctUncertaintyMaxFactor;
  const float kCpuctUncertaintyMinUncertainty;
  const float kCpuctUncertaintyMaxUncertainty;
  const bool kUseCpuctUncertainty;
  const bool kJustFpuUncertainty;
  const float kEasyEvalWeightDecay;
  const float kEasyEvalValueThreshold;
  // --- END Search parameters Members ---

};

} // namespace lczero
