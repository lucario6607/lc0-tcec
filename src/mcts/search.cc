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

#include "mcts/search.h" // Should include node.h, cache.h, etc. now

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <thread>
#include <vector>
#include <limits>
#include <numeric>

// Include necessary headers explicitly if not covered by search.h chain
#include "neural/network.h"
#include "syzygy/syzygy.h"
#include "utils/fastmath.h"
#include "utils/random.h"
#include "utils/spinhelper.h"

namespace lczero {

namespace {
// Maximum delay between outputting "uci info" when nothing interesting happens.
const int kUciInfoMinimumFrequencyMs = 5000;

MoveList MakeRootMoveFilter(const MoveList& searchmoves,
                            SyzygyTablebase* syzygy_tb,
                            const PositionHistory& history, bool fast_play,
                            std::atomic<int>* tb_hits, bool* dtz_success) {
  assert(tb_hits);
  assert(dtz_success);
  // Search moves overrides tablebase.
  if (!searchmoves.empty()) return searchmoves;
  const auto& board = history.Last().GetBoard();
  MoveList root_moves;
  if (!syzygy_tb || !board.castlings().no_legal_castle() ||
      (board.ours() | board.theirs()).count() > syzygy_tb->max_cardinality()) {
    return root_moves;
  }
  if (syzygy_tb->root_probe(
          history.Last(), fast_play || history.DidRepeatSinceLastZeroingMove(),
          &root_moves)) {
    *dtz_success = true;
    tb_hits->fetch_add(1, std::memory_order_acq_rel);
  } else if (syzygy_tb->root_probe_wdl(history.Last(), &root_moves)) {
    tb_hits->fetch_add(1, std::memory_order_acq_rel);
  }
  return root_moves;
}

class MEvaluator {
 public:
  MEvaluator()
      : enabled_{false},
        m_slope_{0.0f},
        m_cap_{0.0f},
        a_constant_{0.0f},
        a_linear_{0.0f},
        a_square_{0.0f},
        q_threshold_{0.0f},
        parent_m_{0.0f} {}

  MEvaluator(const SearchParams& params, const Node* parent = nullptr)
      : enabled_{params.GetMovesLeftMaxEffect() > 0.0f},
        m_slope_{params.GetMovesLeftSlope()},
        m_cap_{params.GetMovesLeftMaxEffect()},
        a_constant_{params.GetMovesLeftConstantFactor()},
        a_linear_{params.GetMovesLeftScaledFactor()},
        a_square_{params.GetMovesLeftQuadraticFactor()},
        q_threshold_{params.GetMovesLeftThreshold()},
        parent_m_{parent ? parent->GetM() : 0.0f},
        parent_within_threshold_{parent ? WithinThreshold(parent, q_threshold_)
                                        : false} {}

  void SetParent(const Node* parent) {
    assert(parent);
    if (enabled_) {
      parent_m_ = parent->GetM();
      parent_within_threshold_ = WithinThreshold(parent, q_threshold_);
    }
  }

  // Calculates the utility for favoring shorter wins and longer losses.
  float GetMUtility(Node* child, float q) const {
     if (!enabled_ || !parent_within_threshold_ || !child) return 0.0f;
    const float child_m = child->GetM();
    float m = std::clamp(m_slope_ * (child_m - parent_m_), -m_cap_, m_cap_);
    m *= FastSign(-q);
    if (q_threshold_ > 0.0f && q_threshold_ < 1.0f) {
      q = std::max(0.0f, (std::abs(q) - q_threshold_)) / (1.0f - q_threshold_);
    }
    m *= a_constant_ + a_linear_ * std::abs(q) + a_square_ * q * q;
    return m;
  }

  float GetMUtility(const EdgeAndNode& child, float q) const {
    if (!enabled_ || !parent_within_threshold_) return 0.0f;
    if (child.GetN() == 0 || !child.node()) return GetDefaultMUtility();
    return GetMUtility(child.node(), q);
  }

  // The M utility to use for unvisited nodes.
  float GetDefaultMUtility() const { return 0.0f; }

 private:
  static bool WithinThreshold(const Node* parent, float q_threshold) {
      if (!parent) return false;
    return std::abs(parent->GetQ(0.0f)) > q_threshold;
  }

  const bool enabled_;
  const float m_slope_;
  const float m_cap_;
  const float a_constant_;
  const float a_linear_;
  const float a_square_;
  const float q_threshold_;
  float parent_m_ = 0.0f;
  bool parent_within_threshold_ = false;
};

// --- Helper functions for PUCT calculation (copied from previous attempts) ---
inline float ComputeUncertaintyFactor(const SearchParams& params, float e) {
  if (e < 0) return 1.0f;
  float min_factor = params.GetCpuctUncertaintyMinFactor();
  float max_factor = params.GetCpuctUncertaintyMaxFactor();
  float min_uncertainty = params.GetCpuctUncertaintyMinUncertainty();
  float max_uncertainty = params.GetCpuctUncertaintyMaxUncertainty();
  e = std::clamp(e * e, min_uncertainty, max_uncertainty);
  float factor = min_factor + (max_factor - min_factor) * (e - min_uncertainty) /
                                  (max_uncertainty - min_uncertainty + 1e-5f);
  return factor;
}

inline float ComputeStdev(const SearchParams& params, float q, float weight,
                          float vs) {
  if (weight <= 1e-9f) return 0.0f;
  float util_sq_avg = vs;
  const float util_sq = q * q;
  util_sq_avg = std::max(util_sq_avg, util_sq);
  const float var_estimate = util_sq_avg - util_sq;
  float stdev_estimate = sqrt(std::max(var_estimate, 0.0f));
  return stdev_estimate;
}

inline float ComputeStdevFactor(const SearchParams& params, float q,
                                float weight, float vs) {
  if (weight <= 1e-9f) return 1.0f;
  float util_sq_avg = vs;
  const float util_sq = q * q;
  util_sq_avg = std::max(util_sq_avg, util_sq);
  const float stdev_prior = params.GetCpuctUtilityStdevPrior();
  const float variance_prior = stdev_prior * stdev_prior;
  const float prior_weight = params.GetCpuctUtilityStdevPriorWeight();
  const float stdev_factor_scale = params.GetCpuctUtilityStdevScale();
  const float denominator = prior_weight + weight - 1.0f;
  if (denominator <= 1e-9f) return 1.0f;
  const float var_estimate =
      ((util_sq + variance_prior) * prior_weight + util_sq_avg * weight) / denominator - util_sq;
  const float stdev_estimate = sqrt(std::max(var_estimate, 0.0f));
  if (stdev_prior <= 1e-9f) return 1.0f;
  float stdev_factor = 1.0f + stdev_factor_scale * (stdev_estimate / stdev_prior - 1.0f);
  return std::max(0.0f, stdev_factor);
}

inline float ComputeDesperationFactor(const SearchParams& params, float q,
  float weight) {
  if (weight <= 1e-9f) return 1.0f;
  const float prior_weight = params.GetDesperationPriorWeight();
  const float low = params.GetDesperationLow(); const float high = params.GetDesperationHigh();
  q = abs(q);
  float factor = (q <= low || q >= high) ? params.GetDesperationMultiplier() : 1.0f;
  return 1.0f + (factor - 1.0f) * weight / (prior_weight + weight);
}

inline float ComputeStdevFactor(const SearchParams& params, Node* node) {
  return node ? ComputeStdevFactor(params, node->GetWL(), node->GetWeight(), node->GetVS()) : 1.0f;
}

inline float ComputeCpuctFactor(const SearchParams& params, float weight,
                                float q, float vs, float e, bool is_root_node) {
  const float stdev_factor = params.GetUseVarianceScaling()
                                 ? ComputeStdevFactor(params, q, weight, vs)
                                 : 1.0f;
  const float uncertainty_factor = params.GetUseCpuctUncertainty()
                                      ? ComputeUncertaintyFactor(params, e) : 1.0f;
  const float desperation_factor =
      params.GetUseDesperation() ?
          ComputeDesperationFactor(params, q, weight) : 1.0f;
  return uncertainty_factor * stdev_factor * desperation_factor;
}

inline float GetFpu(const SearchParams& params, Node* node, bool is_root_node,
                    float draw_score) {
  const auto value = params.GetFpuValue(is_root_node);
  if (!node) return value;
  return params.GetFpuAbsolute(is_root_node)
             ? value
             : fmax(-node->GetQ(-draw_score) - value * std::sqrt(node->GetVisitedPolicy()), -1.0f);
}

inline float GetFpu(const SearchParams& params, Node* node, bool is_root_node,
                    float draw_score, float visited_pol) {
  const auto value = params.GetFpuValue(is_root_node);
   if (!node) return value;
  return params.GetFpuAbsolute(is_root_node)
             ? value
             : fmax(-node->GetQ(-draw_score) - value * std::sqrt(visited_pol), -1.0f);
}

inline float ComputeExploreFactor(const SearchParams& params, float weight, bool is_root_node) {
  const float init = params.GetCpuct(is_root_node);
  const float k = params.GetCpuctFactor(is_root_node);
  const float base = params.GetCpuctBase(is_root_node);
  const float exponent = params.GetCpuctExponent(is_root_node);
  return (init + (k ? k * FastLog((weight + base) / base) : 0.0f)) *
         std::pow(fmax(weight, 1e-5f), exponent);
}

inline float ComputeExploreFactor(const SearchParams& params, float weight, float q,
                          float vs, float e, bool is_root_node) {
  const float base_factor = ComputeExploreFactor(params, weight, is_root_node);
  const float extra_factor = ComputeCpuctFactor(params, weight, q, vs, e, is_root_node);
  return base_factor * extra_factor ;
}

inline float ComputeWeight(const SearchParams& params, float uncertainty) {
  if (!params.GetUseUncertaintyWeighting()) return 1.0f;
  if (uncertainty < 0) return 1.0f;
  const float cap = params.GetUncertaintyWeightingCap();
  const float coefficient = params.GetUncertaintyWeightingCoefficient();
  const float exponent = params.GetUncertaintyWeightingExponent();
  return fmin(cap, coefficient * pow(std::max(0.0f, uncertainty), exponent));
}

inline float ComputePolicyDecayFactor(const SearchParams& params, uint32_t N) {
  const float exponent = params.GetPolicyDecayExponent();
  const float proportionality_factor = params.GetPolicyDecayFactor();
  return (exponent == 0.0f || proportionality_factor == 0.0f)
             ? 1.0f
             : FastExp(-FastLog(1.0f + proportionality_factor * N) * exponent);
}
inline float ComputePolicyDecay(const float factor, const float pol) {
  return factor == 1.0f ? pol : pol / (pol + (1.0f - pol) * factor);
}

float CalculatePUCTScore(const SearchParams& params, Node* parent_node, const EdgeAndNode& child_edge, bool is_root_node, float draw_score) {
    if (!parent_node) return -std::numeric_limits<float>::infinity();
    float fpu = GetFpu(params, parent_node, is_root_node, draw_score);
    float Q = child_edge.HasNode() ? child_edge.GetQ(draw_score) : fpu;
    float U_coeff = ComputeExploreFactor(params, parent_node->GetWeight(), parent_node->GetWL(), parent_node->GetVS(), parent_node->GetE(), is_root_node);
    return Q + child_edge.GetU(U_coeff);
}

// WDLRescale using parameter
double WDLRescaleHelper(float& v, float& d, float wdl_rescale_ratio,
                        float wdl_rescale_diff, float sign, bool invert,
                        float wdl_max_s) {
   return WDLRescale(v, d, wdl_rescale_ratio, wdl_rescale_diff, sign, invert, wdl_max_s);
}

}  // namespace (anonymous)


// Constructor Implementation
Search::Search(NodeTree* dag, Network* network,
               std::unique_ptr<UciResponder> uci_responder,
               const MoveList& searchmoves,
               std::chrono::steady_clock::time_point start_time,
               std::unique_ptr<SearchStopper> stopper, bool infinite,
               bool ponder, const OptionsDict& options, NNCache* cache,
               SyzygyTablebase* syzygy_tb)
    : ok_to_respond_bestmove_(!infinite && !ponder),
      stopper_(std::move(stopper)),
      root_node_(dag ? dag->GetCurrentHead() : nullptr),
      cache_(cache),
      dag_(dag),
      syzygy_tb_(syzygy_tb),
      played_history_(dag ? dag->GetPositionHistory() : PositionHistory()),
      network_(network),
      params_(options),
      searchmoves_(searchmoves),
      start_time_(start_time),
      initial_visits_(root_node_ ? root_node_->GetN() : 0),
      root_move_filter_(MakeRootMoveFilter(
          searchmoves_, syzygy_tb_, played_history_,
          params_.GetSyzygyFastPlay(), &tb_hits_, &root_is_in_dtz_)),
      uci_responder_(std::move(uci_responder)) {

   if (!root_node_) {
        LOGFILE << "Error: Search initialized with null root node. DAG might be uninitialized.";
   }

  if (params_.GetMaxConcurrentSearchers() != 0) {
    pending_searchers_.store(params_.GetMaxConcurrentSearchers(),
                             std::memory_order_release);
  }
  contempt_mode_ = params_.GetContemptMode();
  if (contempt_mode_ == ContemptMode::PLAY) {
    if (infinite) {
      contempt_mode_ = ContemptMode::NONE;
      if (params_.GetWDLRescaleDiff() != 0.0f) {
        std::vector<ThinkingInfo> info(1);
        info.back().comment =
            "WARNING: Contempt mode set to 'disable' as 'play' not supported "
            "for infinite search.";
        if (uci_responder_) uci_responder_->OutputThinkingInfo(&info);
      }
    } else {
      contempt_mode_ = played_history_.IsBlackToMove() != ponder
                           ? ContemptMode::BLACK
                           : ContemptMode::WHITE;
    }
  }
  // Initialize beam state
  current_beam_.clear();
  current_beam_width_ = 0;
  last_beam_update_visits_ = 0;
  next_beam_update_visits_ = (params_.GetRootBeamMaxWidth() > 0 && params_.GetRootBeamUpdateThreshold() > 0)
                              ? params_.GetRootBeamUpdateThreshold()
                              : 0;
  beam_active_ = false;
}

// Destructor Implementation
Search::~Search() {
  Abort();
  Wait();
  {
    SharedMutex::Lock lock(nodes_mutex_);
    CancelSharedCollisions();

#ifndef NDEBUG
    if (root_node_) assert(root_node_->ZeroNInFlight());
#endif
  }

  if (dag_) {
    dag_->TTMaintenance();
    dag_->TTMaintenance();
  }

  LOGFILE << "Search destroyed.";
}

// SendUciInfo Implementation
void Search::SendUciInfo() {
  SharedMutex::SharedLock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);

   if (!root_node_) return;

  const auto max_pv = params_.GetMultiPv();
  std::vector<EdgeAndNode> edges;
  if (beam_active_) {
      std::vector<EdgeAndNode> all_edges;
       for (auto& edge : root_node_->Edges()) { all_edges.push_back(edge); }
       for (const auto& edge : all_edges) {
           bool in_beam = false;
           for (const auto& beam_move : current_beam_) { if (edge.GetMove() == beam_move) { in_beam = true; break; } }
           if (in_beam) edges.push_back(edge);
       }
        std::sort(edges.begin(), edges.end(), [](const auto& a, const auto& b){ return a.GetN() > b.GetN(); });
        if (edges.size() > max_pv) edges.resize(max_pv);
  } else {
      edges = GetBestChildrenNoTemperature(root_node_, max_pv, 0);
  }

  const auto score_type = params_.GetScoreType();
  const auto per_pv_counters = params_.GetPerPvCounters();
  const auto display_cache_usage = params_.GetDisplayCacheUsage();
  const auto draw_score = GetDrawScore(false);
  const float wdl_max_s = params_.GetWDLMaxS();

  std::vector<ThinkingInfo> uci_infos;
  ThinkingInfo common_info;
  common_info.depth = cum_depth_ / (total_playouts_ ? total_playouts_ : 1);
  common_info.seldepth = max_depth_;
  common_info.time = GetTimeSinceStart();
  uint64_t total_nodes;
  std::string reported_nodes = params_.GetReportedNodes();
  if (reported_nodes == "nodes") total_nodes = total_low_nodes_;
  else if (reported_nodes == "queries") total_nodes = total_nn_queries_;
  else total_nodes = total_playouts_;

  if (!per_pv_counters) common_info.nodes = total_playouts_ + initial_visits_;

  if (nps_start_time_) {
    const auto time_since_first_batch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - *nps_start_time_).count();
    if (time_since_first_batch_ms > 0) common_info.nps = total_nodes * 1000 / time_since_first_batch_ms;
  }
  if (display_cache_usage && cache_) common_info.hashfull = cache_->GetSize() * 1000LL / std::max(cache_->GetCapacity(), 1ULL);
  common_info.tb_hits = tb_hits_.load(std::memory_order_acquire);

  int multipv = 0;
  const auto default_q = root_node_ ? -root_node_->GetQ(-draw_score) : 0.0f;
  const auto default_wl = root_node_ ? -root_node_->GetWL() : 0.0f;
  const auto default_d = root_node_ ? root_node_->GetD() : 1.0f;

  for (const auto& edge : edges) {
    ++multipv;
    uci_infos.emplace_back(common_info);
    auto& uci_info = uci_infos.back();
    auto wl = edge.GetWL(default_wl);
    auto d = edge.GetD(default_d);
    float mu_uci = 0.0f;
    if (score_type == "WDL_mu" || (params_.GetWDLRescaleDiff() != 0.0f && contempt_mode_ != ContemptMode::NONE)) {
      auto sign = ((contempt_mode_ == ContemptMode::BLACK) == played_history_.IsBlackToMove()) ? 1.0f : -1.0f;
      mu_uci = WDLRescaleHelper(wl, d, params_.GetWDLRescaleRatio(),
                           contempt_mode_ == ContemptMode::NONE ? 0 : params_.GetWDLRescaleDiff() * params_.GetWDLEvalObjectivity(),
                           sign, true, wdl_max_s);
    }
    const auto q = edge.GetQ(default_q, draw_score);
    if (edge.HasNode() && edge.IsTerminal() && wl != 0.0f) {
      uci_info.mate = std::copysign(std::round(edge.GetM(0.0f) + 1) / 2 + (edge.IsTbTerminal() ? 100 : 0), wl);
    } else if (score_type == "centipawn_with_drawscore") { uci_info.score = 90 * tan(1.5637541897 * q); }
      else if (score_type == "centipawn") { uci_info.score = 90 * tan(1.5637541897 * wl); }
      else if (score_type == "centipawn_2019") { uci_info.score = 295 * wl / (1 - 0.976953126 * std::pow(wl, 14)); }
      else if (score_type == "centipawn_2018") { uci_info.score = 290.680623072 * tan(1.548090806 * wl); }
      else if (score_type == "win_percentage") { uci_info.score = wl * 5000 + 5000; }
      else if (score_type == "Q") { uci_info.score = q * 10000; }
      else if (score_type == "W-L") { uci_info.score = wl * 10000; }
      else if (score_type == "WDL_mu") {
        const float centipawn_fallback_threshold = 0.996f;
        float centipawn_score = 90 * tan(1.5637541897 * wl);
        uci_info.score = (mu_uci != 0.0f && std::abs(wl) + d < centipawn_fallback_threshold && (std::abs(mu_uci) < 1.0f || std::abs(centipawn_score) < std::abs(100 * mu_uci)))
                         ? 100 * mu_uci : centipawn_score;
      }

    auto wdl_w = std::max(0, static_cast<int>(std::round(500.0 * (1.0 + wl - d))));
    auto wdl_l = std::max(0, static_cast<int>(std::round(500.0 * (1.0 - wl - d))));
    auto wdl_d = 1000 - wdl_w - wdl_l;
    if (wdl_d < 0) { wdl_w = std::min(1000, std::max(0, wdl_w + wdl_d / 2)); wdl_l = 1000 - wdl_w; wdl_d = 0; }
    uci_info.wdl = ThinkingInfo::WDL{wdl_w, wdl_d, wdl_l};

    if (network_ && network_->GetCapabilities().has_mlh()) {
       uci_info.moves_left = (edge.HasNode() && root_node_) ? static_cast<int>((1.0f + edge.GetM(1.0f + root_node_->GetM())) / 2.0f) : 0;
    }
    if (max_pv > 1) uci_info.multipv = multipv;
    if (per_pv_counters) uci_info.nodes = edge.GetN();

    bool flip = played_history_.IsBlackToMove();
    int depth = 0;
    auto history = played_history_;
    for (auto iter = edge; iter && iter.HasNode(); iter = GetBestChildNoTemperature(iter.node(), depth), flip = !flip) {
      uci_info.pv.push_back(iter.GetMove(flip));
      history.Append(iter.GetMove());
      if (!iter.node() || history.Last().GetRepetitions() >= 2) break;
      depth += 1;
    }
  }

  if (!uci_infos.empty()) last_outputted_uci_info_ = uci_infos.front();
  if (current_best_edge_ && !edges.empty()) {
    bool found_edge = false;
    for(const auto& edge : edges) { if (edge.edge() == current_best_edge_.edge()) { last_outputted_info_edge_ = current_best_edge_.edge(); found_edge = true; break; } }
    if (!found_edge) last_outputted_info_edge_ = nullptr;
  } else {
     last_outputted_info_edge_ = nullptr;
  }

  if (uci_responder_) uci_responder_->OutputThinkingInfo(&uci_infos);
}

// MaybeOutputInfo Implementation
void Search::MaybeOutputInfo() {
  SharedMutex::Lock nodes_lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  if (!bestmove_is_sent_ && root_node_ && current_best_edge_ &&
      (current_best_edge_.edge() != last_outputted_info_edge_ ||
       last_outputted_uci_info_.depth !=
           static_cast<int>(cum_depth_ /
                            (total_playouts_ ? total_playouts_ : 1)) ||
       last_outputted_uci_info_.seldepth != max_depth_ ||
       last_outputted_uci_info_.time + kUciInfoMinimumFrequencyMs <
           GetTimeSinceStart())) {
    SendUciInfo();
    if (params_.GetLogLiveStats()) {
      SendMovesStats();
    }
    if (stop_.load(std::memory_order_acquire) && !ok_to_respond_bestmove_) {
      std::vector<ThinkingInfo> info(1);
      info.back().comment =
          "WARNING: Search has reached limit and does not make any progress.";
       if (uci_responder_) uci_responder_->OutputThinkingInfo(&info);
    }
  }
}

// Time Functions
int64_t Search::GetTimeSinceStart() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - start_time_)
      .count();
}

int64_t Search::GetTimeSinceFirstBatch() const {
   if (!nps_start_time_) return 0;
   return std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - *nps_start_time_)
       .count();
}

// MaybeTriggerStop Implementation
void Search::MaybeTriggerStop(const IterationStats& stats,
                              StoppersHints* hints) {
  hints->Reset();
  if (params_.GetNpsLimit() > 0) {
    hints->UpdateEstimatedNps(params_.GetNpsLimit());
  }
  SharedMutex::Lock nodes_lock(nodes_mutex_);
  Mutex::Lock lock(counters_mutex_);
  if (bestmove_is_sent_ || !root_node_) return;
  if (total_playouts_ + initial_visits_ == 0) return;

  bool beam_updated = false;
  if (params_.GetRootBeamMaxWidth() > 0 && next_beam_update_visits_ > 0 && root_node_->GetN() >= next_beam_update_visits_) {
      // Check if an update is actually due based on the interval factor logic
       if ((beam_active_ && root_node_->GetN() >= next_beam_update_visits_) || !beam_active_) {
           UpdateRootBeam();
           beam_updated = true;
       }
  }

  if (!stop_.load(std::memory_order_acquire)) {
    if (stopper_->ShouldStop(stats, hints)) FireStopInternal();
  }

  if (stop_.load(std::memory_order_acquire) && ok_to_respond_bestmove_ &&
      !bestmove_is_sent_) {
    if (!beam_updated && params_.GetRootBeamMaxWidth() > 0 && next_beam_update_visits_ > 0 ) {
         // Check again if update is due right before sending bestmove
         if ((beam_active_ && root_node_->GetN() >= next_beam_update_visits_) || !beam_active_) {
             UpdateRootBeam();
         }
    }
    SendUciInfo();
    EnsureBestMoveKnown();
    SendMovesStats();
    BestMoveInfo info(final_bestmove_, final_pondermove_);
    if (uci_responder_) uci_responder_->OutputBestMove(&info);
    stopper_->OnSearchDone(stats);
    bestmove_is_sent_ = true;
    current_best_edge_ = EdgeAndNode();
  } else if (beam_updated) {
      // If beam was updated, re-check best move *if* not stopping yet
       if (!stop_.load(std::memory_order_acquire)) {
            EnsureBestMoveKnown();
       }
  }
}

// GetBestEval Implementation
Eval Search::GetBestEval(Move* move, bool* is_terminal) const {
  SharedMutex::SharedLock lock(nodes_mutex_);
  if (!root_node_) {
      if (move) *move = Move::NO_MOVE;
      if (is_terminal) *is_terminal = true;
      return {0.0f, 1.0f, 0.0f};
  }

  auto best_edges = GetBestChildrenNoTemperature(root_node_, 1, 0);
  EdgeAndNode best_edge = best_edges.empty() ? EdgeAndNode() : best_edges.front();

  float parent_wl = -root_node_->GetWL();
  float parent_d = root_node_->GetD();
  float parent_m = root_node_->GetM();

  if (!best_edge) {
        if (move) *move = Move::NO_MOVE;
        if (is_terminal) *is_terminal = root_node_->IsTerminal();
        return {parent_wl, parent_d, parent_m};
  }

  if (move) *move = best_edge.GetMove(played_history_.IsBlackToMove());
  if (is_terminal) *is_terminal = best_edge.HasNode() && best_edge.IsTerminal();
  return {best_edge.GetWL(parent_wl), best_edge.GetD(parent_d),
          best_edge.HasNode() ? (best_edge.GetM(parent_m - 1) + 1) : parent_m };
}

// GetBestMove Implementation
std::pair<Move, Move> Search::GetBestMove() {
  SharedMutex::Lock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  EnsureBestMoveKnown();
  return {final_bestmove_, final_pondermove_};
}

// GetTotalPlayouts Implementation
std::int64_t Search::GetTotalPlayouts() const {
  SharedMutex::SharedLock lock(nodes_mutex_);
  return total_playouts_;
}

// ResetBestMove Implementation
void Search::ResetBestMove() {
  SharedMutex::Lock nodes_lock(nodes_mutex_);
  Mutex::Lock lock(counters_mutex_);
  bool old_sent = bestmove_is_sent_;
  bestmove_is_sent_ = false;
  EnsureBestMoveKnown();
  bestmove_is_sent_ = old_sent;
}

// EnsureBestMoveKnown Implementation
void Search::EnsureBestMoveKnown() {
  if (bestmove_is_sent_ || !root_node_ || root_node_->GetN() == 0 || !root_node_->HasChildren()) return;

  bool beam_updated_now = false;
   if (params_.GetRootBeamMaxWidth() > 0 && next_beam_update_visits_ > 0 ) {
       if ((beam_active_ && root_node_->GetN() >= next_beam_update_visits_) || !beam_active_) {
            UpdateRootBeam();
            beam_updated_now = true;
       }
  }

  float temperature = params_.GetTemperature();
  const int cutoff_move = params_.GetTemperatureCutoffMove();
  const int decay_delay_moves = params_.GetTempDecayDelayMoves();
  const int decay_moves = params_.GetTempDecayMoves();
  const int moves = played_history_.Last().GetGamePly() / 2;
  if (cutoff_move && (moves + 1) >= cutoff_move) { temperature = params_.GetTemperatureEndgame(); }
  else if (temperature > 1e-5f && decay_moves > 0) {
    if (moves >= decay_delay_moves + decay_moves) temperature = 0.0;
    else if (moves >= decay_delay_moves) temperature *= static_cast<float>(decay_delay_moves + decay_moves - moves) / decay_moves;
    if (temperature < params_.GetTemperatureEndgame()) temperature = params_.GetTemperatureEndgame();
  }

  EdgeAndNode bestmove_edge;
  if (temperature > 1e-5f) { bestmove_edge = GetBestRootChildWithTemperature(temperature); }
  else { bestmove_edge = GetBestChildNoTemperature(root_node_, 0); }

  if (!bestmove_edge) {
      LOGFILE << "Warning: No best move found in EnsureBestMoveKnown. Falling back.";
      bool was_beam_active = beam_active_; beam_active_ = false;
      auto fallback_edges = GetBestChildrenNoTemperature(root_node_, 1, 0);
      beam_active_ = was_beam_active;
      if (!fallback_edges.empty()) bestmove_edge = fallback_edges.front();
      else { final_bestmove_ = Move::NO_MOVE; final_pondermove_ = Move::NO_MOVE; LOGFILE << "Error: Could not determine any best move."; return; }
  }

  final_bestmove_ = bestmove_edge.GetMove(played_history_.IsBlackToMove());
  if (bestmove_edge.HasNode() && bestmove_edge.node()->HasChildren()) {
    final_pondermove_ = GetBestChildNoTemperature(bestmove_edge.node(), 1).GetMove(!played_history_.IsBlackToMove());
  } else {
    final_pondermove_ = Move::NO_MOVE;
  }
}

// GetBestChildrenNoTemperature Implementation
std::vector<EdgeAndNode> Search::GetBestChildrenNoTemperature(Node* parent,
                                                              int count,
                                                              int depth) const {
  if (!parent || parent->GetN() == 0) return {};
  const bool is_root_node = (parent == root_node_);
  const bool is_odd_depth = (depth % 2) == 1;
  const float draw_score = GetDrawScore(is_odd_depth);

  std::vector<EdgeAndNode> edges;
  for (auto& edge : parent->Edges()) {
    if (is_root_node) {
        if (!root_move_filter_.empty()) { if (std::find(root_move_filter_.begin(), root_move_filter_.end(), edge.GetMove()) == root_move_filter_.end()) continue; }
        else if (beam_active_) {
            bool in_beam = false;
            for (const auto& beam_move : current_beam_) { if (edge.GetMove() == beam_move) { in_beam = true; break; } }
            if (!in_beam) continue;
        }
    }
    edges.push_back(edge);
  }

  if (edges.empty()) return {};
  count = std::min(count, static_cast<int>(edges.size()));
  if (count <= 0) return {};
  const auto middle = edges.begin() + count;

  std::partial_sort(edges.begin(), middle, edges.end(),
      [draw_score](const auto& a, const auto& b) {
        enum EdgeRank { kTerminalLoss, kTablebaseLoss, kNonTerminal, kTablebaseWin, kTerminalWin };
        auto GetEdgeRank = [](const EdgeAndNode& edge) {
            const auto wl = edge.GetWL(0.0f);
            if (!edge.HasNode() || edge.GetN() == 0 || !edge.IsTerminal() || wl==0.0f) return kNonTerminal;
            if (edge.IsTbTerminal()) return wl < 0.0 ? kTablebaseLoss : kTablebaseWin;
            return wl < 0.0 ? kTerminalLoss : kTerminalWin;
        };
        const auto a_rank = GetEdgeRank(a); const auto b_rank = GetEdgeRank(b);
        if (a_rank != b_rank) return a_rank > b_rank;
        if (a_rank == kNonTerminal && a.HasNode() && b.HasNode() && a.GetN() != 0 && b.GetN() != 0 && a.IsTerminal() && b.IsTerminal()) {
            if (a.IsTbTerminal() != b.IsTbTerminal()) return a.IsTbTerminal() < b.IsTbTerminal();
            return a.GetM(0.0f) < b.GetM(0.0f);
        }
        if (a_rank == kNonTerminal) {
            if (a.GetWeight() != b.GetWeight()) return a.GetWeight() > b.GetWeight();
            if (a.GetQ(0.0f, draw_score) != b.GetQ(0.0f, draw_score)) return a.GetQ(0.0f, draw_score) > b.GetQ(0.0f, draw_score);
            return a.GetP() > b.GetP();
        }
        if (a_rank > kNonTerminal) { return a.GetM(0.0f) < b.GetM(0.0f); }
        return a.GetM(0.0f) > b.GetM(0.0f);
      });

  edges.resize(count);
  return edges;
}

// GetBestChildNoTemperature Implementation
EdgeAndNode Search::GetBestChildNoTemperature(Node* parent, int depth) const {
  auto res = GetBestChildrenNoTemperature(parent, 1, depth);
  return res.empty() ? EdgeAndNode() : res.front();
}

// GetBestRootChildWithTemperature Implementation
EdgeAndNode Search::GetBestRootChildWithTemperature(float temperature) const {
  const float draw_score = GetDrawScore(false);
  std::vector<float> cumulative_sums;
  std::vector<EdgeAndNode> eligible_edges;
  float sum = 0.0;
  float max_weight = 0.0;
  const float offset = params_.GetTemperatureVisitOffset();
  float max_eval = -std::numeric_limits<float>::infinity();
  if (!root_node_) return EdgeAndNode();
  const float fpu = GetFpu(params_, root_node_, true, draw_score);

  for (auto& edge : root_node_->Edges()) {
    if (!root_move_filter_.empty()) { if (std::find(root_move_filter_.begin(), root_move_filter_.end(), edge.GetMove()) == root_move_filter_.end()) continue; }
    else if (beam_active_) {
        bool in_beam = false;
        for (const auto& beam_move : current_beam_) { if (edge.GetMove() == beam_move) { in_beam = true; break; } }
        if (!in_beam) continue;
    }
    eligible_edges.push_back(edge);
    float current_weight = edge.GetWeight() + offset;
    if (current_weight > max_weight) max_weight = current_weight;
    max_eval = std::max(max_eval, edge.GetQ(fpu, draw_score));
  }

  if (eligible_edges.empty()) return EdgeAndNode();
  max_weight = 0.0f;
  for(const auto& edge : eligible_edges) max_weight = std::max(max_weight, edge.GetWeight() + offset);
  bool use_policy = (max_weight <= 1e-9f);

  const float min_eval = max_eval - params_.GetTemperatureWinpctCutoff() / 50.0f;
  cumulative_sums.reserve(eligible_edges.size());
  std::vector<size_t> final_indices;
  final_indices.reserve(eligible_edges.size());

  for (size_t i = 0; i < eligible_edges.size(); ++i) {
    auto& edge = eligible_edges[i];
    if (edge.GetQ(fpu, draw_score) < min_eval) continue;
    float weight_or_p = use_policy ? edge.GetP() : std::max(0.0f, edge.GetWeight() + offset);
    float norm_factor = use_policy ? 1.0f : max_weight;
    float normalized_val = (norm_factor > 1e-9f) ? (weight_or_p / norm_factor) : 0.0f;
    float temp_inv = (temperature > 1e-9f) ? (1.0f / temperature) : 100.0f; // Handle temp near zero
    float term = (normalized_val > 1e-9f || temp_inv >= 0) ? std::pow(normalized_val, temp_inv) : 0.0f;
    sum += term;
    cumulative_sums.push_back(sum);
    final_indices.push_back(i);
  }

  if (sum <= 1e-9f || final_indices.empty()) {
       LOGFILE << "Warning: Temperature sampling sum is zero or no moves passed filter. Returning best move by weight.";
       EdgeAndNode best_by_weight; float max_w = -std::numeric_limits<float>::infinity();
       for(auto& edge : eligible_edges) { if (edge.GetWeight() > max_w) { max_w = edge.GetWeight(); best_by_weight = edge; } }
       return best_by_weight;
  }

  assert(sum > 0);
  const float toss = Random::Get().GetFloat(cumulative_sums.back());
  int chosen_sum_idx = std::lower_bound(cumulative_sums.begin(), cumulative_sums.end(), toss) - cumulative_sums.begin();
  if (chosen_sum_idx >= final_indices.size()) chosen_sum_idx = final_indices.size() - 1;
  size_t original_idx = final_indices[chosen_sum_idx];
  return eligible_edges[original_idx];
}

// StartThreads Implementation
void Search::StartThreads(size_t how_many) {
  thread_count_.store(how_many, std::memory_order_release);
  Mutex::Lock lock(threads_mutex_);
  if (threads_.size() == 0) {
    threads_.emplace_back([this]() { WatchdogThread(); });
  }
  for (size_t i = 0; i < how_many; i++) {
    threads_.emplace_back([this, i]() {
      SearchWorker worker(this, params_, i);
      worker.RunBlocking();
    });
  }
  LOGFILE << "Search started. "
          << std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - start_time_)
                 .count()
          << "ms already passed.";
}

// RunBlocking Implementation
void Search::RunBlocking(size_t threads) {
  StartThreads(threads);
  Wait();
}

// IsSearchActive Implementation
bool Search::IsSearchActive() const {
  return !stop_.load(std::memory_order_acquire);
}

// PopulateCommonIterationStats Implementation
void Search::PopulateCommonIterationStats(IterationStats* stats) {
  stats->time_since_movestart = GetTimeSinceStart();

  SharedMutex::SharedLock nodes_lock(nodes_mutex_);
  {
    Mutex::Lock counters_lock(counters_mutex_);
    stats->time_since_first_batch = GetTimeSinceFirstBatch();
    if (!nps_start_time_ && total_playouts_ > 0) {
      nps_start_time_ = std::chrono::steady_clock::now();
    }
  }
  if (!root_node_) { // Handle null root node case
       stats->total_visits = initial_visits_;
       stats->total_allocated_nodes = dag_ ? dag_->AllocatedNodeCount() : 0;
       stats->nodes_since_movestart = total_playouts_;
       stats->batches_since_movestart = total_batches_;
       stats->average_depth = 0;
       // ... set other stats to default/zero ...
       return;
  }
  stats->total_visits = total_playouts_ + initial_visits_;
  stats->total_allocated_nodes = dag_ ? dag_->AllocatedNodeCount() : 0;
  stats->nodes_since_movestart = total_playouts_;
  stats->batches_since_movestart = total_batches_;
  stats->average_depth = cum_depth_ / (total_playouts_ ? total_playouts_ : 1);
  stats->edge_n.clear();
  stats->win_found = false;
  stats->may_resign = true;
  stats->num_losing_edges = 0;
  stats->time_usage_hint_ = IterationStats::TimeUsageHint::kNormal;
  stats->mate_depth = std::numeric_limits<int>::max();

  if (root_node_->GetN() > 0) {
    const auto draw_score = GetDrawScore(true);
    const float fpu = GetFpu(params_, root_node_, true, draw_score);
    float max_q_plus_m = -std::numeric_limits<float>::infinity();
    float max_weight = 0.0f;
    bool max_weight_has_max_q_plus_m = true;
    const auto m_evaluator = (network_ && network_->GetCapabilities().has_mlh()) ? MEvaluator(params_, root_node_) : MEvaluator(); // Null check network_
    for (const auto& edge : root_node_->Edges()) {
      if (beam_active_) {
          bool in_beam = false;
          for (const auto& beam_move : current_beam_) { if (edge.GetMove() == beam_move) { in_beam = true; break; } }
          if (!in_beam) continue;
      }

      const auto n = edge.GetN();
      const auto w = edge.GetWeight();
      const auto q = edge.GetQ(fpu, draw_score);
      const auto m = m_evaluator.GetMUtility(edge, q);
      const auto q_plus_m = q + m;
      stats->edge_n.push_back(n);
      if (edge.HasNode()) {
          if (n > 0 && edge.IsTerminal() && edge.GetWL(0.0f) > 0.0f) stats->win_found = true;
          if (n > 0 && edge.IsTerminal() && edge.GetWL(0.0f) < 0.0f) stats->num_losing_edges += 1;
          if (n > 0 && edge.IsTerminal() && edge.GetWL(0.0f) == 1.0f && !edge.IsTbTerminal()) {
            stats->mate_depth = std::min(stats->mate_depth, static_cast<int>(std::round(edge.GetM(0.0f))) / 2 + 1);
          }
      }
      if (n > 0 && q > -0.98f) stats->may_resign = false;
      if (w > max_weight) { max_weight = w; max_weight_has_max_q_plus_m = false; }
      if (max_q_plus_m <= q_plus_m) {
         max_weight_has_max_q_plus_m = (max_q_plus_m == q_plus_m) && (max_weight == w);
         max_q_plus_m = q_plus_m;
         if (max_q_plus_m == q_plus_m && w > max_weight) { max_weight = w; max_weight_has_max_q_plus_m = true; }
         else if (max_q_plus_m == q_plus_m && w == max_weight) { max_weight_has_max_q_plus_m = true; }
      }
    }
    if (!max_weight_has_max_q_plus_m) stats->time_usage_hint_ = IterationStats::TimeUsageHint::kNeedMoreTime;
  }
}

// WatchdogThread Implementation
void Search::WatchdogThread() {
  LOGFILE << "Start a watchdog thread.";
  StoppersHints hints;
  IterationStats stats;
  while (true) {
    PopulateCommonIterationStats(&stats);
    MaybeTriggerStop(stats, &hints);
    MaybeOutputInfo();

    constexpr auto kMaxWaitTimeMs = 100;
    constexpr auto kMinWaitTimeMs = 1;

    Mutex::Lock lock(counters_mutex_);
    if (bestmove_is_sent_) break;

    auto remaining_time = hints.GetEstimatedRemainingTimeMs();
    remaining_time = std::clamp(remaining_time, (int64_t)kMinWaitTimeMs, (int64_t)kMaxWaitTimeMs);

    watchdog_cv_.wait_for(
        lock.get_raw(), std::chrono::milliseconds(remaining_time),
        [this]() { return stop_.load(std::memory_order_acquire); });
  }
  LOGFILE << "End a watchdog thread.";
}

// FireStopInternal Implementation
void Search::FireStopInternal() {
  stop_.store(true, std::memory_order_release);
  watchdog_cv_.notify_all();
}

// Stop Implementation
void Search::Stop() {
  Mutex::Lock lock(counters_mutex_);
  ok_to_respond_bestmove_ = true;
  FireStopInternal();
  LOGFILE << "Stopping search due to `stop` uci command.";
}

// Abort Implementation
void Search::Abort() {
  Mutex::Lock lock(counters_mutex_);
  if (!stop_.load(std::memory_order_acquire) ||
      (!bestmove_is_sent_ && !ok_to_respond_bestmove_)) {
    bestmove_is_sent_ = true;
    FireStopInternal();
  }
  LOGFILE << "Aborting search, if it is still active.";
}

// Wait Implementation
void Search::Wait() {
  Mutex::Lock lock(threads_mutex_);
  while (!threads_.empty()) {
    if (threads_.back().joinable()) threads_.back().join();
    threads_.pop_back();
  }
}

// CancelSharedCollisions Implementation
void Search::CancelSharedCollisions() {
  for (auto& entry : shared_collisions_) {
    auto path = entry.first;
    for (auto it = ++(path.crbegin()); it != path.crend(); ++it) {
        auto* node = std::get<0>(*it);
        if(node) node->CancelScoreUpdate(entry.second);
    }
  }
  shared_collisions_.clear();
}

// GetDrawScore Implementation
float Search::GetDrawScore(bool is_odd_depth) const {
  return (is_odd_depth == played_history_.IsBlackToMove()
              ? params_.GetDrawScore()
              : -params_.GetDrawScore());
}

// UpdateRootBeam Implementation
void Search::UpdateRootBeam() {
    if (params_.GetRootBeamMaxWidth() <= 0 || !root_node_) return;
    if (!root_node_->HasChildren() || root_node_->GetN() == 0) return;

    LOGFILE << "Updating root beam at N=" << root_node_->GetN();

    std::vector<std::pair<float, EdgeAndNode>> scored_edges;
    const float draw_score = GetDrawScore(false);
    bool is_root_node = true;

    for (auto& edge : root_node_->Edges()) {
        if (!root_move_filter_.empty()) { if (std::find(root_move_filter_.begin(), root_move_filter_.end(), edge.GetMove()) == root_move_filter_.end()) continue; }
        float score = CalculatePUCTScore(params_, root_node_, edge, is_root_node, draw_score);
        scored_edges.push_back({score, edge});
    }

    if (scored_edges.empty()) {
        LOGFILE << "No eligible moves found for beam update.";
        beam_active_ = false; next_beam_update_visits_ = 0; return;
    }

    std::sort(scored_edges.begin(), scored_edges.end(), [](const auto& a, const auto& b) {
                  bool a_nan = std::isnan(a.first); bool b_nan = std::isnan(b.first);
                  if (a_nan && b_nan) return false; if (a_nan) return false; if (b_nan) return true;
                  return a.first > b.first;
              });

    const int min_width = std::max(1, params_.GetRootBeamMinWidth());
    const int max_width = std::max(min_width, params_.GetRootBeamMaxWidth());
    int dynamic_width = min_width;
    const float best_score = scored_edges[0].first;
    const float score_margin = std::isnan(best_score) ? 0.0f : params_.GetRootBeamScoreMargin();
    const float score_threshold = std::isnan(best_score) ? -std::numeric_limits<float>::infinity() : best_score - score_margin;

    if (params_.GetRootBeamMinWidth() > 0 && params_.GetRootBeamMinWidth() < params_.GetRootBeamMaxWidth()) {
        for (size_t i = min_width; i < max_width && i < scored_edges.size(); ++i) {
            if (std::isnan(scored_edges[i].first)) continue;
            if (scored_edges[i].first >= score_threshold) dynamic_width = i + 1;
            else break;
        }
         current_beam_width_ = std::min(static_cast<int>(scored_edges.size()), dynamic_width);
    } else {
        current_beam_width_ = std::min(static_cast<int>(scored_edges.size()), max_width);
    }
    current_beam_width_ = std::max(current_beam_width_, std::min(min_width, static_cast<int>(scored_edges.size())));


    current_beam_.clear();
    current_beam_.reserve(current_beam_width_);
    for (int i = 0; i < current_beam_width_; ++i) current_beam_.push_back(scored_edges[i].second.GetMove());

    last_beam_update_visits_ = root_node_->GetN();
    double next_update_double = static_cast<double>(last_beam_update_visits_) * params_.GetRootBeamUpdateIntervalFactor();
    if (next_update_double > static_cast<double>(std::numeric_limits<uint64_t>::max() - 1) ) next_beam_update_visits_ = std::numeric_limits<uint64_t>::max();
    else {
         next_beam_update_visits_ = static_cast<uint64_t>(next_update_double);
         if (next_beam_update_visits_ <= last_beam_update_visits_) next_beam_update_visits_ = last_beam_update_visits_ + 1;
    }

    beam_active_ = true;
    LOGFILE << "Beam updated. Width=" << current_beam_width_ << ", Next update at N=" << next_beam_update_visits_;
    last_outputted_info_edge_ = nullptr;
}

// GetVerboseStats Implementation
std::vector<std::string> Search::GetVerboseStats(Node* node) const {
  if (!node) return {};
  SharedMutex::SharedLock lock(nodes_mutex_);
  const bool is_root = (node == root_node_);
  const bool is_odd_depth = !is_root;
  const bool is_black_to_move = (played_history_.IsBlackToMove() == is_root);
  const float draw_score = GetDrawScore(is_odd_depth);
  const float fpu = GetFpu(params_, node, is_root, draw_score);
  const float U_coeff = ComputeExploreFactor(params_, node->GetWeight(), node->GetWL(), node->GetVS(), node->GetE(), is_root);
  std::vector<EdgeAndNode> edges;
  for (const auto& edge : node->Edges()) {
        if (is_root && beam_active_) {
            bool in_beam = false;
            for (const auto& beam_move : current_beam_) { if (edge.GetMove() == beam_move) { in_beam = true; break; } }
            if (!in_beam) continue;
        }
        edges.push_back(edge);
  }

  std::sort(edges.begin(), edges.end(), [&](const EdgeAndNode& a, const EdgeAndNode& b) {
        float score_a = CalculatePUCTScore(params_, node, a, is_root, draw_score);
        float score_b = CalculatePUCTScore(params_, node, b, is_root, draw_score);
        bool a_nan = std::isnan(score_a); bool b_nan = std::isnan(score_b);
        if (a_nan && b_nan) return false; if (a_nan) return false; if (b_nan) return true;
        return score_a > score_b;
      });

  auto print = [](auto* oss, auto pre, auto v, auto post, auto w, int p = 0) { *oss << pre << std::setw(w) << std::setprecision(p) << v << post; };
  auto print_head = [&](auto* oss, auto label, int i, auto n, auto f, auto p) {
    *oss << std::fixed; print(oss, "", label, " ", 5); print(oss, "(", i, ") ", 4);
    *oss << std::right; print(oss, "N: ", n, " ", 7); print(oss, "(+", f, ") ", 2);
    print(oss, "(P: ", p * 100, "%) ", 5, 2);
   };
  auto print_stats = [&](auto* oss, const auto* n) {
    const auto sign = n == node ? -1 : 1;
    if (n) {
      print(oss, "(WGT: ", n->GetWeight(), ") ", 11, 3); print(oss, "(WL: ", sign * n->GetWL(), ") ", 8, 5);
      print(oss, "(D: ", n->GetD(), ") ", 5, 3); print(oss, "(M: ", n->GetM(), ") ", 4, 1);
      print(oss, "(STD: ", ComputeStdev(params_, n->GetWL(), n->GetWeight(), n->GetVS()), ") ", 6, 5);
      print(oss, "(STDF: ", ComputeStdevFactor(params_, n->GetWL(), n->GetWeight(), n->GetVS()), ") ", 6, 5);
      print(oss, "(UNCF: ", ComputeUncertaintyFactor(params_, n->GetE()), ") ", 6, 5);
      print(oss, "(VS: ", n->GetVS(), ") ", 6, 5); print(oss, "(E: ", n->GetE(), ") ", 6, 5);
      LowNode* low_node = n->GetLowNode();
      if (low_node != nullptr && dag_) {
        CorrHistEntry* cht_entry = dag_->CHTGetOrCreate(low_node->GetCHHash());
        if(cht_entry) {
            print(oss, "(CHW: ", cht_entry->weightSum, ") ", 6, 5);
            print(oss, "(CHD: ", -sign * cht_entry->deltaSum / (cht_entry->weightSum + 1e-6f), ") ", 6, 5);
            print(oss, "(CHN: ", cht_entry->numMembers, ") ", 6);
        }
      } print(oss, "(V: ", sign * n->GetV(), ") ", 6, 5);
    } else { *oss << "(WL:  -.-----) (D: -.---) (M:  -.-) "; }
    print(oss, "(Q: ", n ? sign * n->GetQ(sign * draw_score) : fpu, ") ", 8, 5);
   };
  auto print_tail = [&](auto* oss, const auto* n) {
     const auto sign = n == node ? -1 : 1;
     if (n) {
       auto [lo, up] = n->GetBounds();
       if (sign == -1) { lo = -lo; up = -up; std::swap(lo, up); }
       *oss << (lo == up ? "(T) " : (lo == GameResult::DRAW && up == GameResult::WHITE_WON ? "(W) " : (lo == GameResult::BLACK_WON && up == GameResult::DRAW ? "(L) " : "")));
     }
   };

  std::vector<std::string> infos;
  const auto m_evaluator = (network_ && network_->GetCapabilities().has_mlh()) ? MEvaluator(params_, node) : MEvaluator();
  for (const auto& edge : edges) {
    float Q = edge.GetQ(fpu, draw_score);
    float M = m_evaluator.GetMUtility(edge, Q);
    float U = edge.GetU(U_coeff);
    float S = Q + U + M;
    std::ostringstream oss; oss << std::left;
    print_head(&oss, edge.GetMove(is_black_to_move).as_string(), edge.GetMove().as_nn_index(0), edge.GetN(), edge.GetNInFlight(), edge.GetP());
    print_stats(&oss, edge.node());
    print(&oss, "(U: ", U, ") ", 6, 5); print(&oss, "(S: ", S, ") ", 8, 5);
    print_tail(&oss, edge.node());
    infos.emplace_back(oss.str());
  }

  std::ostringstream oss;
  print_head(&oss, "node ", node->GetNumEdges(), node->GetN(), node->GetNInFlight(), node->GetVisitedPolicy());
  print_stats(&oss, node);
  print_tail(&oss, node);
  oss << std::endl << "Low nodes: " << total_low_nodes_ << " NN queries: " << total_nn_queries_ << " Playouts: " << total_playouts_ + initial_visits_ << std::endl;
	print(&oss, "(U coeff: ", U_coeff, ") ", 15, 2);
  infos.emplace_back(oss.str());
  return infos;
}

// SendMovesStats Implementation
void Search::SendMovesStats() const {
    Mutex::Lock lock(counters_mutex_); // Lock needed to access final_bestmove_
    auto move_stats = GetVerboseStats(root_node_);

  if (params_.GetVerboseStats()) {
    std::vector<ThinkingInfo> infos;
    std::transform(move_stats.begin(), move_stats.end(), std::back_inserter(infos),
                   [](const std::string& line) { ThinkingInfo info; info.comment = line; return info; });
     if (uci_responder_) uci_responder_->OutputThinkingInfo(&infos);
  } else {
    LOGFILE << "=== Move stats:";
    for (const auto& line : move_stats) LOGFILE << line;
  }
  if (root_node_) {
      for (auto& edge : root_node_->Edges()) {
        if (!(edge.GetMove(played_history_.IsBlackToMove()) == final_bestmove_)) continue;
        if (edge.HasNode()) {
          LOGFILE << "--- Opponent moves after: " << final_bestmove_.as_string();
          for (const auto& line : GetVerboseStats(edge.node())) LOGFILE << line;
        }
      }
  }
}

// GetCachedNNEval Implementation
NNCacheLock Search::GetCachedNNEval(const PositionHistory& history) const {
    if (!dag_) return NNCacheLock();
  const auto hash = dag_->GetHistoryHash(history);
  if (!cache_) return NNCacheLock();
  NNCacheLock nneval(cache_, hash);
  return nneval;
}

// --- SearchWorker Implementation ---

// Constructor
SearchWorker::SearchWorker(Search* search, const SearchParams& params, int id)
      : search_(search),
        history_(search_->played_history_), // Copy history
        params_(params),
        moves_left_support_(search_->network_ && // Null check network
                            search_->network_->GetCapabilities().moves_left !=
                            pblczero::NetworkFormat::MOVES_LEFT_NONE) {
    if (search_ && search_->network_) search_->network_->InitThread(id);
    for (int i = 0; i < params.GetTaskWorkersPerSearchWorker(); i++) {
      task_workspaces_.emplace_back();
      task_threads_.emplace_back([this, i]() { this->RunTasks(i); });
    }
}

// Destructor
SearchWorker::~SearchWorker() {
    {
      task_count_.store(-1, std::memory_order_release);
      Mutex::Lock lock(picking_tasks_mutex_);
      exiting_ = true;
      task_added_.notify_all();
    }
    for (size_t i = 0; i < task_threads_.size(); i++) {
      if(task_threads_[i].joinable()) task_threads_[i].join();
    }
}

// RunBlocking
void SearchWorker::RunBlocking() {
  LOGFILE << "Started search thread.";
  try {
    do {
      ExecuteOneIteration();
    } while (search_->IsSearchActive());
  } catch (std::exception& e) {
    std::cerr << "Unhandled exception in worker thread: " << e.what() << std::endl;
    abort();
  }
   LOGFILE << "Ended search thread.";
}

// ExecuteOneIteration
void SearchWorker::ExecuteOneIteration() {
  if (!search_ || !search_->network_) return;

  InitializeIteration(search_->network_->NewComputation());

  if (params_.GetMaxConcurrentSearchers() != 0) {
    std::unique_ptr<SpinHelper> spin_helper;
    if (params_.GetSearchSpinBackoff()) spin_helper = std::make_unique<ExponentialBackoffSpinHelper>();
    else spin_helper = std::make_unique<SpinHelper>();
    while (true) {
        if (search_->stop_.load(std::memory_order_acquire) && search_->GetTotalPlayouts() + search_->initial_visits_ > 0) return;
        int available = search_->pending_searchers_.load(std::memory_order_acquire);
        if (available == 0) { spin_helper->Wait(); continue; }
        if (search_->pending_searchers_.compare_exchange_weak(available, available - 1, std::memory_order_acq_rel)) break;
        else spin_helper->Backoff();
    }
  }

  GatherMinibatch();
  task_count_.store(-1, std::memory_order_release);
  search_->backend_waiting_counter_.fetch_add(1, std::memory_order_relaxed);
  CollectCollisions();

  if (params_.GetMaxConcurrentSearchers() != 0) {
    search_->pending_searchers_.fetch_add(1, std::memory_order_acq_rel);
  }

  RunNNComputation();
  search_->backend_waiting_counter_.fetch_add(-1, std::memory_order_relaxed);
  FetchMinibatchResults();
  DoBackupUpdate();
  UpdateCounters();

   if (params_.GetNpsLimit() > 0) {
    while (search_->IsSearchActive()) {
      int64_t time_since_first_batch_ms = 0;
      { Mutex::Lock lock(search_->counters_mutex_); time_since_first_batch_ms = search_->GetTimeSinceFirstBatch(); }
      if (time_since_first_batch_ms <= 0) time_since_first_batch_ms = search_->GetTimeSinceStart();
      if (time_since_first_batch_ms == 0) time_since_first_batch_ms = 1;
      auto nps = search_->GetTotalPlayouts() * 1e3f / time_since_first_batch_ms;
      if (nps > params_.GetNpsLimit()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
      else break;
    }
  }
}

// InitializeIteration
void SearchWorker::InitializeIteration(
    std::unique_ptr<NetworkComputation> computation) {
  computation_ = std::make_unique<CachingComputation>(
      std::move(computation), search_->network_->GetCapabilities().input_format,
      params_.GetHistoryFill(), search_->cache_);
  computation_->Reserve(params_.GetMiniBatchSize());
  minibatch_.clear();
  minibatch_.reserve(2 * params_.GetMiniBatchSize());
}

// GatherMinibatch
void SearchWorker::GatherMinibatch() {
    uint32_t minibatch_size = 0;
    int cur_n = 0;
    { SharedMutex::SharedLock lock(search_->nodes_mutex_); cur_n = search_->root_node_ ? search_->root_node_->GetN() : 0; }
    int64_t remaining_n = latest_time_manager_hints_.GetEstimatedRemainingPlayouts();
    uint32_t collisions_left = CalculateCollisionsLeft(std::min(static_cast<int64_t>(cur_n), remaining_n), params_);
    number_out_of_order_ = 0;
    int thread_count = search_->thread_count_.load(std::memory_order_acquire);

    while (minibatch_size < params_.GetMiniBatchSize() && number_out_of_order_ < params_.GetMaxOutOfOrderEvals()) {
        if (minibatch_size > 0 && computation_->GetCacheMisses() == 0) return;
        if (thread_count > 1 && minibatch_size > 0 && computation_->GetCacheMisses() > params_.GetIdlingMinimumWork() && thread_count - search_->backend_waiting_counter_.load(std::memory_order_relaxed) > params_.GetThreadIdlingThreshold()) return;

        int new_start = static_cast<int>(minibatch_.size());
        PickNodesToExtend(std::min({collisions_left, params_.GetMiniBatchSize() - minibatch_size, params_.GetMaxOutOfOrderEvals() - number_out_of_order_}));

        int non_collisions = 0;
        for (int i = new_start; i < static_cast<int>(minibatch_.size()); i++) { if (!minibatch_[i].IsCollision()) { ++non_collisions; ++minibatch_size; } }

        { SharedMutex::Lock lock(search_->nodes_mutex_);
          bool needs_wait = false; int ppt_start = new_start;
          if (params_.GetTaskWorkersPerSearchWorker() > 0 && non_collisions >= params_.GetMinimumWorkSizeForProcessing()) {
            const int num_tasks = std::clamp(non_collisions / params_.GetMinimumWorkPerTaskForProcessing(), 2, params_.GetTaskWorkersPerSearchWorker() + 1);
            int per_worker = (num_tasks > 0) ? (non_collisions / num_tasks) : non_collisions;
            needs_wait = true; ResetTasks(); int found = 0;
            for (int i = new_start; i < static_cast<int>(minibatch_.size()); i++) {
              if (minibatch_[i].IsCollision()) continue; ++found;
              if (per_worker > 0 && found == per_worker) {
                picking_tasks_.emplace_back(ppt_start, i + 1); task_count_.fetch_add(1, std::memory_order_acq_rel); ppt_start = i + 1; found = 0;
                if (picking_tasks_.size() == static_cast<size_t>(num_tasks - 1)) break;
              }
            }
             if (ppt_start < static_cast<int>(minibatch_.size()) && num_tasks > picking_tasks_.size() + 1) { picking_tasks_.emplace_back(ppt_start, static_cast<int>(minibatch_.size())); task_count_.fetch_add(1, std::memory_order_acq_rel); ppt_start = static_cast<int>(minibatch_.size()); }
          }
          ProcessPickedTask(ppt_start, static_cast<int>(minibatch_.size()));
          if (needs_wait) WaitForTasks();
        }

        bool some_ooo = false; for (int i = static_cast<int>(minibatch_.size()) - 1; i >= new_start; i--) { if (minibatch_[i].ooo_completed) { some_ooo = true; break; } }
        if (some_ooo) {
          SharedMutex::Lock lock(search_->nodes_mutex_);
          for (int i = static_cast<int>(minibatch_.size()) - 1; i >= new_start; i--) {
            if (minibatch_[i].IsCollision()) {
              for (auto it = ++(minibatch_[i].path.crbegin()); it != minibatch_[i].path.crend(); ++it) { auto* node = std::get<0>(*it); if (node) node->CancelScoreUpdate(minibatch_[i].multivisit); }
              minibatch_.erase(minibatch_.begin() + i);
            } else if (minibatch_[i].ooo_completed) {
              FetchSingleNodeResult(&minibatch_[i], minibatch_[i], 0); DoBackupUpdateSingleNode(minibatch_[i]);
              minibatch_.erase(minibatch_.begin() + i); --minibatch_size; ++number_out_of_order_;
            }
          }
        }

        for (size_t i = new_start; i < minibatch_.size(); i++) {
           if (!minibatch_[i].ShouldAddToInput()) continue;
           if (minibatch_[i].is_cache_hit) computation_->AddInputByHash(minibatch_[i].hash, std::move(minibatch_[i].lock));
           else computation_->AddInput(minibatch_[i].hash, minibatch_[i].history);
        }

        for (size_t i = new_start; i < minibatch_.size(); i++) {
            if (minibatch_[i].IsCollision()) {
                if (minibatch_[i].maxvisit > 0 && collisions_left > minibatch_[i].multivisit) {
                    SharedMutex::Lock lock(search_->nodes_mutex_);
                    int extra = std::min(minibatch_[i].maxvisit, collisions_left) - minibatch_[i].multivisit;
                    minibatch_[i].multivisit += extra;
                    for (auto it = ++(minibatch_[i].path.crbegin()); it != minibatch_[i].path.crend(); ++it) { auto* node = std::get<0>(*it); if (node) node->IncrementNInFlight(extra); }
                }
                if ((collisions_left -= minibatch_[i].multivisit) <= 0) return;
                if (search_->stop_.load(std::memory_order_acquire)) return;
            }
        }
    }
}

// CollectCollisions
void SearchWorker::CollectCollisions() {
  SharedMutex::Lock lock(search_->nodes_mutex_);
  for (const NodeToProcess& node_to_process : minibatch_) {
    if (node_to_process.IsCollision()) {
      search_->shared_collisions_.emplace_back(node_to_process.path,
                                               node_to_process.multivisit);
    }
  }
}

// RunNNComputation
void SearchWorker::RunNNComputation() {
  if(computation_) computation_->ComputeBlocking(params_.GetPolicySoftmaxTemp());
}

// FetchMinibatchResults
void SearchWorker::FetchMinibatchResults() {
  SharedMutex::Lock nodes_lock(search_->nodes_mutex_);
  int idx_in_computation = 0;
  for (auto& node_to_process : minibatch_) {
    FetchSingleNodeResult(&node_to_process, *computation_, idx_in_computation);
    if (node_to_process.ShouldAddToInput()) ++idx_in_computation;
  }
}

// DoBackupUpdate
void SearchWorker::DoBackupUpdate() {
  SharedMutex::Lock lock(search_->nodes_mutex_);
  bool work_done = number_out_of_order_ > 0;
  for (const NodeToProcess& node_to_process : minibatch_) {
    DoBackupUpdateSingleNode(node_to_process);
    if (!node_to_process.IsCollision()) {
      work_done = true;
    }
  }
  if (!work_done && search_->shared_collisions_.empty()) return; // Check shared collisions too
  search_->CancelSharedCollisions();
  search_->total_batches_ += 1;
}

// UpdateCounters
void SearchWorker::UpdateCounters() {
  search_->PopulateCommonIterationStats(&iteration_stats_);
  search_->MaybeTriggerStop(iteration_stats_, &latest_time_manager_hints_);
  search_->MaybeOutputInfo();

  bool work_done = number_out_of_order_ > 0;
  if (!work_done) {
    for (NodeToProcess& node_to_process : minibatch_) {
      if (!node_to_process.IsCollision()) {
        work_done = true;
        break;
      }
    }
  }
  if (!work_done) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

// RunTasks
void SearchWorker::RunTasks(int tid) {
  while (true) {
    PickTask* task = nullptr;
    int id = 0;
    { // Lock scope for task picking
        int spins = 0;
        while (true) {
            int nta = tasks_taken_.load(std::memory_order_acquire);
            int tc = task_count_.load(std::memory_order_acquire);
            if (nta < tc) {
                int val = 0;
                if (task_taking_started_.compare_exchange_weak(val, 1, std::memory_order_acq_rel, std::memory_order_relaxed)) {
                    nta = tasks_taken_.load(std::memory_order_acquire); tc = task_count_.load(std::memory_order_acquire);
                    if (nta < tc) {
                        id = tasks_taken_.fetch_add(1, std::memory_order_acq_rel);
                        if (id < picking_tasks_.size()) task = &picking_tasks_[id]; // Bounds check
                        else { /* Handle error: id out of bounds */ task_taking_started_.store(0, std::memory_order_release); continue; }
                        task_taking_started_.store(0, std::memory_order_release);
                        break;
                    } task_taking_started_.store(0, std::memory_order_release);
                } SpinloopPause(); spins = 0; continue;
            } else if (tc != -1) {
                spins++; if (spins >= 512) { std::this_thread::yield(); spins = 0; } else { SpinloopPause(); } continue;
            }
            spins = 0; Mutex::Lock lock(picking_tasks_mutex_);
            nta = tasks_taken_.load(std::memory_order_acquire); tc = task_count_.load(std::memory_order_acquire);
            if (tc != -1) continue;
            if (nta >= tc && exiting_) return;
            task_added_.wait(lock.get_raw());
            nta = tasks_taken_.load(std::memory_order_acquire); tc = task_count_.load(std::memory_order_acquire);
            if (nta >= tc && exiting_) return;
        }
    } // End lock scope
    if (task != nullptr) {
      switch (task->task_type) {
        case PickTask::kGathering: {
          PickNodesToExtendTask(task->start_path, task->collision_limit,
                                task->history, &(task->results),
                                &(task_workspaces_[tid]));
          break;
        }
        case PickTask::kProcessing: {
          ProcessPickedTask(task->start_idx, task->end_idx);
          break;
        }
      }
       // Ensure id is valid before accessing picking_tasks_ again
      if (id < picking_tasks_.size()) picking_tasks_[id].complete = true;
      completed_tasks_.fetch_add(1, std::memory_order_acq_rel);
    }
  }
}

// ResetTasks
void SearchWorker::ResetTasks() {
  task_count_.store(0, std::memory_order_release);
  tasks_taken_.store(0, std::memory_order_release);
  completed_tasks_.store(0, std::memory_order_release);
  picking_tasks_.clear();
  picking_tasks_.reserve(MAX_TASKS);
}

// WaitForTasks
int SearchWorker::WaitForTasks() {
  while (true) {
    int completed = completed_tasks_.load(std::memory_order_acquire);
    int todo = task_count_.load(std::memory_order_acquire);
    if (todo == completed) return completed;
    SpinloopPause();
  }
}

// PickNodesToExtend (wrapper)
void SearchWorker::PickNodesToExtend(int collision_limit) {
    ResetTasks();
    { Mutex::Lock lock(picking_tasks_mutex_); task_added_.notify_all(); }
    SharedMutex::Lock lock(search_->nodes_mutex_); // Use SharedMutex::Lock
    history_.Trim(search_->played_history_.GetLength());
    // Initial call starts at the root node
    PickNodesToExtendTask({std::make_tuple(search_->root_node_, 0, 0)},
                          collision_limit, history_, &minibatch_,
                          &main_workspace_);

    WaitForTasks(); // Wait for any spawned tasks
    // Collect results from spawned tasks
    for (int i = 0; i < static_cast<int>(picking_tasks_.size()); i++) {
        for (int j = 0; j < static_cast<int>(picking_tasks_[i].results.size()); j++) {
            minibatch_.emplace_back(std::move(picking_tasks_[i].results[j]));
        }
    }
}

// PickNodesToExtendTask (with beam filter integrated)
void SearchWorker::PickNodesToExtendTask(
    const BackupPath& path, int collision_limit, PositionHistory& history,
    std::vector<NodeToProcess>* receiver,
    TaskWorkspace* workspace) NO_THREAD_SAFETY_ANALYSIS {
    assert(!path.empty());
    auto [node, repetitions, moves_left] = path.back();
    if (!node) { LOGFILE << "Error: Null node at start of PickNodesToExtendTask"; return; }

    auto& vtp_buffer = workspace->vtp_buffer;
    auto& visits_to_perform = workspace->visits_to_perform;
    visits_to_perform.clear();
    auto& vtp_last_filled = workspace->vtp_last_filled;
    vtp_last_filled.clear();
    auto& current_path = workspace->current_path;
    current_path.clear();
    auto& full_path = workspace->full_path;
    full_path = path;

    if (receiver->capacity() < 30) receiver->reserve(receiver->size() + 30);

    std::array<float, 256> current_util;
    std::array<bool, 256> visited;
    std::array<float, 256> current_score;
    std::array<float, 256> current_weightstarted;
    constexpr int num_top = 8;
    std::array<float, num_top> top_utils;

    auto& cur_iters = workspace->cur_iters;
    Node::Iterator best_edge;
    Node::Iterator second_best_edge;
    const int64_t best_node_n = search_->current_best_edge_.GetN();

    int passed_off = 0;
    int completed_visits = 0;
    bool is_root_node = node == search_->root_node_;

    // Get beam state once per task invocation if root
    const std::vector<Move> current_beam = is_root_node ? search_->GetCurrentBeam() : std::vector<Move>();
    const bool beam_active_local = is_root_node && search_->IsBeamActive();

    const float even_draw_score = search_->GetDrawScore(false);
    const float odd_draw_score = search_->GetDrawScore(true);
    const auto& root_move_filter = search_->root_move_filter_;
    auto m_evaluator = moves_left_support_ ? MEvaluator(params_) : MEvaluator();
    int max_limit = std::numeric_limits<int>::max();

    current_path.push_back(-1);

    while (current_path.size() > 0) {
        assert(full_path.size() >= path.size());
        if (current_path.back() == -1) {
            int cur_limit = collision_limit;
            if (current_path.size() > 1) {
                size_t parent_level_idx = current_path.size() - 2;
                if (parent_level_idx < visits_to_perform.size() && current_path[parent_level_idx] >= 0 && current_path[parent_level_idx] < 256) {
                    cur_limit = (*visits_to_perform[parent_level_idx])[current_path[parent_level_idx]];
                } else { cur_limit = 0; }
            }

            if (ShouldStopPickingHere(node, is_root_node, repetitions)) {
                if (is_root_node) {
                    if (node->TryStartScoreUpdate()) {
                        cur_limit = std::max(0, cur_limit - 1);
                        receiver->push_back(NodeToProcess::Visit(full_path, history)); // Pass history copy
                        completed_visits++;
                    }
                }
                if (cur_limit > 0) {
                    int max_count = (cur_limit == collision_limit && path.size() == 1 && max_limit > cur_limit) ? max_limit : 0;
                    receiver->push_back(NodeToProcess::Collision(full_path, cur_limit, max_count));
                    completed_visits += cur_limit;
                }
                history.Pop(); full_path.pop_back();
                if (!full_path.empty()) { std::tie(node, repetitions, moves_left) = full_path.back(); is_root_node = (node == search_->root_node_); }
                else { node = nullptr; repetitions = 0; is_root_node = false; }
                current_path.pop_back();
                continue;
            }
            if (is_root_node) node->IncrementNInFlight(cur_limit);

            if (vtp_buffer.size() > 0) { visits_to_perform.push_back(std::move(vtp_buffer.back())); vtp_buffer.pop_back(); }
            else { visits_to_perform.push_back(std::make_unique<std::array<int, 256>>()); }
            vtp_last_filled.push_back(-1);

            int max_needed = node->GetNumEdges();
            for (int i = 0; i < max_needed; i++) { current_util[i] = std::numeric_limits<float>::lowest(); visited[i] = false; }
            for (int i = 0; i < num_top; i++) { top_utils[i] = -std::numeric_limits<float>::infinity(); }

            const float draw_score = (full_path.size() % 2 == 0) ? odd_draw_score : even_draw_score;
            m_evaluator.SetParent(node);
            const float policy_decay_factor = ComputePolicyDecayFactor(params_, node->GetN()); // Use N
            float visited_pol = 0.0f;

            for (Node* child : node->VisitedNodes()) {
                if (!child) continue; int index = child->Index(); if (index < 0 || index >= 256) continue;
                visited_pol += child->GetP(); float q = child->GetQ(draw_score);
                current_util[index] = q + m_evaluator.GetMUtility(child, q); visited[index] = true;
                for (int i = 0; i < num_top; i++) { if (q > top_utils[i]) { for (int j = num_top - 1; j > i; j--) top_utils[j] = top_utils[j - 1]; top_utils[i] = q; break; } }
            }

            const int num_boost_t1 = params_.GetTopPolicyNumBoost(); const int num_boost_t2 = params_.GetTopPolicyTierTwoNumBoost();
            const float min_policy_boost_util_t1 = (num_boost_t1 == 0 || !params_.GetUsePolicyBoosting()) ? std::numeric_limits<float>::infinity() : top_utils[num_boost_t1 - 1];
            const float min_policy_boost_util_t2 = (num_boost_t2 == 0 || !params_.GetUsePolicyBoosting()) ? std::numeric_limits<float>::infinity() : top_utils[num_boost_t2 - 1];
            const float policy_boost_t1 = params_.GetTopPolicyBoost(); const float policy_boost_t2 = params_.GetTopPolicyTierTwoBoost();
            const float fpu = GetFpu(params_, node, is_root_node, draw_score, visited_pol);

            for (int i = 0; i < max_needed; i++) { if (current_util[i] == std::numeric_limits<float>::lowest()) current_util[i] = fpu + m_evaluator.GetDefaultMUtility(); }

            const float puct_mult = ComputeExploreFactor(params_, node->GetWeight(), node->GetWL(), node->GetVS(), node->GetE(), is_root_node);
            int cache_filled_idx = -1;

            while (cur_limit > 0) {
                float best = std::numeric_limits<float>::lowest(); int best_idx = -1;
                float best_without_u = std::numeric_limits<float>::lowest(); float second_best = std::numeric_limits<float>::lowest();
                bool can_exit = false; best_edge.Reset(); second_best_edge.Reset();

                for (int idx = 0; idx < max_needed; ++idx) {
                    if (idx > cache_filled_idx) {
                        if (cache_filled_idx == -1) cur_iters[idx] = node->Edges();
                        else { cur_iters[idx] = cur_iters[cache_filled_idx]; if (cur_iters[idx]) ++cur_iters[idx]; }
                        if (!cur_iters[idx]) break;
                        current_weightstarted[idx] = cur_iters[idx].GetWeightStarted();
                    } else if (!cur_iters[idx]) break;

                    // <<< --- BEAM SEARCH FILTER --- >>>
                    if (is_root_node && beam_active_local) {
                        bool in_beam = false; Move current_move = cur_iters[idx].GetMove();
                        for(const auto& beam_move : current_beam) { if(current_move == beam_move) { in_beam = true; break; } }
                        if (!in_beam) { if (idx > cache_filled_idx) cache_filled_idx++; continue; }
                    }
                    // <<< --- END BEAM FILTER --- >>>

                    float weightstarted = current_weightstarted[idx]; const float util = current_util[idx];
                    if (idx > cache_filled_idx) {
                        float p = cur_iters[idx].GetP(); p = ComputePolicyDecay(policy_decay_factor, p);
                        if (visited[idx]) {
                            if (util >= min_policy_boost_util_t1) p = std::max(p, policy_boost_t1);
                            if (util >= min_policy_boost_util_t2) p = std::max(p, policy_boost_t2);
                        }
                        current_score[idx] = p * puct_mult / (1 + weightstarted) + util;
                        cache_filled_idx++;
                    }

                    if (is_root_node) { /* ... root pruning logic ... */
                        if (cur_iters[idx] != search_->current_best_edge_ && latest_time_manager_hints_.GetEstimatedRemainingPlayouts() < best_node_n - cur_iters[idx].GetN()) continue;
                        if (!root_move_filter.empty() && std::find(root_move_filter.begin(), root_move_filter.end(), cur_iters[idx].GetMove()) == root_move_filter.end()) continue;
                    }

                    float score = current_score[idx];
                    if (score > best) { second_best = best; second_best_edge = best_edge; best = score; best_idx = idx; best_without_u = util; best_edge = cur_iters[idx]; }
                    else if (score > second_best) { second_best = score; second_best_edge = cur_iters[idx]; }
                    if (can_exit) break;
                    if (weightstarted == 0) can_exit = true;
                }

                if (best_idx == -1) { cur_limit = 0; continue; }

                int new_visits = 0;
                if (second_best_edge) { /* ... calculate new_visits based on score diff ... */
                    int estimated_visits_to_change_best = std::numeric_limits<int>::max();
                    if (best_without_u < second_best && (second_best - best_without_u) > 1e-9f) {
                        const auto n1 = current_weightstarted[best_idx] + 1; float numerator = cur_iters[best_idx].GetP() * puct_mult;
                        if (numerator > 0) estimated_visits_to_change_best = static_cast<int>(std::max(1.0f, std::min(numerator / (second_best - best_without_u) - n1 + 1, 1e9f)));
                        else estimated_visits_to_change_best = 1;
                    } else if (best_without_u >= second_best) estimated_visits_to_change_best = 1;
                    max_limit = std::min(max_limit, estimated_visits_to_change_best);
                    new_visits = std::min(cur_limit, estimated_visits_to_change_best);
                } else { new_visits = cur_limit; }
                if (cur_limit > 0 && new_visits <= 0) new_visits = 1;
                new_visits = std::min(new_visits, cur_limit);

                 if (best_idx < 0 || best_idx >= 256 || visits_to_perform.empty() || best_idx >= visits_to_perform.back()->size()) { LOGFILE << "Error: Invalid index/state in VTP update"; break;}
                 if (vtp_last_filled.empty()) { LOGFILE << "Error: vtp_last_filled empty"; break; }
                 auto* vtp_array = visits_to_perform.back()->data();
                 if (best_idx > vtp_last_filled.back()) { std::fill(vtp_array + (vtp_last_filled.back() + 1), vtp_array + best_idx + 1, 0); vtp_last_filled.back() = best_idx; }
                 else if (best_idx < vtp_last_filled.back() && vtp_last_filled.back() >= 0) { /* Already filled */ }
                 else if (best_idx == vtp_last_filled.back()) { /* Already filled */ }
                 else { vtp_last_filled.back() = best_idx; } // If last_filled was -1

                (*visits_to_perform.back())[best_idx] += new_visits;
                cur_limit -= new_visits;

                Node* child_node = best_edge.GetOrSpawnNode(node);
                history.Append(best_edge.GetMove());
                auto [child_repetitions, child_moves_left] = GetRepetitions(full_path.size(), history.Last());
                full_path.push_back({child_node, child_repetitions, child_moves_left});

                if (child_node->TryStartScoreUpdate()) {
                    current_weightstarted[best_idx]++; new_visits -= 1; if (new_visits < 0) new_visits = 0;
                    if (ShouldStopPickingHere(child_node, false, child_repetitions)) {
                        if ((*visits_to_perform.back())[best_idx] > 0) (*visits_to_perform.back())[best_idx] -= 1;
                        receiver->push_back(NodeToProcess::Visit(full_path, history)); completed_visits++;
                    } else {
                        child_node->IncrementNInFlight(new_visits); current_weightstarted[best_idx] += new_visits;
                        current_score[best_idx] = cur_iters[best_idx].GetP() * puct_mult / (1 + current_weightstarted[best_idx]) + current_util[best_idx];
                    }
                } else {
                     child_node->IncrementNInFlight(new_visits); current_weightstarted[best_idx] += new_visits;
                     current_score[best_idx] = cur_iters[best_idx].GetP() * puct_mult / (1 + current_weightstarted[best_idx]) + current_util[best_idx];
                }

                history.Pop(); full_path.pop_back();
            } // End while cur_limit > 0

             is_root_node = false; // Reset after first level
            // Task splitting logic (remains the same)
             if (!vtp_last_filled.empty()) { // Check if empty before accessing back()
                for (int i = 0; i <= vtp_last_filled.back(); i++) {
                    if (!cur_iters[i] || visits_to_perform.empty()) continue; // Safety checks
                    int child_limit = (*visits_to_perform.back())[i];
                    if (params_.GetTaskWorkersPerSearchWorker() > 0 && child_limit > params_.GetMinimumWorkSizeForPicking() && /* ... rest of condition ... */
                        child_limit < ((collision_limit - passed_off - completed_visits) * 2 / 3) &&
                        child_limit + passed_off + completed_visits < collision_limit - params_.GetMinimumRemainingWorkSizeForPicking())
                     {
                      Node* child_node = cur_iters[i].GetOrSpawnNode(node);
                      history.Append(cur_iters[i].GetMove()); auto [child_repetitions, child_moves_left] = GetRepetitions(full_path.size(), history.Last());
                      full_path.push_back({child_node, child_repetitions, child_moves_left});
                      if (!ShouldStopPickingHere(child_node, false, child_repetitions)) {
                        bool passed = false; { Mutex::Lock lock(picking_tasks_mutex_); if (picking_tasks_.size() < MAX_TASKS) { picking_tasks_.emplace_back(full_path, history, child_limit); task_count_.fetch_add(1, std::memory_order_acq_rel); task_added_.notify_all(); passed = true; passed_off += child_limit; } }
                        if (passed) (*visits_to_perform.back())[i] = 0;
                      } history.Pop(); full_path.pop_back();
                    }
                }
            }
        } // End if current_path.back() == -1

        // Select next child or backtrack (remains the same)
        int min_idx = current_path.back(); bool found_child = false;
        if (!vtp_last_filled.empty() && vtp_last_filled.back() > min_idx) {
            int idx = -1;
            for (auto& child : node->Edges()) {
                idx++; if (idx >= 256) break;
                if (idx > min_idx && !visits_to_perform.empty() && (*visits_to_perform.back())[idx] > 0) {
                    current_path.back() = idx; current_path.push_back(-1);
                    node = child.GetOrSpawnNode(node); history.Append(child.GetMove());
                    std::tie(repetitions, moves_left) = GetRepetitions(full_path.size(), history.Last());
                    full_path.push_back({node, repetitions, moves_left});
                    found_child = true; is_root_node = false;
                    break;
                }
                if (!vtp_last_filled.empty() && idx >= vtp_last_filled.back()) break;
            }
        }
        if (!found_child) {
            history.Pop(); full_path.pop_back();
            if (!full_path.empty()) { std::tie(node, repetitions, moves_left) = full_path.back(); is_root_node = (node == search_->root_node_); }
            else { node = nullptr; repetitions = 0; is_root_node = false; }
            current_path.pop_back();
            if (!visits_to_perform.empty()) { vtp_buffer.push_back(std::move(visits_to_perform.back())); visits_to_perform.pop_back(); }
            if (!vtp_last_filled.empty()) { vtp_last_filled.pop_back(); }
        }
    } // End while current_path.size() > 0
} // End PickNodesToExtendTask


// --- Implementations for other SearchWorker methods ---
// (Need full implementations based on uwuplant/lc0)
bool SearchWorker::MaybeAdjustForTerminalOrTransposition(
    Node* n, const LowNode* nl, float& v, float& d, float& m, float& vs,
    uint32_t& n_to_fix, float& weight_to_fix, float& v_delta, float& d_delta,
    float& m_delta, float& vs_delta, bool& update_parent_bounds) const {
     if (!n || !nl) return false;
     if (n->IsTerminal()) { v = n->GetWL(); d = n->GetD(); m = n->GetM(); vs = n->GetVS(); return true; }
     if (nl->IsTransposition() || nl->IsTerminal()) {
        v = -nl->GetWL(); d = nl->GetD(); m = nl->GetM() + 1; vs = nl->GetVS();
        n_to_fix = n->GetN(); weight_to_fix = n->GetWeight();
        v_delta = v - n->GetWL(); d_delta = d - n->GetD(); m_delta = m - n->GetM(); vs_delta = vs - n->GetVS();
        if (params_.GetStickyEndgames()) {
          auto tt = nl->GetTerminalType();
          if (tt != Terminal::NonTerminal) {
            GameResult r = (v == 1.0f) ? GameResult::WHITE_WON : ((v == -1.0f) ? GameResult::BLACK_WON : GameResult::DRAW);
            n->MakeTerminal(r, m, tt); update_parent_bounds = true;
          } else { auto [lower, upper] = nl->GetBounds(); n->SetBounds(-upper, -lower); }
        }
        return true;
     } return false;
}

void SearchWorker::DoBackupUpdateSingleNode(const NodeToProcess& node_to_process) {
     if (node_to_process.IsCollision()) return;
     auto path = node_to_process.path;
     if (path.empty()) { LOGFILE << "Error: Empty path in DoBackupUpdateSingleNode"; return; }
     auto [n, nr, nm] = path.back();
     if (!n) { LOGFILE << "Error: Null node pointer at end of path in DoBackupUpdateSingleNode."; return; }
     bool update_parent_bounds = params_.GetStickyEndgames() && n->IsTerminal() && !n->GetN();
     auto nl = n->GetLowNode();
     float v = 0.0f, d = 0.0f, m = 0.0f, vs = 0.0f;
     uint32_t n_to_fix = 0; float weight_to_fix = 0.0f;
     float v_delta = 0.0f, d_delta = 0.0f, m_delta = 0.0f, vs_delta = 0.0f;
     bool use_correction_history = params_.GetUseCorrectionHistory();
     float ch_delta = 0.0f; CorrHistEntry* ntp_cht_entry = nullptr;
     if (use_correction_history && search_->dag_) {
         ntp_cht_entry = search_->dag_->CHTGetOrCreate(node_to_process.ch_hash);
         if (ntp_cht_entry && ntp_cht_entry->weightSum > 1e-9f) { ch_delta = ntp_cht_entry->deltaSum / ntp_cht_entry->weightSum; }
     }
     float avg_weight = nl ? ComputeWeight(params_, nl->GetE()) : (params_.GetUseUncertaintyWeighting() ? params_.GetUncertaintyWeightingCap() : 1.0f);
     if (nl) n->SetE(nl->GetE()); else n->SetE(-1.0f);
     if (!node_to_process.ShouldAddToInput()) avg_weight *= params_.GetEasyEvalWeightDecay();
     if (nl && nl->GetN() == 0) {
        float wl_corrected = nl->GetWL();
        if (use_correction_history && !nl->IsTwin() && !nl->IsTerminal()) { wl_corrected += params_.GetCorrectionHistoryLambda() * ch_delta; wl_corrected = std::clamp(wl_corrected, -1.0f, 1.0f); }
        nl->FinalizeScoreUpdate(wl_corrected, nl->GetD(), nl->GetM(), nl->GetVS(), node_to_process.multivisit, node_to_process.multivisit * avg_weight, false);
        if (ntp_cht_entry != nullptr && !nl->IsTwin()) { nl->SetCHTEntry(ntp_cht_entry); ntp_cht_entry->numMembers++; }
     }
     if (nr >= 2) { n->SetRepetition(); v = 0.0f; d = 1.0f; m = 1; vs = 0.0f; }
     else if (!MaybeAdjustForTerminalOrTransposition(n, nl, v, d, m, vs, n_to_fix, weight_to_fix, v_delta, d_delta, m_delta, vs_delta, update_parent_bounds)) {
        if (nl) { v = -nl->GetWL(); d = nl->GetD(); m = nl->GetM() + 1; vs = nl->GetVS(); }
        else { v = 0.0f; d = 1.0f; m = 1.0f; vs = 0.0f; LOGFILE << "Warning: LowNode missing in backup where expected."; }
     }
     for (auto it = path.crbegin(); it != path.crend(); ) {
        n = std::get<0>(*it); nr = std::get<1>(*it); nm = std::get<2>(*it);
        if (!n) { LOGFILE << "Error: Null node during backup."; break; }
        n->FinalizeScoreUpdate(v, d, m, vs, node_to_process.multivisit, node_to_process.multivisit * avg_weight);
        if (n_to_fix > 0 && !n->IsTerminal()) { n_to_fix = std::min(n_to_fix, n->GetN()); weight_to_fix = std::min(weight_to_fix, n->GetWeight()); n->AdjustForTerminal(v_delta, d_delta, m_delta, vs_delta, n_to_fix, weight_to_fix); }
        if (nr == 1 && !n->IsTerminal()) { n->SetRepetition(); v = 0.0f; d = 1.0f; m = nm + 1; vs = 0.0f; }
        if (n->IsRepetition()) { n_to_fix = 0; weight_to_fix = 0; }
        if (++it == path.crend()) break;
        auto [p, pr, pm] = *it; if (!p) { LOGFILE << "Error: Null parent during backup."; break; }
        LowNode* pl = p->GetLowNode(); if (!pl) { v = -v; v_delta = -v_delta; m++; continue; }
        if (pl->IsTerminal()) { v = pl->GetWL(); d = pl->GetD(); m = pl->GetM(); vs = pl->GetVS(); n_to_fix = 0; weight_to_fix = 0.0f; }
        pl->FinalizeScoreUpdate(v, d, m, vs, node_to_process.multivisit, node_to_process.multivisit * avg_weight);
        if (n_to_fix > 0) pl->AdjustForTerminal(v_delta, d_delta, m_delta, vs_delta, n_to_fix, weight_to_fix);
        bool old_update_parent_bounds = update_parent_bounds;
        update_parent_bounds = update_parent_bounds && p != search_->root_node_ && !pl->IsTerminal() && MaybeSetBounds(p, m, &n_to_fix, &weight_to_fix, &v_delta, &d_delta, &m_delta, &vs_delta);
        v = -v; v_delta = -v_delta; m++;
        MaybeAdjustForTerminalOrTransposition(p, pl, v, d, m, vs, n_to_fix, weight_to_fix, v_delta, d_delta, m_delta, vs_delta, update_parent_bounds);
        if (p == search_->root_node_ && search_->current_best_edge_) {
            if ((old_update_parent_bounds && n->IsTerminal()) || (n != search_->current_best_edge_.node() && search_->current_best_edge_.GetWeight() <= n->GetWeight())) {
                 search_->current_best_edge_ = search_->GetBestChildNoTemperature(search_->root_node_, 0);
            }
            if (search_->beam_active_ && !search_->current_best_edge_ && search_->root_node_->GetN() > 0) {
                 search_->current_best_edge_ = search_->GetBestChildNoTemperature(search_->root_node_, 0);
            }
        }
     }
     search_->total_playouts_ += node_to_process.multivisit;
     search_->cum_depth_ += node_to_process.path.size() * node_to_process.multivisit;
     search_->max_depth_ = std::max(search_->max_depth_, (uint16_t)node_to_process.path.size());
     if (!node_to_process.is_tt_hit) search_->total_low_nodes_++;
     if (node_to_process.ShouldAddToInput()) search_->total_nn_queries_++;
}

bool SearchWorker::MaybeSetBounds(Node* p, float m, uint32_t* n_to_fix,
                                  float* weight_to_fix, float* v_delta,
                                  float* d_delta, float* m_delta,
                                  float* vs_delta) const {
    auto losing_m = 0.0f; bool prefer_tb = false;
    auto lower = GameResult::BLACK_WON; auto upper = GameResult::BLACK_WON;
    if (!p) return false;
    for (const auto& edge : p->Edges()) {
        if (!edge.HasNode()) continue;
        const auto [edge_lower, edge_upper] = edge.GetBounds();
        lower = std::max(edge_lower, lower); upper = std::max(edge_upper, upper);
        const auto is_tb = edge.IsTbTerminal();
        if (edge_lower == GameResult::WHITE_WON && !is_tb) { prefer_tb = false; break; }
        else if (edge_upper == GameResult::BLACK_WON) { losing_m = std::max(losing_m, edge.GetM(0.0f)); }
        prefer_tb = prefer_tb || is_tb;
    }
    auto pl = p->GetLowNode(); if (!pl) return false;
    if (lower == GameResult::BLACK_WON && upper == GameResult::WHITE_WON) return false;
    else if (lower == upper) {
        *n_to_fix = p->GetN(); *weight_to_fix = p->GetWeight();
        if (*n_to_fix == 0 || *weight_to_fix <= 0.0f) return false;
        pl->MakeTerminal(upper, (upper == GameResult::BLACK_WON ? std::max(losing_m, m) : m), prefer_tb ? Terminal::Tablebase : Terminal::EndOfGame);
        *v_delta = pl->GetWL() + p->GetWL(); *d_delta = pl->GetD() - p->GetD(); *m_delta = pl->GetM() + 1 - p->GetM(); *vs_delta = pl->GetVS() - p->GetVS();
        p->MakeTerminal(-upper, (upper == GameResult::BLACK_WON ? std::max(losing_m, m) : m) + 1.0f, prefer_tb ? Terminal::Tablebase : Terminal::EndOfGame);
    } else { pl->SetBounds(lower, upper); p->SetBounds(-upper, -lower); }
    return true;
}

// ExtendNode
void SearchWorker::ExtendNode(NodeToProcess& picked_node) {
     const auto path = picked_node.path;
     if (path.empty() || !std::get<0>(path.back())) { LOGFILE << "Error: Invalid path/node in ExtendNode"; return; }
     assert(!std::get<0>(path.back())->GetLowNode());
     const PositionHistory& history = picked_node.history;
     const auto& board = history.Last().GetBoard();
     std::vector<Move> legal_moves = board.GenerateLegalMoves();
     auto node = picked_node.node;
     if (legal_moves.empty()) {
        if (board.IsUnderCheck()) node->MakeTerminal(GameResult::WHITE_WON);
        else node->MakeTerminal(GameResult::DRAW);
        return;
     }
     if (node != search_->root_node_) {
        if (!board.HasMatingMaterial()) { node->MakeTerminal(GameResult::DRAW); return; }
        if (history.Last().GetRule50Ply() >= 100) { node->MakeTerminal(GameResult::DRAW); return; }
        if (picked_node.repetitions >= 2) { node->SetRepetition(); node->SetBounds(GameResult::DRAW, GameResult::DRAW); }
        else if (search_->syzygy_tb_ && !search_->root_is_in_dtz_ && board.castlings().no_legal_castle() && history.Last().GetRule50Ply() == 0 && (board.ours() | board.theirs()).count() <= search_->syzygy_tb_->max_cardinality()) {
           ProbeState state; const WDLScore wdl = search_->syzygy_tb_->probe_wdl(history.Last(), &state);
           if (state != FAIL) {
             float m = 0.0f; if (path.size() > 1) { auto parent = std::get<0>(path[path.size() - 2]); if(parent) m = std::max(0.0f, parent->GetM() - 1.0f); }
             if (wdl == WDL_WIN) node->MakeTerminal(GameResult::BLACK_WON, m, Terminal::Tablebase);
             else if (wdl == WDL_LOSS) node->MakeTerminal(GameResult::WHITE_WON, m, Terminal::Tablebase);
             else node->MakeTerminal(GameResult::DRAW, m, Terminal::Tablebase);
             search_->tb_hits_.fetch_add(1, std::memory_order_acq_rel); return;
           }
        }
     }
      if (node->IsRepetition() && !node->IsTerminal()) { picked_node.nn_queried = false; return; }
      if (!node->IsTerminal()) {
          picked_node.nn_queried = true;
          picked_node.hash = search_->dag_->GetHistoryHash(history);
          picked_node.ch_hash = search_->dag_->GetCHHash(history);
          auto tt_low_node = search_->dag_->TTFind(picked_node.hash);
          if (tt_low_node != nullptr) { picked_node.tt_low_node = tt_low_node; picked_node.is_tt_hit = true; }
          else {
             if (params_.GetMoveRuleBucketing()) {
                int my_ply = picked_node.GetRule50Ply(); int bs;
                if (my_ply <= 64) bs = 32; else if (my_ply <= 80) bs = 16; else if (my_ply <= 92) bs = 4; else bs = 1;
                int ply_lo = my_ply / bs * bs; int ply_hi = ply_lo + bs - 1;
                int max_visits = 0; LowNode* twin_low_node = nullptr;
                for (int ply = ply_lo; ply <= ply_hi; ply++) {
                    uint64_t hash = search_->dag_->GetHistoryHash(history, ply);
                    auto low_node = search_->dag_->TTFind(hash);
                    if (low_node != nullptr) { int visits = low_node->GetN(); if (visits > max_visits) { max_visits = visits; twin_low_node = low_node; } }
                }
                if (twin_low_node != nullptr) { picked_node.twin_low_node = twin_low_node; picked_node.is_twin_hit = true; }
             }
             picked_node.lock = NNCacheLock(search_->cache_, picked_node.hash);
             picked_node.is_cache_hit = picked_node.lock;
          }
       } else { picked_node.nn_queried = false; }
}

// GetRepetitions
std::pair<int, int> SearchWorker::GetRepetitions(int depth, const Position& position) {
    const auto repetitions = position.GetRepetitions();
    if (repetitions == 0) return {0, 0};
    if (repetitions >= 2) return {repetitions, 0};
    const auto plies = position.GetPliesSincePrevRepetition();
    if (params_.GetTwoFoldDraws() && depth >= 4 && depth >= plies) return {1, plies};
    return {0, 0};
}

// ShouldStopPickingHere
bool SearchWorker::ShouldStopPickingHere(Node* node, bool is_root_node, int repetitions) {
    constexpr double wl_diff_limit = 0.01; constexpr float d_diff_limit = 0.01f; constexpr float m_diff_limit = 2.0f;
    if (!node) return true;
    if (node->GetN() == 0 || node->IsTerminal()) return true;
    if (is_root_node) return false; // Assert handled externally
    if (repetitions >= 2) return true;
    auto low_node = node->GetLowNode(); if (!low_node) return false;
    if (!low_node->IsTransposition()) return false;
    if (low_node->IsTerminal()) return true;
    auto [low_node_lower, low_node_upper] = low_node->GetBounds(); auto [node_lower, node_upper] = node->GetBounds();
    if (low_node_lower != -node_upper || low_node_upper != -node_lower) return true;
    auto wl_diff = std::abs(static_cast<double>(low_node->GetWL()) + node->GetWL()); if (wl_diff >= wl_diff_limit) return true;
    auto d_diff = std::abs(low_node->GetD() - node->GetD()); if (d_diff >= d_diff_limit) return true;
    auto m_diff = std::abs(low_node->GetM() + 1 - node->GetM()); if (m_diff >= m_diff_limit) return true;
    return false;
}

// ProcessPickedTask
void SearchWorker::ProcessPickedTask(int start_idx, int end_idx) {
  for (int i = start_idx; i < end_idx; i++) {
    auto& picked_node = minibatch_[i];
    if (picked_node.IsCollision()) continue;
    if (picked_node.IsExtendable()) ExtendNode(picked_node);
    picked_node.ooo_completed = params_.GetOutOfOrderEval() && picked_node.CanEvalOutOfOrder();
  }
}

// FetchSingleNodeResult
template <typename Computation>
void SearchWorker::FetchSingleNodeResult(NodeToProcess* node_to_process,
                                         const Computation& computation,
                                         int idx_in_computation) {
    if (!node_to_process || !node_to_process->node) return; // Null checks
    if (!node_to_process->nn_queried) return;
    Node* node = node_to_process->node;
    if (node->IsTerminal() && !node->GetLowNode()) return; // Already terminal, nothing to fetch

    if (!node_to_process->is_tt_hit) {
        if (node_to_process->is_twin_hit) {
            if (!node_to_process->twin_low_node) { LOGFILE << "Error: Twin hit with null twin_low_node"; return; }
            LowNode twin_low_node = *(node_to_process->twin_low_node);
            auto [tt_low_node, is_tt_miss] = search_->dag_->TTGetOrCreate(twin_low_node, node_to_process->hash);
            if (!tt_low_node) { LOGFILE << "Error: Failed to get/create TT node for twin."; return; }
            tt_low_node->MakeTwin(); node_to_process->tt_low_node = tt_low_node;
        } else {
            auto [tt_low_node, is_tt_miss] = search_->dag_->TTGetOrCreate(node_to_process->hash);
            if (!tt_low_node) { LOGFILE << "Error: Failed to get/create TT node."; return; }
            node_to_process->tt_low_node = tt_low_node;
            if (is_tt_miss) {
                auto nn_eval_ptr = computation.GetNNEval(idx_in_computation);
                if(nn_eval_ptr) {
                    auto nn_eval = nn_eval_ptr.get();
                    if (params_.GetWDLRescaleRatio() != 1.0f || (params_.GetWDLRescaleDiff() != 0.0f && search_->contempt_mode_ != ContemptMode::NONE)) {
                        bool root_stm = search_->contempt_mode_ == ContemptMode::WHITE;
                        auto sign = (root_stm ^ node_to_process->history.IsBlackToMove()) ? 1.0f : -1.0f;
                        float v = nn_eval->q; float d = nn_eval->d;
                        WDLRescaleHelper(v, d, params_.GetWDLRescaleRatio(), search_->contempt_mode_ == ContemptMode::NONE ? 0 : params_.GetWDLRescaleDiff(), sign, false, params_.GetWDLMaxS());
                        nn_eval->q = v; nn_eval->d = d;
                    }
                    node_to_process->tt_low_node->SetNNEval(nn_eval); node_to_process->tt_low_node->SetCHHash(node_to_process->ch_hash);
                } else { LOGFILE << "Error: Missing NNEval for index " << idx_in_computation; node->MakeTerminal(GameResult::DRAW); return; }
            }
        }
    }

    if (node_to_process->tt_low_node) {
        if (params_.GetNoiseEpsilon() && node == search_->root_node_) {
            auto low_node = search_->dag_->NonTTAddClone(*node_to_process->tt_low_node);
            if (!low_node) { LOGFILE << "Error: Failed to clone LowNode for noise."; return; }
            node->SetLowNode(low_node);
            ApplyDirichletNoise(node, params_.GetNoiseEpsilon(), params_.GetNoiseAlpha());
            node->SortEdges();
        } else { node->SetLowNode(node_to_process->tt_low_node); }
    } else { LOGFILE << "Error: tt_low_node is null in FetchSingleNodeResult for hash " << node_to_process->hash; node->MakeTerminal(GameResult::DRAW); }
}

}  // namespace lczero
