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

#include "mcts/search.h"

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
#include <vector>    // Added for beam search
#include <limits>    // Added for numeric_limits

#include "mcts/node.h"
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
      : enabled_{true},
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
    if (!enabled_ || !parent_within_threshold_) return 0.0f;
    const float child_m = child->GetM();
    float m = std::clamp(m_slope_ * (child_m - parent_m_), -m_cap_, m_cap_);
    m *= FastSign(-q);
    if (q_threshold_ > 0.0f && q_threshold_ < 1.0f) {
      // This allows a smooth M effect with higher q thresholds, which is
      // necessary for using MLH together with contempt.
      q = std::max(0.0f, (std::abs(q) - q_threshold_)) / (1.0f - q_threshold_);
    }
    m *= a_constant_ + a_linear_ * std::abs(q) + a_square_ * q * q;
    return m;
  }

  float GetMUtility(const EdgeAndNode& child, float q) const {
    if (!enabled_ || !parent_within_threshold_) return 0.0f;
    if (child.GetN() == 0) return GetDefaultMUtility();
    return GetMUtility(child.node(), q);
  }

  // The M utility to use for unvisited nodes.
  float GetDefaultMUtility() const { return 0.0f; }

 private:
  static bool WithinThreshold(const Node* parent, float q_threshold) {
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

}  // namespace

Search::Search(NodeTree* dag, Network* network,
               std::unique_ptr<UciResponder> uci_responder,
               const MoveList& searchmoves,
               std::chrono::steady_clock::time_point start_time,
               std::unique_ptr<SearchStopper> stopper, bool infinite,
               bool ponder, const OptionsDict& options, NNCache* cache,
               SyzygyTablebase* syzygy_tb)
    : ok_to_respond_bestmove_(!infinite && !ponder),
      stopper_(std::move(stopper)),
      root_node_(dag->GetCurrentHead()),
      cache_(cache),
      dag_(dag),
      syzygy_tb_(syzygy_tb),
      played_history_(dag->GetPositionHistory()),
      network_(network),
      params_(options),
      searchmoves_(searchmoves),
      start_time_(start_time),
      initial_visits_(root_node_->GetN()),
      root_move_filter_(MakeRootMoveFilter(
          searchmoves_, syzygy_tb_, played_history_,
          params_.GetSyzygyFastPlay(), &tb_hits_, &root_is_in_dtz_)),
      uci_responder_(std::move(uci_responder)) {
  if (params_.GetMaxConcurrentSearchers() != 0) {
    pending_searchers_.store(params_.GetMaxConcurrentSearchers(),
                             std::memory_order_release);
  }
  contempt_mode_ = params_.GetContemptMode();
  // Make sure the contempt mode is never "play" beyond this point.
  if (contempt_mode_ == ContemptMode::PLAY) {
    if (infinite) {
      // For infinite search disable contempt, only "white"/"black" make sense.
      contempt_mode_ = ContemptMode::NONE;
      // Issue a warning only if contempt mode would have an effect.
      if (params_.GetWDLRescaleDiff() != 0.0f) {
        std::vector<ThinkingInfo> info(1);
        info.back().comment =
            "WARNING: Contempt mode set to 'disable' as 'play' not supported "
            "for infinite search.";
        uci_responder_->OutputThinkingInfo(&info);
      }
    } else {
      // Otherwise set it to the root move's side, unless pondering.
      contempt_mode_ = played_history_.IsBlackToMove() != ponder
                           ? ContemptMode::BLACK
                           : ContemptMode::WHITE;
    }
  }
  // +++ Additions for Beam Search Features +++
  // Initialize beam state
  current_beam_.clear();
  current_beam_width_ = 0;
  last_beam_update_visits_ = 0;
  // Activate only if enabled (MaxWidth > 0) and threshold > 0
  next_beam_update_visits_ = (params_.GetRootBeamMaxWidth() > 0 && params_.GetRootBeamUpdateThreshold() > 0)
                              ? params_.GetRootBeamUpdateThreshold()
                              : 0;
  beam_active_ = false;
  // +++ End Additions +++
}

namespace {
void ApplyDirichletNoise(Node* node, float eps, double alpha) {
  float total = 0;
  std::vector<float> noise;

  for (int i = 0; i < node->GetNumEdges(); ++i) {
    float eta = Random::Get().GetGamma(alpha, 1.0);
    noise.emplace_back(eta);
    total += eta;
  }

  if (total < std::numeric_limits<float>::min()) return;

  int noise_idx = 0;
  for (const auto& child : node->Edges()) {
    auto* edge = child.edge();
    edge->SetP(edge->GetP() * (1 - eps) + eps * noise[noise_idx++] / total);
  }
}
}  // namespace

namespace {
// WDL conversion formula based on random walk model.
inline double WDLRescale(float& v, float& d, float wdl_rescale_ratio,
                         float wdl_rescale_diff, float sign, bool invert,
                         float wdl_max_s = 1.4f) { // Added wdl_max_s parameter
  if (invert) {
    wdl_rescale_diff = -wdl_rescale_diff;
    wdl_rescale_ratio = 1.0f / wdl_rescale_ratio;
  }
  auto w = (1 + v - d) / 2;
  auto l = (1 - v - d) / 2;
  // Safeguard against numerical issues; skip WDL transformation if WDL is too
  // extreme.
  const float eps = 0.0001f;
  if (w > eps && d > eps && l > eps && w < (1.0f - eps) && d < (1.0f - eps) &&
      l < (1.0f - eps)) {
    auto a = FastLog(1 / l - 1);
    auto b = FastLog(1 / w - 1);
    auto s = 2 / (a + b);
    // Safeguard against unrealistically broad WDL distributions coming from
    // the NN. Use wdl_max_s parameter.
    if (!invert && wdl_max_s > 0.0f) s = std::min(wdl_max_s, s); // Check wdl_max_s > 0
    auto mu = (a - b) / (a + b);
    auto s_new = s * wdl_rescale_ratio;
    if (invert) {
      std::swap(s, s_new);
      if (wdl_max_s > 0.0f) s = std::min(wdl_max_s, s); // Apply max_s also when inverting
    }
    auto mu_new = mu + sign * s * s * wdl_rescale_diff;
    auto w_new = FastLogistic((-1.0f + mu_new) / s_new);
    auto l_new = FastLogistic((-1.0f - mu_new) / s_new);
    v = w_new - l_new;
    d = std::max(0.0f, 1.0f - w_new - l_new);
    return mu_new;
  }
  return 0;
}
}  // namespace

void Search::SendUciInfo() REQUIRES(nodes_mutex_) REQUIRES(counters_mutex_) {
  const auto max_pv = params_.GetMultiPv();
  // GetBestChildrenNoTemperature already respects root_move_filter_ if active.
  // If beam is active, we should probably show PVs only from the beam.
  std::vector<EdgeAndNode> edges;
  if (beam_active_) {
      std::vector<EdgeAndNode> all_edges;
       for (auto& edge : root_node_->Edges()) {
           all_edges.push_back(edge);
       }
       // Filter based on beam
       for (const auto& edge : all_edges) {
           bool in_beam = false;
           for (const auto& beam_move : current_beam_) {
               if (edge.GetMove() == beam_move) {
                   in_beam = true;
                   break;
               }
           }
           if (in_beam) {
               edges.push_back(edge);
           }
       }
       // Sort the filtered edges by visits (or another metric if desired for info)
        std::sort(edges.begin(), edges.end(), [](const auto& a, const auto& b){
            return a.GetN() > b.GetN(); // Sort by visits descending
        });
        // Limit to MultiPV after sorting within the beam
        if (edges.size() > max_pv) edges.resize(max_pv);

  } else {
      edges = GetBestChildrenNoTemperature(root_node_, max_pv, 0);
  }

  const auto score_type = params_.GetScoreType();
  const auto per_pv_counters = params_.GetPerPvCounters();
  const auto display_cache_usage = params_.GetDisplayCacheUsage();
  const auto draw_score = GetDrawScore(false);
  const float wdl_max_s = params_.GetWDLMaxS(); // Get max_s for WDL rescaling

  std::vector<ThinkingInfo> uci_infos;

  // Info common for all multipv variants.
  ThinkingInfo common_info;
  common_info.depth = cum_depth_ / (total_playouts_ ? total_playouts_ : 1);
  common_info.seldepth = max_depth_;
  common_info.time = GetTimeSinceStart();
  uint64_t total_nodes;
  std::string reported_nodes = params_.GetReportedNodes();
  if (reported_nodes == "nodes") {
    total_nodes = total_low_nodes_;
  } else if (reported_nodes == "queries") {
    total_nodes = total_nn_queries_;
  } else { // playouts or legacy
    total_nodes = total_playouts_;
  }

  if (!per_pv_counters) {
    // Can't use total_nodes since it won't carry over
    common_info.nodes = total_playouts_ + initial_visits_;
  }
  if (nps_start_time_) {
    const auto time_since_first_batch_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - *nps_start_time_)
            .count();
    if (time_since_first_batch_ms > 0) {
      common_info.nps = total_nodes * 1000 / time_since_first_batch_ms;
    }
  }
  if (display_cache_usage && cache_) { // Add null check for cache_
    common_info.hashfull =
        cache_->GetSize() * 1000LL / std::max(cache_->GetCapacity(), 1ULL); // Ensure 1ULL
  }
  common_info.tb_hits = tb_hits_.load(std::memory_order_acquire);

  int multipv = 0;
  const auto default_q = -root_node_->GetQ(-draw_score);
  const auto default_wl = -root_node_->GetWL();
  const auto default_d = root_node_->GetD();
  for (const auto& edge : edges) {
    ++multipv;
    uci_infos.emplace_back(common_info);
    auto& uci_info = uci_infos.back();
    auto wl = edge.GetWL(default_wl);
    auto d = edge.GetD(default_d);
    float mu_uci = 0.0f;
    if (score_type == "WDL_mu" || (params_.GetWDLRescaleDiff() != 0.0f &&
                                   contempt_mode_ != ContemptMode::NONE)) {
      auto sign = ((contempt_mode_ == ContemptMode::BLACK) ==
                   played_history_.IsBlackToMove())
                      ? 1.0f
                      : -1.0f;
      mu_uci = WDLRescale(
          wl, d, params_.GetWDLRescaleRatio(),
          contempt_mode_ == ContemptMode::NONE
              ? 0
              : params_.GetWDLRescaleDiff() * params_.GetWDLEvalObjectivity(),
          sign, true, wdl_max_s); // Pass wdl_max_s
    }
    const auto q = edge.GetQ(default_q, draw_score);
    if (edge.HasNode() && edge.IsTerminal() && wl != 0.0f) { // Check HasNode
      uci_info.mate = std::copysign(
          std::round(edge.GetM(0.0f) + 1) / 2 + (edge.IsTbTerminal() ? 100 : 0),
          wl);
    } else if (score_type == "centipawn_with_drawscore") {
      uci_info.score = 90 * tan(1.5637541897 * q);
    } else if (score_type == "centipawn") {
      uci_info.score = 90 * tan(1.5637541897 * wl);
    } else if (score_type == "centipawn_2019") {
      uci_info.score = 295 * wl / (1 - 0.976953126 * std::pow(wl, 14));
    } else if (score_type == "centipawn_2018") {
      uci_info.score = 290.680623072 * tan(1.548090806 * wl);
    } else if (score_type == "win_percentage") {
      uci_info.score = wl * 5000 + 5000;
    } else if (score_type == "Q") {
      uci_info.score = q * 10000;
    } else if (score_type == "W-L") {
      uci_info.score = wl * 10000;
    } else if (score_type == "WDL_mu") {
      // Reports the WDL mu value whenever it is reasonable, and defaults to
      // centipawn otherwise.
      const float centipawn_fallback_threshold = 0.996f;
      float centipawn_score = 90 * tan(1.5637541897 * wl);
      uci_info.score =
          mu_uci != 0.0f && std::abs(wl) + d < centipawn_fallback_threshold &&
                  (std::abs(mu_uci) < 1.0f ||
                   std::abs(centipawn_score) < std::abs(100 * mu_uci))
              ? 100 * mu_uci
              : centipawn_score;
    }

    auto wdl_w =
        std::max(0, static_cast<int>(std::round(500.0 * (1.0 + wl - d))));
    auto wdl_l =
        std::max(0, static_cast<int>(std::round(500.0 * (1.0 - wl - d))));
    // Using 1000-w-l so that W+D+L add up to 1000.0.
    auto wdl_d = 1000 - wdl_w - wdl_l;
    if (wdl_d < 0) {
      wdl_w = std::min(1000, std::max(0, wdl_w + wdl_d / 2));
      wdl_l = 1000 - wdl_w;
      wdl_d = 0;
    }
    uci_info.wdl = ThinkingInfo::WDL{wdl_w, wdl_d, wdl_l};
    if (network_->GetCapabilities().has_mlh()) {
       uci_info.moves_left = edge.HasNode() ? static_cast<int>((1.0f + edge.GetM(1.0f + root_node_->GetM())) / 2.0f) : 0; // Check HasNode
    }
    if (max_pv > 1) uci_info.multipv = multipv;
    if (per_pv_counters) uci_info.nodes = edge.GetN();
    bool flip = played_history_.IsBlackToMove();
    int depth = 0;
    auto history = played_history_;
    for (auto iter = edge; iter && iter.HasNode(); // Check iter.HasNode()
         iter = GetBestChildNoTemperature(iter.node(), depth), flip = !flip) {
      uci_info.pv.push_back(iter.GetMove(flip));
      history.Append(iter.GetMove());
      // Last edge was dangling or a draw by repetition, cannot continue.
      if (!iter.node() || history.Last().GetRepetitions() >= 2) break;
      depth += 1;
    }
  }

  if (!uci_infos.empty()) last_outputted_uci_info_ = uci_infos.front();
  if (current_best_edge_ && !edges.empty()) {
    // Ensure the last outputted edge exists in the current (possibly filtered) list
    bool found_edge = false;
    for(const auto& edge : edges) {
        if (edge.edge() == current_best_edge_.edge()) {
            last_outputted_info_edge_ = current_best_edge_.edge();
            found_edge = true;
            break;
        }
    }
    if (!found_edge) {
         // If the previous best edge is no longer in the top PVs (e.g., due to beam),
         // reset the last outputted edge to force an update next time.
         last_outputted_info_edge_ = nullptr;
    }

  } else {
     last_outputted_info_edge_ = nullptr;
  }


  uci_responder_->OutputThinkingInfo(&uci_infos);
}


// Decides whether anything important changed in stats and new info should be
// shown to a user.
void Search::MaybeOutputInfo() {
  SharedMutex::Lock nodes_lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  if (!bestmove_is_sent_ && current_best_edge_ &&
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
      uci_responder_->OutputThinkingInfo(&info);
    }
  }
}

int64_t Search::GetTimeSinceStart() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - start_time_)
      .count();
}

int64_t Search::GetTimeSinceFirstBatch() const REQUIRES(counters_mutex_) {
  if (!nps_start_time_) return 0;
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - *nps_start_time_)
      .count();
}

// Root is depth 0, i.e. even depth.
float Search::GetDrawScore(bool is_odd_depth) const {
  return (is_odd_depth == played_history_.IsBlackToMove()
              ? params_.GetDrawScore()
              : -params_.GetDrawScore());
}


namespace {


inline float ComputeUncertaintyFactor(const SearchParams& params, float e) {
  if (e < 0) return 1.0f; // Handle terminal case where e is -1
  float min_factor = params.GetCpuctUncertaintyMinFactor();
  float max_factor = params.GetCpuctUncertaintyMaxFactor();
  float min_uncertainty = params.GetCpuctUncertaintyMinUncertainty();
  float max_uncertainty = params.GetCpuctUncertaintyMaxUncertainty();
  e = std::clamp(e * e, min_uncertainty, max_uncertainty);
  float factor = min_factor + (max_factor - min_factor) * (e - min_uncertainty) /
                                  (max_uncertainty - min_uncertainty + 1e-5f); // Add f suffix
  return factor;
}

inline float ComputeStdev(const SearchParams& params, float q, float weight,
                          float vs) {
  if (weight <= 1e-9f) return 0.0f; // Avoid division by zero or negative weight, use f suffix
  float util_sq_avg = vs;
  const float util_sq = q * q;
  util_sq_avg = std::max(util_sq_avg,
                         util_sq);  // avoid negative variance

  const float var_estimate = util_sq_avg - util_sq;
  float stdev_estimate = sqrt(std::max(var_estimate, 0.0f));
  return stdev_estimate;
}

inline float ComputeStdevFactor(const SearchParams& params, float q,
                                float weight, float vs) {
  if (weight <= 1e-9f) return 1.0f; // Return neutral factor if no weight, use f suffix
  float util_sq_avg = vs;
  const float util_sq = q * q;
  util_sq_avg = std::max(util_sq_avg,
                         util_sq);  // avoid negative variance

  const float stdev_prior = params.GetCpuctUtilityStdevPrior();
  const float variance_prior = stdev_prior * stdev_prior;
  const float prior_weight = params.GetCpuctUtilityStdevPriorWeight();
  const float stdev_factor_scale = params.GetCpuctUtilityStdevScale();
  const float denominator = prior_weight + weight - 1.0f;

  // Avoid division by zero or near-zero denominator
  if (denominator <= 1e-9f) return 1.0f; // Use f suffix

  const float var_estimate =
      ((util_sq + variance_prior) * prior_weight + util_sq_avg * weight) / denominator - util_sq;


  const float stdev_estimate = sqrt(std::max(var_estimate, 0.0f));

  // Avoid division by zero if prior is zero
  if (stdev_prior <= 1e-9f) return 1.0f; // Use f suffix

  float stdev_factor =
      1.0f + stdev_factor_scale * (stdev_estimate / stdev_prior - 1.0f);
  return std::max(0.0f, stdev_factor); // Ensure non-negative factor
}

inline float ComputeDesperationFactor(const SearchParams& params, float q,
  float weight) {
  if (weight <= 1e-9f) return 1.0f; // Return neutral factor if no weight, use f suffix
  const float prior_weight = params.GetDesperationPriorWeight();
  const float low = params.GetDesperationLow(); const float high = params.GetDesperationHigh();

  q = abs(q);
  float factor = (q <= low || q >= high) ? params.GetDesperationMultiplier() : 1.0f;
  return 1.0f + (factor - 1.0f) * weight / (prior_weight + weight);


}

inline float ComputeStdevFactor(const SearchParams& params, Node* node) {
  return ComputeStdevFactor(params, node->GetWL(), node->GetWeight(),
                            node->GetVS());
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
	// we shouldn't push the value below -1
  return params.GetFpuAbsolute(is_root_node)
             ? value
             : fmax(-node->GetQ(-draw_score) -
                        value * std::sqrt(node->GetVisitedPolicy()), -1.0f);
}

// Faster version for if visited_policy is readily available already.
inline float GetFpu(const SearchParams& params, Node* node, bool is_root_node,
                    float draw_score, float visited_pol) {
  const auto value = params.GetFpuValue(is_root_node);
  return params.GetFpuAbsolute(is_root_node)
             ? value
             : fmax(-node->GetQ(-draw_score) -
                        value * std::sqrt(visited_pol), -1.0f);
}

inline float ComputeExploreFactor(const SearchParams& params, float weight, bool is_root_node) {
  const float init = params.GetCpuct(is_root_node);
  const float k = params.GetCpuctFactor(is_root_node);
  const float base = params.GetCpuctBase(is_root_node);
  const float exponent = params.GetCpuctExponent(is_root_node); // Use getter

  return (init + (k ? k * FastLog((weight + base) / base) : 0.0f)) *
         std::pow(fmax(weight, 1e-5f), exponent); // Ensure 1e-5f
}

inline float ComputeExploreFactor(const SearchParams& params, float weight, float q,
                          float vs, float e, bool is_root_node) {

  const float base_factor = ComputeExploreFactor(params, weight, is_root_node);

  const float extra_factor = ComputeCpuctFactor(params, weight, q, vs, e,
																					  is_root_node);

  return base_factor * extra_factor ;
}


inline float ComputeWeight(const SearchParams& params, float uncertainty) {
  if (!params.GetUseUncertaintyWeighting()) return 1.0f;
  if (uncertainty < 0) return 1.0f; // Handle terminal case where e is -1
  const float cap = params.GetUncertaintyWeightingCap();
  const float coefficient = params.GetUncertaintyWeightingCoefficient();
  const float exponent = params.GetUncertaintyWeightingExponent();
  // Ensure uncertainty is non-negative before pow
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

// Helper to calculate PUCT-like score for beam ranking
float CalculatePUCTScore(const SearchParams& params, Node* parent_node, const EdgeAndNode& child_edge, bool is_root_node, float draw_score) {
    // Use the same FPU logic as in selection
    float fpu = GetFpu(params, parent_node, is_root_node, draw_score);
    // Use the actual Q value if the node exists, otherwise use FPU
    float Q = child_edge.HasNode() ? child_edge.GetQ(draw_score) : fpu;
    // Use the full explore factor calculation
    float U_coeff = ComputeExploreFactor(params, parent_node->GetWeight(), parent_node->GetWL(), parent_node->GetVS(), parent_node->GetE(), is_root_node);
    // GetU uses NStarted which includes virtual losses, suitable for ranking
    return Q + child_edge.GetU(U_coeff);
}


}  // namespace

std::vector<std::string> Search::GetVerboseStats(Node* node) const {
  const bool is_root = (node == root_node_);
  const bool is_odd_depth = !is_root;
  const bool is_black_to_move = (played_history_.IsBlackToMove() == is_root);
  const float draw_score = GetDrawScore(is_odd_depth);
  const float fpu = GetFpu(params_, node, is_root, draw_score);
  const float U_coeff = ComputeExploreFactor(params_, node->GetWeight(), node->GetWL(),
                           node->GetVS(), node->GetE(), is_root);
  std::vector<EdgeAndNode> edges;
  for (const auto& edge : node->Edges()) {
        // +++ Beam Search Filter for Verbose Stats +++
        if (is_root && beam_active_) {
            bool in_beam = false;
            for (const auto& beam_move : current_beam_) {
                if (edge.GetMove() == beam_move) {
                    in_beam = true;
                    break;
                }
            }
            if (!in_beam) continue;
        }
        // +++ End Beam Search Filter +++
        edges.push_back(edge);
  }


  std::sort(
      edges.begin(), edges.end(),
      [&](const EdgeAndNode& a, const EdgeAndNode& b) { // Capture 'this' or pass params
        // Use the same PUCT score for sorting as for beam ranking
        float score_a = CalculatePUCTScore(params_, node, a, is_root, draw_score);
        float score_b = CalculatePUCTScore(params_, node, b, is_root, draw_score);
        return score_a > score_b; // Higher score is better
      });


  auto print = [](auto* oss, auto pre, auto v, auto post, auto w, int p = 0) {
    *oss << pre << std::setw(w) << std::setprecision(p) << v << post;
  };
  auto print_head = [&](auto* oss, auto label, int i, auto n, auto f, auto p) {
    *oss << std::fixed;
    print(oss, "", label, " ", 5);
    print(oss, "(", i, ") ", 4);
    *oss << std::right;
    print(oss, "N: ", n, " ", 7);
    print(oss, "(+", f, ") ", 2);
    print(oss, "(P: ", p * 100, "%) ", 5, 2); // Adjusted precision
  };
  auto print_stats = [&](auto* oss, const auto* n) {
    const auto sign = n == node ? -1 : 1;
    if (n) {
      print(oss, "(WGT: ", n->GetWeight(), ") ", 11, 3);
      print(oss, "(WL: ", sign * n->GetWL(), ") ", 8, 5);
      print(oss, "(D: ", n->GetD(), ") ", 5, 3);
      print(oss, "(M: ", n->GetM(), ") ", 4, 1);
      print(oss, "(STD: ",
            ComputeStdev(params_, n->GetWL(), n->GetWeight(), n->GetVS()), ") ",
            6, 5);
      print(oss, "(STDF: ",
            ComputeStdevFactor(params_, n->GetWL(), n->GetWeight(), n->GetVS()),
            ") ", 6, 5);
      print(oss, "(UNCF: ",
            ComputeUncertaintyFactor(params_, n->GetE()),
            ") ", 6, 5);
      print(oss, "(VS: ", n->GetVS(), ") ", 6, 5);
      print(oss, "(E: ", n->GetE(), ") ", 6, 5);
      LowNode* low_node = n->GetLowNode();
      if (low_node != nullptr && dag_) { // Add null check for dag_
        CorrHistEntry* cht_entry = dag_->CHTGetOrCreate(low_node->GetCHHash());
        if(cht_entry) { // Add null check for cht_entry
            print(oss, "(CHW: ", cht_entry->weightSum, ") ", 6, 5);
            print(oss,
                "(CHD: ", -sign * cht_entry->deltaSum / (cht_entry->weightSum + 0.0001f),
                ") ", 6, 5);
            print(oss, "(CHN: ", cht_entry->numMembers, ") ", 6);
        }
      }
      print(oss, "(V: ", sign * n->GetV(), ") ", 6, 5);
    } else {
      *oss << "(WL:  -.-----) (D: -.---) (M:  -.-) ";
    }
    print(oss, "(Q: ", n ? sign * n->GetQ(sign * draw_score) : fpu, ") ", 8, 5);
  };
  auto print_tail = [&](auto* oss, const auto* n) {
    const auto sign = n == node ? -1 : 1;

    if (n) {
      auto [lo, up] = n->GetBounds();
      if (sign == -1) {
        lo = -lo;
        up = -up;
        std::swap(lo, up);
      }
      *oss << (lo == up
                   ? "(T) "
                   : lo == GameResult::DRAW && up == GameResult::WHITE_WON
                         ? "(W) "
                         : lo == GameResult::BLACK_WON && up == GameResult::DRAW
                               ? "(L) "
                               : "");
    }
  };


  std::vector<std::string> infos;
  const auto m_evaluator = network_->GetCapabilities().has_mlh()
                               ? MEvaluator(params_, node)
                               : MEvaluator();
  for (const auto& edge : edges) {
    float Q = edge.GetQ(fpu, draw_score);
    float M = m_evaluator.GetMUtility(edge, Q);
    float U = edge.GetU(U_coeff); // Calculate U
    float S = Q + U + M; // Calculate total score S
    std::ostringstream oss;
    oss << std::left;
    print_head(&oss, edge.GetMove(is_black_to_move).as_string(),
               edge.GetMove().as_nn_index(0), edge.GetN(), edge.GetNInFlight(),
               edge.GetP());
    print_stats(&oss, edge.node());
    print(&oss, "(U: ", U, ") ", 6, 5); // Print U
    print(&oss, "(S: ", S, ") ", 8, 5); // Print S
    print_tail(&oss, edge.node());
    infos.emplace_back(oss.str());
  }

  // Include stats about the node in similar format to its children above.
  std::ostringstream oss;
  print_head(&oss, "node ", node->GetNumEdges(), node->GetN(),
             node->GetNInFlight(), node->GetVisitedPolicy());
  print_stats(&oss, node);
  print_tail(&oss, node);

  oss << std::endl << "Low nodes: " << total_low_nodes_
       << " NN queries: " << total_nn_queries_
       << " Playouts: " << total_playouts_ + initial_visits_ << std::endl;

	print(&oss, "(U coeff: ", U_coeff, ") ", 15, 2);

  infos.emplace_back(oss.str());
  return infos;
}

void Search::SendMovesStats() const REQUIRES(counters_mutex_) {
  auto move_stats = GetVerboseStats(root_node_);

  if (params_.GetVerboseStats()) {
    std::vector<ThinkingInfo> infos;
    std::transform(move_stats.begin(), move_stats.end(),
                   std::back_inserter(infos), [](const std::string& line) {
                     ThinkingInfo info;
                     info.comment = line;
                     return info;
                   });
    uci_responder_->OutputThinkingInfo(&infos);
  } else {
    LOGFILE << "=== Move stats:";
    for (const auto& line : move_stats) LOGFILE << line;
  }
  for (auto& edge : root_node_->Edges()) {
    if (!(edge.GetMove(played_history_.IsBlackToMove()) == final_bestmove_)) {
      continue;
    }
    if (edge.HasNode()) {
      LOGFILE << "--- Opponent moves after: " << final_bestmove_.as_string();
      for (const auto& line : GetVerboseStats(edge.node())) {
        LOGFILE << line;
      }
    }
  }
}

NNCacheLock Search::GetCachedNNEval(const PositionHistory& history) const {
  const auto hash = dag_->GetHistoryHash(history);
  NNCacheLock nneval(cache_, hash);
  return nneval;
}

void Search::MaybeTriggerStop(const IterationStats& stats,
                              StoppersHints* hints) {
  hints->Reset();
  if (params_.GetNpsLimit() > 0) {
    hints->UpdateEstimatedNps(params_.GetNpsLimit());
  }
  SharedMutex::Lock nodes_lock(nodes_mutex_);
  Mutex::Lock lock(counters_mutex_);
  // Already responded bestmove, nothing to do here.
  if (bestmove_is_sent_) return;
  // Don't stop when the root node is not yet expanded.
  if (total_playouts_ + initial_visits_ == 0) return;

  // Check if beam update threshold is met
  bool beam_updated = false;
  if (params_.GetRootBeamMaxWidth() > 0 && next_beam_update_visits_ > 0 && root_node_->GetN() >= next_beam_update_visits_) {
      UpdateRootBeam(); // Requires nodes_mutex_ and counters_mutex_
      beam_updated = true;
  }

  if (!stop_.load(std::memory_order_acquire)) {
    if (stopper_->ShouldStop(stats, hints)) FireStopInternal();
  }

  // If we are the first to see that stop is needed.
  if (stop_.load(std::memory_order_acquire) && ok_to_respond_bestmove_ &&
      !bestmove_is_sent_) {
    // Ensure beam is updated before final decision if threshold met and not already done
    if (!beam_updated && params_.GetRootBeamMaxWidth() > 0 && next_beam_update_visits_ > 0 && root_node_->GetN() >= next_beam_update_visits_) {
         UpdateRootBeam();
    }
    SendUciInfo();
    EnsureBestMoveKnown();
    SendMovesStats();
    BestMoveInfo info(final_bestmove_, final_pondermove_);
    uci_responder_->OutputBestMove(&info);
    stopper_->OnSearchDone(stats);
    bestmove_is_sent_ = true;
    current_best_edge_ = EdgeAndNode();
  } else if (beam_updated) {
      // If beam was updated, we might need to output info again if best move changed
      EnsureBestMoveKnown(); // Recalculate best move potentially based on new beam
      // Don't call SendUciInfo here directly, let MaybeOutputInfo handle it
  }
}


// Return the evaluation of the actual best child, regardless of temperature
// settings. This differs from GetBestMove, which does obey any temperature
// settings. So, somethimes, they may return results of different possible moves.
Eval Search::GetBestEval(Move* move, bool* is_terminal) const {
  SharedMutex::SharedLock lock(nodes_mutex_);
  // No counter mutex needed for read-only access if called externally
  // If called internally, ensure counters_mutex is held if necessary

  // Use GetBestChildrenNoTemperature which already considers the beam if active
  auto best_edges = GetBestChildrenNoTemperature(root_node_, 1, 0);
  EdgeAndNode best_edge = best_edges.empty() ? EdgeAndNode() : best_edges.front();

  float parent_wl = -root_node_->GetWL();
  float parent_d = root_node_->GetD();
  float parent_m = root_node_->GetM();

  if (!best_edge) { // Handle case where root has no children or no moves in beam
        if (move) *move = Move::NO_MOVE;
        if (is_terminal) *is_terminal = root_node_->IsTerminal(); // Root's terminal status
        return {parent_wl, parent_d, parent_m};
  }

  if (move) *move = best_edge.GetMove(played_history_.IsBlackToMove());
  if (is_terminal) *is_terminal = best_edge.HasNode() && best_edge.IsTerminal(); // Check HasNode
  return {best_edge.GetWL(parent_wl), best_edge.GetD(parent_d),
          best_edge.HasNode() ? (best_edge.GetM(parent_m - 1) + 1) : parent_m }; // Check HasNode
}

std::pair<Move, Move> Search::GetBestMove() {
  SharedMutex::Lock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  EnsureBestMoveKnown();
  return {final_bestmove_, final_pondermove_};
}

std::int64_t Search::GetTotalPlayouts() const {
  // Use SharedLock for read-only access
  SharedMutex::SharedLock lock(nodes_mutex_);
  return total_playouts_;
}

void Search::ResetBestMove() {
  SharedMutex::Lock nodes_lock(nodes_mutex_);
  Mutex::Lock lock(counters_mutex_);
  bool old_sent = bestmove_is_sent_;
  bestmove_is_sent_ = false;
  EnsureBestMoveKnown(); // This will potentially update the beam
  bestmove_is_sent_ = old_sent;
}

// Computes the best move, maybe with temperature (according to the settings).
void Search::EnsureBestMoveKnown() REQUIRES(nodes_mutex_)
    REQUIRES(counters_mutex_) {
  if (bestmove_is_sent_) return;
  if (root_node_->GetN() == 0) return;
  if (!root_node_->HasChildren()) return;

  // +++ Additions for Beam Search Features +++
  // Ensure beam is updated before final decision if threshold met
  bool beam_updated_now = false;
   if (params_.GetRootBeamMaxWidth() > 0 && next_beam_update_visits_ > 0 && root_node_->GetN() >= next_beam_update_visits_) {
       // Check if beam is already active AND next update threshold is reached OR beam is not active yet
       if ((beam_active_ && root_node_->GetN() >= next_beam_update_visits_) || !beam_active_) {
            UpdateRootBeam(); // Requires nodes_mutex_ and counters_mutex_
            beam_updated_now = true;
       }
  }
  // +++ End Additions +++

  float temperature = params_.GetTemperature();
  const int cutoff_move = params_.GetTemperatureCutoffMove();
  const int decay_delay_moves = params_.GetTempDecayDelayMoves();
  const int decay_moves = params_.GetTempDecayMoves();
  const int moves = played_history_.Last().GetGamePly() / 2;

  if (cutoff_move && (moves + 1) >= cutoff_move) {
    temperature = params_.GetTemperatureEndgame();
  } else if (temperature && decay_moves) {
    if (moves >= decay_delay_moves + decay_moves) {
      temperature = 0.0;
    } else if (moves >= decay_delay_moves) {
      temperature *=
          static_cast<float>(decay_delay_moves + decay_moves - moves) /
          decay_moves;
    }
    // don't allow temperature to decay below endgame temperature
    if (temperature < params_.GetTemperatureEndgame()) {
      temperature = params_.GetTemperatureEndgame();
    }
  }

  EdgeAndNode bestmove_edge;
  if (temperature > 1e-5) { // Use a small threshold for floating point comparison
      bestmove_edge = GetBestRootChildWithTemperature(temperature);
  } else {
      // GetBestChildNoTemperature now respects the beam if active
      bestmove_edge = GetBestChildNoTemperature(root_node_, 0);
  }

  // Handle case where no valid move is found (e.g., all moves filtered by beam)
  if (!bestmove_edge) {
      LOGFILE << "Warning: No best move found after applying filters/temperature. Falling back.";
      // Fallback: Get the absolute best move ignoring temperature/beam temporarily
      // This ensures we always output *something*
      // Temporarily deactivate beam for fallback selection
      bool was_beam_active = beam_active_;
      beam_active_ = false;
      auto fallback_edges = GetBestChildrenNoTemperature(root_node_, 1, 0);
      beam_active_ = was_beam_active; // Restore beam status

      if (!fallback_edges.empty()) {
           bestmove_edge = fallback_edges.front();
      } else {
           // If still no move (e.g., root has no children), set to NO_MOVE
           final_bestmove_ = Move::NO_MOVE;
           final_pondermove_ = Move::NO_MOVE;
           LOGFILE << "Error: Could not determine any best move.";
           return; // Exit early
      }
  }


  final_bestmove_ = bestmove_edge.GetMove(played_history_.IsBlackToMove());

  // Find ponder move based on the chosen best move
  if (bestmove_edge.HasNode() && bestmove_edge.node()->HasChildren()) {
    final_pondermove_ = GetBestChildNoTemperature(bestmove_edge.node(), 1)
                            .GetMove(!played_history_.IsBlackToMove());
  } else {
    final_pondermove_ = Move::NO_MOVE; // No ponder if best move leads to terminal/unexpanded
  }
}


// Returns @count children with most visits. Respects beam if active.
std::vector<EdgeAndNode> Search::GetBestChildrenNoTemperature(Node* parent,
                                                              int count,
                                                              int depth) const REQUIRES(nodes_mutex_) {
  // Even if Edges is populated at this point, its a race condition to access
  // the node, so exit quickly.
  if (parent->GetN() == 0) return {};
  const bool is_root_node = (parent == root_node_);
  const bool is_odd_depth = (depth % 2) == 1;
  const float draw_score = GetDrawScore(is_odd_depth);
  // Best child is selected using the following criteria:
  // * Prefer shorter terminal wins / avoid shorter terminal losses.
  // * Largest number of playouts (using weight).
  // * If two nodes have equal number:
  //   * If that number is 0, the one with larger prior wins.
  //   * If that number is larger than 0, the one with larger eval wins.
  std::vector<EdgeAndNode> edges;
  for (auto& edge : parent->Edges()) {
    // Filter based on root_move_filter_ or active beam
    if (is_root_node) {
        if (!root_move_filter_.empty()) {
             if (std::find(root_move_filter_.begin(), root_move_filter_.end(), edge.GetMove()) == root_move_filter_.end()) {
                continue; // Skip if not in searchmoves filter
            }
        } else if (beam_active_) {
            bool in_beam = false;
            for (const auto& beam_move : current_beam_) {
                if (edge.GetMove() == beam_move) {
                    in_beam = true;
                    break;
                }
            }
            if (!in_beam) continue; // Skip if not in beam
        }
    }
    edges.push_back(edge);
  }

  // Handle case where no moves are left after filtering
  if (edges.empty()) return {};

  // Limit count to the number of available edges after filtering
  count = std::min(count, static_cast<int>(edges.size()));
  if (count <= 0) return {}; // Handle count=0 case

  const auto middle = edges.begin() + count;

  std::partial_sort(
      edges.begin(), middle, edges.end(),
      [draw_score](const auto& a, const auto& b) {
        // The function returns "true" when a is preferred to b.

        // Lists edge types from less desirable to more desirable.
        enum EdgeRank {
          kTerminalLoss,
          kTablebaseLoss,
          kNonTerminal,  // Non terminal or terminal draw.
          kTablebaseWin,
          kTerminalWin,
        };

        auto GetEdgeRank = [](const EdgeAndNode& edge) {
          // This default isn't used as wl only checked for case edge is
          // terminal.
          const auto wl = edge.GetWL(0.0f);
          // Not safe to access IsTerminal if GetN is 0.
          if (!edge.HasNode() || edge.GetN() == 0 || !edge.IsTerminal() || wl==0.0f) { // Check HasNode and wl==0 for draw
            return kNonTerminal;
          }
          if (edge.IsTbTerminal()) {
            return wl < 0.0 ? kTablebaseLoss : kTablebaseWin;
          }
          return wl < 0.0 ? kTerminalLoss : kTerminalWin;
        };

        // If moves have different outcomes, prefer better outcome.
        const auto a_rank = GetEdgeRank(a);
        const auto b_rank = GetEdgeRank(b);
        if (a_rank != b_rank) return a_rank > b_rank;

        // If both are terminal draws, try to make it shorter.
        // Not safe to access IsTerminal if GetN is 0.
        if (a_rank == kNonTerminal && a.HasNode() && b.HasNode() && a.GetN() != 0 && b.GetN() != 0 &&
            a.IsTerminal() && b.IsTerminal()) { // Check HasNode
          if (a.IsTbTerminal() != b.IsTbTerminal()) {
            // Prefer non-tablebase draws.
            return a.IsTbTerminal() < b.IsTbTerminal();
          }
          // Prefer shorter draws.
          return a.GetM(0.0f) < b.GetM(0.0f);
        }

        // Neither is terminal, use standard rule.
        if (a_rank == kNonTerminal) {
          // Prefer largest weight then eval then prior.
          if (a.GetWeight() != b.GetWeight())
            return a.GetWeight() > b.GetWeight();
          // Default doesn't matter here so long as they are the same as either
          // both are N==0 (thus we're comparing equal defaults) or N!=0 and
          // default isn't used.
          if (a.GetQ(0.0f, draw_score) != b.GetQ(0.0f, draw_score)) {
            return a.GetQ(0.0f, draw_score) > b.GetQ(0.0f, draw_score);
          }
          return a.GetP() > b.GetP();
        }

        // Both variants are winning, prefer shortest win.
        if (a_rank > kNonTerminal) {
          return a.GetM(0.0f) < b.GetM(0.0f);
        }

        // Both variants are losing, prefer longest losses.
        return a.GetM(0.0f) > b.GetM(0.0f);
      });

  if (count < static_cast<int>(edges.size())) {
    edges.resize(count);
  }
  return edges;
}

// Returns a child with most visits. Respects beam if active.
EdgeAndNode Search::GetBestChildNoTemperature(Node* parent, int depth) const REQUIRES(nodes_mutex_) {
  auto res = GetBestChildrenNoTemperature(parent, 1, depth);
  return res.empty() ? EdgeAndNode() : res.front();
}


// Returns a child of a root chosen according to weighted-by-temperature visit
// count. Respects beam if active.
EdgeAndNode Search::GetBestRootChildWithTemperature(float temperature) const REQUIRES(nodes_mutex_) {
  // Root is at even depth.
  const float draw_score = GetDrawScore(/* is_odd_depth= */ false);

  std::vector<float> cumulative_sums;
  std::vector<EdgeAndNode> eligible_edges; // Store eligible edges
  float sum = 0.0;
  float max_weight = 0.0;
  const float offset = params_.GetTemperatureVisitOffset();
  float max_eval = -std::numeric_limits<float>::infinity(); // Initialize properly
  const float fpu =
      GetFpu(params_, root_node_, /* is_root= */ true, draw_score);

  for (auto& edge : root_node_->Edges()) {
    // Filter based on root_move_filter_ or active beam
    if (!root_move_filter_.empty()) {
        if (std::find(root_move_filter_.begin(), root_move_filter_.end(), edge.GetMove()) == root_move_filter_.end()) {
            continue; // Skip if not in searchmoves filter
        }
    } else if (beam_active_) {
        bool in_beam = false;
        for (const auto& beam_move : current_beam_) {
            if (edge.GetMove() == beam_move) {
                in_beam = true;
                break;
            }
        }
        if (!in_beam) continue; // Skip if not in beam
    }
    eligible_edges.push_back(edge); // Add to eligible list

    if (edge.GetWeight() + offset > max_weight) {
      max_weight = edge.GetWeight() + offset;
    }
    // Update max_eval among eligible moves only
    max_eval = std::max(max_eval, edge.GetQ(fpu, draw_score));
  }

  // Handle case where no moves are eligible
  if (eligible_edges.empty()) return EdgeAndNode();

  // Find the index of the edge that had max_weight
  int max_weight_idx = -1;
  for(int i=0; i<eligible_edges.size(); ++i) {
      if (eligible_edges[i].GetWeight() + offset == max_weight) {
          max_weight_idx = i;
          break;
      }
  }
  // If max_weight was 0 (or negative due to offset), use P value for normalization
  bool use_policy = (max_weight <= 0.0f);


  const float min_eval =
      max_eval - params_.GetTemperatureWinpctCutoff() / 50.0f;
  cumulative_sums.reserve(eligible_edges.size()); // Pre-allocate
  std::vector<size_t> final_indices; // Store indices of moves passing the eval filter
  final_indices.reserve(eligible_edges.size());

  for (size_t i = 0; i < eligible_edges.size(); ++i) {
    auto& edge = eligible_edges[i];
    if (edge.GetQ(fpu, draw_score) < min_eval) continue;

    float weight_or_p = use_policy ? edge.GetP() : std::max(0.0f, edge.GetWeight() + offset);
    float norm_factor = use_policy ? 1.0f : max_weight; // Normalize by max_weight if using weight

    // Avoid division by zero if norm_factor is zero
    float normalized_val = (norm_factor > 1e-9f) ? (weight_or_p / norm_factor) : 0.0f;

    sum += std::pow(normalized_val, 1.0f / temperature); // Ensure temperature != 0
    cumulative_sums.push_back(sum);
    final_indices.push_back(i); // Store original index
  }


  // Check if sum is zero (possible if all eligible moves have zero weight/P and offset is negative)
  if (sum <= 1e-9 || final_indices.empty()) {
       LOGFILE << "Warning: Temperature sampling sum is zero or no moves passed filter. Returning best move by weight.";
       // Fallback to best move by weight among eligible moves
       EdgeAndNode best_by_weight;
       float max_w = -std::numeric_limits<float>::infinity(); // Initialize properly
       for(auto& edge : eligible_edges) {
           if (edge.GetWeight() > max_w) {
               max_w = edge.GetWeight();
               best_by_weight = edge;
           }
       }
       return best_by_weight; // Might be null if eligible_edges was empty
  }

  assert(sum > 0); // Assert sum is positive after check

  const float toss = Random::Get().GetFloat(cumulative_sums.back());
  int chosen_sum_idx =
      std::lower_bound(cumulative_sums.begin(), cumulative_sums.end(), toss) -
      cumulative_sums.begin();

  // Map the index back to the original eligible_edges using final_indices
  size_t original_idx = final_indices[chosen_sum_idx];
  return eligible_edges[original_idx];
}


void Search::StartThreads(size_t how_many) {
  thread_count_.store(how_many, std::memory_order_release);
  Mutex::Lock lock(threads_mutex_);
  // First thread is a watchdog thread.
  if (threads_.size() == 0) {
    threads_.emplace_back([this]() { WatchdogThread(); });
  }
  // Start working threads.
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

void Search::RunBlocking(size_t threads) {
  StartThreads(threads);
  Wait();
}

bool Search::IsSearchActive() const {
  return !stop_.load(std::memory_order_acquire);
}

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
  stats->total_visits = total_playouts_ + initial_visits_;
  stats->total_allocated_nodes = dag_->AllocatedNodeCount();
  stats->nodes_since_movestart = total_playouts_;
  stats->batches_since_movestart = total_batches_;
  stats->average_depth = cum_depth_ / (total_playouts_ ? total_playouts_ : 1);
  stats->edge_n.clear();
  stats->win_found = false;
  stats->may_resign = true;
  stats->num_losing_edges = 0;
  stats->time_usage_hint_ = IterationStats::TimeUsageHint::kNormal;
  stats->mate_depth = std::numeric_limits<int>::max();

  // If root node hasn't finished first visit, none of this code is safe.
  if (root_node_->GetN() > 0) {
    const auto draw_score = GetDrawScore(true); // Depth 1 is odd
    const float fpu =
        GetFpu(params_, root_node_, /* is_root_node */ true, draw_score);
    float max_q_plus_m = -std::numeric_limits<float>::infinity(); // Use infinity
    float max_weight = 0.0f; // Use weight for comparison
    bool max_weight_has_max_q_plus_m = true; // Compare based on weight
    const auto m_evaluator = network_->GetCapabilities().has_mlh()
                                 ? MEvaluator(params_, root_node_)
                                 : MEvaluator();
    for (const auto& edge : root_node_->Edges()) {
      // Consider beam if active
      if (beam_active_) {
          bool in_beam = false;
          for (const auto& beam_move : current_beam_) {
              if (edge.GetMove() == beam_move) {
                  in_beam = true;
                  break;
              }
          }
          if (!in_beam) continue;
      }

      const auto n = edge.GetN();
      const auto w = edge.GetWeight(); // Get weight
      const auto q = edge.GetQ(fpu, draw_score);
      const auto m = m_evaluator.GetMUtility(edge, q);
      const auto q_plus_m = q + m;
      stats->edge_n.push_back(n);
      if (edge.HasNode()) { // Check if node exists
          if (n > 0 && edge.IsTerminal() && edge.GetWL(0.0f) > 0.0f) {
            stats->win_found = true;
          }
          if (n > 0 && edge.IsTerminal() && edge.GetWL(0.0f) < 0.0f) {
            stats->num_losing_edges += 1;
          }
          if (n > 0 && edge.IsTerminal() && edge.GetWL(0.0f) == 1.0f &&
              !edge.IsTbTerminal()) {
            stats->mate_depth =
                std::min(stats->mate_depth,
                        static_cast<int>(std::round(edge.GetM(0.0f))) / 2 + 1);
          }
      }

      // If game is resignable, no need for moving quicker. This allows
      // proving mate when losing anyway for better score output.
      // Hardcoded resign threshold, because there is no available parameter.
      if (n > 0 && q > -0.98f) {
        stats->may_resign = false;
      }
      if (w > max_weight) { // Compare weight
        max_weight = w;
        max_weight_has_max_q_plus_m = false;
      }
      if (max_q_plus_m <= q_plus_m) {
         // Check for strict inequality or equal weight to update max_weight_has_max_q_plus_m
        max_weight_has_max_q_plus_m = (max_q_plus_m == q_plus_m) && (max_weight == w);
        max_q_plus_m = q_plus_m;
        // If q+m is equal, also check weight to decide if max_weight_has_max_q_plus_m needs update
        if (max_q_plus_m == q_plus_m && w > max_weight) {
             max_weight = w;
             max_weight_has_max_q_plus_m = true;
        } else if (max_q_plus_m == q_plus_m && w == max_weight) {
             max_weight_has_max_q_plus_m = true;
        }
      }
    }
    if (!max_weight_has_max_q_plus_m) {
      stats->time_usage_hint_ = IterationStats::TimeUsageHint::kNeedMoreTime;
    }
  }
}

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
    // Only exit when bestmove is responded. It may happen that search threads
    // already all exited, and we need at least one thread that can do that.
    if (bestmove_is_sent_) break;

    auto remaining_time = hints.GetEstimatedRemainingTimeMs();
    if (remaining_time > kMaxWaitTimeMs) remaining_time = kMaxWaitTimeMs;
    if (remaining_time < kMinWaitTimeMs) remaining_time = kMinWaitTimeMs;
    // There is no real need to have max wait time, and sometimes it's fine
    // to wait without timeout at all (e.g. in `go nodes` mode), but we
    // still limit wait time for exotic cases like when pc goes to sleep
    // mode during thinking.
    // Minimum wait time is there to prevent busy wait and other threads
    // starvation.
    watchdog_cv_.wait_for(
        lock.get_raw(), std::chrono::milliseconds(remaining_time),
        [this]() { return stop_.load(std::memory_order_acquire); });
  }
  LOGFILE << "End a watchdog thread.";
}

void Search::FireStopInternal() REQUIRES(counters_mutex_) {
  stop_.store(true, std::memory_order_release);
  watchdog_cv_.notify_all();
}

void Search::Stop() {
  Mutex::Lock lock(counters_mutex_);
  ok_to_respond_bestmove_ = true;
  FireStopInternal();
  LOGFILE << "Stopping search due to `stop` uci command.";
}

void Search::Abort() {
  Mutex::Lock lock(counters_mutex_);
  if (!stop_.load(std::memory_order_acquire) ||
      (!bestmove_is_sent_ && !ok_to_respond_bestmove_)) {
    bestmove_is_sent_ = true;
    FireStopInternal();
  }
  LOGFILE << "Aborting search, if it is still active.";
}

void Search::Wait() {
  Mutex::Lock lock(threads_mutex_);
  while (!threads_.empty()) {
    threads_.back().join();
    threads_.pop_back();
  }
}

void Search::CancelSharedCollisions() REQUIRES(nodes_mutex_) {
  for (auto& entry : shared_collisions_) {
    auto path = entry.first;
    // Iterate backwards, skipping the last node (the collision node itself)
    for (auto it = ++(path.crbegin()); it != path.crend(); ++it) {
        auto* node = std::get<0>(*it);
        if(node) { // Add null check before calling CancelScoreUpdate
            node->CancelScoreUpdate(entry.second);
        }
    }
  }
  shared_collisions_.clear();
}


Search::~Search() {
  Abort();
  Wait();
  {
    SharedMutex::Lock lock(nodes_mutex_);
    CancelSharedCollisions();

#ifndef NDEBUG
    if (root_node_) assert(root_node_->ZeroNInFlight()); // Add null check
#endif
  }

  // Free previously released nodes that were not reused during this search.
  dag_->TTMaintenance();
  dag_->TTMaintenance();

  LOGFILE << "Search destroyed.";
}

// +++ Additions for Beam Search Features +++
void Search::UpdateRootBeam() REQUIRES(nodes_mutex_) REQUIRES(counters_mutex_) {
    if (params_.GetRootBeamMaxWidth() <= 0) return; // Beam search disabled

    // Ensure root has children evaluated
    if (!root_node_->HasChildren() || root_node_->GetN() == 0) return;

    LOGFILE << "Updating root beam at N=" << root_node_->GetN();

    std::vector<std::pair<float, EdgeAndNode>> scored_edges;
    const float draw_score = GetDrawScore(/*is_odd_depth=*/false); // Root is even depth
    bool is_root_node = true;

    for (auto& edge : root_node_->Edges()) {
        // Respect existing searchmoves/TB filter
        if (!root_move_filter_.empty()) {
            if (std::find(root_move_filter_.begin(), root_move_filter_.end(), edge.GetMove()) == root_move_filter_.end()) {
                continue; // Skip if not in searchmoves filter
            }
        }
        // Use PUCT score for ranking
        float score = CalculatePUCTScore(params_, root_node_, edge, is_root_node, draw_score);
        scored_edges.push_back({score, edge});
    }

    if (scored_edges.empty()) {
        LOGFILE << "No eligible moves found for beam update.";
        beam_active_ = false; // Cannot activate beam if no moves
        next_beam_update_visits_ = 0; // Stop trying to update
        return;
    }

    // Sort children by score (descending)
    std::sort(scored_edges.begin(), scored_edges.end(),
              [](const auto& a, const auto& b) {
                  // Handle NaN scores - treat them as lowest possible score
                  bool a_nan = std::isnan(a.first);
                  bool b_nan = std::isnan(b.first);
                  if (a_nan && b_nan) return false; // Keep relative order if both NaN
                  if (a_nan) return false; // NaN is less than any number
                  if (b_nan) return true;  // Any number is greater than NaN
                  return a.first > b.first; // Higher score is better
              });

    // Determine dynamic beam width
    const int min_width = std::max(1, params_.GetRootBeamMinWidth()); // Ensure at least 1 move if possible
    const int max_width = std::max(min_width, params_.GetRootBeamMaxWidth());
    int dynamic_width = min_width;
    const float best_score = scored_edges[0].first;
    // Handle case where best_score might be NaN
    const float score_margin = std::isnan(best_score) ? 0.0f : params_.GetRootBeamScoreMargin();
    const float score_threshold = std::isnan(best_score) ? -std::numeric_limits<float>::infinity() : best_score - score_margin; // Absolute margin

    // Only apply dynamic width logic if RootBeamMinWidth > 0 and < RootBeamMaxWidth
    if (params_.GetRootBeamMinWidth() > 0 && params_.GetRootBeamMinWidth() < params_.GetRootBeamMaxWidth()) {
        for (int i = min_width; i < max_width && i < scored_edges.size(); ++i) {
             // Skip NaN scores when determining dynamic width based on margin
            if (std::isnan(scored_edges[i].first)) continue;
            if (scored_edges[i].first >= score_threshold) {
                dynamic_width = i + 1;
            } else {
                break; // Stop adding moves once they fall below the threshold
            }
        }
         current_beam_width_ = std::min(static_cast<int>(scored_edges.size()), dynamic_width);
    } else {
        // Use fixed width (MaxWidth) if dynamic width is disabled
        current_beam_width_ = std::min(static_cast<int>(scored_edges.size()), max_width);
    }
    // Ensure beam width is at least min_width if there are enough moves
    current_beam_width_ = std::max(current_beam_width_, std::min(min_width, static_cast<int>(scored_edges.size())));


    // Store the beam
    current_beam_.clear();
    current_beam_.reserve(current_beam_width_);
    for (int i = 0; i < current_beam_width_; ++i) {
        current_beam_.push_back(scored_edges[i].second.GetMove());
    }

    // Calculate next update threshold (Geometric Interval)
    last_beam_update_visits_ = root_node_->GetN();
    // Prevent overflow and ensure interval increases meaningfully
    double next_update_double = static_cast<double>(next_beam_update_visits_) * params_.GetRootBeamUpdateIntervalFactor();
    if (next_update_double > static_cast<double>(std::numeric_limits<uint64_t>::max())) {
        next_beam_update_visits_ = std::numeric_limits<uint64_t>::max();
    } else {
         next_beam_update_visits_ = static_cast<uint64_t>(next_update_double);
         // Ensure interval increases by at least 1 visit
        if (next_beam_update_visits_ <= last_beam_update_visits_) {
            next_beam_update_visits_ = last_beam_update_visits_ + 1;
        }
    }

    beam_active_ = true;
    LOGFILE << "Beam updated. Width=" << current_beam_width_ << ", Next update at N=" << next_beam_update_visits_;
}
// +++ End Additions +++


//////////////////////////////////////////////////////////////////////////////
// SearchWorker
//////////////////////////////////////////////////////////////////////////////

void SearchWorker::RunTasks(int tid) {
  while (true) {
    PickTask* task = nullptr;
    int id = 0;
    {
      int spins = 0;
      while (true) {
        int nta = tasks_taken_.load(std::memory_order_acquire);
        int tc = task_count_.load(std::memory_order_acquire);
        if (nta < tc) {
          int val = 0;
          if (task_taking_started_.compare_exchange_weak(
                  val, 1, std::memory_order_acq_rel,
                  std::memory_order_relaxed)) {
            nta = tasks_taken_.load(std::memory_order_acquire);
            tc = task_count_.load(std::memory_order_acquire);
            // We got the spin lock, double check we're still in the clear.
            if (nta < tc) {
              id = tasks_taken_.fetch_add(1, std::memory_order_acq_rel);
              task = &picking_tasks_[id];
              task_taking_started_.store(0, std::memory_order_release);
              break;
            }
            task_taking_started_.store(0, std::memory_order_release);
          }
          SpinloopPause();
          spins = 0;
          continue;
        } else if (tc != -1) {
          spins++;
          if (spins >= 512) {
            std::this_thread::yield();
            spins = 0;
          } else {
            SpinloopPause();
          }
          continue;
        }
        spins = 0;
        // Looks like sleep time.
        Mutex::Lock lock(picking_tasks_mutex_);
        // Refresh them now we have the lock.
        nta = tasks_taken_.load(std::memory_order_acquire);
        tc = task_count_.load(std::memory_order_acquire);
        if (tc != -1) continue;
        if (nta >= tc && exiting_) return;
        task_added_.wait(lock.get_raw());
        // And refresh again now we're awake.
        nta = tasks_taken_.load(std::memory_order_acquire);
        tc = task_count_.load(std::memory_order_acquire);
        if (nta >= tc && exiting_) return;
      }
    }
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
      picking_tasks_[id].complete = true;
      completed_tasks_.fetch_add(1, std::memory_order_acq_rel);
    }
  }
}

void SearchWorker::ExecuteOneIteration() {
  // 1. Initialize internal structures.
  InitializeIteration(search_->network_->NewComputation());

  if (params_.GetMaxConcurrentSearchers() != 0) {
    std::unique_ptr<SpinHelper> spin_helper;
    if (params_.GetSearchSpinBackoff()) {
      spin_helper = std::make_unique<ExponentialBackoffSpinHelper>();
    } else {
      // This is a hard spin lock to reduce latency but at the expense of busy
      // wait cpu usage. If search worker count is large, this is probably a
      // bad idea.
      spin_helper = std::make_unique<SpinHelper>();
    }

    while (true) {
      // If search is stopped, we've not gathered or done anything and we don't
      // want to, so we can safely skip all below. But make sure we have done
      // at least one iteration.
      if (search_->stop_.load(std::memory_order_acquire) &&
          search_->GetTotalPlayouts() + search_->initial_visits_ > 0) {
        return;
      }

      int available =
          search_->pending_searchers_.load(std::memory_order_acquire);
      if (available == 0) {
        spin_helper->Wait();
        continue;
      }

      if (search_->pending_searchers_.compare_exchange_weak(
              available, available - 1, std::memory_order_acq_rel)) {
        break;
      } else {
        spin_helper->Backoff();
      }
    }
  }

  // 2. Gather minibatch.
  GatherMinibatch();
  task_count_.store(-1, std::memory_order_release);
  search_->backend_waiting_counter_.fetch_add(1, std::memory_order_relaxed);

  // 2b. Collect collisions.
  CollectCollisions();

  if (params_.GetMaxConcurrentSearchers() != 0) {
    search_->pending_searchers_.fetch_add(1, std::memory_order_acq_rel);
  }

  // 4. Run NN computation.
  RunNNComputation();
  search_->backend_waiting_counter_.fetch_add(-1, std::memory_order_relaxed);

  // 5. Retrieve NN computations (and terminal values) into nodes.
  FetchMinibatchResults();

  // 6. Propagate the new nodes' information to all their parents in the tree.
  DoBackupUpdate();

  // 7. Update the Search's status and progress information.
  UpdateCounters();

  // If required, waste time to limit nps.
  if (params_.GetNpsLimit() > 0) {
    while (search_->IsSearchActive()) {
      int64_t time_since_first_batch_ms = 0;
      {
        Mutex::Lock lock(search_->counters_mutex_);
        time_since_first_batch_ms = search_->GetTimeSinceFirstBatch();
      }
      if (time_since_first_batch_ms <= 0) {
        time_since_first_batch_ms = search_->GetTimeSinceStart();
      }
      // Prevent division by zero if time is still zero
      if (time_since_first_batch_ms == 0) time_since_first_batch_ms = 1;
      auto nps = search_->GetTotalPlayouts() * 1e3f / time_since_first_batch_ms;
      if (nps > params_.GetNpsLimit()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } else {
        break;
      }
    }
  }
}

// 1. Initialize internal structures.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::InitializeIteration(
    std::unique_ptr<NetworkComputation> computation) {
  computation_ = std::make_unique<CachingComputation>(
      std::move(computation), search_->network_->GetCapabilities().input_format,
      params_.GetHistoryFill(), search_->cache_);
  computation_->Reserve(params_.GetMiniBatchSize());
  minibatch_.clear();
  minibatch_.reserve(2 * params_.GetMiniBatchSize());
}

// 2. Gather minibatch.
// ~~~~~~~~~~~~~~~~~~~~
namespace {
int Mix(int high, int low, float ratio) {
  return static_cast<int>(std::round(static_cast<float>(low) +
                                     static_cast<float>(high - low) * ratio));
}

int CalculateCollisionsLeft(int64_t nodes, const SearchParams& params) {
  // End checked first
  if (nodes >= params.GetMaxCollisionVisitsScalingEnd() && params.GetMaxCollisionVisitsScalingEnd() > 0) { // Check scaling end > 0
    return params.GetMaxCollisionVisits();
  }
  if (nodes <= params.GetMaxCollisionVisitsScalingStart()) {
    return 1;
  }
  // Prevent division by zero if start and end are the same
  if (params.GetMaxCollisionVisitsScalingEnd() <= params.GetMaxCollisionVisitsScalingStart()) {
       return 1;
  }
  return Mix(params.GetMaxCollisionVisits(), 1,
             std::pow((static_cast<float>(nodes) -
                       params.GetMaxCollisionVisitsScalingStart()) /
                          (params.GetMaxCollisionVisitsScalingEnd() -
                           params.GetMaxCollisionVisitsScalingStart()),
                      params.GetMaxCollisionVisitsScalingPower()));
}
}  // namespace

void SearchWorker::GatherMinibatch() {
  // Total number of nodes to process.
  uint32_t minibatch_size = 0;
  int cur_n = 0;
  {
    SharedMutex::SharedLock lock(search_->nodes_mutex_); // Use SharedLock for read
    cur_n = search_->root_node_->GetN();
  }
  // TODO: GetEstimatedRemainingPlayouts has already had smart pruning factor
  // applied, which doesn't clearly make sense to include here...
  int64_t remaining_n =
      latest_time_manager_hints_.GetEstimatedRemainingPlayouts();
  uint32_t collisions_left = CalculateCollisionsLeft(
      std::min(static_cast<int64_t>(cur_n), remaining_n), params_);

  // Number of nodes processed out of order.
  number_out_of_order_ = 0;

  int thread_count = search_->thread_count_.load(std::memory_order_acquire);

  // Gather nodes to process in the current batch.
  // If we had too many nodes out of order, also interrupt the iteration so
  // that search can exit.
  while (minibatch_size < params_.GetMiniBatchSize() &&
         number_out_of_order_ < params_.GetMaxOutOfOrderEvals()) {
    // If there's something to process without touching slow neural net, do it.
    if (minibatch_size > 0 && computation_->GetCacheMisses() == 0) return;

    // If there is backend work to be done, and the backend is idle - exit
    // immediately.
    // Only do this fancy work if there are multiple threads as otherwise we
    // early exit from every batch since there is never another search thread to
    // be keeping the backend busy. Which would mean that threads=1 has a
    // massive nps drop.
    if (thread_count > 1 && minibatch_size > 0 &&
        computation_->GetCacheMisses() > params_.GetIdlingMinimumWork() &&
        thread_count -
                search_->backend_waiting_counter_.load(
                    std::memory_order_relaxed) >
            params_.GetThreadIdlingThreshold()) {
      return;
    }

    int new_start = static_cast<int>(minibatch_.size());

    PickNodesToExtend(
        std::min({collisions_left, params_.GetMiniBatchSize() - minibatch_size,
                  params_.GetMaxOutOfOrderEvals() - number_out_of_order_}));

    // Count the non-collisions.
    int non_collisions = 0;
    for (int i = new_start; i < static_cast<int>(minibatch_.size()); i++) {
      auto& picked_node = minibatch_[i];
      if (picked_node.IsCollision()) {
        continue;
      }
      ++non_collisions;
      ++minibatch_size;
    }

    {
      // This lock must be held until after the task_completed_ wait succeeds
      // below. Since the tasks perform work which assumes they have the lock,
      // even though actually this thread does.
      SharedMutex::Lock lock(search_->nodes_mutex_);

      bool needs_wait = false;
      int ppt_start = new_start;
      if (params_.GetTaskWorkersPerSearchWorker() > 0 &&
          non_collisions >= params_.GetMinimumWorkSizeForProcessing()) {
        const int num_tasks = std::clamp(
            non_collisions / params_.GetMinimumWorkPerTaskForProcessing(), 2,
            params_.GetTaskWorkersPerSearchWorker() + 1);
        // Round down, left overs can go to main thread so it waits less.
        int per_worker = (num_tasks > 0) ? (non_collisions / num_tasks) : non_collisions; // Avoid division by zero
        needs_wait = true;
        ResetTasks();
        int found = 0;
        for (int i = new_start; i < static_cast<int>(minibatch_.size()); i++) {
          auto& picked_node = minibatch_[i];
          if (picked_node.IsCollision()) {
            continue;
          }
          ++found;
          // Ensure per_worker is positive to avoid infinite loop if non_collisions < num_tasks
          if (per_worker > 0 && found == per_worker) {
            picking_tasks_.emplace_back(ppt_start, i + 1);
            task_count_.fetch_add(1, std::memory_order_acq_rel);
            ppt_start = i + 1;
            found = 0;
            if (picking_tasks_.size() == static_cast<size_t>(num_tasks - 1)) {
              break;
            }
          }
        }
        // Add remaining items as the last task if needed
        if (ppt_start < static_cast<int>(minibatch_.size()) && num_tasks > picking_tasks_.size() + 1) {
             picking_tasks_.emplace_back(ppt_start, static_cast<int>(minibatch_.size()));
             task_count_.fetch_add(1, std::memory_order_acq_rel);
             ppt_start = static_cast<int>(minibatch_.size()); // Update start for the main thread part
        }

      }
      ProcessPickedTask(ppt_start, static_cast<int>(minibatch_.size()));
      if (needs_wait) {
        WaitForTasks();
      }
    }
    bool some_ooo = false;
    for (int i = static_cast<int>(minibatch_.size()) - 1; i >= new_start; i--) {
      if (minibatch_[i].ooo_completed) {
        some_ooo = true;
        break;
      }
    }
    if (some_ooo) {
      SharedMutex::Lock lock(search_->nodes_mutex_);
      for (int i = static_cast<int>(minibatch_.size()) - 1; i >= new_start; i--) {
        // If there was any OOO, revert 'all' new collisions - it isn't possible
        // to identify exactly which ones are afterwards and only prune those.
        // This may remove too many items, but hopefully most of the time they
        // will just be added back in the same in the next gather.
        if (minibatch_[i].IsCollision()) {
          for (auto it = ++(minibatch_[i].path.crbegin());
               it != minibatch_[i].path.crend(); ++it) {
             auto* node = std::get<0>(*it);
             if (node) node->CancelScoreUpdate(minibatch_[i].multivisit); // Add null check
          }
          minibatch_.erase(minibatch_.begin() + i);
        } else if (minibatch_[i].ooo_completed) {
          FetchSingleNodeResult(&minibatch_[i], minibatch_[i], 0);
          DoBackupUpdateSingleNode(minibatch_[i]);
          minibatch_.erase(minibatch_.begin() + i);
          --minibatch_size;
          ++number_out_of_order_;
        }
      }
    }
    for (size_t i = new_start; i < minibatch_.size(); i++) {
      // If there was no OOO, there can stil be collisions.
      // There are no OOO though.
      // Also terminals when OOO is disabled.
      if (!minibatch_[i].ShouldAddToInput()) continue;
      if (minibatch_[i].is_cache_hit) {
        // Since minibatch_[i] holds cache lock, this is guaranteed to succeed.
        computation_->AddInputByHash(minibatch_[i].hash,
                                     std::move(minibatch_[i].lock));
      } else {
        computation_->AddInput(minibatch_[i].hash, minibatch_[i].history);
      }
    }

    // Check for stop at the end so we have at least one node.
    for (size_t i = new_start; i < minibatch_.size(); i++) {
      auto& picked_node = minibatch_[i];

      if (picked_node.IsCollision()) {
        // Check to see if we can upsize the collision to exit sooner.
        if (picked_node.maxvisit > 0 &&
            collisions_left > picked_node.multivisit) {
          SharedMutex::Lock lock(search_->nodes_mutex_);
          int extra = std::min(picked_node.maxvisit, collisions_left) -
                      picked_node.multivisit;
          picked_node.multivisit += extra;
          for (auto it = ++(picked_node.path.crbegin());
               it != picked_node.path.crend(); ++it) {
            auto* node = std::get<0>(*it);
            if (node) node->IncrementNInFlight(extra); // Add null check
          }
        }
        if ((collisions_left -= picked_node.multivisit) <= 0) return;
        if (search_->stop_.load(std::memory_order_acquire)) return;
      }
    }
  }
}

void SearchWorker::ProcessPickedTask(int start_idx, int end_idx) {
  for (int i = start_idx; i < end_idx; i++) {
    auto& picked_node = minibatch_[i];
    if (picked_node.IsCollision()) continue;
    // If node is a collision, known as a terminal (win/loss/draw according to
    // the rules of the game) or has a low node, it means that we have already
    // visited this node before and can't extend it.
    if (picked_node.IsExtendable()) {
      // Node was never visited, extend it.
      ExtendNode(picked_node);
    }

    picked_node.ooo_completed =
        params_.GetOutOfOrderEval() && picked_node.CanEvalOutOfOrder();
  }
}

#define MAX_TASKS 100

void SearchWorker::ResetTasks() {
  task_count_.store(0, std::memory_order_release);
  tasks_taken_.store(0, std::memory_order_release);
  completed_tasks_.store(0, std::memory_order_release);
  picking_tasks_.clear();
  // Reserve because resizing breaks pointers held by the task threads.
  picking_tasks_.reserve(MAX_TASKS);
}

int SearchWorker::WaitForTasks() {
  // Spin lock, other tasks should be done soon.
  while (true) {
    int completed = completed_tasks_.load(std::memory_order_acquire);
    int todo = task_count_.load(std::memory_order_acquire);
    if (todo == completed) return completed;
    SpinloopPause();
  }
}

void SearchWorker::PickNodesToExtend(int collision_limit) {
  ResetTasks();
  {
    // While nothing is ready yet - wake the task runners so they are ready to
    // receive quickly.
    Mutex::Lock lock(picking_tasks_mutex_);
    task_added_.notify_all();
  }
  std::vector<Move> empty_movelist;
  // This lock must be held until after the task_completed_ wait succeeds below.
  // Since the tasks perform work which assumes they have the lock, even though
  // actually this thread does.
  SharedMutex::Lock lock(search_->nodes_mutex_);
  history_.Trim(search_->played_history_.GetLength());
  PickNodesToExtendTask({std::make_tuple(search_->root_node_, 0, 0)},
                        collision_limit, history_, &minibatch_,
                        &main_workspace_);

  WaitForTasks();
  for (int i = 0; i < static_cast<int>(picking_tasks_.size()); i++) {
    for (int j = 0; j < static_cast<int>(picking_tasks_[i].results.size());
         j++) {
      minibatch_.emplace_back(std::move(picking_tasks_[i].results[j]));
    }
  }
}

// Depth starts with 0 at root, so number of plies in PV equals depth.
std::pair<int, int> SearchWorker::GetRepetitions(int depth,
                                                 const Position& position) {
  const auto repetitions = position.GetRepetitions();

  if (repetitions == 0) return {0, 0};

  if (repetitions >= 2) return {repetitions, 0};

  const auto plies = position.GetPliesSincePrevRepetition();
  if (params_.GetTwoFoldDraws() && /*repetitions == 1 &&*/ depth >= 4 &&
      depth >= plies) {
    return {1, plies};
  }

  return {0, 0};
}

// Check if PickNodesToExtendTask should stop picking at this @node.
bool SearchWorker::ShouldStopPickingHere(Node* node, bool is_root_node,
                                         int repetitions) {
  constexpr double wl_diff_limit = 0.01; // Use double for comparison
  constexpr float d_diff_limit = 0.01f;
  constexpr float m_diff_limit = 2.0f;

  if (!node) return true; // Stop if node is null

  if (node->GetN() == 0 || node->IsTerminal()) return true;

  // Only stop at root when there is no other option.
  assert(!is_root_node || node == search_->root_node_);
  if (is_root_node) return false;

  // Stop at draws by repetition.
  if (repetitions >= 2) return true;

  // Check if Node and LowNode differ significantly.
  auto low_node = node->GetLowNode();
  if (!low_node) return false; // Cannot compare if low_node doesn't exist yet

  // Only known transpositions can differ.
  if (!low_node->IsTransposition()) return false;

  // LowNode is terminal when Node is not.
  if (low_node->IsTerminal()) return true;

  // Bounds differ (swap).
  auto [low_node_lower, low_node_upper] = low_node->GetBounds();
  auto [node_lower, node_upper] = node->GetBounds();
  if (low_node_lower != -node_upper || low_node_upper != -node_lower)
    return true;

  // WL differs significantly (flip).
  auto wl_diff = std::abs(static_cast<double>(low_node->GetWL()) + node->GetWL()); // Cast to double
  if (wl_diff >= wl_diff_limit) return true;

  // D differs significantly.
  auto d_diff = std::abs(low_node->GetD() - node->GetD());
  if (d_diff >= d_diff_limit) return true;

  // M differs significantly (increment).
  auto m_diff = std::abs(low_node->GetM() + 1 - node->GetM());
  if (m_diff >= m_diff_limit) return true;

  return false;
}

void SearchWorker::PickNodesToExtendTask(
    const BackupPath& path, int collision_limit, PositionHistory& history,
    std::vector<NodeToProcess>* receiver,
    TaskWorkspace* workspace) NO_THREAD_SAFETY_ANALYSIS {
  assert(path.size() == (size_t)history.GetLength() -
                            search_->played_history_.GetLength() + 1);

  // TODO: Bring back pre-cached nodes created outside locks in a way that works
  // with tasks.
  // TODO: pre-reserve visits_to_perform for expected depth and likely maximum
  // width. Maybe even do so outside of lock scope.
  auto& vtp_buffer = workspace->vtp_buffer;
  auto& visits_to_perform = workspace->visits_to_perform;
  visits_to_perform.clear();
  auto& vtp_last_filled = workspace->vtp_last_filled;
  vtp_last_filled.clear();
  auto& current_path = workspace->current_path;
  current_path.clear();
  auto& full_path = workspace->full_path;
  full_path = path;
  assert(full_path.size() > 0);
  auto [node, repetitions, moves_left] = full_path.back();
  // Sometimes receiver is reused, othertimes not, so only jump start if small.
  if (receiver->capacity() < 30) {
    receiver->reserve(receiver->size() + 30);
  }

  // This 1 is 'filled pre-emptively'.
  std::array<float, 256> current_util;
  std::array<bool, 256> visited;

  // These 3 are 'filled on demand'.
  std::array<float, 256> current_score;
  std::array<float, 256> current_weightstarted;


  constexpr int num_top = 8;
  std::array<float, num_top> top_utils;

  auto& cur_iters = workspace->cur_iters;

  Node::Iterator best_edge;
  Node::Iterator second_best_edge;
  // Fetch the current best root node visits for possible smart pruning.
  const int64_t best_node_n = search_->current_best_edge_.GetN();

  int passed_off = 0;
  int completed_visits = 0;

  bool is_root_node = node == search_->root_node_;
  const float even_draw_score = search_->GetDrawScore(false);
  const float odd_draw_score = search_->GetDrawScore(true);
  // Use a local copy of the beam for thread safety during iteration
  const std::vector<Move> current_beam = is_root_node ? search_->GetCurrentBeam() : std::vector<Move>();
  const bool beam_active_local = is_root_node && search_->IsBeamActive();

  const auto& root_move_filter = search_->root_move_filter_;
  auto m_evaluator = moves_left_support_ ? MEvaluator(params_) : MEvaluator();

  int max_limit = std::numeric_limits<int>::max();

  current_path.push_back(-1);
  while (current_path.size() > 0) {
    assert(full_path.size() >= path.size());
    // First prepare visits_to_perform.
    if (current_path.back() == -1) {
      // Need to do n visits, where n is either collision_limit, or comes from
      // visits_to_perform for the current path.
      int cur_limit = collision_limit;
      if (current_path.size() > 1) {
         // Ensure index is valid before accessing visits_to_perform
         if (current_path.size() - 2 < visits_to_perform.size() && current_path[current_path.size() - 2] < 256) {
            cur_limit = (*visits_to_perform[current_path.size() - 2])[current_path[current_path.size() - 2]];
         } else {
             // Handle error or default case if index is invalid
             cur_limit = 0; // Or some other safe default
         }
      }

      // First check if node is terminal or not-expanded. If either than create
      // a collision of appropriate size and pop current_path.
      if (ShouldStopPickingHere(node, is_root_node, repetitions)) {
        if (is_root_node) {
          // Root node is special - since its not reached from anywhere else, so
          // it needs its own logic. Still need to create the collision to
          // ensure the outer gather loop gives up.
          if (node->TryStartScoreUpdate()) {
            cur_limit -= 1;
            // Pass history copy to avoid data race
            receiver->push_back(
                NodeToProcess::Visit(full_path, history));
            completed_visits++;
          }
        }
        // Visits are created elsewhere, just need the collisions here.
        if (cur_limit > 0) {
          int max_count = 0;
          if (cur_limit == collision_limit && path.size() == 1 &&
              max_limit > cur_limit) {
            max_count = max_limit;
          }
          receiver->push_back(
              NodeToProcess::Collision(full_path, cur_limit, max_count));
          completed_visits += cur_limit;
        }
        history.Pop();
        full_path.pop_back();
        if (full_path.size() > 0) {
          std::tie(node, repetitions, moves_left) = full_path.back();
          is_root_node = (node == search_->root_node_); // Update is_root_node
        } else {
          node = nullptr;
          repetitions = 0;
          is_root_node = false; // No longer root if path is empty
        }
        current_path.pop_back();
        continue;
      }
      if (is_root_node) {
        // Root node is again special - needs its n in flight updated separately
        // as its not handled on the path to it, since there isn't one.
        node->IncrementNInFlight(cur_limit);
      }

      // Create visits_to_perform new back entry for this level.
      if (vtp_buffer.size() > 0) {
        visits_to_perform.push_back(std::move(vtp_buffer.back()));
        vtp_buffer.pop_back();
      } else {
        visits_to_perform.push_back(std::make_unique<std::array<int, 256>>());
      }
      vtp_last_filled.push_back(-1);

      // Cache all constant UCT parameters.

      int max_needed = node->GetNumEdges();
      for (int i = 0; i < max_needed; i++) {
        current_util[i] = std::numeric_limits<float>::lowest();
        visited[i] = false;
      }
      for (int i = 0; i < num_top; i++) {
        top_utils[i] = -std::numeric_limits<float>::infinity(); // Use infinity
      }


      // Root depth is 1 here, while for GetDrawScore() it's 0-based, that's why
      // the weirdness.
      const float draw_score =
          (full_path.size() % 2 == 0) ? odd_draw_score : even_draw_score;
      m_evaluator.SetParent(node);
      const float policy_decay_factor =
          ComputePolicyDecayFactor(params_, node->GetWeight());
      float visited_pol = 0.0f;
      for (Node* child : node->VisitedNodes()) {
        int index = child->Index();
        visited_pol += child->GetP();
        float q = child->GetQ(draw_score);
        current_util[index] = q + m_evaluator.GetMUtility(child, q);

        visited[index] = true;

        // we're only counting visited nodes toward top utils
        // since we only boost visited nodes

        for (int i = 0; i < num_top; i++) {
          if (q > top_utils[i]) {
            for (int j = num_top - 1; j > i; j--) {
              top_utils[j] = top_utils[j - 1];
            }
            top_utils[i] = q;
            break;
          }
        }
      }


      const int num_boost_t1 = params_.GetTopPolicyNumBoost();
      const int num_boost_t2 = params_.GetTopPolicyTierTwoNumBoost();

      const float min_policy_boost_util_t1 =
          (num_boost_t1 == 0 || !params_.GetUsePolicyBoosting())
              ? std::numeric_limits<float>::infinity() // Use infinity
              : top_utils[num_boost_t1 - 1];

			const float min_policy_boost_util_t2 =
					(num_boost_t2 == 0 || !params_.GetUsePolicyBoosting())
							? std::numeric_limits<float>::infinity() // Use infinity
							: top_utils[num_boost_t2 - 1];


      const float policy_boost_t1 = params_.GetTopPolicyBoost();
      const float policy_boost_t2 = params_.GetTopPolicyTierTwoBoost();


      const float fpu =
          GetFpu(params_, node, is_root_node, draw_score, visited_pol);
      for (int i = 0; i < max_needed; i++) {
        if (current_util[i] == std::numeric_limits<float>::lowest()) {
          current_util[i] = fpu + m_evaluator.GetDefaultMUtility();
        }
      }


			const float puct_mult =
          ComputeExploreFactor(params_, node->GetWeight(), node->GetWL(),
                               node->GetVS(), node->GetE(), is_root_node);
      int cache_filled_idx = -1;
      while (cur_limit > 0) {
        // Perform UCT for current node.
        float best = std::numeric_limits<float>::lowest();
        int best_idx = -1;
        float best_without_u = std::numeric_limits<float>::lowest();
        float second_best = std::numeric_limits<float>::lowest();
        bool can_exit = false;
        best_edge.Reset();
        second_best_edge.Reset(); // Reset second best edge

        for (int idx = 0; idx < max_needed; ++idx) {
          // Initialize iterator for this index if needed
          if (idx > cache_filled_idx) {
             if (cache_filled_idx == -1) { // First time through
                  cur_iters[idx] = node->Edges();
             } else {
                  cur_iters[idx] = cur_iters[cache_filled_idx]; // Start from previous iterator
                  // Advance only if previous index exists and iterator is valid
                  if (cache_filled_idx >= 0 && cur_iters[cache_filled_idx]) {
                       ++cur_iters[idx];
                  }
             }
             // Check if iterator is valid before accessing
             if (!cur_iters[idx]) {
                 // Reached end of edges unexpectedly or iterator became invalid
                 break; // Exit inner loop
             }
             current_weightstarted[idx] = cur_iters[idx].GetWeightStarted();
          } else {
              // Ensure iterator is still valid even if reusing cached data
              if (!cur_iters[idx]) break;
          }

          // +++ Beam Search Filter +++
          if (is_root_node && beam_active_local) {
              bool in_beam = false;
              Move current_move = cur_iters[idx].GetMove();
              for (const auto& beam_move : current_beam) {
                  if (current_move == beam_move) {
                      in_beam = true;
                      break;
                  }
              }
              if (!in_beam) {
                   if (idx > cache_filled_idx) cache_filled_idx++; // Mark as filled even if skipped
                   continue; // Skip this move if not in beam
              }
          }
          // +++ End Beam Search Filter +++


          float weightstarted = current_weightstarted[idx];
          const float util = current_util[idx];
          if (idx > cache_filled_idx) {
            float p = cur_iters[idx].GetP();

            p = ComputePolicyDecay(policy_decay_factor, p);

            // a small hack to reduce policy on bad moves
            // if (p < 0.01f) p /= 3;
            //if (cur_iters[idx].GetWL(0.0f) < -0.995) p /= 5;
            //else if (cur_iters[idx].GetWL(0.0f) < -0.99) p /= 3;
            //else if (cur_iters[idx].GetWL(0.0f) < -0.95) p /= 2;



            // only boost visited nodes
						if (visited[idx]) {
              if (util >= min_policy_boost_util_t1) {
                p = std::max(p, policy_boost_t1);
              }
              if (util >= min_policy_boost_util_t2) {
                p = std::max(p, policy_boost_t2);
              }
            }


            current_score[idx] =
              p * puct_mult / (1 + weightstarted) + util;
            cache_filled_idx++;
          }
          if (is_root_node) {
            // If there's no chance to catch up to the current best node with
            // remaining playouts, don't consider it.
            // best_move_node_ could have changed since best_node_n was
            // retrieved. To ensure we have at least one node to expand, always
            // include current best node.
            if (cur_iters[idx] != search_->current_best_edge_ &&
                latest_time_manager_hints_.GetEstimatedRemainingPlayouts() <
                    best_node_n - cur_iters[idx].GetN()) {
              continue;
            }
            // If root move filter exists, make sure move is in the list.
            if (!root_move_filter.empty() &&
                std::find(root_move_filter.begin(), root_move_filter.end(),
                          cur_iters[idx].GetMove()) == root_move_filter_.end()) {
              continue;
            }
          }

          float score = current_score[idx];
          if (score > best) {
            second_best = best;
            second_best_edge = best_edge;
            best = score;
            best_idx = idx;
            best_without_u = util;
            best_edge = cur_iters[idx];
          } else if (score > second_best) {
            second_best = score;
            second_best_edge = cur_iters[idx];
          }
          if (can_exit) break;
          if (weightstarted == 0) {
            // One more loop will get 2 unvisited nodes, which is sufficient to
            // ensure second best is correct. This relies upon the fact that
            // edges are sorted in policy decreasing order.
            can_exit = true;
          }
        } // End of for loop through edges

        // Handle case where no edge was selected (e.g., all filtered by beam)
        if (best_idx == -1) {
            // If no move can be selected (e.g., beam filtered everything), break the loop.
            // This might happen if beam becomes very narrow or there are no valid moves.
             LOGFILE << "Warning: No best edge found in PickNodesToExtendTask loop. Breaking.";
             cur_limit = 0; // Ensure loop termination
             continue;
        }


        int new_visits = 0;
        if (second_best_edge) {
          int estimated_visits_to_change_best = std::numeric_limits<int>::max();
          // Avoid division by zero or invalid comparison if second_best is very low
          if (best_without_u < second_best && (second_best - best_without_u) > 1e-9f) {
            const auto n1 = current_weightstarted[best_idx] + 1;
             // Ensure P * puct_mult is positive before division
             float numerator = cur_iters[best_idx].GetP() * puct_mult;
             if (numerator > 0) {
                  estimated_visits_to_change_best = static_cast<int>(
                      std::max(1.0f, std::min(numerator /
                                                      (second_best - best_without_u) -
                                                  n1 + 1,
                                              1e9f)));
             } else {
                 estimated_visits_to_change_best = 1; // If P is zero, need at least 1 visit
             }
          } else if (best_without_u >= second_best) {
              // If best Q is already >= second best score, we might not need many visits
              estimated_visits_to_change_best = 1; // Or potentially calculate differently based on U part
          }
          max_limit = std::min(max_limit, estimated_visits_to_change_best);
          new_visits = std::min(cur_limit, estimated_visits_to_change_best);
        } else {
          // No second best - only one edge, so everything goes in here.
          new_visits = cur_limit;
        }
         // Ensure new_visits is at least 1 if cur_limit > 0
        if (cur_limit > 0 && new_visits <= 0) {
            new_visits = 1;
        }
        // Ensure new_visits doesn't exceed cur_limit
        new_visits = std::min(new_visits, cur_limit);


        // Ensure indices are valid before accessing arrays
        if (best_idx < 0 || best_idx >= 256) {
             LOGFILE << "Error: Invalid best_idx: " << best_idx;
             cur_limit = 0; // Stop processing for safety
             continue;
        }
        if (visits_to_perform.empty() || best_idx >= visits_to_perform.back()->size()) {
             LOGFILE << "Error: Invalid access to visits_to_perform";
             cur_limit = 0;
             continue;
        }
        if (vtp_last_filled.empty() || best_idx < vtp_last_filled.back()) {
           // Fill potentially skipped indices if best_idx jumps ahead
            auto* vtp_array = visits_to_perform.back().get()->data();
            std::fill(vtp_array + (vtp_last_filled.back() + 1),
                     vtp_array + best_idx + 1, 0);
            vtp_last_filled.back() = best_idx; // Update last filled index
        } else if (best_idx > vtp_last_filled.back()) {
            // This case should ideally be handled by the previous block, but double check
            auto* vtp_array = visits_to_perform.back().get()->data();
            std::fill(vtp_array + (vtp_last_filled.back() + 1),
                      vtp_array + best_idx + 1, 0);
            vtp_last_filled.back() = best_idx;
        }

        (*visits_to_perform.back())[best_idx] += new_visits;
        cur_limit -= new_visits;

        Node* child_node = best_edge.GetOrSpawnNode(/* parent */ node);
        history.Append(best_edge.GetMove());
        auto [child_repetitions, child_moves_left] =
            GetRepetitions(full_path.size(), history.Last());
        full_path.push_back({child_node, child_repetitions, child_moves_left});
        if (child_node->TryStartScoreUpdate()) {
          current_weightstarted[best_idx]++;
          new_visits -= 1; // Decrement visits count as one was just used for TryStartScoreUpdate
          if (new_visits < 0) new_visits = 0; // Ensure non-negative

          if (ShouldStopPickingHere(child_node, false, child_repetitions)) {
            // Reduce 1 for the visits_to_perform to ensure the collision
            // created doesn't include this visit.
             if ((*visits_to_perform.back())[best_idx] > 0) { // Only decrement if positive
                (*visits_to_perform.back())[best_idx] -= 1;
             }
            receiver->push_back(NodeToProcess::Visit(full_path, history));
            completed_visits++;
          } else {
            child_node->IncrementNInFlight(new_visits);
            current_weightstarted[best_idx] += new_visits;
          }
           // Update current_score only if the node was not terminal/stopped
           if (!ShouldStopPickingHere(child_node, false, child_repetitions)) {
                current_score[best_idx] = cur_iters[best_idx].GetP() * puct_mult /
                                                (1 + current_weightstarted[best_idx]) +
                                            current_util[best_idx];
           }
        } else {
            // If TryStartScoreUpdate failed (another thread got it),
            // we still need to add the calculated visits to NInFlight
            // for the node that the other thread will process.
             child_node->IncrementNInFlight(new_visits);
             current_weightstarted[best_idx] += new_visits; // Still update our local weight estimate
             current_score[best_idx] = cur_iters[best_idx].GetP() * puct_mult /
                                           (1 + current_weightstarted[best_idx]) +
                                       current_util[best_idx];

        }

        if (best_idx > vtp_last_filled.back() &&
            (*visits_to_perform.back())[best_idx] > 0) {
          vtp_last_filled.back() = best_idx;
        }
        history.Pop();
        full_path.pop_back();
      } // End of while(cur_limit > 0)

      is_root_node = false; // No longer root after the first level
      // Actively do any splits now rather than waiting for potentially long
      // tree walk to get there.
      for (int i = 0; i <= vtp_last_filled.back(); i++) {
        // Ensure iterator is valid before accessing
        if (!cur_iters[i]) continue;

        int child_limit = (*visits_to_perform.back())[i];
        if (params_.GetTaskWorkersPerSearchWorker() > 0 &&
            child_limit > params_.GetMinimumWorkSizeForPicking() &&
            child_limit <
                ((collision_limit - passed_off - completed_visits) * 2 / 3) &&
            child_limit + passed_off + completed_visits <
                collision_limit -
                    params_.GetMinimumRemainingWorkSizeForPicking()) {
          Node* child_node = cur_iters[i].GetOrSpawnNode(/* parent */ node);
          history.Append(cur_iters[i].GetMove());
          auto [child_repetitions, child_moves_left] =
              GetRepetitions(full_path.size(), history.Last());
          full_path.push_back(
              {child_node, child_repetitions, child_moves_left});
          // Don't split if not expanded or terminal.
          if (!ShouldStopPickingHere(child_node, false, child_repetitions)) {
            bool passed = false;
            {
              // Multiple writers, so need mutex here.
              Mutex::Lock lock(picking_tasks_mutex_);
              // Ensure not to exceed size of reservation.
              if (picking_tasks_.size() < MAX_TASKS) {
                picking_tasks_.emplace_back(full_path, history, child_limit);
                task_count_.fetch_add(1, std::memory_order_acq_rel);
                task_added_.notify_all();
                passed = true;
                passed_off += child_limit;
              }
            }
            if (passed) {
              (*visits_to_perform.back())[i] = 0;
            }
          }
          history.Pop();
          full_path.pop_back();
        }
      }
      // Fall through to select the first child.
    } // End of if (current_path.back() == -1)

    int min_idx = current_path.back();
    bool found_child = false;
    if (vtp_last_filled.back() > min_idx) {
      int idx = -1;
      for (auto& child : node->Edges()) {
        idx++;
        if (idx > min_idx && (*visits_to_perform.back())[idx] > 0) {
          current_path.back() = idx;
          current_path.push_back(-1);
          node = child.GetOrSpawnNode(/* parent */ node);
          history.Append(child.GetMove());
          std::tie(repetitions, moves_left) =
              GetRepetitions(full_path.size(), history.Last());
          full_path.push_back({node, repetitions, moves_left});
          found_child = true;
          is_root_node = false; // Not root anymore
          break;
        }
        if (idx >= vtp_last_filled.back()) break;
      }
    }
    if (!found_child) {
      history.Pop();
      full_path.pop_back();
      if (full_path.size() > 0) {
        std::tie(node, repetitions, moves_left) = full_path.back();
         is_root_node = (node == search_->root_node_); // Update is_root_node
      } else {
        node = nullptr;
        repetitions = 0;
        is_root_node = false; // No longer root
      }
      current_path.pop_back();
      // Only pop visits_to_perform and vtp_last_filled if they are not empty
      if (!visits_to_perform.empty()) {
          vtp_buffer.push_back(std::move(visits_to_perform.back()));
          visits_to_perform.pop_back();
      }
      if (!vtp_last_filled.empty()) {
          vtp_last_filled.pop_back();
      }
    }
  } // End of while (current_path.size() > 0)
}


void SearchWorker::ExtendNode(NodeToProcess& picked_node) {
  const auto path = picked_node.path;
  // Check if node exists before accessing it
  if (!std::get<0>(path.back())) {
     LOGFILE << "Error: Null node pointer in ExtendNode.";
     return;
  }
  assert(!std::get<0>(path.back())->GetLowNode());

  const PositionHistory& history = picked_node.history;

  // We don't need the mutex because other threads will see that N=0 and
  // N-in-flight=1 and will not touch this node.
  const auto& board = history.Last().GetBoard();
  std::vector<Move> legal_moves = board.GenerateLegalMoves();

  // Check whether it's a draw/lose by position. Importantly, we must check
  // these before doing the by-rule checks below.
  auto node = picked_node.node;
  if (legal_moves.empty()) {
    // Could be a checkmate or a stalemate
    if (board.IsUnderCheck()) {
      node->MakeTerminal(GameResult::WHITE_WON);
    } else {
      node->MakeTerminal(GameResult::DRAW);
    }
    return;
  }

  // We can shortcircuit these draws-by-rule only if they aren't root;
  // if they are root, then thinking about them is the point.
  if (node != search_->root_node_) {
    if (!board.HasMatingMaterial()) {
      node->MakeTerminal(GameResult::DRAW);
      return;
    }

    if (history.Last().GetRule50Ply() >= 100) {
      node->MakeTerminal(GameResult::DRAW);
      return;
    }

    // Handle repetition draws as pseudo-terminals.
    if (picked_node.repetitions >= 2) {
      // Mark as repetition but don't make terminal yet, let backup handle it
      node->SetRepetition();
      // Need to set bounds appropriately for pseudo-terminal draw
      node->SetBounds(GameResult::DRAW, GameResult::DRAW);

    }
    // Neither by-position or by-rule termination, but maybe it's a TB
    // position.
    else if (search_->syzygy_tb_ && !search_->root_is_in_dtz_ &&
             board.castlings().no_legal_castle() &&
             history.Last().GetRule50Ply() == 0 &&
             (board.ours() | board.theirs()).count() <=
                 search_->syzygy_tb_->max_cardinality()) {
      ProbeState state;
      const WDLScore wdl =
          search_->syzygy_tb_->probe_wdl(history.Last(), &state);
      // Only fail state means the WDL is wrong, probe_wdl may produce correct
      // result with a stat other than OK.
      if (state != FAIL) {
        // TB nodes don't have NN evaluation, assign M from parent node.
        float m = 0.0f;
        if (path.size() > 1) {
          auto parent = std::get<0>(path[path.size() - 2]);
          if (parent) m = std::max(0.0f, parent->GetM() - 1.0f); // Null check parent
        }
        // If the colors seem backwards, check the checkmate check above.
        if (wdl == WDL_WIN) {
          node->MakeTerminal(GameResult::BLACK_WON, m, Terminal::Tablebase);
        } else if (wdl == WDL_LOSS) {
          node->MakeTerminal(GameResult::WHITE_WON, m, Terminal::Tablebase);
        } else {  // Cursed wins and blessed losses count as draws.
          node->MakeTerminal(GameResult::DRAW, m, Terminal::Tablebase);
        }
        search_->tb_hits_.fetch_add(1, std::memory_order_acq_rel);
        return;
      }
    }
  }

  // If node is marked as repetition but not terminal yet, don't query NN
   if (node->IsRepetition() && !node->IsTerminal()) {
       picked_node.nn_queried = false;
       return;
   }

  // If node hasn't been made terminal by rules/TB, query NN
  if (!node->IsTerminal()) {
      picked_node.nn_queried = true;  // Node::SetLowNode() required.

      // Check the transposition table first and NN cache second before asking for
      // NN evaluation.
      picked_node.hash = search_->dag_->GetHistoryHash(history);
      picked_node.ch_hash = search_->dag_->GetCHHash(history);

      auto tt_low_node = search_->dag_->TTFind(picked_node.hash);
      if (tt_low_node != nullptr) {
        picked_node.tt_low_node = tt_low_node;
        picked_node.is_tt_hit = true;
      } else {
        if (params_.GetMoveRuleBucketing()) {
          int my_ply = picked_node.GetRule50Ply();
          int bs;

          if (my_ply <= 64) {
            bs = 32;

          } else if (my_ply <= 80) {
            bs = 16;
          } else if (my_ply <= 92) {
            bs = 4;
          } else {
            bs = 1;
          }
          int ply_lo = my_ply / bs * bs;
          int ply_hi = ply_lo + bs - 1;

          int max_visits = 0;
          LowNode* twin_low_node = nullptr;
          for (int ply = ply_lo; ply <= ply_hi; ply++) {
            uint64_t hash = search_->dag_->GetHistoryHash(history, ply);
            auto low_node = search_->dag_->TTFind(hash);
            if (low_node != nullptr) {
              int visits = low_node->GetN();
              if (visits > max_visits) {
                max_visits = visits;
                twin_low_node = low_node;
              }
            }
          }
          if (twin_low_node != nullptr) {
            picked_node.twin_low_node = twin_low_node;
            picked_node.is_twin_hit = true;
          }
        }  // end if (params_->GetMoveRuleBucketing()) {

        picked_node.lock = NNCacheLock(search_->cache_, picked_node.hash);
        picked_node.is_cache_hit = picked_node.lock;
      }
   } else {
       // If node is already terminal, no need to query NN
       picked_node.nn_queried = false;
   }

}

// 2b. Copy collisions into shared collisions.
void SearchWorker::CollectCollisions() {
  SharedMutex::Lock lock(search_->nodes_mutex_);

  for (const NodeToProcess& node_to_process : minibatch_) {
    if (node_to_process.IsCollision()) {
      search_->shared_collisions_.emplace_back(node_to_process.path,
                                               node_to_process.multivisit);
    }
  }
}

// 4. Run NN computation.
// ~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::RunNNComputation() {
  computation_->ComputeBlocking(params_.GetPolicySoftmaxTemp());
}

// 5. Retrieve NN computations (and terminal values) into nodes.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::FetchMinibatchResults() {
  SharedMutex::Lock nodes_lock(search_->nodes_mutex_);
  // Populate NN/cached results, or terminal results, into nodes.
  int idx_in_computation = 0;
  for (auto& node_to_process : minibatch_) {
    FetchSingleNodeResult(&node_to_process, *computation_, idx_in_computation);
    if (node_to_process.ShouldAddToInput()) ++idx_in_computation;
  }
}

template <typename Computation>
void SearchWorker::FetchSingleNodeResult(NodeToProcess* node_to_process,
                                         const Computation& computation,
                                         int idx_in_computation)
    REQUIRES(search_->nodes_mutex_) {
  if (!node_to_process->nn_queried) return; // Skip if NN wasn't queried

  Node* node = node_to_process->node;
  // If node is already terminal (e.g., set by repetition check), don't overwrite
  if (node->IsTerminal() && !node->GetLowNode()) return;


  if (!node_to_process->is_tt_hit) {
    if (node_to_process->is_twin_hit) {
      LowNode twin_low_node = *(node_to_process->twin_low_node);
      auto [tt_low_node, is_tt_miss] =
          search_->dag_->TTGetOrCreate(twin_low_node, node_to_process->hash);
      assert(tt_low_node != nullptr);
      tt_low_node->MakeTwin();
      node_to_process->tt_low_node = tt_low_node;

    } else {
      auto [tt_low_node, is_tt_miss] =
          search_->dag_->TTGetOrCreate(node_to_process->hash);
      assert(tt_low_node != nullptr);

      node_to_process->tt_low_node = tt_low_node;
      if (is_tt_miss) {
         // Check if NNEval exists before accessing
         auto nn_eval_ptr = computation.GetNNEval(idx_in_computation);
         if(nn_eval_ptr) {
            auto nn_eval = nn_eval_ptr.get();
            if (params_.GetWDLRescaleRatio() != 1.0f ||
                (params_.GetWDLRescaleDiff() != 0.0f &&
                 search_->contempt_mode_ != ContemptMode::NONE)) {
              // Check whether root moves are from the set perspective.
              bool root_stm = search_->contempt_mode_ == ContemptMode::WHITE;
              auto sign = (root_stm ^ node_to_process->history.IsBlackToMove())
                              ? 1.0f
                              : -1.0f;
              float v = nn_eval->q;
              float d = nn_eval->d;
              WDLRescale(v, d, params_.GetWDLRescaleRatio(),
                        search_->contempt_mode_ == ContemptMode::NONE
                            ? 0
                            : params_.GetWDLRescaleDiff(),
                        sign, false, params_.GetWDLMaxS()); // Pass WDLMaxS
              nn_eval->q = v;
              nn_eval->d = d;
            }
            node_to_process->tt_low_node->SetNNEval(nn_eval);
            node_to_process->tt_low_node->SetCHHash(node_to_process->ch_hash);
         } else {
             // Handle case where NNEval is missing (shouldn't happen if ShouldAddToInput was true)
             LOGFILE << "Error: Missing NNEval for index " << idx_in_computation;
             // Potentially make the node terminal or handle error otherwise
             node->MakeTerminal(GameResult::DRAW); // Example fallback
             return; // Skip setting low node
         }
      }
    }
  }

  // Add NN results to node.
  // Ensure LowNode exists before setting
  if (node_to_process->tt_low_node) {
      // Add Dirichlet noise if enabled and at root.
      if (params_.GetNoiseEpsilon() && node == search_->root_node_) {
        auto low_node = search_->dag_->NonTTAddClone(*node_to_process->tt_low_node);
        assert(low_node != nullptr);
        node->SetLowNode(low_node);
        ApplyDirichletNoise(node, params_.GetNoiseEpsilon(),
                            params_.GetNoiseAlpha());
        node->SortEdges();
      } else {
        node->SetLowNode(node_to_process->tt_low_node);
      }
  } else {
       LOGFILE << "Error: tt_low_node is null in FetchSingleNodeResult for hash " << node_to_process->hash;
       // Handle error case, maybe make node terminal?
       node->MakeTerminal(GameResult::DRAW);
  }

}

// 6. Propagate the new nodes' information to all their parents in the tree.
// ~~~~~~~~~~~~~~
void SearchWorker::DoBackupUpdate() {
  // Nodes mutex for doing node updates.
  SharedMutex::Lock lock(search_->nodes_mutex_);

  bool work_done = number_out_of_order_ > 0;
  for (const NodeToProcess& node_to_process : minibatch_) {
    DoBackupUpdateSingleNode(node_to_process);
    if (!node_to_process.IsCollision()) {
      work_done = true;
    }
  }
  if (!work_done) return;
  search_->CancelSharedCollisions();
  search_->total_batches_ += 1;
}

bool SearchWorker::MaybeAdjustForTerminalOrTransposition(
    Node* n, const LowNode* nl, float& v, float& d, float& m, float& vs,
    uint32_t& n_to_fix, float& weight_to_fix, float& v_delta, float& d_delta,
    float& m_delta, float& vs_delta, bool& update_parent_bounds) const {

  if (!n || !nl) return false; // Add null checks

  if (n->IsTerminal()) {
    v = n->GetWL();
    d = n->GetD();
    m = n->GetM();
    vs = n->GetVS();

    return true;
  }

  // Use information from transposition or a new terminal.
  if (nl->IsTransposition() ||
      nl->IsTerminal()) {
    // Adapt information from low node to node by flipping Q sign, bounds,
    // result and incrementing m.
    v = -nl->GetWL();
    d = nl->GetD();
    m = nl->GetM() + 1;
    vs = nl->GetVS();
    // When starting at or going through a transposition/terminal, make sure to
    // use the information it has already acquired.
    n_to_fix = n->GetN();
    weight_to_fix = n->GetWeight();
    v_delta = v - n->GetWL();
    d_delta = d - n->GetD();
    m_delta = m - n->GetM();
    vs_delta = vs - n->GetVS();
    // Update bounds.
    if (params_.GetStickyEndgames()) {
      auto tt = nl->GetTerminalType();
      if (tt != Terminal::NonTerminal) {
        GameResult r;
        if (v == 1.0f) {
          r = GameResult::WHITE_WON;
        } else if (v == -1.0f) {
          r = GameResult::BLACK_WON;
        } else {
          r = GameResult::DRAW;
        }

        n->MakeTerminal(r, m, tt);
        update_parent_bounds = true;
      } else {
        auto [lower, upper] = nl->GetBounds();
        n->SetBounds(-upper, -lower);
      }
    }

    return true;
  }

  return false;
}

// Use information from terminal status or low node to update node and node's
// parent low node and so on until the root is reached. Low node may become a
// transposition and/or get more information even during this batch. Both low
// node and node may adjust bounds and become a terminal during this batch.
void SearchWorker::DoBackupUpdateSingleNode(
    const NodeToProcess& node_to_process) REQUIRES(search_->nodes_mutex_) {
  if (node_to_process.IsCollision()) {
    // Collisions are handled via shared_collisions instead.
    return;
  }

  auto path = node_to_process.path;
  auto [n, nr, nm] = path.back();
  // Check if node 'n' is null before proceeding
  if (!n) {
       LOGFILE << "Error: Null node pointer at the end of path in DoBackupUpdateSingleNode.";
       return;
  }

  // For the first visit to a terminal, maybe update parent bounds too.
  auto update_parent_bounds =
      params_.GetStickyEndgames() && n->IsTerminal() && !n->GetN();
  auto nl = n->GetLowNode();
  float v = 0.0f;
  float d = 0.0f;
  float m = 0.0f;
  float vs = v * v;
  uint32_t n_to_fix = 0;
  float weight_to_fix = 0.0f;
  float v_delta = 0.0f;
  float d_delta = 0.0f;
  float m_delta = 0.0f;
  float vs_delta = 0.0f;

  bool use_correction_history = params_.GetUseCorrectionHistory();

  float ch_delta;
  CorrHistEntry* ntp_cht_entry;

  if (use_correction_history) {
    ntp_cht_entry =
        search_->dag_->CHTGetOrCreate(node_to_process.ch_hash);
    ch_delta = (ntp_cht_entry && ntp_cht_entry->weightSum > 1e-9f) // Add null check and tolerance
                   ? ntp_cht_entry->deltaSum / ntp_cht_entry->weightSum
                   : 0.0f;
  }
  else {
    ch_delta = 0;
    ntp_cht_entry = nullptr;
  }
  float ch_lambda = params_.GetCorrectionHistoryLambda();
  // float ch_alpha = params_.GetCorrectionHistoryAlpha(); // ch_alpha is unused currently


  // Update the low node at the start of the backup path first, but only visit
  // it the first time that backup sees it.
  float avg_weight;
  if (nl) {
    avg_weight = ComputeWeight(params_, nl->GetE());
    n->SetE(nl->GetE());

  } else {
		// game is over so uncertainty is highest possible
    if (params_.GetUseUncertaintyWeighting()) {
      avg_weight = params_.GetUncertaintyWeightingCap();
    }
    else {
      avg_weight = 1.0f;
    }
    n->SetE(-1.0f); // Indicate terminal node
  }

  // Apply EasyEvalWeightDecay if the node evaluation was easy
  if (!node_to_process.ShouldAddToInput()) {
    avg_weight *= params_.GetEasyEvalWeightDecay();
  }


  if (nl && nl->GetN() == 0) {

    float wl_corrected = nl->GetWL();
    if (use_correction_history && !nl->IsTwin() && !nl->IsTerminal()) {
      wl_corrected += ch_lambda * ch_delta;
      wl_corrected = std::clamp(wl_corrected, -1.0f, 1.0f);
    }

    nl->FinalizeScoreUpdate(
       wl_corrected, nl->GetD(), nl->GetM(), nl->GetVS(),
        node_to_process.multivisit,
        node_to_process.multivisit * avg_weight, false);

    // for testing cht is per node
    if (ntp_cht_entry != nullptr && !nl->IsTwin()) {
      nl->SetCHTEntry(ntp_cht_entry);
      ntp_cht_entry->numMembers++;
    }

  }

  if (nr >= 2) {
    // Three-fold itself has to be handled as a terminal to produce relevant
    // results. Unlike two-folds that can keep updating their "real" values.
    n->SetRepetition();
    v = 0.0f;
    d = 1.0f;
    m = 1;
    vs = 0.0f; // Set VS for draw
  } else if (!MaybeAdjustForTerminalOrTransposition(
                 n, nl, v, d, m, vs, n_to_fix, weight_to_fix, v_delta, d_delta,
                 m_delta, vs_delta, update_parent_bounds)) {
    // If there is nothing better, use original NN values adjusted for node.
     if (nl) { // Ensure low node exists before accessing values
        v = -nl->GetWL();
        d = nl->GetD();
        m = nl->GetM() + 1;
        vs = nl->GetVS();
     } else {
         // Handle case where low node doesn't exist (shouldn't happen if not terminal/TT)
         // Fallback or error handling might be needed here. Defaulting to 0 for now.
         v = 0.0f; d = 1.0f; m = 1.0f; vs = 0.0f;
         LOGFILE << "Warning: LowNode missing in backup where expected.";
     }

  }

  // Backup V value up to a root. After 1 visit, V = Q.
  for (auto it = path.crbegin(); it != path.crend();
       /* ++it in the body */) {
    // Get current node 'n' from iterator
    n = std::get<0>(*it);
    nr = std::get<1>(*it);
    nm = std::get<2>(*it);

    // Ensure node 'n' is not null before proceeding
    if (!n) {
        LOGFILE << "Error: Null node pointer encountered during backup path traversal.";
        break; // Stop backup if path is corrupted
    }

    n->FinalizeScoreUpdate(
        v, d, m, vs, node_to_process.multivisit,
        node_to_process.multivisit * avg_weight);
    if (n_to_fix > 0 && !n->IsTerminal()) {
      // First part of the path might be never as it was removed and recreated.
      n_to_fix = std::min(n_to_fix, n->GetN());
      weight_to_fix = std::min(weight_to_fix, n->GetWeight());
      n->AdjustForTerminal(v_delta, d_delta, m_delta, vs_delta, n_to_fix,
                           weight_to_fix);
    }

    // Stop delta update on repetition "terminal" and propagate a draw above
    // repetitions valid on the current path.
    // Only do this after edge update to have good values if play goes here.
    if (nr == 1 && !n->IsTerminal()) {
      n->SetRepetition();
      v = 0.0f;
      d = 1.0f;
      m = nm + 1;
      vs = 0.0f; // Set VS for draw
    }
    if (n->IsRepetition()) {
      n_to_fix = 0;
      weight_to_fix = 0;
    }

    // Nothing left to do without ancestors to update.
    if (++it == path.crend()) break;
    auto [p, pr, pm] = *it;
    // Ensure parent node 'p' is not null
    if (!p) {
        LOGFILE << "Error: Null parent node pointer encountered during backup.";
        break;
    }

    LowNode* pl = p->GetLowNode();

    // Add null check for parent's low node
    if (!pl) {
        LOGFILE << "Error: Null parent LowNode pointer encountered during backup.";
        // This might indicate an issue, perhaps stop backup or handle differently
         v = -v; // Flip values for parent even if LowNode is missing? Decide on handling.
         v_delta = -v_delta;
         m++;
        continue; // Skip LowNode update and bound setting for this parent
    }


    assert(!p->IsTerminal() ||
           (p->IsTerminal() && pl->IsTerminal() && p->GetWL() == -pl->GetWL() &&
            p->GetD() == pl->GetD()));
    // If parent low node is already a (new) terminal, then change propagated
    // values and stop terminal adjustment.


    if (pl->IsTerminal()) {
      v = pl->GetWL();
      d = pl->GetD();
      m = pl->GetM();
      vs = pl->GetVS();
      n_to_fix = 0;
      weight_to_fix = 0.0f;
    }
    pl->FinalizeScoreUpdate(
        v, d, m, vs, node_to_process.multivisit,
        node_to_process.multivisit * avg_weight);
    if (n_to_fix > 0) {
      pl->AdjustForTerminal(v_delta, d_delta, m_delta, vs_delta, n_to_fix,
                            weight_to_fix);
    }



    bool old_update_parent_bounds = update_parent_bounds;
    // Try setting parent bounds except the root or those already terminal.
    update_parent_bounds =
        update_parent_bounds && p != search_->root_node_ && !pl->IsTerminal() &&
        MaybeSetBounds(p, m, &n_to_fix, &weight_to_fix, &v_delta, &d_delta,
                       &m_delta, &vs_delta);

    // Q will be flipped for opponent.
    v = -v;
    v_delta = -v_delta;
    m++;

    MaybeAdjustForTerminalOrTransposition(
        p, pl, v, d, m, vs, n_to_fix, weight_to_fix, v_delta, d_delta, m_delta,
        vs_delta, update_parent_bounds);

    // Update the stats.
    // Best move.
    // If update_parent_bounds was set, we just adjusted bounds on the
    // previous loop or there was no previous loop, so if n is a terminal, it
    // just became that way and could be a candidate for changing the current
    // best edge. Otherwise a visit can only change best edge if its to an edge
    // that isn't already the best and the new n is equal or greater to the old
    // n.
    if (p == search_->root_node_) { // Check against parent 'p' now
         // If the child 'n' (which was just updated) caused the parent 'p' to change state,
         // or if 'n' is now better than the current best edge.
         if ((old_update_parent_bounds && n->IsTerminal()) ||
             (n != search_->current_best_edge_.node() &&
              search_->current_best_edge_.GetWeight() <= n->GetWeight())) {
             // Recalculate best edge for the root 'p'
             search_->current_best_edge_ = search_->GetBestChildNoTemperature(search_->root_node_, 0);
         }
         // If beam just became active, re-evaluate best edge
         if (search_->beam_active_ && !search_->current_best_edge_ && search_->root_node_->GetN() > 0) {
             search_->current_best_edge_ = search_->GetBestChildNoTemperature(search_->root_node_, 0);
         }
    }

    // 'n' for the next iteration is the current parent 'p'
    // (Already handled by the loop structure and `auto [p, pr, pm] = *it;`)
  } // End of backup loop

  search_->total_playouts_ += node_to_process.multivisit;
  search_->cum_depth_ +=
      node_to_process.path.size() * node_to_process.multivisit;
  search_->max_depth_ =
      std::max(search_->max_depth_, (uint16_t)node_to_process.path.size());
  if (!node_to_process.is_tt_hit) {
    search_->total_low_nodes_++;
  }
  if (node_to_process.ShouldAddToInput()) {
    search_->total_nn_queries_++;
  }
}

bool SearchWorker::MaybeSetBounds(Node* p, float m, uint32_t* n_to_fix,
                                  float* weight_to_fix, float* v_delta,
                                  float* d_delta, float* m_delta,
                                  float* vs_delta) const {
  auto losing_m = 0.0f;
  auto prefer_tb = false;

  // Determine the maximum (lower, upper) bounds across all edges.
  // (-1,-1) Loss (initial and lowest bounds)
  // (-1, 0) Can't Win
  // (-1, 1) Regular node
  // ( 0, 0) Draw
  // ( 0, 1) Can't Lose
  // ( 1, 1) Win (highest bounds)
  auto lower = GameResult::BLACK_WON;
  auto upper = GameResult::BLACK_WON;
  for (const auto& edge : p->Edges()) {
    const auto [edge_lower, edge_upper] = edge.GetBounds();
    lower = std::max(edge_lower, lower);
    upper = std::max(edge_upper, upper);

    // Checkmate is the best, so short-circuit.
    const auto is_tb = edge.IsTbTerminal();
    if (edge_lower == GameResult::WHITE_WON && !is_tb) {
      prefer_tb = false;
      break;
    } else if (edge_upper == GameResult::BLACK_WON) {
      // Track the longest loss.
      losing_m = std::max(losing_m, edge.GetM(0.0f));
    }
    prefer_tb = prefer_tb || is_tb;
  }

  // The parent's bounds are flipped from the children (-max(U), -max(L))
  // aggregated as if it was a single child (forced move) of the same bound.
  //       Loss (-1,-1) -> ( 1, 1) Win
  //  Can't Win (-1, 0) -> ( 0, 1) Can't Lose
  //    Regular (-1, 1) -> (-1, 1) Regular
  //       Draw ( 0, 0) -> ( 0, 0) Draw
  // Can't Lose ( 0, 1) -> (-1, 0) Can't Win
  //        Win ( 1, 1) -> (-1,-1) Loss

  // Nothing left to do for ancestors if the parent would be a regular node.
  auto pl = p->GetLowNode();
   // Add null check for parent low node
  if (!pl) return false;

  if (lower == GameResult::BLACK_WON && upper == GameResult::WHITE_WON) {
    return false;
  } else if (lower == upper) {
    // Search can stop at the parent if the bounds can't change anymore, so make
    // it terminal preferring shorter wins and longer losses.
    *n_to_fix = p->GetN();
    *weight_to_fix = p->GetWeight();
    assert(*n_to_fix > 0);
    assert(*weight_to_fix > 0.0f);
    pl->MakeTerminal(
        upper, (upper == GameResult::BLACK_WON ? std::max(losing_m, m) : m),
        prefer_tb ? Terminal::Tablebase : Terminal::EndOfGame);
    // v, d and m will be set in MaybeAdjustForTerminalOrTransposition.
    *v_delta = pl->GetWL() + p->GetWL();
    *d_delta = pl->GetD() - p->GetD();
    *m_delta = pl->GetM() + 1 - p->GetM();
    *vs_delta = pl->GetVS() - p->GetVS();
    p->MakeTerminal(
        -upper,
        (upper == GameResult::BLACK_WON ? std::max(losing_m, m) : m) + 1.0f,
        prefer_tb ? Terminal::Tablebase : Terminal::EndOfGame);
  } else {
    pl->SetBounds(lower, upper);
    p->SetBounds(-upper, -lower);
  }

  // Bounds were set, so indicate we should check the parent too.
  return true;
}


// 7. Update the Search's status and progress information.
//~~~~~~~~~~~~~~~~~~~~
void SearchWorker::UpdateCounters() {
  search_->PopulateCommonIterationStats(&iteration_stats_);
  search_->MaybeTriggerStop(iteration_stats_, &latest_time_manager_hints_);
  search_->MaybeOutputInfo();

  // If this thread had no work, not even out of order, then sleep for some
  // milliseconds. Collisions don't count as work, so have to enumerate to find
  // out if there was anything done.
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

}  // namespace lczero
