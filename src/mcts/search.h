/* ... (copyright notice) ... */
#pragma once

#include <array>
#include <condition_variable>
#include <functional>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <tuple>
#include <vector>
#include <limits>
#include <memory> // Include for std::unique_ptr, std::shared_ptr

#include "chess/callbacks.h"
#include "chess/uciloop.h"
#include "mcts/params.h"
#include "mcts/stoppers/timemgr.h"
#include "neural/cache.h"   // <<< INCLUDE FULL DEFINITION
#include "mcts/node.h"      // <<< INCLUDE FULL DEFINITION
#include "utils/logging.h"
#include "utils/mutex.h"

namespace lczero {

// Forward declarations for types only used as pointers/references
class Network;
class SyzygyTablebase;
class UciResponder;
class SearchStopper;
// PositionHistory, MoveList are included via node.h -> position.h/move.h

typedef std::vector<std::tuple<Node*, int, int>> BackupPath;

class Search {
 public:
  Search(NodeTree* dag, Network* network,
         std::unique_ptr<UciResponder> uci_responder,
         const MoveList& searchmoves,
         std::chrono::steady_clock::time_point start_time,
         std::unique_ptr<SearchStopper> stopper, bool infinite, bool ponder,
         const OptionsDict& options, NNCache* cache,
         SyzygyTablebase* syzygy_tb);

  ~Search();

  void StartThreads(size_t how_many);
  void RunBlocking(size_t threads);
  void Stop();
  void Abort();
  void Wait();
  bool IsSearchActive() const;
  std::pair<Move, Move> GetBestMove();
  Eval GetBestEval(Move* move = nullptr, bool* is_terminal = nullptr) const;
  std::int64_t GetTotalPlayouts() const;
  const SearchParams& GetParams() const { return params_; }

  const std::vector<Move>& GetCurrentBeam() const { return current_beam_; }
  bool IsBeamActive() const { return beam_active_; }

  void ResetBestMove();
  NNCacheLock GetCachedNNEval(const PositionHistory& history) const;

 private:
  // EdgeAndNode is now defined via node.h include

  // Add GUARDED_BY/REQUIRES annotations back if needed for thread safety analysis
  void EnsureBestMoveKnown();
  EdgeAndNode GetBestChildNoTemperature(Node* parent, int depth) const;
  std::vector<EdgeAndNode> GetBestChildrenNoTemperature(Node* parent, int count, int depth) const;
  EdgeAndNode GetBestRootChildWithTemperature(float temperature) const;

  int64_t GetTimeSinceStart() const;
  int64_t GetTimeSinceFirstBatch() const;
  void MaybeTriggerStop(const IterationStats& stats, StoppersHints* hints);
  void MaybeOutputInfo();
  void SendUciInfo();
  void FireStopInternal();

  void UpdateRootBeam();
  void SendMovesStats() const;
  void WatchdogThread();
  void PopulateCommonIterationStats(IterationStats* stats);
  std::vector<std::string> GetVerboseStats(Node* node) const;
  float GetDrawScore(bool is_odd_depth) const;
  void CancelSharedCollisions();

  mutable Mutex counters_mutex_;
  std::atomic<bool> stop_{false};
  std::condition_variable watchdog_cv_;
  bool ok_to_respond_bestmove_;
  bool bestmove_is_sent_;
  Move final_bestmove_;
  Move final_pondermove_;
  std::unique_ptr<SearchStopper> stopper_;

  Mutex threads_mutex_;
  std::vector<std::thread> threads_;

  Node* root_node_;
  NNCache* cache_;
  NodeTree* dag_;
  SyzygyTablebase* syzygy_tb_;
  const PositionHistory& played_history_;

  Network* const network_;
  const SearchParams params_;
  const MoveList searchmoves_;
  const std::chrono::steady_clock::time_point start_time_;
  int64_t initial_visits_;
  bool root_is_in_dtz_ = false;
  std::atomic<int> tb_hits_{0};
  const MoveList root_move_filter_;

  mutable SharedMutex nodes_mutex_;
  EdgeAndNode current_best_edge_; // Now fully defined
  Edge* last_outputted_info_edge_ = nullptr;
  ThinkingInfo last_outputted_uci_info_;
  int64_t total_playouts_ = 0;
  int64_t total_low_nodes_ = 0;
  int64_t total_nn_queries_ = 0;
  int64_t total_batches_ = 0;
  uint16_t max_depth_ = 0;
  uint64_t cum_depth_ = 0;

  std::optional<std::chrono::steady_clock::time_point> nps_start_time_;

  std::atomic<int> pending_searchers_{0};
  std::atomic<int> backend_waiting_counter_{0};
  std::atomic<int> thread_count_{0};

  std::vector<std::pair<const BackupPath, int>> shared_collisions_;

  std::unique_ptr<UciResponder> uci_responder_;
  ContemptMode contempt_mode_;

  std::vector<Move> current_beam_;
  int current_beam_width_ = 0;
  uint64_t next_beam_update_visits_ = 0;
  uint64_t last_beam_update_visits_ = 0;
  bool beam_active_ = false;

  friend class SearchWorker;
};


class CachingComputation; // Forward declare this if needed by SearchWorker

class SearchWorker {
 public:
  SearchWorker(Search* search, const SearchParams& params, int id);
  ~SearchWorker();
  void RunBlocking();
  void ExecuteOneIteration();
  void InitializeIteration(std::unique_ptr<NetworkComputation> computation); // Now fully defined
  void GatherMinibatch();
  void CollectCollisions();
  void RunNNComputation();
  void FetchMinibatchResults();
  void DoBackupUpdate();
  void UpdateCounters();

 private:
  struct NodeToProcess {
    bool IsExtendable() const {
        return !is_collision && node && !node->IsTerminal() && !node->GetLowNode(); // GetLowNode is defined in node.h
    }
    bool IsCollision() const { return is_collision; }
    bool CanEvalOutOfOrder() const {
        return is_tt_hit || is_cache_hit || (node && (node->IsTerminal() || node->GetLowNode())); // GetLowNode is defined
    }
    bool ShouldAddToInput() const {
      return nn_queried && !is_tt_hit && !is_twin_hit;
    }
    int GetRule50Ply() const { return history.Last().GetRule50Ply(); }

    BackupPath path;
    Node* node;
    uint32_t multivisit = 0;
    uint32_t maxvisit = 0;
    float error = 0.0f;
    bool nn_queried = false;
    bool is_tt_hit = false;
    bool is_twin_hit = false;
    bool is_cache_hit = false;
    bool is_collision = false;
    float twin_error;
    uint64_t hash;
    uint64_t ch_hash;
    LowNode* tt_low_node;      // LowNode defined in node.h
    LowNode* twin_low_node;    // LowNode defined in node.h
    NNCacheLock lock;          // NNCacheLock defined in cache.h
    PositionHistory history;   // PositionHistory defined in position.h (via node.h)
    bool ooo_completed = false;
    int repetitions = 0;

    static NodeToProcess Collision(const BackupPath& path, int collision_count,
                                   int max_count) {
      return NodeToProcess(path, collision_count, max_count);
    }
    static NodeToProcess Visit(const BackupPath& path,
                               const PositionHistory& history) {
      return NodeToProcess(path, history);
    }

    void SetR50Bounds(NodeTree* /*dag*/) {}

    // NNEval defined in node.h
    std::shared_ptr<NNEval> GetNNEval(int) const {
        return lock ? lock->eval : nullptr;
    }

    std::string DebugString() const;

   private:
    NodeToProcess(const BackupPath& path, uint32_t multivisit,
                  uint32_t max_count)
        : path(path),
          node(std::get<0>(path.back())),
          multivisit(multivisit),
          maxvisit(max_count),
          is_collision(true),
          repetitions(0),
          tt_low_node(nullptr),
          twin_low_node(nullptr)
          {}
    NodeToProcess(const BackupPath& path, const PositionHistory& in_history)
        : path(path),
          node(std::get<0>(path.back())),
          multivisit(1),
          maxvisit(0),
          is_collision(false),
          history(in_history),
          repetitions(std::get<1>(path.back())),
          tt_low_node(nullptr),
          twin_low_node(nullptr)
          {}
  };

  struct TaskWorkspace {
    std::array<Node::Iterator, 256> cur_iters; // Node::Iterator defined in node.h
    std::vector<std::unique_ptr<std::array<int, 256>>> vtp_buffer;
    std::vector<std::unique_ptr<std::array<int, 256>>> visits_to_perform;
    std::vector<int> vtp_last_filled;
    std::vector<int> current_path;
    BackupPath full_path;
    TaskWorkspace() {
      vtp_buffer.reserve(30);
      visits_to_perform.reserve(30);
      vtp_last_filled.reserve(30);
      current_path.reserve(30);
      full_path.reserve(30);
    }
  };

  struct PickTask {
    enum PickTaskType { kGathering, kProcessing };
    PickTaskType task_type;
    BackupPath start_path;
    Node* start;
    int collision_limit;
    PositionHistory history;
    std::vector<NodeToProcess> results;
    int start_idx;
    int end_idx;
    bool complete = false;

    PickTask(const BackupPath& start_path, const PositionHistory& in_history,
             int collision_limit)
        : task_type(kGathering),
          start_path(start_path),
          start(std::get<0>(start_path.back())),
          collision_limit(collision_limit),
          history(in_history),
          start_idx(0), end_idx(0)
           {}
    PickTask(int start_idx, int end_idx)
        : task_type(kProcessing), start_idx(start_idx), end_idx(end_idx),
          start(nullptr), collision_limit(0)
          {}
  };

  bool MaybeAdjustForTerminalOrTransposition(
      Node* n, const LowNode* nl, float& v, float& d, float& m, float& vs,
      uint32_t& n_to_fix, float& weight_to_fix, float& v_delta, float& d_delta,
      float& m_delta, float& vs_delta, bool& update_parent_bounds) const;
  void DoBackupUpdateSingleNode(const NodeToProcess& node_to_process);
  bool MaybeSetBounds(Node* p, float m, uint32_t* n_to_fix,
                      float* weight_to_fix, float* v_delta, float* d_delta,
                      float* m_delta, float* vs_delta) const;
  // void PickNodesToExtend(int collision_limit); // Definition moved to .cc
  void PickNodesToExtendTask(const BackupPath& path, int collision_limit,
                             PositionHistory& history,
                             std::vector<NodeToProcess>* receiver,
                             TaskWorkspace* workspace);

  std::pair<int, int> GetRepetitions(int depth, const Position& position);
  bool ShouldStopPickingHere(Node* node, bool is_root_node, int repetitions);
  void ProcessPickedTask(int batch_start, int batch_end);
  void ExtendNode(NodeToProcess& picked_node);
  template <typename Computation>
  void FetchSingleNodeResult(NodeToProcess* node_to_process,
                             const Computation& computation,
                             int idx_in_computation);
  void RunTasks(int tid);
  void ResetTasks();
  int WaitForTasks();

  Search* const search_;
  std::vector<NodeToProcess> minibatch_;
  std::unique_ptr<CachingComputation> computation_; // Needs CachingComputation definition
  PositionHistory history_;
  uint32_t number_out_of_order_ = 0;
  const SearchParams& params_;
  std::unique_ptr<Node> precached_node_;
  const bool moves_left_support_;
  IterationStats iteration_stats_;
  StoppersHints latest_time_manager_hints_;

  Mutex picking_tasks_mutex_;
  std::vector<PickTask> picking_tasks_;
  std::atomic<int> task_count_ = -1;
  std::atomic<int> task_taking_started_ = 0;
  std::atomic<int> tasks_taken_ = 0;
  std::atomic<int> completed_tasks_ = 0;
  std::condition_variable task_added_;
  std::vector<std::thread> task_threads_;
  std::vector<TaskWorkspace> task_workspaces_;
  TaskWorkspace main_workspace_;
  bool exiting_ = false;
};

}  // namespace lczero
