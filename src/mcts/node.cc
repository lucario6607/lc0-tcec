/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  ... (License remains the same) ...
*/

#include "mcts/node.h" // Adjusted include path

#include <absl/algorithm/container.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <list>
#include <memory>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <iomanip> // Added for std::setprecision etc.
#include <vector> // Use std::vector

#include "neural/encoder.h"
#include "neural/network.h" // Include for EvalResult etc.
#include "utils/exception.h"
#include "utils/hashcat.h"
#include "chess/position.h" // Included for PositionHistory/MoveList

namespace lczero { // No classic namespace

/////////////////////////////////////////////////////////////////////////
// Node garbage collector
/////////////////////////////////////////////////////////////////////////

namespace {
// Periodicity of garbage collection, milliseconds.
const int kGCIntervalMs = 100;

// Every kGCIntervalMs milliseconds release nodes in a separate GC thread.
class NodeGarbageCollector {
 public:
  NodeGarbageCollector() : gc_thread_([this]() { Worker(); }) {}

  // Takes ownership of a subtree, to dispose it in a separate thread when
  // it has time.
  void AddToGcQueue(std::unique_ptr<Node> node, size_t solid_size = 0) {
    if (!node) return;
    Mutex::Lock lock(gc_mutex_);
    subtrees_to_gc_.emplace_back(std::move(node));
    subtrees_to_gc_solid_size_.push_back(solid_size);
  }

  ~NodeGarbageCollector() {
    // Flips stop flag and waits for a worker thread to stop.
    stop_.store(true);
    gc_thread_.join();
  }

 private:
  void GarbageCollect() {
    while (!stop_.load()) {
      // Node will be released in destructor when mutex is not locked.
      std::unique_ptr<Node> node_to_gc;
      size_t solid_size = 0;
      {
        // Lock the mutex and move last subtree from subtrees_to_gc_ into
        // node_to_gc.
        Mutex::Lock lock(gc_mutex_);
        if (subtrees_to_gc_.empty()) return;
        node_to_gc = std::move(subtrees_to_gc_.back());
        subtrees_to_gc_.pop_back();
        solid_size = subtrees_to_gc_solid_size_.back();
        subtrees_to_gc_solid_size_.pop_back();
      }
      // Solid is a hack...
      if (solid_size != 0) {
        for (size_t i = 0; i < solid_size; i++) {
          node_to_gc.get()[i].~Node();
        }
        std::allocator<Node> alloc;
        alloc.deallocate(node_to_gc.release(), solid_size);
      }
    }
  }

  void Worker() {
    while (!stop_.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kGCIntervalMs));
      GarbageCollect();
    };
  }

  mutable Mutex gc_mutex_;
  std::vector<std::unique_ptr<Node>> subtrees_to_gc_ GUARDED_BY(gc_mutex_); // Use std::vector
  std::vector<size_t> subtrees_to_gc_solid_size_ GUARDED_BY(gc_mutex_); // Use std::vector

  // When true, Worker() should stop and exit.
  std::atomic<bool> stop_{false};
  std::thread gc_thread_;
};

NodeGarbageCollector gNodeGc;
}  // namespace

/////////////////////////////////////////////////////////////////////////
// Edge
/////////////////////////////////////////////////////////////////////////

Move Edge::GetMove(bool as_opponent) const {
  if (!as_opponent) return move_;
  Move m = move_;
  m.Flip();
  return m;
}

// Policy priors (P) are stored in a compressed 16-bit format.
// ... (rest of SetP/GetP comments remain the same) ...

void Edge::SetP(float p) {
  assert(0.0f <= p && p <= 1.0f);
  constexpr int32_t roundings = (1 << 11) - (3 << 28);
  int32_t tmp;
  std::memcpy(&tmp, &p, sizeof(float));
  tmp += roundings;
  p_ = (tmp < 0) ? 0 : static_cast<uint16_t>(tmp >> 12);
}

float Edge::GetP() const {
  // Reshift into place and set the assumed-set exponent bits.
  uint32_t tmp = (static_cast<uint32_t>(p_) << 12) | (3 << 28);
  float ret;
  std::memcpy(&ret, &tmp, sizeof(uint32_t));
  return ret;
}

std::string Edge::DebugString() const {
  std::ostringstream oss;
  oss << "Move: " << move_.ToString(true) << " p_: " << p_
      << " GetP: " << GetP();
  return oss.str();
}

std::unique_ptr<Edge[]> Edge::FromMovelist(const MoveList& moves) {
  std::unique_ptr<Edge[]> edges = std::make_unique<Edge[]>(moves.size());
  auto* edge = edges.get();
  for (const auto move : moves) edge++->move_ = move;
  return edges;
}

/////////////////////////////////////////////////////////////////////////
// Node
/////////////////////////////////////////////////////////////////////////

Node* Node::CreateSingleChildNode(Move m) {
  assert(!edges_);
  assert(!child_);
  edges_ = Edge::FromMovelist({m});
  num_edges_ = 1;
  child_ = std::make_unique<Node>(this, 0);
  return child_.get();
}

void Node::CreateEdges(const MoveList& moves) {
  assert(!edges_);
  assert(!child_);
  edges_ = Edge::FromMovelist(moves);
  num_edges_ = moves.size();
}

Node::ConstIterator Node::Edges() const {
  return {*this, !solid_children_ ? &child_ : nullptr};
}
Node::Iterator Node::Edges() {
  return {*this, !solid_children_ ? &child_ : nullptr};
}

float Node::GetVisitedPolicy() const {
  float sum = 0.0f;
  for (auto* node : VisitedNodes()) sum += GetEdgeToNode(node)->GetP();
  return sum;
}

Edge* Node::GetEdgeToNode(const Node* node) const {
  assert(node->parent_ == this);
  assert(node->index_ < num_edges_);
  return &edges_[node->index_];
}

Edge* Node::GetOwnEdge() const {
  if (parent_ == nullptr) return nullptr; // Root node has no edge to itself
  return GetParent()->GetEdgeToNode(this);
}


std::string Node::DebugString() const {
  std::ostringstream oss;
  oss << " Term:" << static_cast<int>(terminal_type_) << " This:" << this
      << " Parent:" << parent_ << " Index:" << index_
      << " Child:" << child_.get() << " Sibling:" << sibling_.get()
      << " WL:" << wl_ << " N:" << n_ << " N_:" << n_in_flight_
      << " Edges:" << static_cast<int>(num_edges_)
      << " Bounds:" << static_cast<int>(lower_bound_) - 2 << ","
      << static_cast<int>(upper_bound_) - 2 << " Solid:" << solid_children_;
  return oss.str();
}

bool Node::MakeSolid() {
  if (solid_children_ || num_edges_ == 0 || IsTerminal()) return false;
  // Can only make solid if no immediate leaf children are in flight since we
  // allow the search code to hold references to leaf nodes across locks.
  Node* old_child_to_check = child_.get();
  uint32_t total_in_flight = 0;
  while (old_child_to_check != nullptr) {
    if (old_child_to_check->GetN() <= 1 &&
        old_child_to_check->GetNInFlight() > 0) {
      return false;
    }
    if (old_child_to_check->IsTerminal() &&
        old_child_to_check->GetNInFlight() > 0) {
      return false;
    }
    total_in_flight += old_child_to_check->GetNInFlight();
    old_child_to_check = old_child_to_check->sibling_.get();
  }
  // If the total of children in flight is not the same as self, then there are
  // collisions against immediate children (which don't update the GetNInFlight
  // of the leaf) and its not safe.
  if (total_in_flight != GetNInFlight()) {
    return false;
  }
  std::allocator<Node> alloc;
  auto* new_children = alloc.allocate(num_edges_);
  for (int i = 0; i < num_edges_; i++) {
    new (&(new_children[i])) Node(this, i);
  }
  std::unique_ptr<Node> old_child = std::move(child_);
  while (old_child) {
    int index = old_child->index_;
    new_children[index] = std::move(*old_child.get());
    // This isn't needed, but it helps crash things faster if something has gone
    // wrong.
    old_child->parent_ = nullptr;
    gNodeGc.AddToGcQueue(std::move(old_child));
    new_children[index].UpdateChildrenParents();
    old_child = std::move(new_children[index].sibling_);
  }
  // This is a hack.
  child_ = std::unique_ptr<Node>(new_children);
  solid_children_ = true;
  return true;
}

void Node::SortEdges() {
  assert(edges_);
  assert(!child_);
  // Sorting on raw p_ is the same as sorting on GetP() as a side effect of
  // the encoding, and its noticeably faster.
  std::sort(edges_.get(), (edges_.get() + num_edges_),
            [](const Edge& a, const Edge& b) { return a.p_ > b.p_; });
}

void Node::MakeTerminal(GameResult result, float plies_left, Terminal type) {
  if (type != Terminal::TwoFold) SetBounds(result, result);
  terminal_type_ = type;
  m_ = plies_left;
  if (result == GameResult::DRAW) {
    wl_ = 0.0f;
    d_ = 1.0f;
  } else if (result == GameResult::WHITE_WON) {
    wl_ = 1.0f;
    d_ = 0.0f;
  } else if (result == GameResult::BLACK_WON) {
    wl_ = -1.0f;
    d_ = 0.0f;
    // Terminal losses have no uncertainty and no reason for their U value to be
    // comparable to another non-loss choice. Force this by clearing the policy.
    if (GetParent() != nullptr) GetOwnEdge()->SetP(0.0f);
  }
}

void Node::MakeNotTerminal() {
  terminal_type_ = Terminal::NonTerminal;
  n_ = 0;

  // If we have edges, we've been extended (1 visit), so include children too.
  if (edges_) {
    n_++;
    for (const auto& child : Edges()) {
      const auto n = child.GetN();
      if (n > 0) {
        n_ += n;
        // Flip Q for opponent.
        // Default values don't matter as n is > 0.
        wl_ += -child.GetWL(0.0f) * n;
        d_ += child.GetD(0.0f) * n;
      }
    }

    // Recompute with current eval (instead of network's) and children's eval.
    if (n_ > 0) { // Avoid division by zero
        wl_ /= n_;
        d_ /= n_;
    } else { // If n_ is still 0 after loop (only possible if n_ initially 0 and no children had visits)
        wl_ = 0.0;
        d_ = 0.0; // Or perhaps 1.0 if representing an unknown state? Check semantics.
    }
  }
}

void Node::SetBounds(GameResult lower, GameResult upper) {
  lower_bound_ = lower;
  upper_bound_ = upper;
}

bool Node::TryStartScoreUpdate() {
  if (n_ == 0 && n_in_flight_ > 0) return false;
  ++n_in_flight_;
  return true;
}

void Node::CancelScoreUpdate(int multivisit) { n_in_flight_ -= multivisit; }

void Node::FinalizeScoreUpdate(float v, float d, float m, int multivisit) {
  // Recompute Q.
  const uint32_t n_new = n_ + multivisit;
  if (n_new > 0) { // Avoid division by zero
    wl_ += multivisit * (v - wl_) / n_new;
    d_ += multivisit * (d - d_) / n_new;
    m_ += multivisit * (m - m_) / n_new;
  } else {
    // Should not happen if multivisit > 0, but handle defensively
    wl_ = v;
    d_ = d;
    m_ = m;
  }

  // Increment N.
  n_ += multivisit;
  // Decrement virtual loss.
  assert(n_in_flight_ >= (uint32_t)multivisit);
  n_in_flight_ -= multivisit;
}

void Node::AdjustForTerminal(float v, float d, float m, int multivisit) {
  // Recompute Q.
  assert(n_ > 0); // Should only adjust nodes with visits
  wl_ += multivisit * v / n_;
  d_ += multivisit * d / n_;
  m_ += multivisit * m / n_;
}

void Node::RevertTerminalVisits(float v, float d, float m, int multivisit) {
  // Compute new n_ first, as reducing a node to 0 visits is a special case.
  const int n_new = n_ - multivisit;
  if (n_new <= 0) {
    // If n_new == 0, reset all relevant values to 0.
    wl_ = 0.0;
    d_ = 1.0;
    m_ = 0.0;
    n_ = 0;
  } else {
    // Recompute Q and M. Avoid division by zero.
    wl_ -= multivisit * (v - wl_) / n_new;
    d_ -= multivisit * (d - d_) / n_new;
    m_ -= multivisit * (m - m_) / n_new;
    // Decrement N.
    n_ -= multivisit;
  }
}

void Node::UpdateChildrenParents() {
  if (!solid_children_) {
    Node* cur_child = child_.get();
    while (cur_child != nullptr) {
      cur_child->parent_ = this;
      cur_child = cur_child->sibling_.get();
    }
  } else {
    Node* child_array = child_.get();
    for (int i = 0; i < num_edges_; i++) {
      child_array[i].parent_ = this;
    }
  }
}

void Node::ReleaseChildren() {
  gNodeGc.AddToGcQueue(std::move(child_), solid_children_ ? num_edges_ : 0);
}

void Node::ReleaseChildrenExceptOne(Node* node_to_save) {
  if (solid_children_) {
    std::unique_ptr<Node> saved_node;
    if (node_to_save != nullptr) {
      saved_node = std::make_unique<Node>(this, node_to_save->index_);
      *saved_node = std::move(*node_to_save);
    }
    gNodeGc.AddToGcQueue(std::move(child_), num_edges_);
    child_ = std::move(saved_node);
    if (child_) {
      child_->UpdateChildrenParents();
    }
    solid_children_ = false;
  } else {
    // Stores node which will have to survive (or nullptr if it's not found).
    std::unique_ptr<Node> saved_node;
    // Pointer to unique_ptr, so that we could move from it.
    for (std::unique_ptr<Node>* node = &child_; *node;
         node = &(*node)->sibling_) {
      // If current node is the one that we have to save.
      if (node->get() == node_to_save) {
        // Kill all remaining siblings.
        gNodeGc.AddToGcQueue(std::move((*node)->sibling_));
        // Save the node, and take the ownership from the unique_ptr.
        saved_node = std::move(*node);
        break;
      }
    }
    // Make saved node the only child. (kills previous siblings).
    gNodeGc.AddToGcQueue(std::move(child_));
    child_ = std::move(saved_node);
  }
  if (!child_) {
    num_edges_ = 0;
    edges_.reset();  // Clear edges list.
  }
}

/////////////////////////////////////////////////////////////////////////
// EdgeAndNode
/////////////////////////////////////////////////////////////////////////

std::string EdgeAndNode::DebugString() const {
  if (!edge_) return "(no edge)";
  return edge_->DebugString() + " " +
         (node_ ? node_->DebugString() : "(no node)");
}

// Removed NodeTree definitions from node.cc

}  // namespace lczero
