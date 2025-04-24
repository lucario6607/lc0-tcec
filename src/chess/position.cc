/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

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

#include "chess/position.h"

#include <cstring>
#include <iostream>

#include "utils/hashcat.h"

namespace lczero {
////////////////////////////////////////////////////////////////////////////////
// Position
////////////////////////////////////////////////////////////////////////////////

Position::Position(const ChessBoard& board, int rule50_ply, int game_ply)
    : board_(board), rule50_ply_(rule50_ply), game_ply_(game_ply) {
  // Generate hash.
  hash_ = board_.Hash();
  ch_hash_ = board_.CHHash();
  repetitions_ = 0;
  // This is the most efficient way to detect first repetition, but slow to
  // detect second or third. Luckily they are very rare.
  if (rule50_ply > 0) {
    PositionHistory history(*this);
    for (int i = 0; i < rule50_ply; ++i) {
      history.Pop();
      if (history.Last().Hash() == hash_) {
        repetitions_ = 1;
        plies_since_prev_repetition_ = i + 1;
        break;
      }
    }
  }
}

Position::Position(const Position& parent, Move move) {
  const auto& pboard = parent.board_;
  bool was_capture = pboard.IsCapture(move);
  bool was_pawn = pboard.IsPawnMove(move);
  board_ = pboard.ApplyMove(move);
  hash_ = board_.Hash();
  ch_hash_ = board_.CHHash();
  // Update draw counters.
  game_ply_ = parent.game_ply_ + 1;
  rule50_ply_ = (was_capture || was_pawn) ? 0 : parent.rule50_ply_ + 1;
  // Update repetition counter.
  repetitions_ = 0;
  if (rule50_ply_ > 0) {
    if (parent.Hash() == hash_) {
      repetitions_ = parent.repetitions_ + 1;
      plies_since_prev_repetition_ = 1;
    } else if (parent.repetitions_ > 0 && parent.rule50_ply_ > 0) {
      // If parent wasn't first repetition, we could be second or third.
      PositionHistory history(*this);
      for (int i = 0; i < rule50_ply_; ++i) {
        history.Pop();
        if (history.Last().Hash() == hash_) {
          repetitions_ = history.Last().repetitions_ + 1;
          plies_since_prev_repetition_ = i + 1;
          break;
        }
      }
    }
  }
}

// Returns number of previous repetitions of the current position.
// @return 0 - never repeated, 1 - repeated once, 2+ - repeated twice or more.
int Position::GetRepetitions() const { return repetitions_; }

int Position::GetPliesSincePrevRepetition() const {
  return plies_since_prev_repetition_;
}

bool Position::IsDraw() const {
  return (!board_.HasMatingMaterial() || (rule50_ply_ >= 100) ||
          (repetitions_ >= 2));
}

std::vector<Position::InputPlanes> EncodePositionHistory(
    pblczero::NetworkFormat format, const PositionHistory& history,
    const Position& position, int max_planes) {
  auto input_planes = EncodePosition(format, position.GetBoard(), max_planes);
  if (max_planes == Position::InputPlanes::kTotalMoveCount) return input_planes;
  const auto& board = position.GetBoard();
  auto hist_iter = history.rbegin();
  // Iterate over past positions and fill input history.
  for (int p = 0; p < 7; ++p) {
    if (hist_iter != history.rend()) ++hist_iter;
    const auto& prev_board =
        (hist_iter != history.rend()) ? hist_iter->GetBoard() : board;
    const auto planes =
        EncodePosition(format, prev_board, Position::InputPlanes::kTotal);
    // Copy 2 planes (our pieces, their pieces).
    input_planes[Position::InputPlanes::kTotal + p * 2 + 0] = planes[0];
    input_planes[Position::InputPlanes::kTotal + p * 2 + 1] = planes[1];
  }
  return input_planes;
}

std::vector<Position::InputPlanes> EncodePositionForNN(
    const Position& position, const SearchParams& params) {
  PositionHistory history(position);
  int planes = Position::InputPlanes::kTotal;
  if (params.GetCacheHistoryLength() == 0) {
    FillEmptyHistory fill_mode = params.GetHistoryFill();
    if (fill_mode != FillEmptyHistory::NO) {
      if (fill_mode == FillEmptyHistory::ALWAYS ||
          position.GetFenString() != kStartingFen) {
        history.FillFrom(position);
        planes = Position::InputPlanes::kTotalHistory;
      }
    }
  }

  return EncodePositionHistory(pblczero::NetworkFormat::INPUT_CLASSICAL_112,
                               history, position, planes);
}

Position::InputPlanes& Position::InputPlanes::operator=(
    const Position::InputPlanes& p) {
  std::memcpy(data.data(), p.data.data(), data.size());
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
// PositionHistory
////////////////////////////////////////////////////////////////////////////////
uint64_t PositionHistory::HashLast(int count, int r50_ply) const {
  auto hist_iter = rbegin();
  uint64_t hash = hist_iter->Hash();
  if (r50_ply >= 0) hash ^= utils::HashCat(2, static_cast<uint64_t>(r50_ply));
  ++hist_iter;
  --count;
  for (int i = 0; i < count && hist_iter != rend(); ++i, ++hist_iter) {
    hash = utils::HashCat(hash, hist_iter->Hash());
  }
  return hash;
}


uint64_t PositionHistory::CHHash() const {
  auto hist_iter = rbegin();
  uint64_t hash = hist_iter->CHHash();

  ++hist_iter;
  if (hist_iter == rend()) return 0;
  hash = utils::HashCat(hash, hist_iter->CHHash());

  // const Move last_move = LastMove(); // WARNING: Unused variable
  return hash;

}



}  // namespace lczero
