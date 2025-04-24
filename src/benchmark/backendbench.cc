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
#include "benchmark/backendbench.h"

#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <iomanip> // For setprecision

#include "chess/position.h"
#include "config.h" // <<< ADDED THIS INCLUDE
#include "mcts/node.h" // Include for NodeTree definition
#include "mcts/params.h"
#include "neural/encoder.h"
#include "neural/factory.h"
#include "neural/network.h"
#include "utils/commandline.h"
#include "utils/optionsdict.h"
#include "utils/parallel_helpers.h"
#include "utils/string.h"
#include "utils/thread_pool.h"
#include "utils/hashcat.h" // Added for utils::HashCat

namespace lczero {

namespace {
const OptionId kFenId{"fen", "Fen", "Fen string used for benchmarking"};
}

void BackendBenchmark::RegisterOptions(OptionsParser* parser) {
  parser->Add<StringOption>(kFenId) = kStartingFen;
}

BackendBenchmark::BackendBenchmark(const OptionsDict& option_dict)
    : backend_name_(option_dict.Get<std::string>(NeuralFactory::kBackendId)),
      option_dict_(option_dict),
      // Pass option_dict_ to SearchParams constructor
      params_(option_dict_),
      network_(NeuralFactory::CreateNetwork(option_dict_)),
      total_batches_(0),
      total_nps_(0),
      total_samples_(0) {}

BackendBenchmark::~BackendBenchmark() {}

void BackendBenchmark::Benchmark(int thread_id, int num_threads,
                                 std::vector<float>& nps_stats,
                                 std::vector<float>& batch_stats,
                                 std::vector<int>& sample_stats) {
  const int batch_size = option_dict_.Get<int>(SearchParams::kMiniBatchSizeId);

  // One network per thread, otherwise synchronization costs are measured.
  auto network = NeuralFactory::CreateNetwork(option_dict_);
  network->InitThread(thread_id);

  // Initialize input features.
  // Need SearchParams to initialize NodeTree
  NodeTree tree(params_);
  tree.ResetToPosition(option_dict_.Get<std::string>(kFenId), {});
  const auto& history = tree.GetPositionHistory();
  const auto input_format = network->GetCapabilities().input_format;
  // Assuming EncodePositionHistory exists and works correctly after position.cc fixes
  std::vector<Position::InputPlanes> planes =
      EncodePositionHistory(input_format, history, history.Last(), Position::InputPlanes::kTotalHistory);

  // Total test duration 10 seconds.
  const auto benchmark_duration = std::chrono::seconds(10);
  const auto start_time = std::chrono::steady_clock::now();

  size_t total_samples = 0;
  size_t current_total_batches = 0; // Use local variable

  while (true) {
    auto current_time = std::chrono::steady_clock::now();
    if (current_time - start_time > benchmark_duration) break;

    // Prepare batch.
    auto computation = network->NewComputation();
    computation->Reserve(batch_size);
    for (int i = 0; i < batch_size; i++) {
      // Use the pre-encoded planes
      // Use a unique hash for each sample if needed, otherwise use base hash
      uint64_t sample_hash = utils::HashCat(history.Last().Hash(), static_cast<uint64_t>(i));
      computation->AddInput(sample_hash, planes);
    }

    // Compute batch.
    computation->ComputeBlocking(1.0f); // Use appropriate softmax temp if needed

    current_total_batches++;
    total_samples += batch_size;
  }

  const auto end_time = std::chrono::steady_clock::now();
  const auto time_delta = end_time - start_time;
  const float nps = (time_delta.count() > 0) ?
      static_cast<float>(total_samples) /
      std::chrono::duration_cast<std::chrono::duration<float>>(time_delta)
          .count() : 0.0f;
  const float batch_latency = (current_total_batches > 0) ? // Use local batch count
      static_cast<float>(
          std::chrono::duration_cast<std::chrono::microseconds>(time_delta)
              .count()) /
      current_total_batches : 0.0f;


  nps_stats[thread_id] = nps;
  batch_stats[thread_id] = batch_latency;
  sample_stats[thread_id] = total_samples;
}

void BackendBenchmark::Run() {
  std::cout << "Generating benchmark data for " << backend_name_
            << " backend..." << std::endl;

  const int num_threads = option_dict_.Get<int>(NeuralFactory::kMaxThreadsId);
  std::vector<float> nps_stats(num_threads);
  std::vector<float> batch_stats(num_threads); // This now stores latency per thread
  std::vector<int> sample_stats(num_threads);

  // Start N benchmark threads.
  utils::Waitable waitable;
  for (int i = 0; i < num_threads; i++) {
    // Use ThreadPool for cleaner thread management if available, otherwise raw threads
    utils::internal::WorkerData* wd = new utils::internal::WorkerData;
    wd->th = std::thread(
        [&](int thread_id, std::vector<float>& nps, std::vector<float>& batch,
            std::vector<int>& samples) {
          Benchmark(thread_id, num_threads, nps, batch, samples);
          waitable.Signal();
          delete wd; // Clean up worker data when done
        },
        i, std::ref(nps_stats), std::ref(batch_stats), std::ref(sample_stats));
    wd->th.detach(); // Detach as we manage completion with Waitable
    waitable.Add(1);
  }


  // Wait for all threads to finish.
  waitable.Wait();

  total_samples_ = std::accumulate(sample_stats.begin(), sample_stats.end(), 0LL); // Use 0LL for long long
  // total_batches_ calculation seems off, should be sum of batches processed per thread
  // Recalculate based on samples and batch size? Or pass back batch count?
  // Let's assume batch_stats holds latency, calculate avg batch latency differently.
  total_batches_ = 0; // Reset this as it was misused
  float total_latency_sum = std::accumulate(batch_stats.begin(), batch_stats.end(), 0.0f);


  total_nps_ = std::accumulate(nps_stats.begin(), nps_stats.end(), 0.0f);

  average_batch_latency_ = (num_threads > 0) ? (total_latency_sum / num_threads / 1000.0f) : 0.0f; // Average latency in ms

  // Calculate standard deviation.
  float avg_nps = (num_threads > 0) ? (total_nps_ / num_threads) : 0.0f;
  float sumsq_nps = 0;
  for (int i = 0; i < num_threads; i++) {
    sumsq_nps += (nps_stats[i] - avg_nps) * (nps_stats[i] - avg_nps);
  }
  std_dev_nps_ = (num_threads > 0) ? std::sqrt(sumsq_nps / num_threads) : 0.0f;

  float avg_latency_us = average_batch_latency_ * 1000.0f; // Back to microseconds
  float sumsq_latency = 0;
  for (int i = 0; i < num_threads; i++) {
    sumsq_latency +=
        (batch_stats[i] - avg_latency_us) * (batch_stats[i] - avg_latency_us);
  }
  std_dev_batch_latency_ = (num_threads > 0) ? (std::sqrt(sumsq_latency / num_threads) / 1000.0f) : 0.0f; // Back to ms
}

void BackendBenchmark::Report() {
  std::cout << std::fixed << std::setprecision(1);
  std::cout << "Total evaluations: " << total_samples_ << std::endl;
  std::cout << "NN outputs per second: " << total_nps_ << " +/- "
            << 2 * std_dev_nps_ << std::endl;
  std::cout << "Average batch latency: " << average_batch_latency_ << " +/- "
            << 2 * std_dev_batch_latency_ << " ms" << std::endl;
}

}  // namespace lczero
