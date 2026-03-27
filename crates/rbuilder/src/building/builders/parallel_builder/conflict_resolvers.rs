use ahash::{HashMap, HashSet};
use alloy_primitives::{Address, U256};
use derivative::Derivative;
use eyre::Result;
use itertools::Itertools;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction::{Incoming, Outgoing};
use rand::{seq::SliceRandom, SeedableRng};
use rayon::{prelude::*, ThreadPool};
use reth::providers::StateProvider;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::trace;

use super::{
    conflict_task_generator::{update_default_graph_study_record, DefaultGraphStudyRecord},
    simulation_cache::{CachedSimulationState, SharedSimulationCache},
    Algorithm, ConflictTask, ResolutionResult,
};

use crate::{
    building::{
        evm_inspector::UsedStateTrace, BlockBuildingContext, BlockState, ExecutionError,
        ExecutionResult, PartialBlock, ThreadBlockBuildingContext,
    },
    primitives::{OrderId, SimulatedOrder},
};

const MAX_ORIENTATION_SEARCH_NODES: usize = 16;
const MIN_SEQUENCES_FOR_PARALLEL_EVAL: usize = 16;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ConflictGraphStats {
    pub edge_count: usize,
    pub density: f64,
    pub max_degree: usize,
    pub has_missing_traces: bool,
}

#[derive(Debug, Clone)]
struct OrientationProblem {
    hard_precedence: Vec<u32>,
    soft_conflicts: Vec<(usize, usize)>,
}

#[derive(Debug)]
struct TaskExecutionData {
    order_ids_by_index: Vec<OrderId>,
    order_id_to_index: HashMap<OrderId, usize>,
}

impl TaskExecutionData {
    fn new(task: &ConflictTask) -> Self {
        let order_ids_by_index: Vec<_> = task
            .group
            .orders
            .iter()
            .map(|sim_order| sim_order.order.id())
            .collect();
        let order_id_to_index = order_ids_by_index
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, order_id)| (order_id, idx))
            .collect();

        Self {
            order_ids_by_index,
            order_id_to_index,
        }
    }
}

pub(crate) fn analyze_conflict_graph(orders: &[Arc<SimulatedOrder>]) -> ConflictGraphStats {
    let n = orders.len();
    if n < 2 {
        return ConflictGraphStats::default();
    }

    if orders.iter().any(|order| order.used_state_trace.is_none()) {
        return ConflictGraphStats {
            has_missing_traces: true,
            ..Default::default()
        };
    }

    let mut edge_count = 0;
    let mut degrees = vec![0usize; n];
    for left in 0..n {
        for right in (left + 1)..n {
            if orders_conflict(&orders[left], &orders[right]) {
                edge_count += 1;
                degrees[left] += 1;
                degrees[right] += 1;
            }
        }
    }

    let complete_graph_edges = n * (n - 1) / 2;
    let density = if complete_graph_edges == 0 {
        0.0
    } else {
        edge_count as f64 / complete_graph_edges as f64
    };

    ConflictGraphStats {
        edge_count,
        density,
        max_degree: degrees.into_iter().max().unwrap_or(0),
        has_missing_traces: false,
    }
}

/// Context for resolving conflicts in merging tasks.

#[derive(Derivative)]
#[derivative(Debug)]
pub struct ResolverContext {
    #[derivative(Debug = "ignore")]
    pub state: Arc<dyn StateProvider>,
    pub ctx: BlockBuildingContext,
    pub cancellation_token: CancellationToken,
    pub simulation_cache: Arc<SharedSimulationCache>,
    pub sequence_thread_pool: Arc<ThreadPool>,
    pub max_group_parallelism: usize,
    pub graph_study_collector: Option<Arc<parking_lot::Mutex<Vec<DefaultGraphStudyRecord>>>>,
}

impl ResolverContext {
    /// Creates a new `ResolverContext`.
    ///
    /// # Arguments
    ///
    /// * `provider_factory` - Factory for creating state providers.
    /// * `ctx` - Context for block building.
    /// * `cancellation_token` - Token for cancelling operations.
    /// * `cache` - Optional cached reads for optimization.
    /// * `simulation_cache` - Shared cache for simulation results.
    pub fn new(
        state: Arc<dyn StateProvider>,
        ctx: BlockBuildingContext,
        cancellation_token: CancellationToken,
        simulation_cache: Arc<SharedSimulationCache>,
        sequence_thread_pool: Arc<ThreadPool>,
        max_group_parallelism: usize,
        graph_study_collector: Option<Arc<parking_lot::Mutex<Vec<DefaultGraphStudyRecord>>>>,
    ) -> Self {
        ResolverContext {
            state,
            ctx,
            cancellation_token,
            simulation_cache,
            sequence_thread_pool,
            max_group_parallelism,
            graph_study_collector,
        }
    }

    /// Runs a merging task and returns the best [ResolutionResult] found.
    ///
    /// # Arguments
    ///
    /// * `task` - The [ConflictTask] to run.
    ///
    /// # Returns
    ///
    /// The best [ResolutionResult] and corresponding sequence of order indices found.
    pub fn run_conflict_task(&mut self, task: ConflictTask) -> Result<ResolutionResult> {
        trace!(
            "run_conflict_task: {:?} with algorithm {:?}",
            task.group.id,
            task.algorithm
        );

        let sequence_to_try = generate_sequences_of_orders_to_try(&task);
        if let Some(collector) = &self.graph_study_collector {
            update_default_graph_study_record(
                collector,
                task.group.id,
                task.algorithm,
                Some(sequence_to_try.len()),
                None,
            );
        }
        let task_execution_data = TaskExecutionData::new(&task);
        let best = self.evaluate_sequences(sequence_to_try, &task, &task_execution_data)?;

        let mut best_resolution_result = ResolutionResult {
            total_profit: U256::ZERO,
            sequence_of_orders: vec![],
        };
        self.update_best_result(best, &mut best_resolution_result);

        trace!(
            "Resolved conflict task {:?} with profit: {:?} and algorithm: {:?}",
            task.group.id,
            best_resolution_result.total_profit,
            task.algorithm
        );
        if let Some(collector) = &self.graph_study_collector {
            update_default_graph_study_record(
                collector,
                task.group.id,
                task.algorithm,
                None,
                Some(best_resolution_result.total_profit),
            );
        }
        Ok(best_resolution_result)
    }

    fn evaluate_sequences(
        &self,
        sequence_to_try: Vec<Vec<usize>>,
        task: &ConflictTask,
        task_execution_data: &TaskExecutionData,
    ) -> Result<ResolutionResult> {
        if sequence_to_try.is_empty() {
            return Ok(ResolutionResult {
                total_profit: U256::ZERO,
                sequence_of_orders: vec![],
            });
        }

        let desired_parallelism =
            sequence_parallelism_budget(sequence_to_try.len(), self.max_group_parallelism);
        if desired_parallelism <= 1 {
            let mut resolver_ctx = self.clone_for_parallel_batch();
            return resolver_ctx.process_sequence_batch(
                &sequence_to_try,
                task,
                task_execution_data,
            );
        }

        let batch_size = sequence_to_try.len().div_ceil(desired_parallelism);
        let state = self.state.clone();
        let ctx = self.ctx.clone();
        let cancellation_token = self.cancellation_token.clone();
        let simulation_cache = self.simulation_cache.clone();
        let sequence_thread_pool = Arc::clone(&self.sequence_thread_pool);
        let max_group_parallelism = self.max_group_parallelism;

        let best = self.sequence_thread_pool.install(|| {
            sequence_to_try
                .par_chunks(batch_size)
                .map(|batch| {
                    let mut resolver_ctx = ResolverContext {
                        state: state.clone(),
                        ctx: ctx.clone(),
                        cancellation_token: cancellation_token.clone(),
                        simulation_cache: simulation_cache.clone(),
                        sequence_thread_pool: Arc::clone(&sequence_thread_pool),
                        max_group_parallelism,
                        graph_study_collector: self.graph_study_collector.clone(),
                    };
                    resolver_ctx.process_sequence_batch(batch, task, task_execution_data)
                })
                .try_reduce_with(|a: ResolutionResult, b: ResolutionResult| {
                    Ok(if a.total_profit >= b.total_profit {
                        a
                    } else {
                        b
                    })
                })
        });

        Ok(best.transpose()?.unwrap_or(ResolutionResult {
            total_profit: U256::ZERO,
            sequence_of_orders: vec![],
        }))
    }

    fn clone_for_parallel_batch(&self) -> Self {
        Self {
            state: self.state.clone(),
            ctx: self.ctx.clone(),
            cancellation_token: self.cancellation_token.clone(),
            simulation_cache: self.simulation_cache.clone(),
            sequence_thread_pool: Arc::clone(&self.sequence_thread_pool),
            max_group_parallelism: self.max_group_parallelism,
            graph_study_collector: self.graph_study_collector.clone(),
        }
    }

    /// Updates the best result if a better one is found.
    ///
    /// # Arguments
    ///
    /// * `new_result` - The newly processed result.
    /// * `best_result` - The current best result to update.
    fn update_best_result(
        &mut self,
        new_result: ResolutionResult,
        best_result: &mut ResolutionResult,
    ) {
        if best_result.total_profit < new_result.total_profit {
            best_result.total_profit = new_result.total_profit;
            best_result.sequence_of_orders = new_result.sequence_of_orders;
        }
    }

    fn process_sequence_batch(
        &mut self,
        sequences: &[Vec<usize>],
        task: &ConflictTask,
        task_execution_data: &TaskExecutionData,
    ) -> Result<ResolutionResult> {
        let mut best_resolution_result = ResolutionResult {
            total_profit: U256::ZERO,
            sequence_of_orders: vec![],
        };

        for sequence_of_orders in sequences {
            let (resolution_result, _state) = self.process_sequence_of_orders(
                sequence_of_orders,
                task,
                task_execution_data,
                self.state.clone(),
            )?;
            self.update_best_result(resolution_result, &mut best_resolution_result);
        }

        Ok(best_resolution_result)
    }

    /// Processes a single sequence of orders, utilizing the simulation cache.
    ///
    /// # Arguments
    ///
    /// * `sequence_of_orders` - The order of transaction indices to process.
    /// * `task` - The current conflict task.
    /// * `state_provider` - The state provider for the current block.
    ///
    /// # Returns
    ///
    /// A tuple containing the resolution result and the final block state.
    fn process_sequence_of_orders(
        &mut self,
        sequence_of_orders: &[usize],
        task: &ConflictTask,
        task_execution_data: &TaskExecutionData,
        state_provider: Arc<dyn StateProvider>,
    ) -> Result<(ResolutionResult, BlockState)> {
        // @todo actually reuse it for the duration of the block
        let mut local_ctx = ThreadBlockBuildingContext::default();

        let full_sequence_of_orders =
            self.initialize_full_order_ids_vec(sequence_of_orders, task_execution_data);

        // Check for cached simulation state
        let (cached_state_option, cached_up_to_index) = self
            .simulation_cache
            .get_cached_state(&full_sequence_of_orders);

        // Initialize state and partial block
        let mut partial_block = PartialBlock::new(true);
        let mut state = self.initialize_block_state(state_provider);
        partial_block.pre_block_call(&self.ctx, &mut local_ctx, &mut state)?;

        // Initialize sequenced_order_result
        let mut sequenced_order_result = self.initialize_result_order_sequence(
            &cached_state_option,
            &task_execution_data.order_id_to_index,
        );

        let mut total_profit = cached_state_option
            .as_ref()
            .map_or(U256::ZERO, |cached| cached.total_profit);

        let mut per_order_profits = cached_state_option
            .as_ref()
            .map_or(Vec::new(), |cached| cached.per_order_profits.clone());

        // Prepare the sequence of orders to try, skipping already cached orders
        let mut remaining_orders = sequence_of_orders[cached_up_to_index..].to_vec();
        remaining_orders.reverse(); // Use as a stack: pop from the end

        let mut pending_orders: HashMap<(Address, u64), usize> = HashMap::default();

        // Processing loop
        while let Some(order_idx) = remaining_orders.pop() {
            if self.cancellation_token.is_cancelled() {
                return Err(eyre::eyre!("Cancelled"));
            }

            let sim_order = &task.group.orders[order_idx];
            match partial_block.commit_order(
                sim_order,
                &self.ctx,
                &mut local_ctx,
                &mut state,
                &|_| Ok(()),
            )? {
                Ok(res) => self.handle_successful_commit(
                    res,
                    sim_order,
                    order_idx,
                    &mut pending_orders,
                    &mut remaining_orders,
                    &mut sequenced_order_result,
                    &mut total_profit,
                    &mut per_order_profits,
                ),
                Err(err) => self.handle_err(&err, sim_order, &mut pending_orders, order_idx),
            }
        }

        self.store_simulation_state(
            &full_sequence_of_orders,
            &state,
            total_profit,
            &per_order_profits,
        );

        let resolution_result = ResolutionResult {
            total_profit,
            sequence_of_orders: sequenced_order_result,
        };
        Ok((resolution_result, state))
    }

    /// Helper function to handle a successful commit of an order.
    #[allow(clippy::too_many_arguments)]
    fn handle_successful_commit(
        &mut self,
        res: ExecutionResult,
        sim_order: &SimulatedOrder,
        order_idx: usize,
        pending_orders: &mut HashMap<(Address, u64), usize>,
        remaining_orders: &mut Vec<usize>,
        sequenced_order_result: &mut Vec<(usize, U256)>,
        total_profit: &mut U256,
        per_order_profits: &mut Vec<(OrderId, U256)>,
    ) {
        for (address, nonce) in res.nonces_updated {
            if let Some(pending_order) = pending_orders.remove(&(address, nonce)) {
                remaining_orders.push(pending_order);
            }
        }
        let order_id = sim_order.order.id();
        *total_profit += res.coinbase_profit;
        per_order_profits.push((order_id, res.coinbase_profit));
        sequenced_order_result.push((order_idx, res.coinbase_profit));
    }

    /// Helper function to handle an error in committing an order.
    fn handle_err(
        &mut self,
        err: &ExecutionError,
        sim_order: &SimulatedOrder,
        pending_orders: &mut HashMap<(Address, u64), usize>,
        order_idx: usize,
    ) {
        if let Some((address, nonce)) = err.try_get_tx_too_high_error(&sim_order.order) {
            pending_orders.insert((address, nonce), order_idx);
        };
    }

    /// Initializes a vector of full order ids corresponding to the sequence of orders.
    fn initialize_full_order_ids_vec(
        &self,
        sequence_of_orders: &[usize],
        task_execution_data: &TaskExecutionData,
    ) -> Vec<OrderId> {
        sequence_of_orders
            .iter()
            .map(|&idx| task_execution_data.order_ids_by_index[idx])
            .collect()
    }

    /// Initializes the tuple of (order_idx, profit) for the resolution result using the cached state if available.
    fn initialize_result_order_sequence(
        &self,
        cached_state_option: &Option<Arc<CachedSimulationState>>,
        order_id_to_index: &HashMap<OrderId, usize>,
    ) -> Vec<(usize, U256)> {
        if let Some(cached_state) = &cached_state_option {
            cached_state
                .per_order_profits
                .iter()
                .filter_map(|(order_id, profit)| {
                    order_id_to_index.get(order_id).map(|&idx| (idx, *profit))
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        }
    }

    /// Initializes the block state, using a cached state if available.
    fn initialize_block_state(&mut self, state_provider: Arc<dyn StateProvider>) -> BlockState {
        BlockState::new_arc(state_provider)
    }

    /// Stores the simulation state in the cache.
    fn store_simulation_state(
        &self,
        full_order_ids: &[OrderId],
        state: &BlockState,
        total_profit: U256,
        per_order_profits: &[(OrderId, U256)],
    ) {
        let (bundle_state, _) = state.clone().into_parts();
        let cached_simulation_state = CachedSimulationState {
            bundle_state,
            total_profit,
            per_order_profits: per_order_profits.to_owned(),
        };
        self.simulation_cache
            .store_cached_state(full_order_ids, cached_simulation_state);
    }
}

/// Generates different sequences of orders to try based on the conflict task command.
///
/// # Arguments
///
/// * `task` - The conflict task containing the algorithm for generating sequences.
///
/// # Returns
///
/// A vector of different sequences of order indices to try.
fn generate_sequences_of_orders_to_try(task: &ConflictTask) -> Vec<Vec<usize>> {
    match task.algorithm {
        Algorithm::Greedy => generate_greedy_sequence(task, false),
        Algorithm::ReverseGreedy => generate_greedy_sequence(task, true),
        Algorithm::Length => generate_length_based_sequence(task),
        Algorithm::AllPermutations => generate_all_permutations(task),
        Algorithm::Random { seed, count } => generate_random_permutations(task, seed, count),
        Algorithm::PermutationsWithNonces => generate_all_permutations_with_nonces(task),
        Algorithm::OrientationSearch => generate_orientation_search_sequences(task),
        Algorithm::BestOfN => analyze_group_conflicts_and_find_best(task), // BestOfN implementation
    }
}

/// Generates random permutations of sequences of order indices.
///
/// # Arguments
///
/// * `task` - The current conflict task.
/// * `seed` - Seed for the random number generator.
/// * `count` - Number of random permutations to generate.
///
/// # Returns
///
/// A vector of randomly generated sequences of order indices.
fn generate_random_permutations(task: &ConflictTask, seed: u64, count: usize) -> Vec<Vec<usize>> {
    let mut sequences_of_orders = vec![];

    let order_group = &task.group;
    let mut indexes = (0..order_group.orders.len()).collect::<Vec<_>>();
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    for _ in 0..count {
        indexes.shuffle(&mut rng);
        sequences_of_orders.push(indexes.clone());
    }

    sequences_of_orders
}

/// Generates all possible permutations of sequences of order indices.
///
/// # Arguments
///
/// * `task` - The current conflict task.
///
/// # Returns
///
/// A vector of all possible sequences of order indices.
fn generate_all_permutations(task: &ConflictTask) -> Vec<Vec<usize>> {
    let order_group = &task.group;
    let sequences_of_orders = (0..order_group.orders.len()).collect::<Vec<_>>();
    sequences_of_orders
        .into_iter()
        .permutations(order_group.orders.len())
        .collect()
}

fn generate_orientation_search_sequences(task: &ConflictTask) -> Vec<Vec<usize>> {
    let order_group = &task.group;
    if order_group.orders.len() <= 1 {
        return vec![(0..order_group.orders.len()).collect()];
    }

    let Some(problem) = build_orientation_problem(order_group.orders.as_ref()) else {
        return generate_all_permutations_with_nonces(task);
    };

    let n = order_group.orders.len();
    let mut reach = problem.hard_precedence;
    let mut sequences = Vec::new();
    let mut seen = HashSet::default();

    enumerate_orientations(
        n,
        &problem.soft_conflicts,
        0,
        &mut reach,
        &mut seen,
        &mut sequences,
    );

    if sequences.is_empty() {
        generate_all_permutations_with_nonces(task)
    } else {
        sequences
    }
}

/// Generates static sequences of order indices based on gas price and coinbase profit.
///
/// # Arguments
///
/// * `task` - The current conflict task.
/// * `reverse` - Whether to reverse the sorting order (e.g. sorting by min coinbase profit and mev_gas_price)
///
/// # Returns
///
/// A vector of static sequences of order indices, sorted by coinbase profit and mev_gas_price.
fn generate_greedy_sequence(task: &ConflictTask, reverse: bool) -> Vec<Vec<usize>> {
    let order_group = &task.group;

    let create_sequence = |value_extractor: fn(&SimulatedOrder) -> U256| {
        let mut ids_and_value: Vec<_> = order_group
            .orders
            .iter()
            .enumerate()
            .map(|(idx, order)| (idx, value_extractor(order)))
            .collect();

        ids_and_value.sort_by(|a, b| {
            if reverse {
                a.1.cmp(&b.1)
            } else {
                b.1.cmp(&a.1)
            }
        });
        ids_and_value.into_iter().map(|(idx, _)| idx).collect()
    };

    vec![
        create_sequence(|sim_order| sim_order.sim_value.full_profit_info().coinbase_profit()),
        create_sequence(|sim_order| sim_order.sim_value.full_profit_info().mev_gas_price()),
    ]
}

/// Generates length based sequences of order indices based on the length of the orders.
/// e.g. prioritizes longer bundles first
///
/// # Arguments
///
/// * `task` - The current conflict task.
///
/// # Returns
///
/// A vector of length based sequences of order indices.
fn generate_length_based_sequence(task: &ConflictTask) -> Vec<Vec<usize>> {
    let mut sequences_of_orders = vec![];
    let order_group = &task.group;

    let mut order_data: Vec<(usize, usize, U256)> = order_group
        .orders
        .iter()
        .enumerate()
        .map(|(idx, order)| {
            (
                idx,
                order.order.list_txs().len(),
                order.sim_value.full_profit_info().coinbase_profit(),
            )
        })
        .collect();

    // Sort by length (descending) and then by profit (descending) as a tie-breaker
    order_data.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.2.cmp(&a.2)));

    // Extract the sorted indices
    let length_based_sequence: Vec<usize> = order_data.into_iter().map(|(idx, _, _)| idx).collect();

    sequences_of_orders.push(length_based_sequence);
    sequences_of_orders
}

fn find_permutations_recursive(
    graph: &DiGraph<Arc<SimulatedOrder>, ()>,
    current_permutation_indices: &mut Vec<NodeIndex>,
    in_degrees: &mut HashMap<NodeIndex, usize>,
    total_nodes: usize,
    all_permutations_indices: &mut Vec<Vec<NodeIndex>>,
) {
    if current_permutation_indices.len() == total_nodes {
        all_permutations_indices.push(current_permutation_indices.clone());
        return;
    }

    let mut candidates: Vec<NodeIndex> = Vec::new();
    for node_idx in graph.node_indices() {
        if !current_permutation_indices.contains(&node_idx)
            && *in_degrees.get(&node_idx).unwrap_or(&0) == 0
        {
            // only consider nodes with in-degree 0
            candidates.push(node_idx);
        }
    }

    candidates.sort_unstable(); // Sort by NodeIndex, which implements Ord

    if candidates.is_empty() && current_permutation_indices.len() != total_nodes {
        return;
    }

    for &candidate_node_idx in &candidates {
        current_permutation_indices.push(candidate_node_idx);

        let mut affected_neighbors: Vec<NodeIndex> = Vec::new();
        for edge in graph.edges_directed(candidate_node_idx, Outgoing) {
            let target_node_idx = edge.target();
            if let Some(degree) = in_degrees.get_mut(&target_node_idx) {
                *degree -= 1;
                affected_neighbors.push(target_node_idx);
            }
        }

        find_permutations_recursive(
            graph,
            current_permutation_indices,
            in_degrees,
            total_nodes,
            all_permutations_indices,
        );

        for neighbor_idx in affected_neighbors {
            if let Some(degree) = in_degrees.get_mut(&neighbor_idx) {
                *degree += 1;
            }
        }
        current_permutation_indices.pop();
    }
}

// optimal solution: each group contains exlsuivity
fn analyze_group_conflicts_and_find_best(task: &ConflictTask) -> Vec<Vec<usize>> {
    let order_group = &task.group;
    let orders = &order_group.orders;

    // Collect the nonce sets for each order (assuming Vec<u64> or similar)
    let _order_nonces: Vec<_> = orders
        .iter()
        .map(|order_arc| order_arc.order.nonces())
        .collect();

    // Define a function to compute the "value" of an order, here using coinbase_profit()
    // You can replace it with other metrics if needed
    fn order_value(order: &SimulatedOrder) -> U256 {
        order.sim_value.full_profit_info().coinbase_profit()
    }

    // Create index and value pairs for sorting
    let mut idx_and_value: Vec<(usize, U256)> = orders
        .iter()
        .enumerate()
        .map(|(idx, order_arc)| (idx, order_value(order_arc)))
        .collect();

    // Sort orders by value in descending order, prioritizing higher-value orders
    idx_and_value.sort_by(|a, b| b.1.cmp(&a.1));

    let mut selected = Vec::new();
    let best_order_idx = idx_and_value[0].0;
    selected.push(best_order_idx);
    vec![selected]

    // let mut used_nonces = std::collections::HashSet::new();

    // for (idx, _) in idx_and_value {
    //     let nonces = &order_nonces[idx];

    //     // Check if there is any nonce conflict with already selected orders
    //     let conflict = nonces.iter().any(|nonce| used_nonces.contains(nonce));

    //     if !conflict {
    //         // No conflict, select this order
    //         selected.push(idx);
    //         for nonce in nonces {
    //             used_nonces.insert(nonce.clone());
    //         }
    //     }
    //     // Skip orders with nonce conflicts
    // }

    // // Return a vector containing one group of selected order indices
    // // Modify as needed if you want multiple groups or different solutions
    // vec![selected]
}

fn generate_all_permutations_with_nonces(task: &ConflictTask) -> Vec<Vec<usize>> {
    let order_group = &task.group;
    let mut graph = DiGraph::<Arc<SimulatedOrder>, ()>::new();
    let mut node_indices: Vec<NodeIndex> = vec![];
    let mut index_map: HashMap<NodeIndex, usize> = HashMap::default();

    for (i, order_arc) in order_group.orders.iter().enumerate() {
        let node_index = graph.add_node(order_arc.clone());
        node_indices.push(node_index);
        index_map.insert(node_index, i);
    }

    let mut nonce_to_node_map: HashMap<(Address, u64), NodeIndex> = HashMap::default();

    for (original_order_idx, order_arc) in order_group.orders.iter().enumerate() {
        let current_node_idx = node_indices[original_order_idx];
        for nonce in order_arc.nonces() {
            let key = (nonce.address, nonce.nonce);
            nonce_to_node_map.insert(key, current_node_idx);
        }
    }

    // Create edges based on nonce relationships
    // For each order_i, find order_j such that nonce_j.address == nonce_i.address
    // and nonce_j.nonce == nonce_i.nonce - 1
    for (order_i_original_idx, order_i_arc) in order_group.orders.iter().enumerate() {
        let node_i_idx = node_indices[order_i_original_idx]; // NodeIndex for order_i

        for nonce_i in order_i_arc.nonces() {
            // We are looking for an order_j with nonce_j such that:
            // nonce_j.address == nonce_i.address
            // nonce_j.nonce == nonce_i.nonce - 1
            // This means an edge from order_j's node to order_i's node.

            if nonce_i.nonce == 0 {
                // Or whatever your minimum nonce value is
                // Cannot have a nonce that is `nonce_i.value - 1` if current is 0
                continue;
            }
            let target_nonce_value_for_j = nonce_i.nonce - 1;
            let lookup_key = (nonce_i.address, target_nonce_value_for_j);

            if let Some(&node_j_idx) = nonce_to_node_map.get(&lookup_key) {
                // We found an order_j (represented by node_j_idx) that has the preceding nonce.
                // Add an edge from node_j (the one with nonce N-1) to node_i (the one with nonce N).
                // Ensure it's not an edge to itself if an order could somehow contain (addr, N) and (addr, N-1).
                // The problem implies distinct orders, which `node_j_idx != node_i_idx` would check.
                // However, the map construction ensures `node_j_idx` is the node for the order *containing* that specific (addr, N-1) nonce.
                // If order_i and order_j are different orders, their node indices will be different.
                if node_j_idx != node_i_idx {
                    // Avoid self-loops based on this specific logic
                    graph.add_edge(node_j_idx, node_i_idx, ());
                }
            }
        }
    }

    let mut all_permutations_indices: Vec<Vec<NodeIndex>> = Vec::new();
    let mut current_permutation_indices: Vec<NodeIndex> = Vec::new();

    let mut in_degrees: HashMap<NodeIndex, usize> = graph
        .node_indices()
        .map(|node_idx| (node_idx, graph.edges_directed(node_idx, Incoming).count()))
        .collect();

    let node_count = graph.node_count();

    for node_idx in graph.node_indices() {
        in_degrees.entry(node_idx).or_insert(0);
    }

    find_permutations_recursive(
        &graph,
        &mut current_permutation_indices,
        &mut in_degrees,
        node_count,
        &mut all_permutations_indices,
    );

    let sequences_of_orders: Vec<Vec<usize>> = all_permutations_indices
        .into_iter()
        .map(|node_idx_vec| {
            node_idx_vec
                .into_iter()
                .map(|node_idx| index_map[&node_idx])
                .collect()
        })
        .collect();

    // println!("length of sequences_of_orders: {}", sequences_of_orders.len());

    sequences_of_orders
}

fn build_orientation_problem(orders: &[Arc<SimulatedOrder>]) -> Option<OrientationProblem> {
    if orders.is_empty() || orders.len() > MAX_ORIENTATION_SEARCH_NODES {
        return None;
    }

    if orders.iter().any(|order| order.used_state_trace.is_none()) {
        return None;
    }

    let n = orders.len();
    let mut hard_precedence = vec![0u32; n];
    let mut nonce_to_order: HashMap<(Address, u64), usize> = HashMap::default();

    for (idx, order) in orders.iter().enumerate() {
        for nonce in order.order.nonces() {
            nonce_to_order.insert((nonce.address, nonce.nonce), idx);
        }
    }

    for (idx, order) in orders.iter().enumerate() {
        for nonce in order.order.nonces() {
            if nonce.nonce == 0 {
                continue;
            }

            if let Some(&prev_idx) = nonce_to_order.get(&(nonce.address, nonce.nonce - 1)) {
                if prev_idx != idx {
                    hard_precedence[prev_idx] |= bit(idx);
                }
            }
        }
    }

    if !close_transitive_closure(&mut hard_precedence, n) {
        return None;
    }

    let mut soft_conflicts = Vec::new();
    for left in 0..n {
        for right in (left + 1)..n {
            if !orders_conflict(&orders[left], &orders[right]) {
                continue;
            }

            if hard_precedence[left] & bit(right) != 0 || hard_precedence[right] & bit(left) != 0 {
                continue;
            }

            soft_conflicts.push((left, right));
        }
    }

    Some(OrientationProblem {
        hard_precedence,
        soft_conflicts,
    })
}

fn enumerate_orientations(
    node_count: usize,
    soft_conflicts: &[(usize, usize)],
    edge_idx: usize,
    reach: &mut [u32],
    seen: &mut HashSet<Vec<usize>>,
    sequences: &mut Vec<Vec<usize>>,
) {
    if edge_idx == soft_conflicts.len() {
        if let Some(sequence) = canonical_topological_order(reach, node_count) {
            if seen.insert(sequence.clone()) {
                sequences.push(sequence);
            }
        }
        return;
    }

    let (left, right) = soft_conflicts[edge_idx];
    let left_before_right = reach[left] & bit(right) != 0;
    let right_before_left = reach[right] & bit(left) != 0;

    if left_before_right && right_before_left {
        return;
    }
    if left_before_right {
        enumerate_orientations(
            node_count,
            soft_conflicts,
            edge_idx + 1,
            reach,
            seen,
            sequences,
        );
        return;
    }
    if right_before_left {
        enumerate_orientations(
            node_count,
            soft_conflicts,
            edge_idx + 1,
            reach,
            seen,
            sequences,
        );
        return;
    }

    let snapshot = reach.to_vec();
    if add_precedence_edge(reach, left, right, node_count) {
        enumerate_orientations(
            node_count,
            soft_conflicts,
            edge_idx + 1,
            reach,
            seen,
            sequences,
        );
    }

    reach.copy_from_slice(&snapshot);
    if add_precedence_edge(reach, right, left, node_count) {
        enumerate_orientations(
            node_count,
            soft_conflicts,
            edge_idx + 1,
            reach,
            seen,
            sequences,
        );
    }
    reach.copy_from_slice(&snapshot);
}

fn add_precedence_edge(reach: &mut [u32], from: usize, to: usize, node_count: usize) -> bool {
    if from == to || reach[to] & bit(from) != 0 {
        return false;
    }

    let mut predecessors_mask = bit(from);
    for idx in 0..node_count {
        if reach[idx] & bit(from) != 0 {
            predecessors_mask |= bit(idx);
        }
    }

    let successors_mask = reach[to] | bit(to);
    for idx in 0..node_count {
        if predecessors_mask & bit(idx) != 0 {
            reach[idx] |= successors_mask;
            if reach[idx] & bit(idx) != 0 {
                return false;
            }
        }
    }
    true
}

fn close_transitive_closure(reach: &mut [u32], node_count: usize) -> bool {
    for pivot in 0..node_count {
        let pivot_bit = bit(pivot);
        let pivot_reach = reach[pivot];
        for node in 0..node_count {
            if reach[node] & pivot_bit != 0 {
                reach[node] |= pivot_reach;
            }
        }
    }

    (0..node_count).all(|idx| reach[idx] & bit(idx) == 0)
}

fn canonical_topological_order(reach: &[u32], node_count: usize) -> Option<Vec<usize>> {
    let mut remaining = if node_count == 32 {
        u32::MAX
    } else {
        (1u32 << node_count) - 1
    };
    let mut sequence = Vec::with_capacity(node_count);

    while remaining != 0 {
        let mut next_node = None;
        for node in 0..node_count {
            let node_bit = bit(node);
            if remaining & node_bit == 0 {
                continue;
            }

            let mut has_incoming = false;
            for other in 0..node_count {
                if other == node || remaining & bit(other) == 0 {
                    continue;
                }
                if reach[other] & node_bit != 0 {
                    has_incoming = true;
                    break;
                }
            }

            if !has_incoming {
                next_node = Some(node);
                break;
            }
        }

        let Some(node) = next_node else {
            return None;
        };
        sequence.push(node);
        remaining &= !bit(node);
    }

    Some(sequence)
}

fn bit(idx: usize) -> u32 {
    1u32 << idx
}

fn sequence_parallelism_budget(sequence_count: usize, max_group_parallelism: usize) -> usize {
    if max_group_parallelism <= 1 || sequence_count < MIN_SEQUENCES_FOR_PARALLEL_EVAL {
        return 1;
    }

    let scaled_parallelism = (sequence_count / MIN_SEQUENCES_FOR_PARALLEL_EVAL).next_power_of_two();
    scaled_parallelism.min(max_group_parallelism).max(1)
}

fn orders_conflict(left: &SimulatedOrder, right: &SimulatedOrder) -> bool {
    match (&left.used_state_trace, &right.used_state_trace) {
        (Some(left_trace), Some(right_trace)) => traces_conflict(left_trace, right_trace),
        _ => true,
    }
}

fn traces_conflict(left: &UsedStateTrace, right: &UsedStateTrace) -> bool {
    left.read_slot_values
        .keys()
        .any(|key| right.written_slot_values.contains_key(key))
        || right
            .read_slot_values
            .keys()
            .any(|key| left.written_slot_values.contains_key(key))
        || left
            .read_slot_values
            .keys()
            .any(|key| code_write_contains(right, key.address))
        || right
            .read_slot_values
            .keys()
            .any(|key| code_write_contains(left, key.address))
        || left
            .written_slot_values
            .keys()
            .any(|key| code_write_contains(right, key.address))
        || right
            .written_slot_values
            .keys()
            .any(|key| code_write_contains(left, key.address))
        || left
            .created_contracts
            .iter()
            .chain(left.destructed_contracts.iter())
            .any(|address| code_write_contains(right, *address))
        || left
            .created_contracts
            .iter()
            .chain(left.destructed_contracts.iter())
            .any(|address| {
                right
                    .read_slot_values
                    .keys()
                    .any(|key| key.address == *address)
                    || right
                        .written_slot_values
                        .keys()
                        .any(|key| key.address == *address)
            })
        || right
            .created_contracts
            .iter()
            .chain(right.destructed_contracts.iter())
            .any(|address| {
                left.read_slot_values
                    .keys()
                    .any(|key| key.address == *address)
                    || left
                        .written_slot_values
                        .keys()
                        .any(|key| key.address == *address)
            })
}

fn code_write_contains(trace: &UsedStateTrace, address: Address) -> bool {
    trace.created_contracts.contains(&address) || trace.destructed_contracts.contains(&address)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use ahash::HashSet;
    use alloy_consensus::TxLegacy;
    use alloy_primitives::{Address, TxHash, B256, U256};
    use reth::primitives::TransactionSigned;
    use reth_primitives::{Recovered, Transaction};
    use uuid::Uuid;

    use super::*;
    use crate::{
        building::{
            builders::parallel_builder::{ConflictGroup, GroupId, TaskPriority},
            evm_inspector::{SlotKey, UsedStateTrace},
        },
        primitives::{
            Bundle, Metadata, Order, SimValue, SimulatedOrder,
            TransactionSignedEcRecoveredWithBlobs, LAST_BUNDLE_VERSION,
        },
    };

    struct DataGenerator {
        last_used_id: u64,
    }
    impl DataGenerator {
        pub fn new() -> DataGenerator {
            DataGenerator { last_used_id: 0 }
        }

        pub fn create_u64(&mut self) -> u64 {
            self.last_used_id += 1;
            self.last_used_id
        }

        pub fn create_u256(&mut self) -> U256 {
            U256::from(self.create_u64())
        }

        pub fn create_hash(&mut self) -> TxHash {
            TxHash::from(self.create_u256())
        }

        pub fn create_tx(&mut self) -> Recovered<TransactionSigned> {
            let tx_legacy = TxLegacy {
                nonce: self.create_u64(),
                ..Default::default()
            };
            Recovered::new_unchecked(
                TransactionSigned::new_unchecked(
                    Transaction::Legacy(tx_legacy),
                    alloy_primitives::Signature::test_signature(),
                    self.create_hash(),
                ),
                Address::default(),
            )
        }

        pub fn create_order_with_length(
            &mut self,
            coinbase_profit: U256,
            mev_gas_price: U256,
            num_of_orders: usize,
        ) -> Arc<SimulatedOrder> {
            let mut txs = Vec::new();
            for _ in 0..num_of_orders {
                txs.push(
                    TransactionSignedEcRecoveredWithBlobs::new_no_blobs(self.create_tx()).unwrap(),
                );
            }

            let sim_value = SimValue::new_test_no_gas(coinbase_profit, mev_gas_price);

            let bundle = Bundle {
                block: Some(0),
                min_timestamp: None,
                max_timestamp: None,
                txs,
                reverting_tx_hashes: Vec::new(),
                hash: B256::ZERO,
                uuid: Uuid::new_v4(),
                replacement_data: None,
                signer: None,
                metadata: Metadata::default(),
                dropping_tx_hashes: Vec::new(),
                refund: None,
                version: LAST_BUNDLE_VERSION,
            };

            Arc::new(SimulatedOrder {
                order: Order::Bundle(bundle),
                used_state_trace: None,
                sim_value,
            })
        }

        pub fn create_order_with_trace(
            &mut self,
            read: Option<SlotKey>,
            write: Option<SlotKey>,
            coinbase_profit: U256,
        ) -> Arc<SimulatedOrder> {
            let mut trace = UsedStateTrace::default();
            if let Some(read) = read {
                trace
                    .read_slot_values
                    .insert(read, self.create_hash().into());
            }
            if let Some(write) = write {
                trace
                    .written_slot_values
                    .insert(write, self.create_hash().into());
            }

            let sim_value = SimValue::new_test_no_gas(coinbase_profit, U256::ZERO);
            let tx = TransactionSignedEcRecoveredWithBlobs::new_no_blobs(self.create_tx()).unwrap();
            Arc::new(SimulatedOrder {
                order: Order::Tx(crate::primitives::MempoolTx { tx_with_blobs: tx }),
                used_state_trace: Some(trace),
                sim_value,
            })
        }
    }

    // Helper function to create an order group
    fn create_mock_order_group(
        id: GroupId,
        orders: Vec<Arc<SimulatedOrder>>,
        conflicting_ids: HashSet<GroupId>,
    ) -> ConflictGroup {
        ConflictGroup {
            id,
            orders: Arc::new(orders),
            conflicting_group_ids: Arc::new(conflicting_ids.into_iter().collect()),
        }
    }

    fn create_mock_task(
        group_idx: usize,
        group: ConflictGroup,
        algorithm: Algorithm,
        priority: TaskPriority,
        created_at: Instant,
    ) -> ConflictTask {
        ConflictTask {
            group_idx,
            group,
            algorithm,
            priority,
            created_at,
        }
    }

    #[test]
    fn test_all_permutations() {
        let mut data_generator = DataGenerator::new();
        let group = create_mock_order_group(
            1,
            vec![
                data_generator.create_order_with_length(U256::from(100), U256::from(100), 1), // index: 0, Length 1, profit 100, mev_gas_price 100
                data_generator.create_order_with_length(U256::from(200), U256::from(200), 3), // index: 1, Length 3, profit 200, mev_gas_price 200
                data_generator.create_order_with_length(U256::from(150), U256::from(150), 2), // index: 2, Length 2, profit 150, mev_gas_price 150
            ],
            HashSet::default(),
        );

        let task = create_mock_task(
            0,
            group,
            Algorithm::AllPermutations,
            TaskPriority::Low,
            Instant::now(),
        );

        let sequences = generate_sequences_of_orders_to_try(&task);
        assert_eq!(sequences.len(), 6);
        assert_eq!(sequences[0], vec![0, 1, 2]);
        assert_eq!(sequences[1], vec![0, 2, 1]);
        assert_eq!(sequences[2], vec![1, 0, 2]);
        assert_eq!(sequences[3], vec![1, 2, 0]);
        assert_eq!(sequences[4], vec![2, 0, 1]);
        assert_eq!(sequences[5], vec![2, 1, 0]);
    }

    #[test]
    fn test_generate_length_based_sequence() {
        let mut data_generator = DataGenerator::new();

        let orders = vec![
            data_generator.create_order_with_length(U256::from(100), U256::from(100), 1), // Length 1, profit 100
            data_generator.create_order_with_length(U256::from(200), U256::from(200), 3), // Length 3, profit 200
            data_generator.create_order_with_length(U256::from(150), U256::from(150), 2), // Length 2, profit 150
            data_generator.create_order_with_length(U256::from(300), U256::from(300), 1), // Length 1, profit 300
        ];

        let group = create_mock_order_group(1, orders, HashSet::default());

        let task = create_mock_task(
            0,
            group,
            Algorithm::Length,
            TaskPriority::Low,
            Instant::now(),
        );

        let sequences = generate_sequences_of_orders_to_try(&task);
        assert_eq!(sequences.len(), 1);
        assert_eq!(sequences[0], vec![1, 2, 3, 0]);
    }

    #[test]
    fn test_max_profit_and_mev_gas_price_sequences() {
        let mut data_generator = DataGenerator::new();
        let group = create_mock_order_group(
            1,
            vec![
                data_generator.create_order_with_length(U256::from(100), U256::from(300), 1), // index: 0, Length 1, profit 100, mev_gas_price 300
                data_generator.create_order_with_length(U256::from(200), U256::from(150), 3), // index: 1, Length 3, profit 200, mev_gas_price 150
                data_generator.create_order_with_length(U256::from(150), U256::from(200), 2), // index: 2, Length 2, profit 150, mev_gas_price 200
                data_generator.create_order_with_length(U256::from(300), U256::from(100), 1), // index: 3, Length 1, profit 300, mev_gas_price 100
            ],
            HashSet::default(),
        );

        let task = create_mock_task(
            0,
            group,
            Algorithm::Greedy,
            TaskPriority::Low,
            Instant::now(),
        );

        let sequences = generate_sequences_of_orders_to_try(&task);
        assert_eq!(sequences.len(), 2);

        // Coinbase profit is the first
        assert_eq!(sequences[0], vec![3, 1, 2, 0]);
        // MEV gas price is the second
        assert_eq!(sequences[1], vec![0, 2, 1, 3]);
    }

    #[test]
    fn test_reverse_max_profit_and_mev_gas_price_sequences() {
        let mut data_generator = DataGenerator::new();
        let group = create_mock_order_group(
            1,
            vec![
                data_generator.create_order_with_length(U256::from(100), U256::from(300), 1), // index: 0, Length 1, profit 100, mev_gas_price 300
                data_generator.create_order_with_length(U256::from(200), U256::from(150), 3), // index: 1, Length 3, profit 200, mev_gas_price 150
                data_generator.create_order_with_length(U256::from(150), U256::from(200), 2), // index: 2, Length 2, profit 150, mev_gas_price 200
                data_generator.create_order_with_length(U256::from(300), U256::from(100), 1), // index: 3, Length 1, profit 300, mev_gas_price 100
            ],
            HashSet::default(),
        );

        let task = create_mock_task(
            0,
            group,
            Algorithm::ReverseGreedy,
            TaskPriority::Low,
            Instant::now(),
        );

        let sequences = generate_sequences_of_orders_to_try(&task);
        assert_eq!(sequences.len(), 2);

        // Coinbase profit is the first
        assert_eq!(sequences[0], vec![0, 2, 1, 3]);
        // MEV gas price is the second
        assert_eq!(sequences[1], vec![3, 1, 2, 0]);
    }

    #[test]
    fn test_orientation_search_reduces_independent_swaps() {
        let mut data_generator = DataGenerator::new();
        let slot_a = SlotKey {
            address: Address::repeat_byte(0x11),
            key: B256::from(U256::from(1)),
        };
        let slot_b = SlotKey {
            address: Address::repeat_byte(0x22),
            key: B256::from(U256::from(2)),
        };

        let group = create_mock_order_group(
            1,
            vec![
                data_generator.create_order_with_trace(Some(slot_a.clone()), None, U256::from(100)),
                data_generator.create_order_with_trace(
                    Some(slot_b.clone()),
                    Some(slot_a),
                    U256::from(200),
                ),
                data_generator.create_order_with_trace(None, Some(slot_b), U256::from(150)),
            ],
            HashSet::default(),
        );

        let task = create_mock_task(
            0,
            group,
            Algorithm::OrientationSearch,
            TaskPriority::Low,
            Instant::now(),
        );

        let sequences = generate_sequences_of_orders_to_try(&task);
        assert_eq!(sequences.len(), 4);
        assert!(sequences.contains(&vec![0, 1, 2]));
        assert!(sequences.contains(&vec![0, 2, 1]));
        assert!(sequences.contains(&vec![1, 0, 2]));
        assert!(sequences.contains(&vec![2, 1, 0]));
    }
}
