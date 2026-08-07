use ahash::HashMap;
use alloy_primitives::{Address, U256};
use derivative::Derivative;
use eyre::Result;
use itertools::Itertools;
use rand::{seq::SliceRandom, SeedableRng};
use rayon::prelude::*;
use reth_errors::ProviderResult;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::trace;

use super::{
    conflict_task_generator::{update_default_graph_study_record, DefaultGraphStudyRecord},
    simulation_cache::{CachedSimulationState, SharedSimulationCache},
    Algorithm, ConflictTask, ResolutionResult,
};

use crate::building::{
    cached_reads::CachedDB, BlockBuildingContext, BlockState, ExecutionError, ExecutionResult,
    PartialBlock, ThreadBlockBuildingContext,
};
use crate::provider::StateProviderSource;
use rbuilder_primitives::{evm_inspector::UsedStateTrace, OrderId, SimulatedOrder};

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ConflictGraphStats {
    pub edge_count: usize,
    pub density: f64,
    pub max_degree: usize,
    pub has_missing_traces: bool,
}

const MIN_LENGTH_FOR_RECURSIVE_DEFAULT: usize = 8;
const RECURSIVE_DEFAULT_SEED_COUNT: usize = MIN_LENGTH_FOR_RECURSIVE_DEFAULT - 1;
const RECURSIVE_DEFAULT_SAMPLE_SEED: u64 = 0x06511;

#[derive(Debug, Clone)]
struct InteractionGraph {
    neighbors: Vec<Vec<usize>>,
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
    pub source: StateProviderSource,
    pub ctx: BlockBuildingContext,
    pub cancellation_token: CancellationToken,
    pub simulation_cache: Arc<SharedSimulationCache>,
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
        source: StateProviderSource,
        ctx: BlockBuildingContext,
        cancellation_token: CancellationToken,
        simulation_cache: Arc<SharedSimulationCache>,
        graph_study_collector: Option<Arc<parking_lot::Mutex<Vec<DefaultGraphStudyRecord>>>>,
    ) -> Self {
        ResolverContext {
            source,
            ctx,
            cancellation_token,
            simulation_cache,
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

        let (best_resolution_result, candidate_sequence_count) =
            if matches!(task.algorithm, Algorithm::RecursiveDefault) {
                self.resolve_recursive_default_task(&task)?
            } else {
                let sequence_to_try = generate_sequences_of_orders_to_try(&task);
                let best_resolution_result =
                    self.evaluate_sequences_for_task(&task, sequence_to_try.clone())?;
                (best_resolution_result, sequence_to_try.len())
            };

        if let Some(collector) = &self.graph_study_collector {
            update_default_graph_study_record(
                collector,
                task.group.id,
                task.algorithm,
                Some(candidate_sequence_count),
                None,
            );
        }

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

    fn evaluate_sequences_for_task(
        &mut self,
        task: &ConflictTask,
        sequence_to_try: Vec<Vec<usize>>,
    ) -> Result<ResolutionResult> {
        let mut best_resolution_result = ResolutionResult {
            total_profit: U256::ZERO,
            sequence_of_orders: vec![],
        };

        if sequence_to_try.len() <= 1 {
            for sequence_of_orders in sequence_to_try {
                let (resolution_result, _state) =
                    self.process_sequence_of_orders(sequence_of_orders, task, self.source.clone())?;
                self.update_best_result(resolution_result, &mut best_resolution_result);
            }
        } else {
            let source = self.source.clone();
            let ctx = self.ctx.clone();
            let cancellation_token = self.cancellation_token.clone();
            let simulation_cache = Arc::clone(&self.simulation_cache);

            let best_parallel_result = sequence_to_try
                .into_par_iter()
                .map(|sequence_of_orders| {
                    let mut resolver_ctx = ResolverContext::new(
                        source.clone(),
                        ctx.clone(),
                        cancellation_token.clone(),
                        Arc::clone(&simulation_cache),
                        None,
                    );
                    resolver_ctx
                        .process_sequence_of_orders(
                            sequence_of_orders,
                            task,
                            resolver_ctx.source.clone(),
                        )
                        .map(|(resolution_result, _state)| resolution_result)
                })
                .try_reduce_with(|left, right| {
                    Ok(if left.total_profit >= right.total_profit {
                        left
                    } else {
                        right
                    })
                })
                .transpose()?;

            if let Some(best_parallel_result) = best_parallel_result {
                self.update_best_result(best_parallel_result, &mut best_resolution_result);
            }
        }

        Ok(best_resolution_result)
    }

    fn resolve_recursive_default_task(
        &mut self,
        task: &ConflictTask,
    ) -> Result<(ResolutionResult, usize)> {
        let orders = task.group.orders.as_ref();
        let interaction_graph = build_interaction_graph(orders);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(RECURSIVE_DEFAULT_SAMPLE_SEED);
        let indices = (0..orders.len()).collect::<Vec<_>>();

        let mut solve_small = |subproblem_indices: &[usize]| {
            self.solve_small_recursive_default_subproblem(task, subproblem_indices)
        };

        let (sequence_of_orders, mut candidate_sequence_count) =
            solve_recursive_default_indices_with_solver(
                &interaction_graph,
                &indices,
                &mut rng,
                &mut solve_small,
            )?;

        let (resolution_result, _state) =
            self.process_sequence_of_orders(sequence_of_orders, task, self.source.clone())?;
        candidate_sequence_count += 1;
        Ok((resolution_result, candidate_sequence_count))
    }

    fn solve_small_recursive_default_subproblem(
        &mut self,
        root_task: &ConflictTask,
        indices: &[usize],
    ) -> Result<(Vec<usize>, usize)> {
        if indices.is_empty() {
            return Ok((Vec::new(), 0));
        }

        let subtask = create_recursive_subtask(root_task, indices);
        let candidate_sequences = generate_all_permutations_with_nonces(&subtask);
        let candidate_sequence_count = candidate_sequences.len();
        let best_result = self.evaluate_sequences_for_task(&subtask, candidate_sequences)?;
        Ok((
            remap_resolution_result_indices(&best_result, indices),
            candidate_sequence_count,
        ))
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
        sequence_of_orders: Vec<usize>,
        task: &ConflictTask,
        source: StateProviderSource,
    ) -> Result<(ResolutionResult, BlockState<CachedDB>)> {
        // @todo actually reuse it for the duration of the block
        let mut local_ctx = ThreadBlockBuildingContext::default();

        let order_id_to_index = self.initialize_order_id_to_index_map(task);
        let full_sequence_of_orders = self.initialize_full_order_ids_vec(&sequence_of_orders, task);

        // Check for cached simulation state
        let (cached_state_option, cached_up_to_index) = self
            .simulation_cache
            .get_cached_state(&full_sequence_of_orders);

        // Initialize state and partial block
        let mut partial_block = PartialBlock::new(true);
        let mut state = self.initialize_block_state(source)?;
        partial_block.pre_block_call(&self.ctx, &mut state)?;

        // Initialize sequenced_order_result
        let mut sequenced_order_result =
            self.initialize_result_order_sequence(&cached_state_option, &order_id_to_index);

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
                #[allow(clippy::result_large_err)]
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

    /// Initializes a HashMap of order id to index.
    fn initialize_order_id_to_index_map(&self, task: &ConflictTask) -> HashMap<OrderId, usize> {
        task.group
            .orders
            .iter()
            .enumerate()
            .map(|(idx, sim_order)| (sim_order.order.id(), idx))
            .collect()
    }

    /// Initializes a vector of full order ids corresponding to the sequence of orders.
    fn initialize_full_order_ids_vec(
        &self,
        sequence_of_orders: &[usize],
        task: &ConflictTask,
    ) -> Vec<OrderId> {
        sequence_of_orders
            .iter()
            .map(|&idx| task.group.orders[idx].order.id())
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
    fn initialize_block_state(
        &mut self,
        source: StateProviderSource,
    ) -> ProviderResult<BlockState<CachedDB>> {
        let cached = CachedDB::new(
            source.state_provider()?,
            self.ctx.shared_cached_reads.clone(),
        );
        Ok(BlockState::new(cached))
    }

    /// Stores the simulation state in the cache.
    fn store_simulation_state(
        &self,
        full_order_ids: &[OrderId],
        state: &BlockState<CachedDB>,
        total_profit: U256,
        per_order_profits: &[(OrderId, U256)],
    ) {
        let cached_simulation_state = CachedSimulationState {
            bundle_state: state.clone_bundle(),
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
        Algorithm::RecursiveDefault => {
            unreachable!("RecursiveDefault is handled directly in ResolverContext")
        }
        Algorithm::Random { seed, count } => generate_random_permutations(task, seed, count),
        Algorithm::PermutationsWithNonces => generate_all_permutations_with_nonces(task),
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

fn solve_recursive_default_indices_with_solver<F>(
    interaction_graph: &InteractionGraph,
    indices: &[usize],
    rng: &mut rand::rngs::SmallRng,
    solve_small: &mut F,
) -> Result<(Vec<usize>, usize)>
where
    F: FnMut(&[usize]) -> Result<(Vec<usize>, usize)>,
{
    if indices.is_empty() {
        return Ok((Vec::new(), 0));
    }

    if indices.len() < MIN_LENGTH_FOR_RECURSIVE_DEFAULT {
        return solve_small(indices);
    }

    let seed_indices = sample_seed_indices(indices, rng);
    solve_recursive_default_from_seed_indices_with_solver(
        interaction_graph,
        indices,
        &seed_indices,
        rng,
        solve_small,
    )
}

fn solve_recursive_default_from_seed_indices_with_solver<F>(
    interaction_graph: &InteractionGraph,
    indices: &[usize],
    seed_indices: &[usize],
    rng: &mut rand::rngs::SmallRng,
    solve_small: &mut F,
) -> Result<(Vec<usize>, usize)>
where
    F: FnMut(&[usize]) -> Result<(Vec<usize>, usize)>,
{
    let residual_components =
        partition_residual_components(indices, &seed_indices, interaction_graph);
    let (mut sequence, mut candidate_sequence_count) = solve_small(seed_indices)?;

    for component in residual_components {
        let (component_sequence, component_candidate_count) =
            solve_recursive_default_indices_with_solver(
                interaction_graph,
                &component,
                rng,
                solve_small,
            )?;
        sequence.extend(component_sequence);
        candidate_sequence_count += component_candidate_count;
    }

    Ok((sequence, candidate_sequence_count))
}

fn sample_seed_indices(indices: &[usize], rng: &mut rand::rngs::SmallRng) -> Vec<usize> {
    let sample_size = RECURSIVE_DEFAULT_SEED_COUNT.min(indices.len());
    indices
        .choose_multiple(rng, sample_size)
        .copied()
        .collect::<Vec<_>>()
}

fn partition_residual_components(
    indices: &[usize],
    seed_indices: &[usize],
    interaction_graph: &InteractionGraph,
) -> Vec<Vec<usize>> {
    let mut is_seed = vec![false; interaction_graph.neighbors.len()];
    for &seed_idx in seed_indices {
        is_seed[seed_idx] = true;
    }

    let residual_indices = indices
        .iter()
        .copied()
        .filter(|idx| {
            !is_seed[*idx]
                && seed_indices
                    .iter()
                    .all(|seed_idx| !interaction_graph.has_edge(*idx, *seed_idx))
        })
        .collect::<Vec<_>>();

    connected_components(&residual_indices, interaction_graph)
}

fn connected_components(
    indices: &[usize],
    interaction_graph: &InteractionGraph,
) -> Vec<Vec<usize>> {
    if indices.is_empty() {
        return Vec::new();
    }

    let mut in_subproblem = vec![false; interaction_graph.neighbors.len()];
    for &idx in indices {
        in_subproblem[idx] = true;
    }

    let mut visited = vec![false; interaction_graph.neighbors.len()];
    let mut components = Vec::new();

    for &start_idx in indices {
        if visited[start_idx] {
            continue;
        }

        let mut stack = vec![start_idx];
        let mut component = Vec::new();
        visited[start_idx] = true;

        while let Some(node_idx) = stack.pop() {
            component.push(node_idx);
            for &neighbor_idx in &interaction_graph.neighbors[node_idx] {
                if in_subproblem[neighbor_idx] && !visited[neighbor_idx] {
                    visited[neighbor_idx] = true;
                    stack.push(neighbor_idx);
                }
            }
        }

        component.sort_unstable();
        components.push(component);
    }

    components.sort_by_key(|component| component[0]);
    components
}

fn build_interaction_graph(orders: &[Arc<SimulatedOrder>]) -> InteractionGraph {
    let node_count = orders.len();
    let mut neighbors = vec![Vec::new(); node_count];

    for left in 0..node_count {
        for right in (left + 1)..node_count {
            if orders_conflict(&orders[left], &orders[right]) {
                add_interaction_edge(&mut neighbors, left, right);
            }
        }
    }

    let mut nonce_to_node_map: HashMap<(Address, u64), usize> = HashMap::default();
    for (idx, order) in orders.iter().enumerate() {
        for nonce in order.nonces() {
            nonce_to_node_map.insert((nonce.address, nonce.nonce), idx);
        }
    }

    for (idx, order) in orders.iter().enumerate() {
        for nonce in order.nonces() {
            if nonce.nonce == 0 {
                continue;
            }

            if let Some(&prev_idx) = nonce_to_node_map.get(&(nonce.address, nonce.nonce - 1)) {
                if prev_idx != idx {
                    add_interaction_edge(&mut neighbors, prev_idx, idx);
                }
            }
        }
    }

    InteractionGraph { neighbors }
}

fn add_interaction_edge(neighbors: &mut [Vec<usize>], left: usize, right: usize) {
    if !neighbors[left].contains(&right) {
        neighbors[left].push(right);
    }
    if !neighbors[right].contains(&left) {
        neighbors[right].push(left);
    }
}

impl InteractionGraph {
    fn has_edge(&self, left: usize, right: usize) -> bool {
        self.neighbors[left].contains(&right)
    }
}

fn create_recursive_subtask(root_task: &ConflictTask, indices: &[usize]) -> ConflictTask {
    let orders = indices
        .iter()
        .map(|&idx| root_task.group.orders[idx].clone())
        .collect::<Vec<_>>();

    ConflictTask {
        group_idx: root_task.group_idx,
        algorithm: Algorithm::PermutationsWithNonces,
        priority: root_task.priority,
        group: super::ConflictGroup {
            id: root_task.group.id,
            orders: Arc::new(orders),
            conflicting_group_ids: root_task.group.conflicting_group_ids.clone(),
        },
        created_at: root_task.created_at,
    }
}

fn remap_resolution_result_indices(result: &ResolutionResult, indices: &[usize]) -> Vec<usize> {
    result
        .sequence_of_orders
        .iter()
        .map(|(local_idx, _)| indices[*local_idx])
        .collect()
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
    successors: &[Vec<usize>],
    current_permutation_indices: &mut Vec<usize>,
    in_degrees: &mut [usize],
    total_nodes: usize,
    all_permutations_indices: &mut Vec<Vec<usize>>,
) {
    if current_permutation_indices.len() == total_nodes {
        all_permutations_indices.push(current_permutation_indices.clone());
        return;
    }

    let mut candidates = Vec::new();
    for (node_idx, degree) in in_degrees.iter().enumerate() {
        if !current_permutation_indices.contains(&node_idx) && *degree == 0 {
            candidates.push(node_idx);
        }
    }

    candidates.sort_unstable();

    if candidates.is_empty() && current_permutation_indices.len() != total_nodes {
        return;
    }

    for &candidate_node_idx in &candidates {
        current_permutation_indices.push(candidate_node_idx);

        let mut affected_neighbors = Vec::new();
        for &target_node_idx in &successors[candidate_node_idx] {
            in_degrees[target_node_idx] -= 1;
            affected_neighbors.push(target_node_idx);
        }

        find_permutations_recursive(
            successors,
            current_permutation_indices,
            in_degrees,
            total_nodes,
            all_permutations_indices,
        );

        for neighbor_idx in affected_neighbors {
            in_degrees[neighbor_idx] += 1;
        }
        current_permutation_indices.pop();
    }
}

fn analyze_group_conflicts_and_find_best(task: &ConflictTask) -> Vec<Vec<usize>> {
    let order_group = &task.group;
    let orders = &order_group.orders;

    fn order_value(order: &SimulatedOrder) -> U256 {
        order.sim_value.full_profit_info().coinbase_profit()
    }

    let mut idx_and_value: Vec<(usize, U256)> = orders
        .iter()
        .enumerate()
        .map(|(idx, order_arc)| (idx, order_value(order_arc)))
        .collect();
    idx_and_value.sort_by(|a, b| b.1.cmp(&a.1));

    let best_order_idx = idx_and_value[0].0;
    vec![vec![best_order_idx]]
}

fn generate_all_permutations_with_nonces(task: &ConflictTask) -> Vec<Vec<usize>> {
    generate_all_permutations_with_nonces_for_orders(task.group.orders.as_ref())
}

fn generate_all_permutations_with_nonces_for_orders(
    orders: &[Arc<SimulatedOrder>],
) -> Vec<Vec<usize>> {
    let node_count = orders.len();
    let mut nonce_to_node_map: HashMap<(Address, u64), usize> = HashMap::default();

    for (original_order_idx, order_arc) in orders.iter().enumerate() {
        for nonce in order_arc.nonces() {
            nonce_to_node_map.insert((nonce.address, nonce.nonce), original_order_idx);
        }
    }

    let mut successors = vec![Vec::new(); node_count];
    let mut in_degrees = vec![0usize; node_count];

    for (order_i_original_idx, order_i_arc) in orders.iter().enumerate() {
        for nonce_i in order_i_arc.nonces() {
            if nonce_i.nonce == 0 {
                continue;
            }
            let lookup_key = (nonce_i.address, nonce_i.nonce - 1);

            if let Some(&node_j_idx) = nonce_to_node_map.get(&lookup_key) {
                if node_j_idx != order_i_original_idx
                    && !successors[node_j_idx].contains(&order_i_original_idx)
                {
                    successors[node_j_idx].push(order_i_original_idx);
                    in_degrees[order_i_original_idx] += 1;
                }
            }
        }
    }

    let mut all_permutations_indices = Vec::new();
    let mut current_permutation_indices = Vec::new();

    find_permutations_recursive(
        &successors,
        &mut current_permutation_indices,
        &mut in_degrees,
        node_count,
        &mut all_permutations_indices,
    );

    all_permutations_indices
}

pub(crate) fn orders_conflict(left: &SimulatedOrder, right: &SimulatedOrder) -> bool {
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
    use reth_ethereum_primitives::Transaction;
    use reth_primitives_traits::Recovered;
    use uuid::Uuid;

    use super::*;
    use crate::building::builders::parallel_builder::{ConflictGroup, GroupId, TaskPriority};
    use rbuilder_primitives::{
        evm_inspector::{SlotKey, UsedStateTrace},
        Bundle, Metadata, Order, SimValue, SimulatedOrder, TransactionSignedEcRecoveredWithBlobs,
        LAST_BUNDLE_VERSION,
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
                refund_identity: None,
                version: LAST_BUNDLE_VERSION,
                external_hash: None,
            };

            Arc::new(SimulatedOrder::new(
                Arc::new(Order::Bundle(bundle)),
                sim_value,
                None,
            ))
        }

        fn create_slot(key: u64) -> SlotKey {
            SlotKey {
                address: Address::ZERO,
                key: B256::from(alloy_primitives::U256::from(key)),
            }
        }

        pub fn create_traced_order(
            &mut self,
            profit: u64,
            read_slots: &[u64],
            write_slots: &[u64],
        ) -> Arc<SimulatedOrder> {
            let tx = TransactionSignedEcRecoveredWithBlobs::new_no_blobs(self.create_tx()).unwrap();
            let mut trace = UsedStateTrace::default();
            for slot in read_slots {
                trace.read_slot_values.insert(
                    Self::create_slot(*slot),
                    B256::from(alloy_primitives::U256::from(*slot + 100)),
                );
            }
            for slot in write_slots {
                trace.written_slot_values.insert(
                    Self::create_slot(*slot),
                    B256::from(alloy_primitives::U256::from(*slot + 200)),
                );
            }

            Arc::new(SimulatedOrder::new(
                Arc::new(Order::Bundle(Bundle {
                    block: Some(0),
                    min_timestamp: None,
                    max_timestamp: None,
                    txs: vec![tx],
                    reverting_tx_hashes: Vec::new(),
                    hash: B256::ZERO,
                    uuid: Uuid::new_v4(),
                    replacement_data: None,
                    signer: None,
                    metadata: Metadata::default(),
                    dropping_tx_hashes: Vec::new(),
                    refund: None,
                    refund_identity: None,
                    version: LAST_BUNDLE_VERSION,
                    external_hash: None,
                })),
                SimValue::new_test_no_gas(U256::from(profit), U256::from(profit)),
                Some(trace),
            ))
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
    fn test_partition_residual_components_discards_seed_conflicts() {
        let mut data_generator = DataGenerator::new();
        let orders = vec![
            data_generator.create_traced_order(10, &[], &[1]),
            data_generator.create_traced_order(11, &[], &[2]),
            data_generator.create_traced_order(12, &[], &[3]),
            data_generator.create_traced_order(13, &[], &[4]),
            data_generator.create_traced_order(14, &[], &[5]),
            data_generator.create_traced_order(15, &[], &[6]),
            data_generator.create_traced_order(16, &[], &[7]),
            data_generator.create_traced_order(17, &[], &[100]),
            data_generator.create_traced_order(18, &[100], &[]),
            data_generator.create_traced_order(19, &[1], &[]),
        ];
        let interaction_graph = build_interaction_graph(&orders);
        let indices = (0..orders.len()).collect::<Vec<_>>();
        let seed_indices = (0..7).collect::<Vec<_>>();

        let components = partition_residual_components(&indices, &seed_indices, &interaction_graph);

        assert_eq!(components, vec![vec![7, 8]]);
    }

    #[test]
    fn test_recursive_default_appends_residual_component_solutions_without_cartesian() {
        let mut data_generator = DataGenerator::new();
        let orders = vec![
            data_generator.create_traced_order(10, &[], &[1]),
            data_generator.create_traced_order(11, &[], &[2]),
            data_generator.create_traced_order(12, &[], &[3]),
            data_generator.create_traced_order(13, &[], &[4]),
            data_generator.create_traced_order(14, &[], &[5]),
            data_generator.create_traced_order(15, &[], &[6]),
            data_generator.create_traced_order(16, &[], &[7]),
            data_generator.create_traced_order(17, &[], &[100]),
            data_generator.create_traced_order(18, &[100], &[]),
        ];
        let interaction_graph = build_interaction_graph(&orders);
        let indices = (0..orders.len()).collect::<Vec<_>>();
        let seed_indices = (0..7).collect::<Vec<_>>();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(RECURSIVE_DEFAULT_SAMPLE_SEED);
        let mut solve_small = |subproblem_indices: &[usize]| -> Result<(Vec<usize>, usize)> {
            Ok((subproblem_indices.to_vec(), 1))
        };

        let (sequence, candidate_sequence_count) =
            solve_recursive_default_from_seed_indices_with_solver(
                &interaction_graph,
                &indices,
                &seed_indices,
                &mut rng,
                &mut solve_small,
            )
            .unwrap();

        assert_eq!(sequence, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(candidate_sequence_count, 2);
    }

    #[test]
    fn test_remap_resolution_result_indices_maps_local_sequence_back_to_parent() {
        let resolution_result = ResolutionResult {
            total_profit: U256::ZERO,
            sequence_of_orders: vec![
                (1, U256::from(10)),
                (0, U256::from(20)),
                (2, U256::from(30)),
            ],
        };
        let indices = vec![4, 7, 9];

        let remapped = remap_resolution_result_indices(&resolution_result, &indices);

        assert_eq!(remapped, vec![7, 4, 9]);
    }

    #[test]
    fn test_solve_recursive_default_indices_delegates_small_subproblems() {
        let mut data_generator = DataGenerator::new();
        let orders = vec![
            data_generator.create_traced_order(10, &[], &[1]),
            data_generator.create_traced_order(11, &[], &[2]),
            data_generator.create_traced_order(12, &[], &[3]),
        ];
        let interaction_graph = build_interaction_graph(&orders);
        let indices = (0..orders.len()).collect::<Vec<_>>();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(RECURSIVE_DEFAULT_SAMPLE_SEED);
        let mut solve_small = |subproblem_indices: &[usize]| -> Result<(Vec<usize>, usize)> {
            Ok((subproblem_indices.iter().rev().copied().collect(), 6))
        };

        let (sequence, candidate_sequence_count) = solve_recursive_default_indices_with_solver(
            &interaction_graph,
            &indices,
            &mut rng,
            &mut solve_small,
        )
        .unwrap();

        assert_eq!(sequence, vec![2, 1, 0]);
        assert_eq!(candidate_sequence_count, 6);
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
}
