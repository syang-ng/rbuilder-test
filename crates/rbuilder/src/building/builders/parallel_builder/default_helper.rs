use alloy_primitives::U256;
use itertools::Itertools;
use parking_lot::Mutex;
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};
use rbuilder_primitives::SimulatedOrder;
use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};

use super::{
    conflict_resolvers::{analyze_conflict_graph, ConflictGraphStats},
    task::ConflictTask,
    Algorithm, ConflictGroup, GroupId, TaskPriority,
};

const NUMBER_OF_RANDOM_TASKS: usize = 50;
const MIN_LENGTH_FOR_ORIENTATION_SEARCH: usize = 8;
const MAX_FALLBACK_PERMUTATION_ORDERS: usize = MIN_LENGTH_FOR_ORIENTATION_SEARCH - 1;
const MAX_LENGTH_FOR_ORIENTATION_SEARCH: usize = 14;
const MAX_CONFLICT_EDGES_FOR_ORIENTATION_SEARCH: usize = 15;
const MAX_DENSITY_FOR_ORIENTATION_SEARCH: f64 = 0.40;
const MAX_LENGTH_FOR_PATH_LIKE_ORIENTATION_SEARCH: usize = 16;
const DEFAULT_PERMUTATION_SAMPLE_SEED: u64 = 0x06511;
const FALLBACK_PERMUTATION_SEQUENCE_BUDGET: usize = 5_040;
const FALLBACK_PERMUTATION_WORK_BUDGET: usize =
    FALLBACK_PERMUTATION_SEQUENCE_BUDGET * MAX_FALLBACK_PERMUTATION_ORDERS;

static DEFAULT_GRAPH_STUDY_CAPTURE_ENABLED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DefaultGraphStudyRecord {
    pub group_id: GroupId,
    pub algorithm: String,
    pub original_order_count: usize,
    pub selected_order_count: usize,
    pub edge_count: usize,
    pub density: f64,
    pub max_degree: usize,
    pub has_missing_traces: bool,
    pub orientation_eligible: bool,
    pub random_task_added: bool,
    pub candidate_sequence_count: Option<usize>,
    pub best_profit: Option<U256>,
}

thread_local! {
    static DEFAULT_GRAPH_STUDY_RECORDS: RefCell<Option<Arc<Mutex<Vec<DefaultGraphStudyRecord>>>>> = const { RefCell::new(None) };
}

pub(crate) fn start_default_graph_study_capture() {
    if !DEFAULT_GRAPH_STUDY_CAPTURE_ENABLED.load(Ordering::Relaxed) {
        return;
    }
    DEFAULT_GRAPH_STUDY_RECORDS.with(|records| {
        *records.borrow_mut() = Some(Arc::new(Mutex::new(Vec::new())));
    });
}

pub(crate) fn set_default_graph_study_capture_enabled(enabled: bool) {
    DEFAULT_GRAPH_STUDY_CAPTURE_ENABLED.store(enabled, Ordering::Relaxed);
}

pub(crate) fn take_default_graph_study_records() -> Vec<DefaultGraphStudyRecord> {
    DEFAULT_GRAPH_STUDY_RECORDS.with(|records| {
        records
            .borrow_mut()
            .take()
            .map(|records| records.lock().clone())
            .unwrap_or_default()
    })
}

pub(crate) fn current_default_graph_study_collector(
) -> Option<Arc<Mutex<Vec<DefaultGraphStudyRecord>>>> {
    DEFAULT_GRAPH_STUDY_RECORDS.with(|records| records.borrow().clone())
}

fn default_graph_study_enabled() -> bool {
    DEFAULT_GRAPH_STUDY_RECORDS.with(|records| records.borrow().is_some())
}

fn record_default_graph_study(record: DefaultGraphStudyRecord) {
    DEFAULT_GRAPH_STUDY_RECORDS.with(|records| {
        if let Some(records) = records.borrow().as_ref() {
            records.lock().push(record);
        }
    });
}

pub(crate) fn update_default_graph_study_record(
    collector: &Arc<Mutex<Vec<DefaultGraphStudyRecord>>>,
    group_id: GroupId,
    algorithm: Algorithm,
    candidate_sequence_count: Option<usize>,
    best_profit: Option<U256>,
) {
    let algorithm = format!("{:?}", algorithm);
    let mut records = collector.lock();
    if let Some(record) = records
        .iter_mut()
        .find(|record| record.group_id == group_id && record.algorithm == algorithm)
    {
        if let Some(candidate_sequence_count) = candidate_sequence_count {
            record.candidate_sequence_count = Some(candidate_sequence_count);
        }
        if let Some(best_profit) = best_profit {
            record.best_profit = Some(best_profit);
        }
    }
}

pub(crate) fn get_default_tasks_for_group(
    group: &ConflictGroup,
    priority: TaskPriority,
) -> Vec<ConflictTask> {
    let mut tasks = vec![];
    let created_at = Instant::now();

    if group.orders.len() >= MIN_LENGTH_FOR_ORIENTATION_SEARCH {
        let orders = &group.orders;
        let order_nonces: Vec<_> = orders
            .iter()
            .map(|order_arc| order_arc.order.nonces())
            .collect();

        let mut idx_and_value: Vec<(usize, U256)> = orders
            .iter()
            .enumerate()
            .map(|(idx, order_arc)| (idx, order_value(order_arc)))
            .collect();
        idx_and_value.sort_by(|a, b| b.1.cmp(&a.1));

        let mut used_nonces = std::collections::HashSet::new();
        let mut selected = vec![];

        for (idx, _) in idx_and_value {
            let nonces = &order_nonces[idx];
            let conflict = nonces.iter().any(|nonce| used_nonces.contains(nonce));

            if !conflict {
                selected.push(idx);
                for nonce in nonces {
                    used_nonces.insert(nonce.clone());
                }
            }
        }

        let selected_orders: Vec<_> = selected.iter().map(|&idx| orders[idx].clone()).collect();
        let selected_order_count = selected_orders.len();
        let graph_study_enabled = default_graph_study_enabled();
        let should_analyze_conflict_graph = graph_study_enabled
            || (MIN_LENGTH_FOR_ORIENTATION_SEARCH..=MAX_LENGTH_FOR_PATH_LIKE_ORIENTATION_SEARCH)
                .contains(&selected_order_count);
        let selected_stats = if should_analyze_conflict_graph {
            analyze_conflict_graph(&selected_orders)
        } else {
            ConflictGraphStats::default()
        };
        let orientation_eligible = should_analyze_conflict_graph
            && should_use_orientation_search_for_stats(selected_order_count, &selected_stats);

        let target_contracts: Vec<_> = selected_orders
            .iter()
            .flat_map(|order_arc| {
                order_arc
                    .order
                    .list_txs()
                    .into_iter()
                    .filter_map(|(tx, _)| tx.to())
            })
            .collect();
        let overlap_rate_of_target_contracts = target_contracts.iter().counts_by(|addr| *addr);
        let max_overlap_contracts = overlap_rate_of_target_contracts
            .values()
            .max()
            .cloned()
            .unwrap_or(0);
        let denominator_contracts = target_contracts.len();
        let percentage_contracts = if denominator_contracts > 0 {
            (max_overlap_contracts as f64) / (denominator_contracts as f64) * 100.0
        } else {
            0.0
        };
        let random_task_added = percentage_contracts >= 20.0;

        if random_task_added {
            tasks.push(ConflictTask {
                group_idx: group.id,
                algorithm: Algorithm::Random {
                    seed: group.id as u64,
                    count: NUMBER_OF_RANDOM_TASKS,
                },
                priority,
                group: ConflictGroup {
                    id: group.id,
                    orders: Arc::new(selected_orders.clone()),
                    conflicting_group_ids: group.conflicting_group_ids.clone(),
                },
                created_at,
            });
        }

        if orientation_eligible {
            tasks.push(ConflictTask {
                group_idx: group.id,
                algorithm: Algorithm::OrientationSearch,
                priority,
                group: ConflictGroup {
                    id: group.id,
                    orders: Arc::new(selected_orders),
                    conflicting_group_ids: group.conflicting_group_ids.clone(),
                },
                created_at,
            });
        } else {
            let exhaustive_group = build_fallback_permutation_group(group, selected_orders);
            tasks.push(ConflictTask {
                group_idx: group.id,
                algorithm: Algorithm::AllPermutations,
                priority,
                group: exhaustive_group,
                created_at,
            });
        }

        if default_graph_study_enabled() {
            for task in &tasks {
                record_default_graph_study(DefaultGraphStudyRecord {
                    group_id: group.id,
                    algorithm: format!("{:?}", task.algorithm),
                    original_order_count: group.orders.len(),
                    selected_order_count,
                    edge_count: selected_stats.edge_count,
                    density: selected_stats.density,
                    max_degree: selected_stats.max_degree,
                    has_missing_traces: selected_stats.has_missing_traces,
                    orientation_eligible,
                    random_task_added,
                    candidate_sequence_count: None,
                    best_profit: None,
                });
            }
        }
    } else {
        tasks.push(ConflictTask {
            group_idx: group.id,
            algorithm: Algorithm::PermutationsWithNonces,
            priority,
            group: group.clone(),
            created_at,
        });

        if default_graph_study_enabled() {
            let stats = analyze_conflict_graph(group.orders.as_ref());
            for task in &tasks {
                record_default_graph_study(DefaultGraphStudyRecord {
                    group_id: group.id,
                    algorithm: format!("{:?}", task.algorithm),
                    original_order_count: group.orders.len(),
                    selected_order_count: group.orders.len(),
                    edge_count: stats.edge_count,
                    density: stats.density,
                    max_degree: stats.max_degree,
                    has_missing_traces: stats.has_missing_traces,
                    orientation_eligible: should_use_orientation_search_for_stats(
                        group.orders.len(),
                        &stats,
                    ),
                    random_task_added: false,
                    candidate_sequence_count: None,
                    best_profit: None,
                });
            }
        }
    }

    tasks
}

fn order_value(order: &SimulatedOrder) -> U256 {
    order.sim_value.full_profit_info().coinbase_profit()
}

fn build_fallback_permutation_group(
    group: &ConflictGroup,
    selected_orders: Vec<Arc<SimulatedOrder>>,
) -> ConflictGroup {
    match selected_orders.len() {
        len if len <= MAX_FALLBACK_PERMUTATION_ORDERS => ConflictGroup {
            id: group.id,
            orders: Arc::new(selected_orders),
            conflicting_group_ids: group.conflicting_group_ids.clone(),
        },
        _ => {
            let mut rng = SmallRng::seed_from_u64(DEFAULT_PERMUTATION_SAMPLE_SEED);
            let sample: Vec<_> = selected_orders
                .choose_multiple(&mut rng, MAX_FALLBACK_PERMUTATION_ORDERS)
                .cloned()
                .collect();
            ConflictGroup {
                id: group.id,
                orders: Arc::new(sample),
                conflicting_group_ids: group.conflicting_group_ids.clone(),
            }
        }
    }
}

fn should_use_orientation_search_for_stats(n: usize, stats: &ConflictGraphStats) -> bool {
    if !(MIN_LENGTH_FOR_ORIENTATION_SEARCH..=MAX_LENGTH_FOR_PATH_LIKE_ORIENTATION_SEARCH)
        .contains(&n)
    {
        return false;
    }
    if stats.has_missing_traces {
        return false;
    }
    if estimated_orientation_work_upper_bound_exceeds_budget(n, stats.edge_count) {
        return false;
    }

    if n <= MAX_LENGTH_FOR_ORIENTATION_SEARCH
        && stats.edge_count <= MAX_CONFLICT_EDGES_FOR_ORIENTATION_SEARCH
        && stats.density <= MAX_DENSITY_FOR_ORIENTATION_SEARCH
    {
        return true;
    }

    n <= MAX_LENGTH_FOR_PATH_LIKE_ORIENTATION_SEARCH
        && stats.edge_count <= n + 2
        && stats.max_degree <= 2
}

fn estimated_orientation_work_upper_bound_exceeds_budget(
    order_count: usize,
    conflict_edge_count: usize,
) -> bool {
    let sequence_upper_bound = estimated_orientation_sequence_upper_bound(conflict_edge_count);
    order_count
        .checked_mul(sequence_upper_bound)
        .unwrap_or(usize::MAX)
        > FALLBACK_PERMUTATION_WORK_BUDGET
}

fn estimated_orientation_sequence_upper_bound(conflict_edge_count: usize) -> usize {
    if conflict_edge_count >= usize::BITS as usize {
        usize::MAX
    } else {
        1usize << conflict_edge_count
    }
}
