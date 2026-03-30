use alloy_primitives::U256;
use parking_lot::Mutex;
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

const MIN_LENGTH_FOR_EXPANDED_SELECTION: usize = 8;

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

    if group.orders.len() >= MIN_LENGTH_FOR_EXPANDED_SELECTION {
        let graph_study_enabled = default_graph_study_enabled();
        let group_stats = if graph_study_enabled {
            analyze_conflict_graph(group.orders.as_ref())
        } else {
            ConflictGraphStats::default()
        };
        tasks.push(ConflictTask {
            group_idx: group.id,
            algorithm: Algorithm::RecursiveDefault,
            priority,
            group: group.clone(),
            created_at,
        });

        if default_graph_study_enabled() {
            for task in &tasks {
                record_default_graph_study(DefaultGraphStudyRecord {
                    group_id: group.id,
                    algorithm: format!("{:?}", task.algorithm),
                    original_order_count: group.orders.len(),
                    selected_order_count: group.orders.len(),
                    edge_count: group_stats.edge_count,
                    density: group_stats.density,
                    max_degree: group_stats.max_degree,
                    has_missing_traces: group_stats.has_missing_traces,
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
                    candidate_sequence_count: None,
                    best_profit: None,
                });
            }
        }
    }

    tasks
}

#[cfg(test)]
mod tests {
    use ahash::HashSet;
    use alloy_consensus::TxLegacy;
    use alloy_primitives::{Address, B256};
    use rbuilder_primitives::{
        evm_inspector::{SlotKey, UsedStateTrace},
        Bundle, Metadata, Order, SimValue, SimulatedOrder, TransactionSignedEcRecoveredWithBlobs,
        LAST_BUNDLE_VERSION,
    };
    use reth::primitives::TransactionSigned;
    use reth_primitives::{Recovered, Transaction};
    use uuid::Uuid;

    use super::*;

    struct DataGenerator {
        next_nonce: u64,
    }

    impl DataGenerator {
        fn new() -> Self {
            Self { next_nonce: 0 }
        }

        fn create_tx(&mut self) -> Recovered<TransactionSigned> {
            let nonce = self.next_nonce;
            self.next_nonce += 1;
            Recovered::new_unchecked(
                TransactionSigned::new_unchecked(
                    Transaction::Legacy(TxLegacy {
                        nonce,
                        ..Default::default()
                    }),
                    alloy_primitives::Signature::test_signature(),
                    B256::from(alloy_primitives::U256::from(nonce + 1)),
                ),
                Address::default(),
            )
        }

        fn create_slot(key: u64) -> SlotKey {
            SlotKey {
                address: Address::ZERO,
                key: B256::from(alloy_primitives::U256::from(key)),
            }
        }

        fn create_order(
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

    fn create_group(id: GroupId, orders: Vec<Arc<SimulatedOrder>>) -> ConflictGroup {
        ConflictGroup {
            id,
            orders: Arc::new(orders),
            conflicting_group_ids: Arc::new(HashSet::<GroupId>::default().into_iter().collect()),
        }
    }

    #[test]
    fn test_large_default_group_keeps_full_group_for_recursive_default() {
        let mut data = DataGenerator::new();
        let orders = vec![
            data.create_order(10, &[], &[1]),
            data.create_order(11, &[1], &[2]),
            data.create_order(12, &[2], &[3]),
            data.create_order(13, &[3], &[4]),
            data.create_order(14, &[4], &[5]),
            data.create_order(15, &[5], &[6]),
            data.create_order(16, &[6], &[7]),
            data.create_order(17, &[], &[]),
            data.create_order(18, &[], &[]),
        ];
        let group = create_group(1, orders);

        let tasks = get_default_tasks_for_group(&group, TaskPriority::High);

        assert_eq!(tasks.len(), 1);
        assert!(matches!(tasks[0].algorithm, Algorithm::RecursiveDefault));
        assert_eq!(tasks[0].group.orders.len(), group.orders.len());
    }

    #[test]
    fn test_large_default_group_uses_single_recursive_default_task() {
        let mut data = DataGenerator::new();
        let orders = (0..9)
            .map(|idx| data.create_order(100 + idx, &[], &[idx + 1]))
            .collect();
        let group = create_group(2, orders);

        let tasks = get_default_tasks_for_group(&group, TaskPriority::High);

        assert_eq!(tasks.len(), 1);
        assert!(matches!(tasks[0].algorithm, Algorithm::RecursiveDefault));
    }
}
