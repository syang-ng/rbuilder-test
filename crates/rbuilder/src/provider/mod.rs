use std::sync::{
    atomic::{AtomicU64, Ordering},
    mpsc, Arc,
};

use crate::{
    building::ThreadBlockBuildingContext, live_builder::simulation::SimulatedOrderCommand,
    roothash::RootHashError,
};
use alloy_consensus::Header;
use alloy_eips::BlockNumHash;
use alloy_primitives::{Address, BlockHash, BlockNumber, Bytes, B256};
use eth_sparse_mpt::utils::{HashMap, HashSet};
use reth_errors::ProviderResult;
use reth_provider::StateProviderBox;
use revm::database::BundleState;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProviderAccessStats {
    pub provider_opens: u64,
    pub consistency_checks: u64,
    pub health_scans: u64,
}

static PROVIDER_OPEN_COUNT: AtomicU64 = AtomicU64::new(0);
static PROVIDER_CONSISTENCY_CHECK_COUNT: AtomicU64 = AtomicU64::new(0);
static PROVIDER_HEALTH_SCAN_COUNT: AtomicU64 = AtomicU64::new(0);

/// Resets process-wide diagnostics used by the single-block benchmark command.
///
/// Production code never resets these counters. The relaxed atomics are deliberately kept at
/// provider-open/check granularity rather than on every account or storage read.
pub fn reset_provider_access_stats() {
    PROVIDER_OPEN_COUNT.store(0, Ordering::Relaxed);
    PROVIDER_CONSISTENCY_CHECK_COUNT.store(0, Ordering::Relaxed);
    PROVIDER_HEALTH_SCAN_COUNT.store(0, Ordering::Relaxed);
}

pub fn provider_access_stats() -> ProviderAccessStats {
    ProviderAccessStats {
        provider_opens: PROVIDER_OPEN_COUNT.load(Ordering::Relaxed),
        consistency_checks: PROVIDER_CONSISTENCY_CHECK_COUNT.load(Ordering::Relaxed),
        health_scans: PROVIDER_HEALTH_SCAN_COUNT.load(Ordering::Relaxed),
    }
}

pub(crate) fn record_provider_consistency_check() {
    PROVIDER_CONSISTENCY_CHECK_COUNT.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_provider_health_scan() {
    PROVIDER_HEALTH_SCAN_COUNT.fetch_add(1, Ordering::Relaxed);
}

pub fn benchmark_metrics_enabled() -> bool {
    std::env::var_os("RBUILDER_BENCHMARK_METRICS").is_some()
}

/// Opens a [`StateProviderBox`] for a fixed parent block from a shared [`StateProviderFactory`].
///
/// This is the unit shared across building threads instead of a single already-opened provider:
/// the factory handle and the block id are `Send + Sync`, so the source is too, and each consumer
/// opens its own `Send`-only provider on demand. Pairing the factory with the parent block also
/// removes the class of bugs where a provider is opened for the wrong block.
#[derive(Clone)]
pub struct StateProviderSource {
    factory: Arc<dyn StateProviderFactory>,
    parent_hash: BlockHash,
}

impl StateProviderSource {
    pub fn new(factory: Arc<dyn StateProviderFactory>, parent_hash: BlockHash) -> Self {
        Self {
            factory,
            parent_hash,
        }
    }

    /// Prepares a provider factory snapshot for one fixed parent block.
    ///
    /// Factories that need a consistency check before serving state can return a checked snapshot
    /// from [`StateProviderFactory::prepare_for_parent`]. Other factories keep their existing
    /// behavior through the default `None` implementation.
    pub fn new_prepared(
        factory: Arc<dyn StateProviderFactory>,
        parent_hash: BlockHash,
    ) -> ProviderResult<Self> {
        let factory = match factory.prepare_for_parent(parent_hash)? {
            Some(prepared) => prepared,
            None => factory,
        };
        Ok(Self::new(factory, parent_hash))
    }

    /// Opens a fresh state provider for the configured parent block.
    pub fn state_provider(&self) -> ProviderResult<StateProviderBox> {
        PROVIDER_OPEN_COUNT.fetch_add(1, Ordering::Relaxed);
        self.factory.history_by_block_hash(self.parent_hash)
    }

    pub fn parent_hash(&self) -> BlockHash {
        self.parent_hash
    }
}

pub mod ipc_state_provider;
pub mod reth_prov;
pub mod state_provider_factory_from_provider_factory;

/// Main trait to interact with the chain data.
/// Allows to create different backends for chain data access without implementing lots of interfaces as would happen with reth_provider::StateProviderFactory
/// since it only asks for what we really use.
pub trait StateProviderFactory: Send + Sync {
    /// Returns a checked factory snapshot for work tied to `parent_hash`.
    ///
    /// The default keeps the factory unchanged. Implementations should only override this when a
    /// one-time preparation step can replace expensive checks on every state-provider open.
    fn prepare_for_parent(
        &self,
        _parent_hash: BlockHash,
    ) -> ProviderResult<Option<Arc<dyn StateProviderFactory>>> {
        Ok(None)
    }

    fn latest(&self) -> ProviderResult<StateProviderBox>;

    fn history_by_block_number(&self, block: BlockNumber) -> ProviderResult<StateProviderBox>;

    fn history_by_block_hash(&self, block: BlockHash) -> ProviderResult<StateProviderBox>;

    fn header(&self, block_hash: &BlockHash) -> ProviderResult<Option<Header>>;

    fn block_hash(&self, number: BlockNumber) -> ProviderResult<Option<B256>>;

    fn best_block_number(&self) -> ProviderResult<BlockNumber>;

    fn header_by_number(&self, num: u64) -> ProviderResult<Option<Header>>;

    fn last_block_number(&self) -> ProviderResult<BlockNumber>;

    fn root_hasher(&self, parent_num_hash: BlockNumHash) -> ProviderResult<Box<dyn RootHasher>>;
}

impl<T> StateProviderFactory for Arc<T>
where
    T: StateProviderFactory + ?Sized,
{
    fn prepare_for_parent(
        &self,
        parent_hash: BlockHash,
    ) -> ProviderResult<Option<Arc<dyn StateProviderFactory>>> {
        (**self).prepare_for_parent(parent_hash)
    }

    fn latest(&self) -> ProviderResult<StateProviderBox> {
        (**self).latest()
    }

    fn history_by_block_number(&self, block: BlockNumber) -> ProviderResult<StateProviderBox> {
        (**self).history_by_block_number(block)
    }

    fn history_by_block_hash(&self, block: BlockHash) -> ProviderResult<StateProviderBox> {
        (**self).history_by_block_hash(block)
    }

    fn header(&self, block_hash: &BlockHash) -> ProviderResult<Option<Header>> {
        (**self).header(block_hash)
    }

    fn block_hash(&self, number: BlockNumber) -> ProviderResult<Option<B256>> {
        (**self).block_hash(number)
    }

    fn best_block_number(&self) -> ProviderResult<BlockNumber> {
        (**self).best_block_number()
    }

    fn header_by_number(&self, num: u64) -> ProviderResult<Option<Header>> {
        (**self).header_by_number(num)
    }

    fn last_block_number(&self) -> ProviderResult<BlockNumber> {
        (**self).last_block_number()
    }

    fn root_hasher(&self, parent_num_hash: BlockNumHash) -> ProviderResult<Box<dyn RootHasher>> {
        (**self).root_hasher(parent_num_hash)
    }
}

/// trait that computes the roothash for a new block assuming a predefine parent block (given in StateProviderFactory::root_hasher)
/// Ideally, it caches information in each roothash is computes (state_root) so the next one is faster.
/// Before using all run_prefetcher to allow the RootHasher start a prefetcher task that will pre cache root state trie nodes
/// based on what it sees on the simulations.
pub trait RootHasher: std::fmt::Debug + Send + Sync {
    /// Must be called once before using.
    /// This is too specific and prone to error (you may forget to call it), maybe it's a better idea to pass this to StateProviderFactory::root_hasher and let each RootHasher decide what to do?
    fn run_prefetcher(&self, simulated_orders: mpsc::Receiver<SimulatedOrderCommand>);

    /// State root for changes outcome on top of parent block.
    /// Incermental change is a list of accounts that are changed for the block since the last call to state_root
    fn state_root(
        &self,
        outcome: &BundleState,
        incremental_change: &[Address],
        local_ctx: &mut ThreadBlockBuildingContext,
    ) -> Result<B256, RootHashError>;

    /// Generate the account proof for the target address.
    /// NOTE: Proof targets are required to be loaded in the bundle state of [`ExecutionOutcome`].
    /// If the accounts are missing from the bundle state, the method will return "KeyNotFound" error.
    fn account_proofs(
        &self,
        outcome: &BundleState,
        addresses: &HashSet<Address>,
        local_ctx: &mut ThreadBlockBuildingContext,
    ) -> Result<HashMap<Address, Vec<Bytes>>, RootHashError>;
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    use reth_db::DatabaseError;
    use reth_errors::ProviderError;

    use super::*;

    #[derive(Clone)]
    struct CountingPreparingFactory {
        prepare_count: Arc<AtomicUsize>,
        history_count: Arc<AtomicUsize>,
        prepared: bool,
    }

    impl CountingPreparingFactory {
        fn unavailable<T>() -> ProviderResult<T> {
            Err(ProviderError::Database(DatabaseError::Other(
                "counting test factory has no provider".to_owned(),
            )))
        }
    }

    impl StateProviderFactory for CountingPreparingFactory {
        fn prepare_for_parent(
            &self,
            _parent_hash: BlockHash,
        ) -> ProviderResult<Option<Arc<dyn StateProviderFactory>>> {
            if self.prepared {
                return Ok(None);
            }
            self.prepare_count.fetch_add(1, AtomicOrdering::Relaxed);
            Ok(Some(Arc::new(Self {
                prepare_count: Arc::clone(&self.prepare_count),
                history_count: Arc::clone(&self.history_count),
                prepared: true,
            })))
        }

        fn latest(&self) -> ProviderResult<StateProviderBox> {
            Self::unavailable()
        }

        fn history_by_block_number(&self, _block: BlockNumber) -> ProviderResult<StateProviderBox> {
            Self::unavailable()
        }

        fn history_by_block_hash(&self, _block: BlockHash) -> ProviderResult<StateProviderBox> {
            self.history_count.fetch_add(1, AtomicOrdering::Relaxed);
            Self::unavailable()
        }

        fn header(&self, _block_hash: &BlockHash) -> ProviderResult<Option<Header>> {
            Self::unavailable()
        }

        fn block_hash(&self, _number: BlockNumber) -> ProviderResult<Option<B256>> {
            Self::unavailable()
        }

        fn best_block_number(&self) -> ProviderResult<BlockNumber> {
            Self::unavailable()
        }

        fn header_by_number(&self, _num: u64) -> ProviderResult<Option<Header>> {
            Self::unavailable()
        }

        fn last_block_number(&self) -> ProviderResult<BlockNumber> {
            Self::unavailable()
        }

        fn root_hasher(
            &self,
            _parent_num_hash: BlockNumHash,
        ) -> ProviderResult<Box<dyn RootHasher>> {
            Self::unavailable()
        }
    }

    #[test]
    fn prepared_source_is_checked_once_and_opened_per_concurrent_worker() {
        let prepare_count = Arc::new(AtomicUsize::new(0));
        let history_count = Arc::new(AtomicUsize::new(0));
        let factory = Arc::new(CountingPreparingFactory {
            prepare_count: Arc::clone(&prepare_count),
            history_count: Arc::clone(&history_count),
            prepared: false,
        });
        let source = StateProviderSource::new_prepared(factory, B256::ZERO).unwrap();

        std::thread::scope(|scope| {
            for _ in 0..60 {
                let source = source.clone();
                scope.spawn(move || assert!(source.state_provider().is_err()));
            }
        });

        assert_eq!(prepare_count.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(history_count.load(AtomicOrdering::Relaxed), 60);
    }
}
