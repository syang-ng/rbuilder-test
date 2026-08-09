//! Caching layer for database, used to minimize disk access.
//! Frequently repeated account and storage reads stay worker-local, with a shared cache used to
//! avoid duplicate provider reads across workers.

use ahash::{HashMap, RandomState};
use alloy_primitives::{Address, B256, U256};
use dashmap::DashMap;
use reth::revm::database::StateProviderDatabase;
use reth_errors::ProviderError;
use reth_provider::{StateProvider, StateProviderBox};
use revm::{bytecode::Bytecode, state::AccountInfo, Database as RevmDatabase};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use tracing::info;

/// Database cache shared bewteen multiple threads.
/// It should be created for unique parent block.
#[derive(Debug, Clone, Default)]
pub struct SharedCachedReads {
    pub account_info: DashMap<Address, Option<AccountInfo>, RandomState>,
    pub storage: DashMap<(Address, U256), U256, RandomState>,

    pub code_by_hash: DashMap<B256, Bytecode, RandomState>,
    pub block_hash: DashMap<u64, B256, RandomState>,

    pub shared_hit_count: Arc<AtomicU64>,
    pub shared_miss_count: Arc<AtomicU64>,
}

impl Drop for SharedCachedReads {
    fn drop(&mut self) {
        let shared_hit_count = self.shared_hit_count.load(Ordering::Relaxed);
        let shared_miss_count = self.shared_miss_count.load(Ordering::Relaxed);
        let total_reads = shared_hit_count + shared_miss_count;
        let shared_hit_perc = if total_reads == 0 {
            0.0
        } else {
            100.0 * shared_hit_count as f64 / total_reads as f64
        };
        info!(
            shared_hit_count,
            shared_miss_count, shared_hit_perc, "Storage cache stats"
        );
    }
}

/// Database that wraps a reth state provider with a shared read cache.
/// Intentionally not `Clone` since StateProvider is not cloneable.
pub struct CachedDB {
    state_provider: StateProviderBox,
    shared_cache: Arc<SharedCachedReads>,
    parent_hash: Option<B256>,
    local_account_info: HashMap<Address, Option<AccountInfo>>,
    local_storage: HashMap<(Address, U256), U256>,
    pending_cache_hits: u64,
    pending_cache_misses: u64,
}

impl std::fmt::Debug for CachedDB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedDB")
            .field("parent_hash", &self.parent_hash)
            .field("local_account_info_len", &self.local_account_info.len())
            .field("local_storage_len", &self.local_storage.len())
            .finish_non_exhaustive()
    }
}

impl CachedDB {
    pub fn new(state_provider: StateProviderBox, shared_cache: Arc<SharedCachedReads>) -> Self {
        Self {
            state_provider,
            shared_cache,
            parent_hash: None,
            local_account_info: HashMap::default(),
            local_storage: HashMap::default(),
            pending_cache_hits: 0,
            pending_cache_misses: 0,
        }
    }

    /// Creates a cache scoped to a fixed parent block.
    pub fn new_for_parent(
        state_provider: StateProviderBox,
        shared_cache: Arc<SharedCachedReads>,
        parent_hash: B256,
    ) -> Self {
        let mut db = Self::new(state_provider, shared_cache);
        db.parent_hash = Some(parent_hash);
        db
    }

    /// Rebinds this worker cache to a provider/shared cache for parent_hash.
    ///
    /// Base-state reads are valid across candidates for the same parent. A parent change clears
    /// all worker-local reads before the database can be used again.
    pub fn rebind_for_parent(
        &mut self,
        state_provider: StateProviderBox,
        shared_cache: Arc<SharedCachedReads>,
        parent_hash: B256,
    ) {
        let parent_changed = self.parent_hash != Some(parent_hash);
        self.flush_stats();
        self.state_provider = state_provider;
        self.shared_cache = shared_cache;
        if parent_changed {
            self.local_account_info.clear();
            self.local_storage.clear();
        }
        self.parent_hash = Some(parent_hash);
    }

    fn inner_db(&self) -> StateProviderDatabase<&dyn StateProvider> {
        StateProviderDatabase::new(&*self.state_provider)
    }

    fn record_cache_hit(&mut self) {
        self.pending_cache_hits = self.pending_cache_hits.saturating_add(1);
    }

    fn record_cache_miss(&mut self) {
        self.pending_cache_misses = self.pending_cache_misses.saturating_add(1);
    }

    fn flush_stats(&mut self) {
        let hits = std::mem::take(&mut self.pending_cache_hits);
        let misses = std::mem::take(&mut self.pending_cache_misses);
        if hits != 0 {
            self.shared_cache
                .shared_hit_count
                .fetch_add(hits, Ordering::Relaxed);
        }
        if misses != 0 {
            self.shared_cache
                .shared_miss_count
                .fetch_add(misses, Ordering::Relaxed);
        }
    }
}

impl Drop for CachedDB {
    fn drop(&mut self) {
        self.flush_stats();
    }
}

impl RevmDatabase for CachedDB {
    type Error = ProviderError;

    fn basic(&mut self, address: Address) -> Result<Option<AccountInfo>, Self::Error> {
        if let Some(data) = self.local_account_info.get(&address).cloned() {
            self.record_cache_hit();
            return Ok(data);
        }
        if let Some(data) = self
            .shared_cache
            .account_info
            .get(&address)
            .map(|data| data.clone())
        {
            self.record_cache_hit();
            self.local_account_info.insert(address, data.clone());
            return Ok(data);
        }
        self.record_cache_miss();
        let result = self.inner_db().basic(address)?;
        self.shared_cache
            .account_info
            .insert(address, result.clone());
        self.local_account_info.insert(address, result.clone());
        Ok(result)
    }

    fn code_by_hash(&mut self, code_hash: B256) -> Result<Bytecode, Self::Error> {
        if let Some(data) = self
            .shared_cache
            .code_by_hash
            .get(&code_hash)
            .map(|data| data.clone())
        {
            self.record_cache_hit();
            return Ok(data);
        }
        self.record_cache_miss();
        let data = self.inner_db().code_by_hash(code_hash)?;
        self.shared_cache
            .code_by_hash
            .insert(code_hash, data.clone());
        Ok(data)
    }

    fn storage(&mut self, address: Address, index: U256) -> Result<U256, Self::Error> {
        let key = (address, index);
        if let Some(data) = self.local_storage.get(&key).copied() {
            self.record_cache_hit();
            return Ok(data);
        }
        if let Some(data) = self.shared_cache.storage.get(&key).map(|data| *data) {
            self.record_cache_hit();
            self.local_storage.insert(key, data);
            return Ok(data);
        }
        self.record_cache_miss();
        let result = self.inner_db().storage(address, index)?;
        self.shared_cache.storage.insert(key, result);
        self.local_storage.insert(key, result);
        Ok(result)
    }

    fn block_hash(&mut self, number: u64) -> Result<B256, Self::Error> {
        if let Some(data) = self.shared_cache.block_hash.get(&number).map(|data| *data) {
            self.record_cache_hit();
            return Ok(data);
        }
        self.record_cache_miss();
        let data = self.inner_db().block_hash(number)?;
        self.shared_cache.block_hash.insert(number, data);
        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::building::testing::test_chain_state::{BlockArgs, NamedAddr, TestChainState};
    #[test]
    fn worker_local_account_and_storage_bypass_shared_and_batch_stats() -> eyre::Result<()> {
        let chain = TestChainState::new(BlockArgs::default())?;
        let address = chain.named_address(NamedAddr::User(0))?;
        let slot = U256::from(3);
        let storage_value = U256::from(9);
        let account_info = AccountInfo {
            balance: U256::from(11),
            nonce: 7,
            ..Default::default()
        };
        let shared_cache = Arc::new(SharedCachedReads::default());
        shared_cache
            .account_info
            .insert(address, Some(account_info.clone()));
        shared_cache.storage.insert((address, slot), storage_value);

        let mut db = CachedDB::new(
            chain.provider_factory().latest()?,
            Arc::clone(&shared_cache),
        );
        assert_eq!(db.basic(address)?, Some(account_info.clone()));
        assert_eq!(db.storage(address, slot)?, storage_value);

        shared_cache.account_info.clear();
        shared_cache.storage.clear();
        assert_eq!(db.basic(address)?, Some(account_info));
        assert_eq!(db.storage(address, slot)?, storage_value);

        assert_eq!(shared_cache.shared_hit_count.load(Ordering::Relaxed), 0);
        assert_eq!(shared_cache.shared_miss_count.load(Ordering::Relaxed), 0);
        drop(db);
        assert_eq!(shared_cache.shared_hit_count.load(Ordering::Relaxed), 4);
        assert_eq!(shared_cache.shared_miss_count.load(Ordering::Relaxed), 0);
        Ok(())
    }

    #[test]
    fn local_miss_uses_shared_before_provider_across_workers() -> eyre::Result<()> {
        let chain = TestChainState::new(BlockArgs::default())?;
        let address = chain.named_address(NamedAddr::User(0))?;
        let shared_cache = Arc::new(SharedCachedReads::default());

        let first_result = {
            let mut first_worker = CachedDB::new(
                chain.provider_factory().latest()?,
                Arc::clone(&shared_cache),
            );
            first_worker.basic(address)?
        };
        assert_eq!(shared_cache.shared_hit_count.load(Ordering::Relaxed), 0);
        assert_eq!(shared_cache.shared_miss_count.load(Ordering::Relaxed), 1);

        let second_result = {
            let mut second_worker = CachedDB::new(
                chain.provider_factory().latest()?,
                Arc::clone(&shared_cache),
            );
            second_worker.basic(address)?
        };
        assert_eq!(second_result, first_result);
        assert_eq!(shared_cache.shared_hit_count.load(Ordering::Relaxed), 1);
        assert_eq!(shared_cache.shared_miss_count.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn rebind_to_new_parent_clears_local_reads_and_flushes_old_stats() -> eyre::Result<()> {
        let chain = TestChainState::new(BlockArgs::default())?;
        let address = chain.named_address(NamedAddr::User(0))?;
        let parent_a = B256::from([1_u8; 32]);
        let parent_b = B256::from([2_u8; 32]);
        let account_a = AccountInfo {
            balance: U256::from(1),
            nonce: 1,
            ..Default::default()
        };
        let account_b = AccountInfo {
            balance: U256::from(2),
            nonce: 2,
            ..Default::default()
        };
        let shared_a = Arc::new(SharedCachedReads::default());
        let shared_b = Arc::new(SharedCachedReads::default());
        shared_a
            .account_info
            .insert(address, Some(account_a.clone()));
        shared_b
            .account_info
            .insert(address, Some(account_b.clone()));

        let mut db = CachedDB::new_for_parent(
            chain.provider_factory().latest()?,
            Arc::clone(&shared_a),
            parent_a,
        );
        assert_eq!(db.basic(address)?, Some(account_a));
        assert_eq!(shared_a.shared_hit_count.load(Ordering::Relaxed), 0);

        db.rebind_for_parent(
            chain.provider_factory().latest()?,
            Arc::clone(&shared_b),
            parent_b,
        );
        assert_eq!(db.parent_hash, Some(parent_b));
        assert_eq!(shared_a.shared_hit_count.load(Ordering::Relaxed), 1);
        assert_eq!(db.basic(address)?, Some(account_b));
        drop(db);

        assert_eq!(shared_b.shared_hit_count.load(Ordering::Relaxed), 1);
        assert_eq!(shared_a.shared_miss_count.load(Ordering::Relaxed), 0);
        assert_eq!(shared_b.shared_miss_count.load(Ordering::Relaxed), 0);
        Ok(())
    }
}
