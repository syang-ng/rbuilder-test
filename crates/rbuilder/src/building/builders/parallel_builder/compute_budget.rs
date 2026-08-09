use eyre::{bail, Result, WrapErr};
use rayon::{ThreadPool, ThreadPoolBuilder};
#[cfg(target_os = "linux")]
use std::{collections::HashSet, fs};
use std::{fmt, sync::Arc, thread};

/// A reusable Rayon pool for permutation candidate evaluation in backtests.
///
/// Live builders intentionally do not use this type and retain their existing worker behavior.
#[derive(Clone)]
pub struct CandidateExecutor {
    pool: Arc<ThreadPool>,
    candidate_threads: usize,
}

impl fmt::Debug for CandidateExecutor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CandidateExecutor")
            .field("candidate_threads", &self.candidate_threads)
            .finish_non_exhaustive()
    }
}

impl CandidateExecutor {
    /// Creates a standalone candidate executor, as used by single-block backtests.
    pub fn new(candidate_threads: usize) -> Result<Self> {
        let physical_core_limit = available_physical_cores();
        validate_compute_budget(1, candidate_threads, physical_core_limit)?;
        Self::new_for_lane(candidate_threads, 0)
    }

    fn new_for_lane(candidate_threads: usize, lane_index: usize) -> Result<Self> {
        let pool = ThreadPoolBuilder::new()
            .num_threads(candidate_threads)
            .thread_name(move |worker_index| {
                format!("backtest-candidate-{lane_index}-{worker_index}")
            })
            .build()
            .wrap_err_with(|| {
                format!(
                    "failed to create candidate pool for lane {lane_index} with \
                     {candidate_threads} threads"
                )
            })?;
        Ok(Self {
            pool: Arc::new(pool),
            candidate_threads,
        })
    }

    /// Runs work in this executor. Nested Rayon operations inherit this pool instead of the
    /// process-global pool.
    pub fn install<Operation, Output>(&self, operation: Operation) -> Output
    where
        Operation: FnOnce() -> Output + Send,
        Output: Send,
    {
        self.pool.install(operation)
    }

    pub fn candidate_threads(&self) -> usize {
        self.candidate_threads
    }
}

/// Fixed compute budget for a range backtest.
///
/// One candidate pool is created per block lane and reused for every batch in the range. A lane's
/// coordinator only waits in [`CandidateExecutor::install`]; all block and candidate computation
/// runs on the lane pool.
#[derive(Debug)]
pub struct BacktestComputeBudget {
    block_concurrency: usize,
    candidate_threads: usize,
    physical_core_limit: usize,
    lanes: Vec<CandidateExecutor>,
}

impl BacktestComputeBudget {
    pub fn new(block_concurrency: usize, candidate_threads: usize) -> Result<Self> {
        Self::new_with_physical_core_limit(
            block_concurrency,
            candidate_threads,
            available_physical_cores(),
        )
    }

    fn new_with_physical_core_limit(
        block_concurrency: usize,
        candidate_threads: usize,
        physical_core_limit: usize,
    ) -> Result<Self> {
        validate_compute_budget(block_concurrency, candidate_threads, physical_core_limit)?;
        let lanes = (0..block_concurrency)
            .map(|lane_index| CandidateExecutor::new_for_lane(candidate_threads, lane_index))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            block_concurrency,
            candidate_threads,
            physical_core_limit,
            lanes,
        })
    }

    pub fn block_concurrency(&self) -> usize {
        self.block_concurrency
    }

    pub fn candidate_threads(&self) -> usize {
        self.candidate_threads
    }

    pub fn physical_core_limit(&self) -> usize {
        self.physical_core_limit
    }

    pub fn lane(&self, lane_index: usize) -> &CandidateExecutor {
        &self.lanes[lane_index]
    }

    /// Runs one batch with at most one block assigned to each lane.
    ///
    /// The short-lived coordinator threads block in `install`; the operation itself executes on
    /// the persistent lane pool. Calling this repeatedly therefore reuses exactly the same pools.
    pub fn run_batch<Input, Output, Operation>(
        &self,
        inputs: Vec<Input>,
        operation: Operation,
    ) -> Result<Vec<Output>>
    where
        Input: Send,
        Output: Send,
        Operation: Fn(usize, CandidateExecutor, Input) -> Output + Sync,
    {
        if inputs.len() > self.block_concurrency {
            bail!(
                "backtest batch contains {} blocks but block_concurrency is {}",
                inputs.len(),
                self.block_concurrency
            );
        }
        Ok(thread::scope(|scope| {
            let operation = &operation;
            let handles = inputs
                .into_iter()
                .enumerate()
                .map(|(lane_index, input)| {
                    let executor = self.lane(lane_index).clone();
                    let executor_for_operation = executor.clone();
                    scope.spawn(move || {
                        executor
                            .install(move || operation(lane_index, executor_for_operation, input))
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
                })
                .collect()
        }))
    }
}

fn validate_compute_budget(
    block_concurrency: usize,
    candidate_threads: usize,
    physical_core_limit: usize,
) -> Result<()> {
    if block_concurrency == 0 {
        bail!("block_concurrency must be greater than zero");
    }
    if candidate_threads == 0 {
        bail!("candidate_threads must be greater than zero");
    }
    if physical_core_limit == 0 {
        bail!("available physical core count must be greater than zero");
    }
    let requested_threads = block_concurrency
        .checked_mul(candidate_threads)
        .ok_or_else(|| eyre::eyre!("block_concurrency * candidate_threads overflowed"))?;
    if requested_threads > physical_core_limit {
        bail!(
            "backtest compute budget exceeds available physical cores: \
             block_concurrency ({block_concurrency}) * candidate_threads ({candidate_threads}) = \
             {requested_threads}, available physical cores = {physical_core_limit}"
        );
    }
    Ok(())
}

/// Best-effort count of physical cores available to this process.
///
/// On Linux this intersects the process CPU affinity with package/core topology. The logical CPU
/// budget reported by the standard library remains an upper bound so cgroup quotas are respected.
/// Other platforms use `num_cpus` physical-core detection, also capped by available parallelism.
pub fn available_physical_cores() -> usize {
    let available_logical = thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    let detected_physical = num_cpus::get_physical().max(1);
    #[cfg(target_os = "linux")]
    if let Some(physical) = linux_available_physical_cores() {
        return physical.min(available_logical).max(1);
    }
    detected_physical.min(available_logical).max(1)
}

#[cfg(target_os = "linux")]
fn linux_available_physical_cores() -> Option<usize> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let allowed = status.lines().find_map(|line| {
        line.strip_prefix("Cpus_allowed_list:")
            .map(str::trim)
            .and_then(parse_cpu_list)
    })?;
    let mut cores = HashSet::new();
    for cpu in allowed {
        let topology = format!("/sys/devices/system/cpu/cpu{cpu}/topology");
        let core_id = fs::read_to_string(format!("{topology}/core_id")).ok()?;
        let package_id = fs::read_to_string(format!("{topology}/physical_package_id")).ok()?;
        cores.insert((package_id.trim().to_owned(), core_id.trim().to_owned()));
    }
    (!cores.is_empty()).then_some(cores.len())
}

#[cfg(target_os = "linux")]
fn parse_cpu_list(value: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in value.split(',') {
        let part = part.trim();
        if let Some((start, end)) = part.split_once('-') {
            let start = start.parse::<usize>().ok()?;
            let end = end.parse::<usize>().ok()?;
            if start > end {
                return None;
            }
            cpus.extend(start..=end);
        } else {
            cpus.push(part.parse::<usize>().ok()?);
        }
    }
    Some(cpus)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Barrier,
    };

    #[test]
    fn requested_sweep_combinations_fit_sixty_physical_cores() {
        for (block_concurrency, candidate_threads) in [(1, 60), (2, 30), (4, 15), (6, 10), (10, 6)]
        {
            validate_compute_budget(block_concurrency, candidate_threads, 60).unwrap();
        }
        assert!(validate_compute_budget(2, 31, 60).is_err());
        assert!(validate_compute_budget(0, 60, 60).is_err());
        assert!(validate_compute_budget(1, 0, 60).is_err());
        assert!(validate_compute_budget(1, 1, 0).is_err());
        assert!(validate_compute_budget(usize::MAX, 2, usize::MAX).is_err());
    }

    #[test]
    fn candidate_pool_is_reused_across_lane_installs() {
        let budget = BacktestComputeBudget::new_with_physical_core_limit(2, 1, 2).unwrap();
        let run = || {
            budget
                .run_batch(vec![(), ()], |lane_index, executor, ()| {
                    (
                        lane_index,
                        Arc::as_ptr(&executor.pool) as usize,
                        rayon::current_num_threads(),
                    )
                })
                .unwrap()
        };

        let first_batch = run();
        let second_batch = run();
        assert_eq!(first_batch, second_batch);
        assert_eq!(first_batch[0].2, 1);
        assert_eq!(first_batch[1].2, 1);
        assert_ne!(first_batch[0].1, first_batch[1].1);
        assert!(budget.run_batch(vec![(), (), ()], |_, _, ()| ()).is_err());
    }

    #[test]
    fn nested_install_stays_on_the_same_lane_worker() {
        let executor = CandidateExecutor::new_for_lane(2, 0).unwrap();
        let nested_executor = executor.clone();

        executor.install(move || {
            let outer_thread = thread::current().id();
            let inner_thread = nested_executor.install(|| thread::current().id());
            assert_eq!(outer_thread, inner_thread);
        });
    }

    #[test]
    fn active_candidate_work_never_exceeds_total_compute_budget() {
        let budget = BacktestComputeBudget::new_with_physical_core_limit(2, 2, 4).unwrap();
        let barrier = Arc::new(Barrier::new(4));
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        budget
            .run_batch(vec![(), ()], |_, _, ()| {
                let barrier = Arc::clone(&barrier);
                let active = Arc::clone(&active);
                let peak = Arc::clone(&peak);
                (0..2usize).into_par_iter().for_each(|_| {
                    let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(current, Ordering::SeqCst);
                    barrier.wait();
                    active.fetch_sub(1, Ordering::SeqCst);
                });
            })
            .unwrap();

        assert_eq!(peak.load(Ordering::SeqCst), 4);
        assert_eq!(active.load(Ordering::SeqCst), 0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parses_linux_cpu_affinity_lists() {
        assert_eq!(
            parse_cpu_list("0-2,8,10-11"),
            Some(vec![0, 1, 2, 8, 10, 11])
        );
        assert_eq!(parse_cpu_list("4"), Some(vec![4]));
        assert_eq!(parse_cpu_list("3-1"), None);
    }
}
