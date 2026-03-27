//! Backtest app to build a single block in a similar way as we do in live.
//! It gets the orders from a HistoricalDataStorage, simulates the orders and then runs the building algorithms.
//! It outputs the best algorithm (most profit) so we can check for improvements in our [crate::building::builders::BlockBuildingAlgorithm]s
//! BlockBuildingAlgorithm are defined on the config file but selected on the command line via "--builders"
//! Sample call:
//! backtest-build-block --config /home/happy_programmer/config.toml --builders mgp-ordering --builders mp-ordering 19380913 --show-orders --show-missing

use ahash::HashMap;
use alloy_primitives::{utils::format_ether, TxHash};

use crate::{
    backtest::{
        execute::{backtest_prepare_orders_from_building_context, BacktestBlockInput},
        OrdersWithTimestamp,
    },
    building::{
        builders::{
            parallel_builder::conflict_task_generator::{
                set_default_graph_study_capture_enabled, start_default_graph_study_capture,
                take_default_graph_study_records, DefaultGraphStudyRecord,
            },
            BacktestSimulateBlockInput,
        },
        BlockBuildingContext, ExecutionResult, NullPartialBlockExecutionTracer,
    },
    live_builder::cli::LiveBuilderConfig,
    provider::StateProviderFactory,
    utils::elapsed_ms,
};
use clap::Parser;
use rbuilder_primitives::{order_statistics::OrderStatistics, Order, OrderId, SimulatedOrder};
use std::{
    fs::File,
    io::{self, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

#[derive(Parser, Debug)]
pub struct BuildBlockCfg {
    #[clap(long, help = "Config file path", env = "RBUILDER_CONFIG")]
    pub config: PathBuf,
    #[clap(long, help = "Show all available orders")]
    pub show_orders: bool,
    #[clap(long, help = "Show order data and top of block simulation results")]
    pub show_sim: bool,
    #[clap(long, help = "don't build block")]
    pub no_block_building: bool,
    #[clap(
        long,
        help = "builders to build block with (see config builders)",
        default_value = "mp-ordering"
    )]
    pub builders: Vec<String>,
    #[clap(
        long,
        help = "Traces block building execution (shows all executed orders and txs)"
    )]
    pub trace_block_building: bool,
    #[clap(
        long,
        help = "Shows any order and sim order containing this tx hash. Example: --show-tx-extra-data 0x4905f253e997236afecddb080e38028227b083c4d9921209df7fda192f0ec428"
    )]
    pub show_tx_extra_data: Option<TxHash>,
    #[clap(
        long,
        help = "Path to csv file to write default builder graph-study output to"
    )]
    pub graph_stats_csv: Option<PathBuf>,
}

/// Provides all the orders needed to simulate the construction of a block.
/// It also provides the needed context to execute those orders.
pub trait OrdersSource<ConfigType, ProviderType>
where
    ConfigType: LiveBuilderConfig,
    ProviderType: StateProviderFactory + Clone + 'static,
{
    fn config(&self) -> &ConfigType;
    /// Orders available to build blocks with their time of arrival.
    fn available_orders(&self) -> Vec<OrdersWithTimestamp>;
    /// Start of the slot for the block.
    /// Usually all the orders will arrive before block_time_as_unix_ms + 4secs (max get_header time from validator to relays).
    fn block_time_as_unix_ms(&self) -> u64;

    /// ugly: it takes BaseConfig but not all implementations need it.....
    fn create_provider_factory(&self) -> eyre::Result<ProviderType>;

    fn create_block_building_context(&self) -> eyre::Result<BlockBuildingContext>;

    /// Prints any stats specific to the particular OrdersSource implementation (eg: parameters, block simulation)
    fn print_custom_stats(&self, provider: ProviderType) -> eyre::Result<()>;
}

pub async fn run_backtest_build_block<ConfigType, OrdersSourceType, ProviderType>(
    build_block_cfg: BuildBlockCfg,
    orders_source: OrdersSourceType,
) -> eyre::Result<()>
where
    ConfigType: LiveBuilderConfig,
    ProviderType: StateProviderFactory + Clone + 'static,
    OrdersSourceType: OrdersSource<ConfigType, ProviderType>,
{
    let total_start = Instant::now();
    let mut step_start = Instant::now();
    let ctx = orders_source.create_block_building_context()?;
    print_backtest_timing("create_block_building_context", step_start, total_start);

    step_start = Instant::now();
    let config = orders_source.config();
    config.base_config().setup_tracing_subscriber()?;
    print_backtest_timing("setup_tracing_subscriber", step_start, total_start);

    let mut graph_stats_csv_output = if let Some(path) = &build_block_cfg.graph_stats_csv {
        let mut graph_stats_csv_output = GraphStatsCSVWriter::new(path)?;
        graph_stats_csv_output.write_header()?;
        Some(graph_stats_csv_output)
    } else {
        None
    };
    set_default_graph_study_capture_enabled(graph_stats_csv_output.is_some());

    step_start = Instant::now();
    let available_orders = orders_source.available_orders();
    let mut order_statistics = OrderStatistics::new();
    for order in &available_orders {
        order_statistics.add(&order.order);
    }
    println!("mev_blocker_price {}", format_ether(ctx.mev_blocker_price));
    println!("Available orders: {}", available_orders.len());
    println!("Available orders: {}", available_orders.len());
    println!("Order statistics: {order_statistics:?}");
    print_backtest_timing("load_available_orders", step_start, total_start);

    step_start = Instant::now();
    let provider_factory = orders_source.create_provider_factory()?;
    orders_source.print_custom_stats(provider_factory.clone())?;
    print_backtest_timing("create_provider_and_stats", step_start, total_start);

    step_start = Instant::now();
    let BacktestBlockInput { sim_orders, .. } = backtest_prepare_orders_from_building_context(
        ctx.clone(),
        available_orders.clone(),
        provider_factory.clone(),
    )?;
    print_backtest_timing("simulate_available_orders", step_start, total_start);

    if let Some(tx_hash) = build_block_cfg.show_tx_extra_data {
        step_start = Instant::now();
        print_orders_with_tx_hash(tx_hash, &available_orders, &sim_orders);
        print_backtest_timing("show_tx_extra_data", step_start, total_start);
    }

    if build_block_cfg.show_orders {
        step_start = Instant::now();
        print_order_and_timestamp(&available_orders, orders_source.block_time_as_unix_ms());
        print_backtest_timing("show_orders", step_start, total_start);
    }

    if build_block_cfg.show_sim {
        step_start = Instant::now();
        let order_and_timestamp: HashMap<OrderId, u64> = available_orders
            .iter()
            .map(|order| (order.order.id(), order.timestamp_ms))
            .collect();
        print_simulated_orders(
            &sim_orders,
            &order_and_timestamp,
            orders_source.block_time_as_unix_ms(),
        );
        print_backtest_timing("show_sim", step_start, total_start);
    }

    if !build_block_cfg.no_block_building {
        println!(
            "[backtest-build-block] start_block_building total_elapsed_ms={:.2}",
            elapsed_ms(total_start)
        );
        let winning_builder = build_block_cfg
            .builders
            .iter()
            .filter_map(|builder_name: &String| {
                println!(
                    "[backtest-build-block] start_builder builder={} total_elapsed_ms={:.2}",
                    builder_name,
                    elapsed_ms(total_start)
                );
                let input = BacktestSimulateBlockInput {
                    ctx: ctx.clone(),
                    builder_name: builder_name.clone(),
                    sim_orders: &sim_orders,
                    provider: provider_factory.clone(),
                };
                start_default_graph_study_capture();
                let build_res = if build_block_cfg.trace_block_building {
                    let build_start = Instant::now();
                    let build_res = config.build_backtest_block(
                        builder_name,
                        input,
                        crate::backtest::build_block::full_partial_block_execution_tracer::FullPartialBlockExecutionTracer::new(),
                    );
                    let build_time_ms = build_start.elapsed().as_millis() as u64;
                    (build_res, build_time_ms)
                } else {
                    let build_start = Instant::now();
                    let build_res = config.build_backtest_block(
                        builder_name,
                        input,
                        NullPartialBlockExecutionTracer {},
                    );
                    let build_time_ms = build_start.elapsed().as_millis() as u64;
                    (build_res, build_time_ms)
                };
                let (build_res, build_time_ms) = build_res;
                let graph_study_records = take_default_graph_study_records();
                if let Some(graph_stats_csv_output) = &mut graph_stats_csv_output {
                    if let Err(err) = graph_stats_csv_output.write_builder_records(
                        ctx.block(),
                        builder_name,
                        &graph_study_records,
                    ) {
                        println!(
                            "Failed to write graph-study csv for builder {builder_name}: {err:?}"
                        );
                    }
                }
                if let Err(err) = &build_res {
                    println!("Error building block: {err:?}");
                    return None;
                }
                let block = build_res.ok()?;
                println!(
                    "Built block {} with builder: {builder_name:?}",
                    ctx.block()
                );
                println!("Builder profit: {}", format_ether(block.trace.bid_value));
                println!("Builder time:   {} ms", build_time_ms);
                println!(
                    "[backtest-build-block] finish_builder builder={} build_time_ms={} total_elapsed_ms={:.2}",
                    builder_name,
                    build_time_ms,
                    elapsed_ms(total_start)
                );
                println!(
                    "Number of used orders: {}",
                    block.trace.included_orders.len()
                );
                block.trace.included_orders.iter().for_each(print_order_execution_result);
                Some((builder_name.clone(), block.trace.bid_value))
            })
            .max_by_key(|(_, value)| *value);

        if let Some((builder_name, value)) = winning_builder {
            println!(
                "Winning builder: {} with profit: {}",
                builder_name,
                format_ether(value)
            );
        }
    }

    println!(
        "[backtest-build-block] done total_elapsed_ms={:.2}",
        elapsed_ms(total_start)
    );

    Ok(())
}

fn print_backtest_timing(step: &str, step_start: Instant, total_start: Instant) {
    println!(
        "[backtest-build-block] step={} step_ms={:.2} total_elapsed_ms={:.2}",
        step,
        elapsed_ms(step_start),
        elapsed_ms(total_start)
    );
}

#[derive(Debug)]
struct GraphStatsCSVWriter {
    file: File,
}

impl GraphStatsCSVWriter {
    fn new(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        Ok(Self { file })
    }

    fn write_header(&mut self) -> io::Result<()> {
        writeln!(
            self.file,
            "block_number,builder_name,group_id,algorithm,original_order_count,selected_order_count,edge_count,density,max_degree,has_missing_traces,orientation_eligible,random_task_added,candidate_sequence_count,best_profit"
        )?;
        self.file.flush()
    }

    fn write_builder_records(
        &mut self,
        block_number: u64,
        builder_name: &str,
        records: &[DefaultGraphStudyRecord],
    ) -> io::Result<()> {
        for record in records {
            writeln!(
                self.file,
                "{},{},{},{},{},{},{},{:.6},{},{},{},{},{},{}",
                block_number,
                builder_name,
                record.group_id,
                record.algorithm,
                record.original_order_count,
                record.selected_order_count,
                record.edge_count,
                record.density,
                record.max_degree,
                record.has_missing_traces,
                record.orientation_eligible,
                record.random_task_added,
                record
                    .candidate_sequence_count
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                record
                    .best_profit
                    .map(|value| value.to_string())
                    .unwrap_or_default()
            )?;
        }
        self.file.flush()
    }
}

fn print_order(order: &Order) {
    println!("{}", order.id());
    if let Order::Bundle(_) = order {
        for (tx, _) in order.list_txs() {
            println!("      ↳ {:?}", tx.hash());
        }
    }
}

fn print_sim_order(sim_order: &SimulatedOrder) {
    print_order(&sim_order.order);
    let sim_value = &sim_order.sim_value;
    let profit_info = [
        ("full", sim_value.full_profit_info()),
        ("non_mempool", sim_value.non_mempool_profit_info()),
    ];
    for (name, profit_info) in profit_info {
        println!(
            "      * {name}: coinbase_profit {} mev_gas_price {}",
            format_ether(profit_info.coinbase_profit()),
            format_ether(profit_info.mev_gas_price())
        );
    }
    println!("      * gas_used {:?}", sim_value.gas_used());
}

fn print_orders_with_tx_hash(
    tx_hash: TxHash,
    available_orders: &[OrdersWithTimestamp],
    sim_orders: &[Arc<SimulatedOrder>],
) {
    println!("---- BEGIN Orders with tx hash: {:?}", tx_hash);
    println!("ORDERS:");

    available_orders
        .iter()
        .map(|order_with_timestamp| &order_with_timestamp.order)
        .filter(|order| order.list_txs().iter().any(|(tx, _)| tx.hash() == tx_hash))
        .for_each(|order| print_order(order));
    println!("\nSIM ORDERS:");
    sim_orders
        .iter()
        .filter(|order| {
            order
                .order
                .list_txs()
                .iter()
                .any(|(tx, _)| tx.hash() == tx_hash)
        })
        .for_each(|sim_order| print_sim_order(sim_order.as_ref()));
    println!("---- END Orders with tx hash: {:?}", tx_hash);
}

fn print_order_execution_result(order_result: &ExecutionResult) {
    println!(
        "{:<74} gas: {:>8} profit: {}",
        order_result.order.id().to_string(),
        order_result.space_used.gas,
        format_ether(order_result.coinbase_profit),
    );
    if let Order::Bundle(_) = order_result.order.as_ref() {
        for tx in order_result.tx_infos.iter().map(|info| &info.tx) {
            println!("      ↳ {:?}", tx.hash());
        }

        for (to, value) in &order_result.paid_kickbacks {
            println!(
                "      $ Paid kickback to: {:?} value: {}",
                to,
                format_ether(*value)
            );
        }

        if let Some(delayed_kickback) = &order_result.delayed_kickback {
            println!(
                "      $ Delayed kickback to: {:?} value: {} tx_fee: {} paid at end of block: {}",
                delayed_kickback.recipient,
                format_ether(delayed_kickback.payout_value),
                format_ether(delayed_kickback.payout_tx_fee),
                delayed_kickback.should_pay_in_block
            );
        }
    }
}

/// Convert a timestamp in milliseconds to the slot time relative to the given block timestamp.
fn timestamp_ms_to_slot_time(timestamp_ms: u64, block_timestamp: u64) -> i64 {
    (block_timestamp * 1000) as i64 - (timestamp_ms as i64)
}

/// Print the available orders sorted by timestamp.
fn print_order_and_timestamp(orders_with_ts: &[OrdersWithTimestamp], block_time_as_unix_ms: u64) {
    println!("---- BEGIN Orders and timestamp:");
    let mut order_by_ts = orders_with_ts.to_vec();
    order_by_ts.sort_by_key(|owt| owt.timestamp_ms);
    for owt in order_by_ts {
        let id = owt.order.id();
        println!(
            "{:>74} ts: {}",
            id.to_string(),
            timestamp_ms_to_slot_time(owt.timestamp_ms, block_time_as_unix_ms)
        );
        for (tx, optional) in owt.order.list_txs() {
            println!("    {:?} {:?}", tx.hash(), optional);
            println!(
                "        from: {:?} to: {:?} nonce: {}",
                tx.signer(),
                tx.to(),
                tx.nonce()
            )
        }
    }
    println!("---- END Orders and timestamp");
}

/// Print information about simulated orders.
fn print_simulated_orders(
    sim_orders: &[Arc<SimulatedOrder>],
    order_and_timestamp: &HashMap<OrderId, u64>,
    block_time_as_unix_ms: u64,
) {
    println!("Simulated orders: ({} total)", sim_orders.len());
    let mut sorted_orders = sim_orders.to_owned();
    sorted_orders.sort_by_key(|order| order.sim_value.full_profit_info().coinbase_profit());
    sorted_orders.reverse();
    for order in sorted_orders {
        let order_timestamp = order_and_timestamp
            .get(&order.order.id())
            .copied()
            .unwrap_or_default();

        let slot_time_ms = timestamp_ms_to_slot_time(order_timestamp, block_time_as_unix_ms);

        println!(
            "{:>74} slot_time_ms: {:>8}, gas: {:>8} profit: {}",
            order.order.id().to_string(),
            slot_time_ms,
            order.sim_value.gas_used(),
            format_ether(order.sim_value.full_profit_info().coinbase_profit()),
        );
    }
    println!();
}
