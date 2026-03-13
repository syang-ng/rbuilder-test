//! Implementation of [`DataSource`] to bring private transactions.
//! It downloads all the needed parquet files and keeps them cached for future use.
use crate::{
    backtest::{
        fetch::data_source::{
            get_full_slot_data_from_data, BlockRef, DataSource, DatasourceData,
            FullSlotDatasourceData,
        },
        OrdersWithTimestamp,
    },
    primitives::{
        serialize::{RawBundle, RawOrder, RawShareBundle, TxEncoding, BUNDLE_VERSION_V1},
        Order,
    },
};
use alloy_primitives::{Bytes, B256, U64};
use async_trait::async_trait;
use csv::Reader;
use eyre::WrapErr;

use serde::{Deserialize, Deserializer};
use std::{
    collections::HashSet,
    fs::{create_dir_all, File},
    path::{Path, PathBuf},
    str::FromStr,
};
use time::{
    format_description::well_known::Rfc3339, macros::format_description, Duration, OffsetDateTime,
};
use tracing::trace;
use uuid::Uuid;
use zip::ZipArchive;

fn parse_with_utc<'de, D>(deserializer: D) -> Result<OffsetDateTime, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    let rfc3339_string = format!("{}Z", s.replace(' ', "T"));
    OffsetDateTime::parse(&rfc3339_string, &Rfc3339).map_err(serde::de::Error::custom)
}

fn deserialize_flexible_integer<'de, D>(deserializer: D) -> Result<Option<i64>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum IntOrFloat {
        Int(i64),
        Float(f64),
    }
    // First, deserialize into an Option of our helper enum.
    // serde will correctly handle an empty field as `None`.
    let optional_value = Option::<IntOrFloat>::deserialize(deserializer)?;

    match optional_value {
        // If we got a value, convert it to i64 and wrap it back in Some()
        Some(IntOrFloat::Int(i)) => Ok(Some(i)),
        Some(IntOrFloat::Float(f)) => Ok(Some(f as i64)),
        // If the value was None to begin with, just pass it through.
        None => Ok(None),
    }
}

fn deserialize_hex_to_bytes32<'de, D>(deserializer: D) -> Result<[u8; 32], D::Error>
where
    D: Deserializer<'de>,
{
    let hex_str = String::deserialize(deserializer)?;
    let s_no_prefix = hex_str.strip_prefix("0x").unwrap_or(&hex_str);
    let bytes = hex::decode(s_no_prefix).map_err(serde::de::Error::custom)?;
    if bytes.len() != 32 {
        return Err(serde::de::Error::custom("Expected 32 bytes"));
    }
    let mut array = [0u8; 32];
    array.copy_from_slice(&bytes);
    Ok(array)
}

#[derive(Debug, Deserialize)]
struct BundleRecord {
    #[serde(deserialize_with = "parse_with_utc")]
    inserted_at: OffsetDateTime,
    bundle_hash: String,
    param_block_number: u64,
    param_signed_txs: String,
    param_reverting_tx_hashes: String,
    signing_address: String,
    replacement_uuid: Option<Uuid>,
    #[serde(deserialize_with = "deserialize_flexible_integer")]
    param_timestamp: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct ShareBundleRecord {
    #[serde(deserialize_with = "parse_with_utc")]
    received_at: OffsetDateTime,
    #[serde(deserialize_with = "deserialize_hex_to_bytes32")]
    bundle_hash: [u8; 32],
    body: String,
}

/// Constructs the path to the CSV file containing private transactions for a specific day.
fn path_bundles(data_dir: &Path, day: &str) -> PathBuf {
    data_dir.join(format!("bundles/{}.csv.zip", day))
}

fn path_sbundles(data_dir: &Path, day: &str) -> PathBuf {
    data_dir.join(format!("sbundles/{}.csv", day))
}

/// Reads a CSV file containing private transaction bundles, filters them by block number
/// and returns a vector of `BundleRecord` sorted by insertion time.
fn read_filtered_bundles(path: &Path, block: u64) -> eyre::Result<Vec<BundleRecord>> {
    if !path.exists() {
        return Err(eyre::eyre!("File not found: {}", path.display()));
    }

    let file = File::open(path)?;
    let mut archive = ZipArchive::new(file)?;

    if archive.len() == 0 {
        return Err(eyre::eyre!("ZIP archive is empty: {}", path.display()));
    }

    let csv_file = archive.by_index(0)?;
    let mut rdr = Reader::from_reader(csv_file);

    let mut bundles = Vec::new();

    for result in rdr.deserialize() {
        let record: BundleRecord = result?;
        if record.param_block_number == block {
            bundles.push(record);
        }
    }

    bundles.sort_by_key(|r| r.inserted_at);

    Ok(bundles)
}

fn read_filtered_sbundles(
    path: &Path,
    from: OffsetDateTime,
    to: OffsetDateTime,
) -> eyre::Result<Vec<ShareBundleRecord>> {
    if !path.exists() {
        return Err(eyre::eyre!("File not found: {}", path.display()));
    }

    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);

    let mut sbundles = Vec::new();

    for result in rdr.deserialize() {
        let record: ShareBundleRecord = result?;
        if record.received_at >= from && record.received_at <= to {
            sbundles.push(record);
        }
    }

    sbundles.sort_by_key(|r| r.received_at);

    Ok(sbundles)
}

/// Scans date-sharded CSV files over a given range, filters bundles for a specific
/// block, and returns them sorted by insertion time.
fn get_bundles_from_file(
    data_dir: &Path,
    from: OffsetDateTime,
    to: OffsetDateTime,
    block: u64,
) -> eyre::Result<Vec<BundleRecord>> {
    let mut bundles = Vec::new();
    let date_format = format_description!("[year]-[month]-[day]-[hour]");

    let mut current_date = from;
    // Loop through each day in the range (inclusive).
    while current_date <= to {
        let date_str = current_date.format(&date_format)?;
        let path = path_bundles(data_dir, &date_str);
        println!("Reading bundles from file: {}", path.display());
        bundles.extend(read_filtered_bundles(&path, block)?);
        current_date += Duration::hours(1);
    }

    if bundles.is_empty() {
        return Err(eyre::eyre!(
            "No bundles found for block {} in the specified range",
            block
        ));
    }
    // Sort all collected bundles by their insertion timestamp.
    bundles.sort_by_key(|r| r.inserted_at);
    Ok(bundles)
}

/// Gets all the OrdersWithTimestamp in the given interval.
/// Simulation info is set to None.
/// It checks for pre-downloaded csv files on data_dir and downloads only the missing ones.
pub fn get_bundles(
    data_dir: &Path,
    from: OffsetDateTime,
    to: OffsetDateTime,
    block: u64,
) -> eyre::Result<Vec<OrdersWithTimestamp>> {
    let bundles = get_bundles_from_file(data_dir, from, to, block)
        .wrap_err_with(|| format!("Failed to get private transactions for block {}", block))?;

    let bundle_result = bundles
        .into_iter()
        .map(
            |BundleRecord {
                 inserted_at,
                 bundle_hash,
                 param_block_number,
                 param_signed_txs,
                 param_reverting_tx_hashes,
                 signing_address,
                 replacement_uuid,
                 param_timestamp,
             }|
             -> eyre::Result<OrdersWithTimestamp> {
                let txs = param_signed_txs
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(Bytes::from_str)
                    .collect::<Result<Vec<_>, _>>()
                    .wrap_err_with(|| format!("Failed to parse txs for bundle {}", bundle_hash))?;
                let reverting_tx_hashes = param_reverting_tx_hashes
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(B256::from_str)
                    .collect::<Result<_, _>>()
                    .wrap_err_with(|| {
                        format!(
                            "Failed to parse reverting tx hashes for bundle {}",
                            bundle_hash
                        )
                    })?;

                if txs.is_empty() {
                    return Err(eyre::eyre!("Bundle {} has no txs", bundle_hash));
                }

                let signing_address = Some(signing_address.parse().wrap_err_with(|| {
                    format!("Failed to parse signing address for bundle {}", bundle_hash)
                })?);

                let raw_bundle = RawBundle {
                    block_number: Some(U64::from(block)),
                    txs,
                    reverting_tx_hashes,
                    dropping_tx_hashes: Default::default(),
                    replacement_uuid,
                    uuid: replacement_uuid,
                    signing_address: signing_address,
                    min_timestamp: param_timestamp.map(|ts| ts.try_into().unwrap_or_default()),
                    max_timestamp: None,
                    replacement_nonce: replacement_uuid.and(Some(0)),
                    refund_percent: None,
                    refund_recipient: None,
                    refund_tx_hashes: None,
                    first_seen_at: None,
                    version: Some(BUNDLE_VERSION_V1.to_owned()),
                };

                let order = RawOrder::Bundle(raw_bundle)
                    .decode(TxEncoding::NoBlobData)
                    .wrap_err_with(|| format!("Failed to parse bundle {}", bundle_hash))?;

                Ok(OrdersWithTimestamp {
                    timestamp_ms: (inserted_at.unix_timestamp_nanos() / 1_000_000).try_into()?,
                    order,
                })
            },
        )
        .collect::<Vec<_>>();

    let bundles = bundle_result
        .into_iter()
        .filter_map(|res| match res {
            Ok(bundle) => Some(bundle),
            Err(err) => {
                tracing::warn!(err = ?err, "Failed to parse bundle");
                None
            }
        })
        .collect();

    Ok(bundles)
}

fn get_sbundles_from_file(
    data_dir: &Path,
    from: OffsetDateTime,
    to: OffsetDateTime,
    block: u64,
) -> eyre::Result<Vec<ShareBundleRecord>> {
    let mut sbundles = Vec::new();
    let date_format = format_description!("[year]-[month]-[day]");

    let mut current_date = from;
    // Loop through each day in the range (inclusive).
    while current_date.date() <= to.date() {
        let date_str = current_date.format(&date_format)?;
        let path = path_sbundles(data_dir, &date_str);

        sbundles.extend(read_filtered_sbundles(&path, from, to)?);
        current_date += Duration::days(1);
    }

    // skip sbundle absence error for now
    // if sbundles.is_empty() {
    //     return Err(eyre::eyre!(
    //         "No share bundles found for block {} in the specified range",
    //         block
    //     ));
    // }

    // Sort all collected share bundles by their insertion timestamp.
    sbundles.sort_by_key(|r| r.received_at);
    Ok(sbundles)
}

pub fn get_sbundles(
    data_dir: &Path,
    from: OffsetDateTime,
    to: OffsetDateTime,
    block: u64,
) -> eyre::Result<Vec<OrdersWithTimestamp>> {
    let simulated_bundles = get_sbundles_from_file(data_dir, from, to, block)
        .wrap_err_with(|| format!("Failed to get share bundles for block {}", block))?;

    let bundles = simulated_bundles.into_iter().map(|v| (v, false));

    let bundles = bundles
        .map(
            |(record, used_sbundle)| -> eyre::Result<(u64, RawShareBundle, B256)> {
                let ShareBundleRecord {
                    received_at,
                    bundle_hash,
                    body,
                } = record;

                let hash = (bundle_hash.len() == 32)
                    .then(|| B256::from_slice(&bundle_hash))
                    .ok_or_else(|| eyre::eyre!("Invalid hash length"))?;
                let mut bundle = serde_json::from_str::<RawShareBundle>(&body)
                    .wrap_err_with(|| format!("Failed to parse share bundle {:?}", hash))?;
                // if it was used by the live builder we are sure that it has correct block range
                // so we modify it here to correct db overwrites
                if used_sbundle {
                    bundle.inclusion.block = U64::from(block);
                    bundle.inclusion.max_block = None;
                }

                Ok((
                    (received_at.unix_timestamp_nanos() / 1_000_000).try_into()?,
                    bundle,
                    hash,
                ))
            },
        )
        .collect::<Result<Vec<_>, _>>()?;

    let mut result = Vec::with_capacity(bundles.len());
    let mut inserted_bundles: HashSet<B256> = HashSet::default();

    for (timestamp_ms, bundle, hash) in bundles {
        if inserted_bundles.contains(&hash) {
            continue;
        }
        let from = bundle.inclusion.block.to::<u64>();
        let to = bundle
            .inclusion
            .max_block
            .unwrap_or(bundle.inclusion.block)
            .to::<u64>();

        if !(from <= block && block <= to) {
            continue;
        }

        let raw_order = RawOrder::ShareBundle(bundle);

        let order: Order = {
            let initial_attempt = raw_order.clone().decode(TxEncoding::NoBlobData);

            match initial_attempt {
                Ok(order) => order,
                Err(err) => {
                    // now we try to decode it with blob data
                    raw_order
                        .decode(TxEncoding::WithBlobData)
                        .wrap_err_with(|| {
                            format!(
                                "Failed to decode share bundle {:?} with blob data: {}",
                                hash, err
                            )
                        })?
                }
            }
        };

        result.push(OrdersWithTimestamp {
            timestamp_ms,
            order,
        });
        inserted_bundles.insert(hash);
    }

    Ok(result)
}

#[derive(Debug, Clone)]
pub struct PrivateTransactionsDatasource {
    path: PathBuf,
}

/// Implementation of DataSource via dumped private transactions
/// It's just a wrapper on get_bundles
#[async_trait]
impl DataSource for PrivateTransactionsDatasource {
    async fn get_data(&self, block: BlockRef) -> eyre::Result<DatasourceData> {
        let (from, to) = {
            let block_time = OffsetDateTime::from_unix_timestamp(block.block_timestamp as i64)?;
            (
                block_time - Duration::minutes(3),
                // we look ahead by 5 seconds in case block bid was delayed relative to the timestamp
                block_time + Duration::seconds(5),
            )
        };
        let bundles = get_bundles(self.path.as_path(), from, to, block.block_number)
            .wrap_err_with(|| {
                format!(
                    "Failed to get private transactions for block {} at timestamp {}",
                    block.block_number, block.block_timestamp
                )
            })?;

        trace!(
            "Fetched unfiltered private transactions, count: {}",
            bundles.len()
        );

        // let sbundles = get_sbundles(
        //     self.path.as_path(),
        //     from,
        //     to,
        //     block.block_number
        // )
        // .wrap_err_with(|| {
        //     format!(
        //         "Failed to get share bundles for block {} at timestamp {}",
        //         block.block_number, block.block_timestamp
        //     )
        // })?;

        // trace!(
        //     "Fetched unfiltered share bundles, count: {}",
        //     sbundles.len()
        // );

        Ok(DatasourceData {
            orders: bundles.into_iter().collect(),
            // .into_iter()
            // .chain(sbundles.into_iter())
            // .collect(),
            built_block_data: None,
        })
    }

    async fn get_full_slot_data(&self, block: BlockRef) -> eyre::Result<FullSlotDatasourceData> {
        get_full_slot_data_from_data(self, block).await
    }

    fn clone_box(&self) -> Box<dyn DataSource> {
        Box::new(self.clone())
    }
}

impl PrivateTransactionsDatasource {
    pub fn new(path: impl Into<PathBuf>) -> Result<Self, std::io::Error> {
        let path: PathBuf = path.into();

        // create the directory if it doesn't exist
        create_dir_all(&path)?;
        create_dir_all(path.join("bundles"))?;
        create_dir_all(path.join("sbundles"))?;

        Ok(Self { path })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use test_utils::ignore_if_env_not_set;
    use time::macros::datetime;

    #[ignore_if_env_not_set("MEMPOOL_DATADIR")]
    #[tokio::test]
    async fn test_get_mempool_transactions() {
        let data_dir = std::env::var("MEMPOOL_DATADIR").expect("MEMPOOL_DATADIR not set");

        let source = PrivateTransactionsDatasource::new(data_dir).unwrap();
        let block = BlockRef {
            block_number: 18048817,
            block_timestamp: datetime!(2023-09-04 23:59:00 UTC).unix_timestamp() as u64,
            landed_block_hash: None,
        };

        let txs = source.get_data(block).await.unwrap().orders;
        assert_eq!(txs.len(), 1732);
    }
}
