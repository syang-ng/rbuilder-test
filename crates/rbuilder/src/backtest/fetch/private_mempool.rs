//! Implementation of [`DataSource`] to bring private transactions.
//! It downloads all the needed parquet files and keeps them cached for future use.
use crate::backtest::{
    fetch::data_source::{
        get_full_slot_data_from_data, BlockRef, DataSource, DatasourceData, FullSlotDatasourceData,
    },
    OrdersWithTimestamp,
};
use rbuilder_primitives::{
    serialize::{
        RawBundle, RawBundleMetadata, RawOrder, TxEncoding, BUNDLE_VERSION_V1,
    },
};
use alloy_primitives::{Bytes, B256, U64};
use async_trait::async_trait;
use csv::Reader;
use eyre::WrapErr;

use serde::{Deserialize, Deserializer};
use std::{
    fs::{create_dir_all, File}, path::{Path, PathBuf}, str::FromStr
};
use time::{macros::format_description, Duration, OffsetDateTime, format_description::well_known::Rfc3339};
use tracing::trace;
use uuid::Uuid;
use zip::ZipArchive;



fn parse_with_utc<'de, D>(deserializer: D) -> Result<OffsetDateTime, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    let rfc3339_string = format!("{}Z", s.replace(' ', "T"));
    OffsetDateTime::parse(&rfc3339_string, &Rfc3339)
        .map_err(serde::de::Error::custom)
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

/// Constructs the path to the CSV file containing private transactions for a specific day.
fn path_bundles(data_dir: &Path, day: &str) -> PathBuf {
    data_dir.join(format!("bundles/{}.csv.zip", day))
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
                | BundleRecord {
                    inserted_at,
                    bundle_hash,
                    param_block_number: _,
                    param_signed_txs,
                    param_reverting_tx_hashes,
                    signing_address,
                    replacement_uuid,
                    param_timestamp,
                }| -> eyre::Result<OrdersWithTimestamp> {
                    let txs = param_signed_txs
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .map(Bytes::from_str)
                        .collect::<Result<Vec<_>, _>>()
                        .wrap_err_with(|| {
                            format!("Failed to parse txs for bundle {}", bundle_hash)
                        })?;
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
                        metadata: RawBundleMetadata {
                            version: Some(BUNDLE_VERSION_V1.to_owned()),
                            block_number: Some(U64::from(block)),
                            reverting_tx_hashes,
                            dropping_tx_hashes: Default::default(),
                            replacement_uuid,
                            uuid: replacement_uuid,
                            signing_address,
                            refund_identity: None,
                            min_timestamp: param_timestamp
                                .map(|ts| ts.try_into().unwrap_or_default()),
                            max_timestamp: None,
                            replacement_nonce: replacement_uuid.map(|_| 0),
                            refund_percent: None,
                            refund_recipient: None,
                            refund_tx_hashes: None,
                            delayed_refund: None,
                            bundle_hash: B256::from_str(&bundle_hash).ok(),
                            disable_cross_region_sharing: false,
                        },
                        txs,
                    };

                    let order = RawOrder::Bundle(raw_bundle)
                        .decode(TxEncoding::NoBlobData)
                        .wrap_err_with(|| format!("Failed to parse bundle {}", bundle_hash))?;

                    Ok(OrdersWithTimestamp {
                        timestamp_ms: (inserted_at.unix_timestamp_nanos() / 1_000_000)
                            .try_into()?,
                        order: order.into(),
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
        let bundles = get_bundles(
            self.path.as_path(),
            from,
            to,
            block.block_number,
        )
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

        Ok(DatasourceData {
            orders: bundles,
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
