# MinHash LSH Index (Near-Duplicate Search)

The MinHash LSH index finds near duplicates of a text: rows whose content is
essentially the same as the query with a few edits, such as reposts with a
different footer, mirror pages, or copy-pasted records with small changes. It
scores rows by the **estimated Jaccard similarity** of their token shingle
sets, a literal measure of overlap that needs no model. Unlike the Full-Text
Search index, which ranks rows by term relevance, and unlike a vector index,
which finds rows that mean the same thing, this index answers "which rows are
copies of this one".

The index is approximate: candidates are found with locality sensitive hashing
(LSH) over MinHash signatures and ranked by comparing signatures, so the
reported similarity is an estimate, and rows whose similarity is below the
index's threshold are usually not found.

## High-Level Architecture

Every indexed text is turned into a fixed-length **signature** of `num_hashes`
values, and the signature is split into `num_bands` **bands** whose hashes are
the lookup keys. Two rows become candidates for each other when they share at
least one band key; candidates are then ranked by the fraction of equal
signature values.

```
              text
               |  tokenize, shingle, hash
               v
   +-----------------------------------------+
   |   signature: num_hashes 16-bit values   |   1 - (fraction of equal values)
   +-----------------------------------------+   = Jaccard distance
       |  split into num_bands bands, hash each
       v
   band keys ------------> bands table   (candidate lookup: rows sharing a key)
   signature ------------> signature table  (ranking the candidates)
```

The index uses Lance's **Segmented Index** architecture: each segment covers a
disjoint group of fragments and holds its own bands table and signature table.
Scores depend only on the two signatures being compared, never on corpus
statistics, so results from different segments merge exactly.

```
                     +----------------------------------------+
                     |            Lance Dataset               |
                     |   (Disjoint groups of Fragments 0..N)  |
                     +----------------------------------------+
                                         |
                                         v
                     +----------------------------------------+
                     |            Segmented Index             |
                     |  +-----------+ +-----------+ +-------+ |
                     |  | Segment 1 | | Segment 2 | | ...   | |
                     |  | bands +   | | bands +   | |       | |
                     |  | signatures| | signatures| |       | |
                     |  +-----------+ +-----------+ +-------+ |
                     +----------------------------------------+
```

## Index Details

```protobuf
%%% proto.message.MinHashLshIndexDetails %%%
```

| Parameter           | Default | Effect                                                                                                                                                                                                         |
|:--------------------|:--------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `num_hashes`        | 128     | Signature length. More hashes make the similarity estimate more precise (error about `1 / sqrt(num_hashes)`) and the signature table larger (`8 + 2 * num_hashes` bytes per row).                              |
| `num_bands`         | 16      | Number of bands; `num_hashes` must be a multiple of it. With `num_hashes` it sets the similarity threshold `(1 / num_bands) ^ (num_bands / num_hashes)`. The bands table costs `12 * num_bands` bytes per row. |
| `shingle_size`      | 3       | Tokens per shingle. Shorter shingles tolerate more edits but let unrelated texts that share common phrases look alike; 5 suits long documents, 2 very short texts.                                             |
| `tokenizer`         | full text search default, without stemming and stop-word removal | The tokenizer, recorded as the Full-Text Search index records it. Stemming and stop-word removal are off by default because merging different words inflates similarity.                         |
| `signature_version` | 0       | Version of the signature procedure, including its hash seed. Managed by Lance, not a user parameter.                                                                                                           |

Reference points: the probability that a row with true Jaccard similarity `J`
becomes a candidate is `1 - (1 - J^r)^b` with `b = num_bands` and
`r = num_hashes / num_bands`.

| `num_hashes` / `num_bands` | Threshold | Found at J = 0.9 | Found at J = 0.7 | Bytes per row |
|:---------------------------|:----------|:-----------------|:-----------------|:--------------|
| 64 / 8                     | 0.77      | 99%              | 38%              | 232           |
| 128 / 16 (default)         | 0.71      | 99.99%           | 61%              | 456           |
| 96 / 16                    | 0.63      | 100%             | 87%              | 392           |

Bands of fewer than six values admit a noticeable share of unrelated rows into
the candidate set on tables of billions of rows and are better left to offline
use.

## Signature Generation

The build side and the query side run the same procedure, so a query is
comparable with every stored row:

1. Tokenize the text with the configured tokenizer.
2. Form shingles of `shingle_size` consecutive tokens (joined with the byte
   `0x1F`). A text with fewer tokens than `shingle_size` forms one shingle of
   all its tokens; a text with no tokens (NULL, empty, whitespace only) has no
   signature, is not indexed, and never appears in results.
3. Hash every shingle to 64 bits with XXH64 (seed 42).
4. Apply `num_hashes` permutations of the form `a_i * x + b_i mod 2^64`, with
   coefficients drawn from a SplitMix64 generator (seed 42, odd `a_i`), and
   keep the minimum of each permutation over all shingles of the text.
5. Store each minimum as 16 bits: `(c_i * min mod 2^64) >> 48` with an odd
   multiplier `c_i` from the same generator. A minimum shrinks as the text
   grows, so its raw high bits would agree between any two long texts;
   multiply-shift hashing keeps the chance that two different minima collide
   below `2^-15`, negligible against the `1 / num_hashes` resolution of the
   estimate.

The estimated Jaccard similarity of two signatures is the fraction of
positions whose values are equal; the reported `_distance` is
`1 - estimated similarity`. Every constant of the procedure is fixed by
`signature_version`, and every segment of an index must carry identical
details, so signatures written by different segments or at query time are
always comparable. Changing a parameter, or upgrading a tokenizer dictionary,
requires rebuilding the index.

## Storage Layout

Each segment consists of two Lance files:

- `signatures.lance` — one row per indexed document: its `_rowid` and its
  signature as a fixed-size list of `num_hashes` 16-bit values. The row number
  is the segment-local document id, so a segment holds at most `2^32`
  documents; larger tables use several segments. Rows are fixed width, so one
  document's signature is a single ranged read at a known offset.
- `bands.lance` — one row per (band key, document id) pair, sorted by band key
  and then document id. A bucket is the run of rows sharing a key. The file is
  divided into logical pages of 4096 rows, and a page table holding the
  largest key of every page is stored in a global buffer of the file and kept
  in memory when the index is open. The band key puts the band number in its
  high 8 bits and a hash of the band's values in the low 56 bits, so the keys
  of one band are contiguous.

Both files also record the index parameters and the index version as schema
metadata, so a mismatch with the index details is detected when the index is
opened.

## Segments, Appends and Merging

- **Building**: `create_index` reads the text column, signs every row and
  writes one segment. Distributed builds create one uncommitted segment per
  worker over disjoint fragment sets and commit them together; the commit
  rejects segments whose details differ.
- **Unindexed appends**: fragments appended after the build are not covered
  by any segment. `optimize_indices` adds a segment over them. Until then a
  query still searches those rows, signing them on the fly with the index
  parameters, unless the scan asks for indexed rows only (`fast_search`).
- **Segment merging**: merging segments rebuilds one segment from the stored
  signatures without tokenizing the text again — the signatures of the
  surviving rows are concatenated, band keys are recomputed and the bands
  table is rewritten. A merge whose result would exceed `2^32` documents is
  rejected and the segments stay separate.
- **Deletes and compaction**: deleted rows are filtered at query time; after a
  compaction the stored row ids are remapped like those of every other scalar
  index.

## Query Evaluation

When a search is submitted (`nearest = MinHashQuery(text, column)`):

1. The query text is signed with the parameters recorded in the index, and its
   band keys are computed.
2. In every segment, each band key is looked up through the page table: a
   binary search finds the pages holding the key's bucket, and those pages are
   fetched in one scattered read. The document ids in the buckets form the
   candidate set.
3. The candidates' signatures are read — from memory when resident, otherwise
   with scattered reads, or with a sequential scan when the candidates cover a
   large share of the segment — and each candidate's Jaccard distance to the
   query is computed. Rows removed by filters or deletions are skipped.
4. Each segment keeps its `limit` closest rows; rows not covered by any
   segment are scored the same way by the query engine.
5. The per-segment results are merged by distance, and the `limit` closest
   rows are returned with their `_distance`.
