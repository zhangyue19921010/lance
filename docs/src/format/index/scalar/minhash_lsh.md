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

| Parameter           | Default                                                          | Range                                         | Effect                                                                                                                                                                                                         |
|:--------------------|:-----------------------------------------------------------------|:----------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `num_hashes`        | 128                                                              | 1 to 4096, a multiple of `num_bands`          | Signature length. More hashes make the similarity estimate more precise (error about `1 / sqrt(num_hashes)`) and the signature table larger (`8 + 2 * num_hashes` bytes per row).                              |
| `num_bands`         | 16                                                               | 1 to 256                                      | Number of bands. With `num_hashes` it sets the similarity threshold `(1 / num_bands) ^ (num_bands / num_hashes)`. The bands table costs `12 * num_bands` bytes per row.                                        |
| `shingle_size`      | 3                                                                | at least 1                                    | Tokens per shingle. Shorter shingles tolerate more edits but let unrelated texts that share common phrases look alike; 5 suits long documents, 2 very short texts.                                             |
| `tokenizer`         | full text search default, without stemming and stop-word removal | the subset described under [Tokenizer](#tokenizer) | The tokenizer, recorded as the Full-Text Search index records it. Stemming and stop-word removal are off by default because merging different words inflates similarity.                                  |
| `signature_version` | 0                                                                | 0                                             | Version of the signature procedure, including its hash seed and the band key hash. Managed by Lance, not a user parameter.                                                                                     |
| `tokenizer_fingerprint` | absent                                                       | present exactly when the tokenizer loads resources from the language model home | XXH64 fingerprint of those resources, defined under [Tokenizer](#tokenizer). Managed by Lance, not a user parameter.                                                                       |

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

### Validation

A reader validates the details before it allocates memory or reads an index
file, and rejects the index when:

- a parameter is outside its range above (this includes a `num_hashes` that is
  not a multiple of `num_bands`);
- `tokenizer` is absent, or asks for a setting outside the supported subset;
- `signature_version` is not a version the reader implements (only 0 exists);
- `tokenizer_fingerprint` is present for a tokenizer that loads no resources,
  or absent for one that does;
- the tokenizer loads resources and the reader does not implement the
  fingerprint.

The ranges bound every derived size: a signature row is at most
`8 + 2 * 4096` bytes, a band at most 4096 values and a query at most 256 band
keys, so no combination of parameters can overflow a size computation or drive
an unbounded allocation.

### Tokenizer

The tokenizer is recorded as a `lance.table.InvertedIndexDetails`, the message
the Full-Text Search index persists, and is rebuilt from that message alone at
query time. The supported subset is exactly what the message records:

| Fields                                                                                                                                                                | Role                                                                                    |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------|
| `base_tokenizer`, `language`, `max_token_length`, `lower_case`, `stem`, `remove_stop_words`, `ascii_folding`, `min_ngram_length`, `max_ngram_length`, `prefix_only`, `code_config` | Shape the token stream and are applied by the index.                                    |
| `with_position`, `block_size`, `document_granularity`, `posting_format_version`                                                                                       | Describe full text search postings; carried unchanged and ignored.                      |
| custom stop words, document-level text extraction (`lance_tokenizer`)                                                                                                 | Cannot be recorded in the message; a build that asks for them is rejected.              |

The default is the `simple` tokenizer with `language = English`,
`max_token_length = 40`, `lower_case = true`, `ascii_folding = true`,
`stem = false` and `remove_stop_words = false`.

The details identify every input of tokenization, in one of two ways:

- **Tokenizers whose data ships with Lance** (`simple`, `whitespace`, `raw`,
  `ngram`, `code`, `icu`, `icu/split`; the ICU segmentation data is compiled
  into Lance): their token stream for a given text and configuration is part
  of `signature_version`, and `tokenizer_fingerprint` is absent.
- **Tokenizers that load resources from the language model home**
  (`jieba`, `jieba/*`, `lindera/*`, which read the directory
  `LANCE_LANGUAGE_MODEL_HOME/<base_tokenizer>/`): the details carry
  `tokenizer_fingerprint`, the XXH64 hash (seed 0) of the directory's
  contents. The hash consumes, for every regular file below the directory in
  ascending order of its relative path bytes, following symbolic links: the
  relative path as UTF-8 bytes with `/` separators, one `0x00` byte, the file
  length as 8 little-endian bytes, and the file contents. The writer computes
  it when the index is built; a reader recomputes it from its own deployment
  when it opens the index and rejects a mismatch, since the query would be
  tokenized differently from the rows (the index must be rebuilt, or the
  resources restored). An implementation that does not compute the
  fingerprint rejects these tokenizers when an index is created and when one
  is opened.

## Signature Generation

The build side and the query side run the same procedure, so a query is
comparable with every stored row. All arithmetic below is on unsigned 64-bit
integers modulo `2^64`, and every multi-byte value is little-endian.

1. **Tokenize** the text with the configured tokenizer. A token is the UTF-8
   bytes of the token text the tokenizer emits, in emission order.
2. **Shingle**: a shingle is `shingle_size` consecutive tokens joined with the
   single byte `0x1F`; a text of `n` tokens has `n - shingle_size + 1`
   shingles, one starting at every token position. A text with fewer tokens
   than `shingle_size` but at least one token forms one shingle of all its
   tokens. A text with no tokens (NULL, empty, whitespace only, or every token
   discarded by the tokenizer) has no signature, is not indexed, and never
   appears in results.
3. **Hash** every shingle to 64 bits: `x = XXH64(shingle bytes, seed = 42)`.
4. **Coefficients** come from a SplitMix64 generator whose state `s` starts at
   42. One draw is:

    ```
    s    = s + 0x9E3779B97F4A7C15
    z    = s
    z    = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z    = (z ^ (z >> 27)) * 0x94D049BB133111EB
    draw = z ^ (z >> 31)
    ```

    The draws are consumed in this order: for `i = 0 .. num_hashes - 1`,
    `a_i = draw() | 1` and then `b_i = draw()`; after all pairs, for
    `i = 0 .. num_hashes - 1`, `c_i = draw() | 1`. The coefficients depend on
    `num_hashes` only.
5. **Permute and take the minima**: `m_i = min over all shingles x of
   (a_i * x + b_i)`, comparing the full 64-bit values.
6. **Compress** each minimum to 16 bits: `signature[i] = (c_i * m_i) >> 48`.
   A minimum shrinks as the text grows, so its raw high bits would agree
   between any two long texts; multiply-shift hashing keeps the chance that
   two different minima collide below `2^-15`, negligible against the
   `1 / num_hashes` resolution of the estimate.

The estimated Jaccard similarity of two signatures is the fraction of
positions whose values are equal; the reported `_distance` is
`1 - estimated similarity`, and rows at equal distance are ordered by row id.
Every constant of the procedure, and the band key below, is fixed by
`signature_version`; every segment of an index must carry identical details,
so signatures written by different segments or at query time are always
comparable. Changing a parameter, or upgrading a tokenizer dictionary,
requires rebuilding the index.

### Band Keys

The signature is split into `num_bands` bands of `r = num_hashes / num_bands`
consecutive values; band `j` holds `signature[j * r .. (j + 1) * r)`. Its key
is

```
band_key_j = (j << 56) | (XXH64(band bytes, seed = 42) & 0x00FFFFFFFFFFFFFF)
```

where the band bytes are the band's `r` values, each written as 2 little-endian
bytes, in signature order. The band number in the high byte keeps the keys of
one band contiguous in the sorted bands table.

### Known-Answer Vectors

An implementation of the procedure must reproduce these values, computed with
`num_hashes = 4`, `num_bands = 2`, `shingle_size = 2`, the default tokenizer
and `signature_version = 0`.

Coefficients:

| `i` | `a_i`              | `b_i`              | `c_i`              |
|:----|:-------------------|:-------------------|:-------------------|
| 0   | `0xbdd732262feb6e95` | `0x28efe333b266f103` | `0x5705b8770b3d7dd5` |
| 1   | `0x47526757130f9f53` | `0x581ce1ff0e4ae394` | `0x9e54d738297f77af` |
| 2   | `0x09bc585a244823f3` | `0xde4431fa3c80db06` | `0x3474724a775b19bf` |
| 3   | `0x37e9671c45376d5d` | `0xccf635ee9e9e2fa4` | `0x7e348a0e451650bf` |

Text `"The quick brown fox"` tokenizes to `the`, `quick`, `brown`, `fox`:

| Shingle bytes           | `XXH64`            |
|:------------------------|:-------------------|
| `the` `0x1F` `quick`    | `0x4c0b0d36203ce55f` |
| `quick` `0x1F` `brown`  | `0x09c7ae895d82b184` |
| `brown` `0x1F` `fox`    | `0x68e7f4b1be7b4c1f` |

| `i` | `m_i`              | `signature[i]` |
|:----|:-------------------|:---------------|
| 0   | `0x0959767d77eafad7` | `0xb00f`         |
| 1   | `0x464e4457275cd2a1` | `0x59d6`         |
| 2   | `0x50f41804abaa5973` | `0x511a`         |
| 3   | `0x8944bd01067b09e7` | `0xcd2c`         |

Band keys: `0x00f30f3501d458ea` (band 0), `0x01aec5df28d04915` (band 1).

Text `"Fox"` has one token, fewer than `shingle_size`, so its only shingle is
`fox` with `XXH64` `0x07c0130ab04388d5`; the minima are `0x2bb8ade505081afc`,
`0x0a273c6ff9a78ba3`, `0x802f20573838dc35`, `0x1ef46a8d372c9605`, the signature is
`0x2970 0xd6db 0x65f8 0xadfc`, and the band keys are `0x003cf6e803312eca` and
`0x017c5504fbc83f36`.

The texts `""` and `"  \n"` have no tokens and therefore no signature.

## Storage Layout

Each segment consists of two Lance files. The details in the index metadata
are authoritative; both files repeat them as schema metadata so that a file
that does not belong to the details is detected when the segment is opened.

### Signature File Schema

`signatures.lance` holds one row per indexed document. Row `i` is document
`i` of the segment: the row number is the segment-local document id, so a
segment holds at most `2^32` documents, and larger tables use several
segments.

```python
pa.schema(
    [
        pa.field("_rowid", pa.uint64(), nullable=False),
        pa.field(
            "signature",
            pa.list_(pa.field("item", pa.uint16(), nullable=False), num_hashes),
            nullable=False,
            metadata={
                "lance-encoding:structural-encoding": "fullzip",
                "lance-encoding:compression": "none",
            },
        ),
    ],
    metadata={
        "minhash_lsh_details": "<hexadecimal serialized MinHashLshIndexDetails>",
        "minhash_lsh_index_version": "0",
    },
)
```

`_rowid` is the row id of the indexed row as the dataset hands it to the
index: the row address, or the stable row id when the dataset has stable row
ids enabled, like every scalar index. `signature` is the signature of
[Signature Generation](#signature-generation). The field metadata of
`signature` selects the full-zip structural encoding without compression, so
every row occupies the same `2 * num_hashes` bytes and a reader fetches the
signature of document `i` by row number as one ranged read. Rows are written
in document id order and never reordered.

### Bands File Schema

`bands.lance` holds one row per (band key, document id) pair, sorted by
`band_key` and then `doc_id`, ascending. A bucket is the run of rows sharing
a key.

```python
pa.schema(
    [
        pa.field(
            "band_key",
            pa.uint64(),
            nullable=False,
            metadata={"lance-encoding:compression": "none"},
        ),
        pa.field(
            "doc_id",
            pa.uint32(),
            nullable=False,
            metadata={"lance-encoding:compression": "none"},
        ),
    ],
    metadata={
        "minhash_lsh_details": "<hexadecimal serialized MinHashLshIndexDetails>",
        "minhash_lsh_index_version": "0",
        "minhash_lsh_num_docs": "<decimal document count>",
        "minhash_lsh_page_rows": "4096",
        "minhash_lsh_page_table_buffer": "<decimal global buffer index>",
    },
)
```

`band_key` is the key of [Band Keys](#band-keys); `doc_id` is the
segment-local document id, the row number in `signatures.lance`. Both
columns are stored without compression, so any run of rows is one ranged
read per column. The file is divided into logical pages of
`minhash_lsh_page_rows` rows: page `p` holds rows
`[p * page_rows, (p + 1) * page_rows)`, and the last page may be shorter. A
bucket may span any number of pages.

### Schema Metadata

| Key                             | File       | Value                                                                                                                                                                                                |
|:--------------------------------|:-----------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `minhash_lsh_details`           | both       | Lower-case hexadecimal encoding of the serialized `MinHashLshIndexDetails` the segment was built from. A reader decodes it and requires it to describe, field by field, the same details as the index metadata. |
| `minhash_lsh_index_version`     | both       | Decimal file layout version; `0` for the layout described here. Independent of `signature_version`.                                                                                                 |
| `minhash_lsh_num_docs`          | `bands`    | Decimal number of documents of the segment; equals the row count of `signatures.lance`.                                                                                                             |
| `minhash_lsh_page_rows`         | `bands`    | Decimal number of rows per logical page; positive. Writers use `4096`.                                                                                                                              |
| `minhash_lsh_page_table_buffer` | `bands`    | Decimal index of the global buffer of `bands.lance` that holds the page table.                                                                                                                      |

### Page Table

The page table is a global buffer of `bands.lance` holding
`ceil(num_rows / page_rows)` little-endian UInt64 values; entry `p` is the
largest `band_key` of page `p`, that is the key of its last row. It is read
when the segment is opened and kept in memory while the index is open. A
bucket for key `K` starts in the first page whose entry is `>= K` and ends in
the first page whose entry is `> K` (or the last page when there is none);
the pages strictly between hold nothing but that bucket. Both positions are
binary searches over the table, so a bucket's page range is known without
reading the file.

### Opening a Segment

A reader opens a segment in this order and treats a failed check as
corruption of the named file, except where noted:

1. Parse and validate the index details ([Validation](#validation)). When
   they carry a `tokenizer_fingerprint`, recompute it from the deployment
   and reject a mismatch as unsupported, not as corruption.
2. Open both files. In each, the schema must match its definition above
   exactly in field names, types, nullability (including the list item) and
   list size, which must equal `num_hashes`; `minhash_lsh_details` must be
   present, decode, validate and equal the index details field by field; and
   `minhash_lsh_index_version` must be present. A version greater than the
   one the reader implements is rejected as unsupported, not as corruption.
3. In `bands.lance`, `minhash_lsh_page_rows` must be positive and
   `minhash_lsh_num_docs` must equal the row count of `signatures.lance`.
4. Read the page table buffer; its length must be a multiple of 8 and its
   entry count must be `ceil(num_rows / page_rows)` for the row count of
   `bands.lance`.

While answering queries, a page that does not decode to non-null `UInt64`
and `UInt32` columns, and a `doc_id` that is not below
`minhash_lsh_num_docs`, are corruption of `bands.lance`.

The fragments a segment covers are recorded in the index metadata of the
dataset, never derived from the stored `_rowid` values (which do not identify
fragments once stable row ids are enabled).

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
2. In every segment, each band key is looked up through the
   [page table](#page-table), which yields the first and the last page of the
   key's bucket. The boundary pages of all buckets are fetched in one
   scattered read; the pages in between, when a bucket spans more than two
   pages, are streamed in bounded windows. The document ids in the buckets
   form the candidate set.
3. The candidates' signatures are read — from memory when resident, otherwise
   with scattered reads, or with a sequential scan when the candidates cover a
   large share of the segment — and each candidate's Jaccard distance to the
   query is computed. Rows removed by filters or deletions are skipped.
4. Each segment keeps its `limit` closest rows; rows not covered by any
   segment are scored the same way by the query engine.
5. The per-segment results are merged by distance, and the `limit` closest
   rows are returned with their `_distance`.
